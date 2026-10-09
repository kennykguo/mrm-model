"""WDM bank supervisor: line identification, assignment, supply-limit sequencing, barrel shift (PAT-1) and
sequential acquisition of the rings of one bus (ARCHITECTURE section 6 item 10; 07 P1-12, P1-1, P0-1).

Sources: PAT-1 (US20240405908A1: heaters only heat; when the power to hold a map is excessive, lower the heater power
so that every ring moves one channel and the map rotates, the wrap-around ring taking another order), PAT-3 (ramp to
the monitor maximum), [HAT24] (rings assigned sequentially; a ring placed on an occupied line shows a much smaller
rise), [CHOI] (lock-to-deterministic / cyclic / any; sequential tuning fails by "stealing"), 07 P1-12 (supply limit
``array.heater_total_power_max_w`` / ``heater_supply_current_max_a``, enforced by the plant with a droop of every
heater), docs/handoff/integrator.md section 4 (the cold-ring hazard).

The hazard (g26 nominal, measured on the reference plant): every ring's cold resonance sits 0.73 nm (about 2 FWHM)
blue of the line three channels up, so at laser turn-on rings self-heat onto a FOREIGN line (in the nominal world
seven of the eight rings sit 0.2 nm red of line n + 3 with the heaters off). A PAT-3 ramp-to-the-first-maximum then
locks every ring on a wrong line, and two rings can take the same line, the downstream one starved and lost
(tests/verification/test_ctrl_b_ramp.py). A heater-up sweep of one FSR crosses every line once; with the g26 grid
(8 lines at 400 GHz, FSR 1.6-2.0 THz) the crossings alternate at about half a grid step and leave ONE empty gap per
FSR (between lines 4 and 3, where a ninth line would be), which identifies every crossing.

Phases, per batch element (vectorized ring FSMs of mrm.control.baselines.ramp_acquire underneath)
-------------------------------------------------------------------------------------------------
WAIT      start delay (the laser turn-on settles).
MAP       cold start with a remapping policy (cyclic / any) only: the bank map that plans the assignment. Rings in
          BUS ORDER, one at a time (supply sequencing: one sweeping heater): the ring sweeps up from w = 0 and
          records every resonance it passes until L + 1 (one FSR) or ``map_w_max``; it is then parked from below
          at the floor after its first resonance, between lines (a drive below the first resonance is
          history-dependent: a line whose hot branch reaches w = 0 holds the ring after the heater drops). Upstream
          rings are therefore off-line while a downstream ring maps; a ring captured on a foreign line at turn-on is
          released by its own sweep. The maps are labelled jointly (label_bank_by_pattern: one FSR and curvature
          for the bank).
ASSIGN    policy ``fixed`` (spec.ring_to_line), ``cyclic`` (PAT-1 barrel shifts of the declared map: the declared
          map when the predicted lock currents fit the supply budget, else the feasible shift with the smallest
          total drive, lock-to-cyclic) or ``any`` (minimum total drive, a linear assignment). ``cyclic``/``any`` need
          ``tx.assignment`` bound to the controller (07 P1-12); otherwise the policy is ``fixed``. A warm start
          takes the resonances and labels of the factory table (mrm.control.calibration.factory_cal).
ACQUIRE   rings in bus order, one at a time. Cold start: a FRESH map of the ring first (the resonance comb drifts
          by up to half a spacing between a bank map and an acquisition: the rings locked before it heat it through
          the crosstalk, the ms tails settle, so neither a stored window nor a count from a park is reliable; the
          fresh map, labelled with the bank fit's FSR and curvature, is), then an up-search from 0 in a window of
          +-half a spacing around the target resonance, the true peak on the DOWN scan, the re-approach and the
          red-side lock (ramp_acquire). Warm start: the window around the table's resonance, shifted by the table's
          temperature correction and the drift of the rings acquired so far, reaching 0.75 of the spacing below
          and 0.35 above (crosstalk makes resonances come early). The projected supply (the codes held now + this
          ring at its target) is checked before the ring starts; a ring that does not fit is left at code 0 and
          flagged ``supply_limited``. Verification (wrong line, [HAT24]): the acquired peak must reach
          ``verify_height_frac`` of the mapped height of that resonance (a line taken upstream shows a small peak),
          else the ring is flagged ``wrong_line`` and re-acquired once on the next order of its line; and the lock
          and the resonance passed just below it must sit where the map put them (:meth:`_verify_lock`), else the
          lock is not vouched for. Telemetry ``line_confident`` is the only flag that vouches for a ring's line
          (07 P1-20); in sampled worlds about half the rings end on a foreign line, never vouched for
          (docs/handoff/controllers_b.md section 6).
TRACK     every ring on its red-side loop. PAT-1 online: when a locked ring's heater stays above
          ``excess_code_frac`` of full scale, or the estimated supply above the budget, for ``excess_steps`` steps,
          the cyclic policy rotates the map by one channel towards lower power (every ring to the next bluer line,
          the wrap-around ring one order down) and re-acquires ring by ring, each window shifted by that ring's own
          drift since its acquisition; with the fixed policy the event is only flagged.

Everything is measured (monitor codes, the controller's own heater codes, the temperature sensor through the
factory table); the nominal design is used only through the knowledge gate (heater law and supply limits for the
budget, FSR window for the pattern fit). Truth and sampled parameters are never read (07 P1-1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mrm.control.api import BaseController, ControllerSpec, Observation
from mrm.control.baselines.ramp_acquire import (ST_FAIL, ST_HOLD, ST_LOCK, ST_PARK, RampFSM, RampParams,
                                                monitor_mean)
from mrm.control.calibration import (FactoryCalTable, code_of_w, drop_first, fsr_prior_hz, heater_current_a,
                                     heater_power_w, label_bank_by_pattern, label_lines_by_pattern, law_exponents,
                                     with_first)
from mrm.control.registry import register

__all__ = ["WdmParams", "WdmSupervisor", "PH_WAIT", "PH_MAP", "PH_ASSIGN", "PH_ACQUIRE", "PH_TRACK", "PHASE_NAMES"]

PH_WAIT, PH_MAP, PH_ASSIGN, PH_ACQUIRE, PH_TRACK = 0, 1, 2, 3, 4
PHASE_NAMES = ("wait", "map", "assign", "acquire", "track")


@dataclass
class WdmParams(RampParams):
    """:class:`RampParams` (per-ring acquisition and lock) plus the supervisor's settings (module docstring).

    policy: fixed | cyclic | any (cyclic and any need tx.assignment bound to the controller); map_w_max: top of the
    map sweep (drive units): a map records L + 2 resonances (the first is left out of the line fit, being often a
    landing from a state captured at the sweep start), so it must pass one FSR plus two resonances (0.68-0.75 at
    g26 nominal; 0.85 by default); map_dw: map sweep step (None: fwhm_w / 2.5 from the nominal linewidth, else
    0.006); supply_margin: usable fraction of the nominal supply limits; excess_code_frac,
    excess_steps: PAT-1 "excessive power" test in TRACK; verify_height_frac: wrong-line test of an acquired peak
    against its mapped height; window_frac: half-width of the acquisition window above the target resonance, as a
    fraction of the local resonance spacing; start_delay_s: wait before the map (laser turn-on), counted from the
    first step after a reset; w_min_target: lowest drive a lock target may have (0, off, by default). PAT-1 [0050]
    keeps heater headroom so the heater can back off when the chip warms (a target at w ~ 0 is often a partial lock:
    its peak below w = 0), which a positive floor (e.g. 3 nominal linewidths, ~0.045 at g26) buys by taking such a
    line one order up; but under a tight supply the lowest-power map IS the one at w ~ 0 (the 0.10 A barrel shift of
    tests/verification/test_ctrl_b_wdm_slow.py: with a 0.045 floor three rings could not be placed), so it is off.
    The online PAT-1 rotation (towards lower power) never applies it.
    """
    policy: str = "cyclic"
    map_w_max: float = 0.85
    map_dw: float | None = None
    supply_margin: float = 0.95
    excess_code_frac: float = 0.97
    excess_steps: int = 200
    verify_height_frac: float = 0.5
    window_frac: float = 0.5
    start_delay_s: float = 100e-6
    w_min_target: float = 0.0


@register("wdm_supervisor")
class WdmSupervisor(BaseController):
    """WDM bank supervisor (module docstring). Writes ``tx.heater`` and, when bound, ``tx.assignment``.

    ``WdmSupervisor(params, factory_table=table)`` or :meth:`load_factory_cal` before ``reset`` gives the warm start
    (no map phase). Telemetry (B, N): state (ring FSM state), phase, locked (the ring's loop is closed), detuning_est_pm,
    integrator, integrator_saturated, target_line, assigned_line, map_events, map_ok (the line labels of the ring
    come from a confident pattern fit, or from a factory table entry marked ok), wrong_line (the acquired peak failed
    the height check), line_confident (locked and map_ok and not wrong_line: a lock the supervisor vouches for; a
    ``locked`` ring with line_confident 0 sits on a line it could not verify, 07 P1-20), supply_limited,
    captured_at_start, shift (barrel shift in use), p_est_mw (nominal heater power estimate, NaN without
    heater_nominal knowledge).
    """
    name = "wdm_supervisor"
    Params = WdmParams

    def __init__(self, params=None, *, factory_table: FactoryCalTable | None = None, **overrides):
        super().__init__(params, **overrides)
        self.table = factory_table

    @classmethod
    def from_params(cls, params):
        return cls(dict(params or {}))

    def load_factory_cal(self, table: FactoryCalTable | None) -> None:
        """Warm start (07 P1-1): use this factory table at the next reset (None = cold start)."""
        self.table = table

    # ------------------------------------------------------------------------------------------------ reset
    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        p = self.params
        if p.policy not in ("fixed", "cyclic", "any"):
            raise ValueError(f"WdmSupervisor: policy {p.policy!r} must be fixed, cyclic or any")
        self.fsm = f = RampFSM(spec, p)
        self.d = f.design
        B, N, L = spec.n_batch, spec.n_rings, spec.n_lines
        self.B, self.N, self.L = B, N, L
        self.bound_assignment = "tx.assignment" in spec.actuators
        self.policy = p.policy if self.bound_assignment else "fixed"
        r2l = spec.ring_to_line if spec.ring_to_line is not None else tuple(range(N))
        self.declared = np.array([int(x) % L for x in r2l], dtype=np.int64)
        self.order = [int(x) for x in spec.bus_order]
        self.dw_map = float(p.map_dw) if p.map_dw is not None else (self.d.fwhm_w / 2.5 if self.d.fwhm_w else 0.006)
        # the acquisition's up-search passes several resonances (COUNT search) at the map's step; the peak itself
        # is measured on the slower DOWN scan (dw_down)
        self.dw_acq = max(f.dw_up0, self.dw_map) if p.dw_up is None else f.dw_up0
        self.delay_ticks = int(round(p.start_delay_s / spec.tick_s))
        self.t0: int | None = None                    # tick of the first step after reset (start delay origin)
        self._synced = False                          # codes taken from tx.heater.applied at the first step
        self.fits: list = [None] * B                  # bank pattern fit of each world (cyclic / any cold start)
        self.fresh_fits: list = [[] for _ in range(B)]   # (fsr, exponent, curvature) of confident fresh-map fits
        self.exponents = law_exponents(self.d, f.q)   # drive-law exponents tried by the pattern fits
        self.w_min_target = float(p.w_min_target)
        z = lambda v=0.0, dt=np.float64: np.full((B, N), v, dtype=dt)
        self.phase = np.full(B, PH_WAIT, np.int64)
        self.cursor = np.zeros(B, np.int64)          # index into the bus order (MAP and ACQUIRE)
        self.busy = np.zeros(B, bool)                 # the cursor ring has been started
        self.events = [[np.zeros(0) for _ in range(N)] for _ in range(B)]
        self.ev_floor_w = [[np.zeros(0) for _ in range(N)] for _ in range(B)]
        self.ev_height = [[np.zeros(0) for _ in range(N)] for _ in range(B)]
        self.labels = [[np.zeros(0, np.int64) for _ in range(N)] for _ in range(B)]
        self.map_ok = z(False, bool)
        self.map_events = z(0, np.int64)
        self.target_line = z(-1, np.int64)
        self.target_ev = z(-1, np.int64)
        self.assigned = np.repeat(self.declared[None, :], B, axis=0).astype(np.int64)
        self.shift = np.zeros(B, np.int64)
        self.wrong = z(False, bool)
        self.supply_limited = z(False, bool)
        self.captured = z(False, bool)
        self.retried = z(False, bool)
        self.w_lock0 = z(np.nan)
        self.excess = np.zeros(B, np.int64)
        self.drift = np.zeros(B)                      # drive drift since the map, from the rings acquired so far
        self.drifts = [[] for _ in range(B)]
        self.drift_ring = z(np.nan)                   # per-ring drift measured at an online barrel shift
        self.retarget = np.zeros(B, bool)             # ACQUIRE after an online barrel shift (rings move one by one)
        self.drift_used = z(0.0)                      # drift applied to each ring's last acquisition window
        self.moved = z(False, bool)                   # the supervisor has moved the ring since the reset
        self.fresh = z(False, bool)                   # the ring's map was taken right before its acquisition
        self.mapping = z(False, bool)
        self.map_capped = z(False, bool)              # the fresh map was cut short by the supply budget
        self.drift_meas = z(np.nan)                   # comb drift measured at the lock (_verify_lock)
        self.acq_uncertain = z(False, bool)           # the lock does not sit where the map says (not vouched)
        self.y = z(np.nan)
        self.t_die_c = np.full(B, np.nan)
        # every ring holds its applied code until the supervisor moves it
        f.state[:] = ST_HOLD

    # ------------------------------------------------------------------------------------------------ helpers
    def _power(self, codes) -> np.ndarray | None:
        if self.d.heater_law is None or self.d.p_drive_fs_w is None:
            return None
        return heater_power_w(codes, self.d)

    def _current(self, codes) -> np.ndarray | None:
        if self.d.heater_law is None or not self.d.r_cold_ohm:
            return None
        return heater_current_a(codes, self.d)

    def _budget_ok(self, codes_b: np.ndarray) -> bool:
        """Nominal supply estimate of one element's heater codes against the margin x limit (True if unknown)."""
        p, i = self._power(codes_b), self._current(codes_b)
        ok = True
        if p is not None and self.d.supply_power_max_w is not None:
            ok &= float(np.sum(p)) <= self.params.supply_margin * self.d.supply_power_max_w
        if i is not None and self.d.supply_current_max_a is not None:
            ok &= float(np.sum(i)) <= self.params.supply_margin * self.d.supply_current_max_a
        return bool(ok)

    def _w_cap(self, b, n, w_max: float) -> float:
        """Largest drive of ring n (<= w_max) that keeps the element's nominal supply estimate within the budget
        with every other ring at its present code (w_max when the supply is unknown)."""
        f = self.fsm
        codes = f.code[b].copy()
        ws = np.linspace(0.0, w_max, 41)
        ok_w = 0.0
        for w in ws:
            codes[n] = int(code_of_w(w, f.code_max, f.q))
            if not self._budget_ok(codes):
                break
            ok_w = float(w)
        return ok_w

    def _ring_events(self, b, n):
        """(w (K,), line (K,), floor_w (K,), height (K,)) of ring n: factory table or own map."""
        if self.table is not None:
            w, ln = self.table.events(b, n)
            fl = np.full(w.size, np.nan)
            sel = np.isfinite(self.table.w[b, n])
            o = np.argsort(self.table.w[b, n][sel])
            h = (self.table.height[b, n] - self.table.floor[b, n])[sel][o]     # rise above the floor, as a map
            if w.size > 1:
                fl[1:] = w[:-1] + 0.4 * np.diff(w)            # the table has no floor drives: 40 % into the gap
            fl[0] = max(w[0] - 0.6 * (np.diff(w).min() if w.size > 1 else 0.05), 0.0) if w.size else 0.0
            shift = self._table_shift(b)
            return w + shift, ln, np.maximum(fl + shift, 0.0), h
        return self.events[b][n], self.labels[b][n], self.ev_floor_w[b][n], self.ev_height[b][n]

    def _table_shift(self, b) -> float:
        """Drive shift of the factory table to the present die temperature (nominal dlambda/dT and tuning)."""
        t = self.table
        d = self.d
        if (t is None or not np.isfinite(self.t_die_c[b]) or not np.isfinite(t.t_cal_c[b]) or not d.s0_m_per_k
                or not d.tuning_m_per_w_drive):
            return 0.0
        return -(self.t_die_c[b] - float(t.t_cal_c[b])) * d.s0_m_per_k / d.tuning_m_per_w_drive

    def _target_index(self, b, n, line, w_min=None, w_below=None):
        """Index of the resonance of ``line`` to acquire: the lowest at or above w_min (default: the headroom floor
        ``w_min_target``), or (w_below) the highest below that drive."""
        if w_min is None:
            w_min = self.w_min_target
        w, ln, _, _ = self._ring_events(b, n)
        idx = np.flatnonzero(ln == line)
        if idx.size == 0:
            return -1
        if w_below is not None:
            cand = idx[w[idx] < w_below]
            return int(cand[np.argmax(w[cand])]) if cand.size else -1
        cand = idx[w[idx] >= w_min]
        return int(cand[np.argmin(w[cand])]) if cand.size else -1

    def _plan_assignment(self, b) -> None:
        """Choose the map (policy) and the target resonance of every ring for element b."""
        N, L = self.N, self.L
        cost = np.full((N, L), np.inf)
        evi = np.full((N, L), -1, np.int64)
        for n in range(N):
            w, ln, _, _ = self._ring_events(b, n)
            for m in range(L):
                k = self._target_index(b, n, m)
                if k >= 0:
                    cost[n, m] = w[k]
                    evi[n, m] = k

        def feasible(lines):
            if np.any(~np.isfinite(cost[np.arange(N), lines])):
                return False
            # the lock sits a little above the resonance found by the up-search: same margin as ACQUIRE's check
            codes = code_of_w(cost[np.arange(N), lines] + 2.0 * self.dw_acq, self.fsm.code_max, self.fsm.q)
            return self._budget_ok(codes)

        lines = self.declared.copy()
        shift = 0
        if self.policy == "cyclic":
            # PAT-1: the next shift towards lower heater power is +1 (every ring one channel bluer); try the
            # declared map first, then the shifts in order of their total drive
            shifts = sorted(range(L), key=lambda s: (s != 0, float(np.sum(np.where(
                np.isfinite(cost[np.arange(N), (self.declared + s) % L]), cost[np.arange(N), (self.declared + s) % L],
                10.0)))))
            for s in shifts:
                cand = (self.declared + s) % L
                if feasible(cand):
                    lines, shift = cand, s
                    break
            else:
                # no shift fits the budget: the one with the least total drive (the supply sequencing of ACQUIRE
                # then flags the rings that do not fit) rather than the declared map, which may be the costliest
                tot = [float(np.sum(cost[np.arange(N), (self.declared + s) % L])) for s in range(L)]
                if np.isfinite(min(tot)):
                    shift = int(np.argmin(tot))
                    lines = (self.declared + shift) % L
        elif self.policy == "any":
            from scipy.optimize import linear_sum_assignment
            c = np.where(np.isfinite(cost), cost, 1e3)
            r, col = linear_sum_assignment(c[:, :L])
            lines = np.full(N, -1, np.int64)
            lines[r] = col
        self.shift[b] = shift
        self.target_line[b] = lines
        self.assigned[b] = lines
        self.target_ev[b] = evi[np.arange(N), np.clip(lines, 0, L - 1)]

    def _start_ring(self, b, n, from_zero: bool = False) -> bool:
        """Start the acquisition of ring n of element b in a window around its target resonance k.

        The window is half the local spacing on each side of the resonance's drive plus the drift estimate: zero for
        a fresh map taken just before (cold start), the ring's own drift for an online barrel shift, else the median
        drift of the rings acquired so far. With a factory table (warm start) the window reaches 0.75 of the spacing
        below and 0.35 above, since the rings acquired before this one heat it through the crosstalk (resonances
        come at lower drive). ``from_zero``: drop to code 0 and search up from there (the state reached at a drive
        depends on the approach under bistability; an up-search from 0 meets the target from blue like the map's
        sweep did); otherwise step to half a spacing below the window.
        """
        f = self.fsm
        k = int(self.target_ev[b, n])
        w, _, _, _ = self._ring_events(b, n)
        if k < 0 or k >= w.size:
            return False
        sp = np.diff(w)
        med = float(np.median(sp)) if sp.size else 0.05
        below = float(w[k] - w[k - 1]) if k >= 1 else med
        above = float(w[k + 1] - w[k]) if k + 1 < w.size else med
        dr = self.drift_ring[b, n]
        if self.fresh[b, n] and not self.retarget[b]:
            used = 0.0
        else:
            used = dr if np.isfinite(dr) else float(self.drift[b])
        self.drift_used[b, n] = used
        c = float(w[k] + used)
        lo_frac, hi_frac = (0.75, 0.35) if (self.table is not None and not self.retarget[b]) else \
            (self.params.window_frac, self.params.window_frac)
        lo = max(c - lo_frac * below, 0.0)
        hi = c + hi_frac * above
        mask = np.zeros((self.B, self.N), bool)
        mask[b, n] = True
        cur = float(f.w[b, n])
        start = max(lo - 0.5 * below, 0.0)
        if from_zero:
            f.start(mask, w_from=0.0, w_lo=lo, w_hi=hi)
        else:
            f.start(mask, w_from=cur if start <= cur < lo else start, w_lo=lo, w_hi=hi)
        self.moved[b, n] = True
        self.acq_uncertain[b, n] = False
        return True

    def _verify_lock(self, b, n) -> None:
        """Check a fresh lock of ring n against its map (cold start) or table (warm start); sets ``acq_uncertain``.

        Under a common drift d of the ring's comb since the map, the resonance found must sit at w[k] + d and the
        last resonance the up-search passed below it (``RampFSM.sk_w``) at w[k - 1] + d, with d within 0.3 of the
        target's local spacing of the drift the window was centred with (0 after a fresh map). A ring warmer than
        during its map (the ms tail of the map's own excursion, strong self-heating: up to a whole spacing on
        sampled g26 worlds) finds a neighbour of the target inside the window; then the passed resonance is
        displaced from w[k - 1] even when the found one looks right (window-A pairs, where neighbours are 1 nm
        apart). Such a lock is not vouched for (telemetry line_confident 0, 07 P1-20). Only the resonances next to
        the target are compared: an up-search from w = 0 often sees one or two extra resonances at its start that
        the map did not (the start capture depends on the ring's temperature). The window itself is not moved: on
        the near-periodic g26 pattern a drift of one spacing cannot be told from a start one resonance later (tried
        in review: moving the window broke worlds that the fixed window gets right)."""
        f = self.fsm
        k = int(self.target_ev[b, n])
        w, _, _, _ = self._ring_events(b, n)
        if k < 0 or k >= w.size or not np.isfinite(f.w_event[b, n]):
            return
        sp = np.diff(w)
        med = float(np.median(sp)) if sp.size else 0.05
        below = float(w[k] - w[k - 1]) if k >= 1 else med
        above = float(w[k + 1] - w[k]) if k + 1 < w.size else med
        tol = 0.3 * min(below, above)
        used = float(self.drift_used[b, n])
        we = float(f.w_event[b, n])
        bad = abs(we - w[k] - used) > tol
        cnt = int(min(f.sk_n[b, n], f.sk_w.shape[2]))
        sv = f.sk_w[b, n, :cnt]
        sv = sv[np.isfinite(sv) & (sv < we)]
        if k >= 1 and sv.size:
            s_last = float(sv.max())
            self.drift_meas[b, n] = s_last - float(w[k - 1])
            bad |= abs(self.drift_meas[b, n] - used) > tol
        else:
            self.drift_meas[b, n] = we - float(w[k])
        self.acq_uncertain[b, n] = bool(bad)

    # ------------------------------------------------------------------------------------------------ step
    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        p = self.params
        f = self.fsm
        y, valid, clip = monitor_mean(obs, None if np.all(np.isnan(self.y)) else self.y)
        self.y = y
        if self.spec.temp_sensors:
            ts = self.spec.temp_sensors[0]
            code = np.asarray(obs.sensors[ts.channel], dtype=float)[:, 0]
            got = np.asarray(obs.new[ts.channel], dtype=bool).reshape(self.B, -1)[:, 0] & (code > 0)
            self.t_die_c = np.where(got, code * ts.lsb_k + ts.offset_c, self.t_die_c)
        if not self._synced:
            app = np.asarray(obs.sensors["tx.heater.applied"], dtype=np.int64)
            f.code = app.copy()
            f.u = app.astype(float)
            self._synced = True
            self.t0 = int(obs.tick)
        for b in range(self.B):
            ph = self.phase[b]
            if ph == PH_WAIT:
                if obs.tick - self.t0 >= self.delay_ticks:
                    # the bank map is needed only to plan a remap (cyclic / any); a fixed map is acquired ring by
                    # ring from fresh maps, a warm start from the factory table
                    need_map = self.table is None and self.policy != "fixed"
                    self.phase[b] = PH_MAP if need_map else PH_ASSIGN
                    self.cursor[b] = 0
                    self.busy[b] = False
            if self.phase[b] == PH_MAP:
                self._map_step(b, y)
            if self.phase[b] == PH_ASSIGN:
                if self.table is None and self.policy != "fixed":
                    self._label(b)
                if self.table is None and self.policy == "fixed":
                    self.target_line[b] = self.declared
                    self.assigned[b] = self.declared
                    self.target_ev[b] = -1                      # found in the ring's fresh map
                else:
                    self._plan_assignment(b)
                if self.table is not None:
                    self.map_ok[b] = np.asarray(self.table.ok[b], bool)     # the factory labels are the map
                f.dw_up[b] = self.dw_acq
                # cold start: the rings stay at their map parks (between lines, low power), the state their count
                # refers to; warm start: they have not moved since the laser turn-on
                self.drift[b] = 0.0
                self.phase[b] = PH_ACQUIRE
                self.cursor[b] = 0
                self.busy[b] = False
            if self.phase[b] == PH_ACQUIRE:
                self._acquire_step(b)
            if self.phase[b] == PH_TRACK:
                self._track_step(b)
        codes = f.step(y, valid, clip)
        out = {"tx.heater": codes}
        if self.bound_assignment:
            out["tx.assignment"] = self.assigned.astype(np.int64)
        return out

    def _map_step(self, b, y):
        f = self.fsm
        if self.cursor[b] >= self.N:
            self.phase[b] = PH_ASSIGN
            return
        n = self.order[self.cursor[b]]
        mask = np.zeros((self.B, self.N), bool)
        mask[b, n] = True
        if not self.busy[b]:
            f.dw_up[b] = self.dw_map
            f.start_map(mask, w_from=0.0, w_hi=self.params.map_w_max, n_events=self.L + 2)
            self.busy[b] = True
            return
        if f.map_done[b, n] and f.state[b, n] == ST_HOLD:
            k = self._take_map(b, n)
            # park between lines, approached from below: at the floor before the SECOND resonance. The state at a
            # drive below the first resonance depends on history (a line whose hot branch reaches w = 0 holds the
            # ring after the heater drops), while the floor after the first resonance is reached on that line's far
            # red tail from any state at w = 0
            fw = self.ev_floor_w[b][n]
            park = float(fw[1]) if k >= 2 and np.isfinite(fw[1]) else 0.0
            f.park_from_below(mask, park)
            self.moved[b, n] = True
            self.cursor[b] += 1
            self.busy[b] = False
            if self.cursor[b] >= self.N:
                self.phase[b] = PH_ASSIGN

    def _take_map(self, b, n) -> int:
        """Store ring n's map (drives, floor drives, heights above the floor) from the FSM; flag a ring that sat
        on a line at its start (the monitor high at w = 0, then falling: captured by self-heating). Returns the
        number of resonances."""
        f = self.fsm
        k = int(min(f.ev_n[b, n], f.ev_w.shape[2]))
        w = f.ev_w[b, n, :k].copy()
        o = np.argsort(w)
        self.events[b][n] = w[o]
        self.ev_floor_w[b][n] = f.ev_wfloor[b, n, :k][o].copy()
        self.ev_height[b][n] = (f.ev_y[b, n, :k] - f.ev_floor[b, n, :k])[o].copy()
        self.map_events[b, n] = k
        ys = f.y_start[b, n]
        hmed = float(np.median(self.ev_height[b][n])) if k else np.inf
        fl = float(np.nanmin(f.ev_floor[b, n, :k])) if k else ys
        self.captured[b, n] |= bool(np.isfinite(ys) and ys - fl > 0.3 * hmed)
        return k

    def _label_fresh(self, b, n) -> None:
        """Line labels of ring n's fresh map: the bank fit's FSR, exponent and curvature narrow the search when a
        bank map was taken (cyclic / any); else, once two rings of this world have confident fresh-map fits, their
        median FSR, exponent and curvature narrow it (one design, one heater family: the same narrowing as the bank
        fit, built up ring by ring); else the knowledge-gated FSR window. The narrowing removes labellings that need
        another FSR, which a single map's ambiguity test counts against it (g26 nominal: ring 2 at 1.96 alone)."""
        grid = self.spec.line_grid_hz
        w = self.events[b][n]
        if grid is None or w.size < 3:
            self.labels[b][n] = np.full(w.size, -1, np.int64)
            self.map_ok[b, n] = False
            return
        bank = self.fits[b]
        wf = drop_first([w], self.L)[0]
        ref = None
        if bank is not None and bank[n].get("ok", False) and np.isfinite(bank[n].get("fsr_hz", np.nan)):
            ref = (float(bank[n]["fsr_hz"]), float(bank[n].get("exponent", 1.0)), float(bank[n]["curvature"]))
        elif len(self.fresh_fits[b]) >= 2:
            ff = np.array(self.fresh_fits[b])
            gm = float(np.median(ff[:, 1]))
            sel = ff[ff[:, 1] == gm] if np.any(ff[:, 1] == gm) else ff
            ref = (float(np.median(sel[:, 0])), gm, float(np.median(sel[:, 2])))
        if ref is not None:
            # narrowed around this world's confident fits; the narrow window's border is not the prior's, so a fit
            # on it is not suspicious (require_interior False)
            fsr, gm, c = ref
            fit = label_lines_by_pattern(wf, grid, fsr_range_hz=(0.985 * fsr, 1.015 * fsr), n_fsr=61,
                                         curvature_range=(c - 0.15, c + 0.15), n_curv=13, exponents=(gm,),
                                         require_interior=False)
        else:
            fit = label_lines_by_pattern(wf, grid, fsr_range_hz=fsr_prior_hz(self.d), exponents=self.exponents)
        if fit["ok"] and bank is None:
            self.fresh_fits[b].append((float(fit["fsr_hz"]), float(fit["exponent"]), float(fit["curvature"])))
        fit = with_first(fit, w, grid)
        self.labels[b][n] = np.asarray(fit["lines"], np.int64)
        self.map_ok[b, n] = bool(fit["ok"])

    def _label(self, b):
        grid = self.spec.line_grid_hz
        evs = [self.events[b][n] for n in range(self.N)]
        if grid is None:
            # without the line grid nothing can be labelled: keep the declared map on the first resonance
            for n in range(self.N):
                self.labels[b][n] = np.full(evs[n].size, -1, np.int64)
            return
        fits = label_bank_by_pattern(drop_first(evs, self.L), grid, fsr_range_hz=fsr_prior_hz(self.d),
                                     exponents=self.exponents)
        fits = [with_first(fits[n], evs[n], grid) for n in range(self.N)]
        for n in range(self.N):
            self.labels[b][n] = np.asarray(fits[n]["lines"], np.int64)
            self.map_ok[b, n] = bool(fits[n]["ok"])
        self.fits[b] = fits

    def _acquire_step(self, b):
        f = self.fsm
        if self.cursor[b] >= self.N:
            self.phase[b] = PH_TRACK
            self.retarget[b] = False
            return
        n = self.order[self.cursor[b]]
        if self.table is None and not self.retarget[b] and not self.fresh[b, n]:
            # cold start: a fresh map of this ring right before its acquisition, under the present thermal state
            # (rings acquired before it are locked and warm it through the crosstalk; the comb drifts by up to
            # half a spacing between a bank map and an acquisition, too much for a window or a count)
            if not self.mapping[b, n]:
                # supply sequencing: the map sweep may only climb as high as the budget left by the rings held now
                w_cap = self._w_cap(b, n, self.params.map_w_max)
                if w_cap < 0.1:
                    self.supply_limited[b, n] = True
                    self.cursor[b] += 1
                    return
                self.map_capped[b, n] = w_cap < self.params.map_w_max
                mask = np.zeros((self.B, self.N), bool)
                mask[b, n] = True
                f.dw_up[b] = self.dw_map
                f.start_map(mask, w_from=0.0, w_hi=w_cap, n_events=self.L + 2)
                self.mapping[b, n] = True
                return
            if not (f.map_done[b, n] and f.state[b, n] == ST_HOLD):
                return
            self.mapping[b, n] = False
            self._take_map(b, n)
            self._label_fresh(b, n)
            f.dw_up[b] = self.dw_acq
            self.fresh[b, n] = True
            k = self._target_index(b, n, int(self.target_line[b, n]))
            self.target_ev[b, n] = k
            if k < 0:
                self.supply_limited[b, n] |= bool(self.map_capped[b, n])
                self.assigned[b, n] = -1 if self.bound_assignment else self.assigned[b, n]
                mask = np.zeros((self.B, self.N), bool)
                mask[b, n] = True
                f.park(mask, 0.0)
                self.cursor[b] += 1
                return
        if not self.busy[b]:
            # projected supply: rings already acquired at their codes + this ring at its target
            codes = f.code[b].copy()
            k = int(self.target_ev[b, n])
            w, _, _, _ = self._ring_events(b, n)
            if 0 <= k < w.size:
                codes[n] = int(code_of_w(w[k] + 2 * f.dw_up[b, n], f.code_max, f.q))
            if not self._budget_ok(codes) or k < 0:
                self.supply_limited[b, n] = k >= 0
                self.assigned[b, n] = -1 if k < 0 else self.assigned[b, n]
                self.cursor[b] += 1
                return
            # from 0 after a fresh map or from a table (the up-search meets the target from blue, as the map did);
            # after an online barrel shift from the held code (the ring keeps the heat its drift was measured with)
            if self._start_ring(b, n, from_zero=not self.retarget[b]):
                self.busy[b] = True
            else:
                self.cursor[b] += 1
            return
        st = f.state[b, n]
        if st == ST_LOCK:
            # wrong-line check: the acquired peak against the mapped height of that resonance
            w, ln, _, h = self._ring_events(b, n)
            k = int(self.target_ev[b, n])
            h_map = float(h[k]) if 0 <= k < h.size and np.isfinite(h[k]) else np.nan
            h_now = float(f.y_pk[b, n] - f.y_bg[b, n])
            low = bool(np.isfinite(h_map) and h_now < self.params.verify_height_frac * h_map)
            self.wrong[b, n] = low                      # judged on the latest lock (a successful retry clears it)
            if low and not self.retried[b, n]:
                self.retried[b, n] = True
                # occupied or wrong resonance: try the next order of the same line
                k2 = self._target_index(b, n, int(self.target_line[b, n]), w_min=w[k] + 1e-9)
                if k2 >= 0:
                    self.target_ev[b, n] = k2
                    self._start_ring(b, n, from_zero=True)
                    return
            self._verify_lock(b, n)
            k = int(self.target_ev[b, n])
            w, _, _, _ = self._ring_events(b, n)
            if 0 <= k < w.size and np.isfinite(f.w_event[b, n]):
                dk = float(f.w_event[b, n] - w[k])
                sp = np.diff(w)
                spacing = float(np.median(sp)) if sp.size else 0.05
                if not self.fresh[b, n]:
                    self.drifts[b].append(dk)
                    self.drift[b] = float(np.median(self.drifts[b]))
                self.drift_used[b, n] = dk
            self.w_lock0[b, n] = f.w[b, n] - self.drift_used[b, n]      # in map coordinates
            self.cursor[b] += 1
            self.busy[b] = False
        elif st == ST_FAIL:
            self.assigned[b, n] = -1
            self.cursor[b] += 1
            self.busy[b] = False

    def _track_step(self, b):
        f = self.fsm
        p = self.params
        locked = f.state[b] == ST_LOCK
        hot = np.any(locked & (f.code[b] >= p.excess_code_frac * f.code_max))
        over = not self._budget_ok(np.where(locked, f.code[b], 0))
        self.excess[b] = self.excess[b] + 1 if (hot or over) else 0
        if self.excess[b] < p.excess_steps:
            return
        self.excess[b] = 0
        if self.policy != "cyclic":
            self.supply_limited[b] |= locked
            return
        # PAT-1: rotate by one channel towards lower power (every ring one line bluer, the wrap-around ring one order
        # down) and re-acquire ring by ring in bus order, each window shifted by that ring's own drive drift since
        # its acquisition (the reason for the shift: the chip cooled).
        drift = np.where(np.isfinite(self.w_lock0[b]), f.w[b] - self.w_lock0[b], 0.0)
        # every ring's drift is measured now, while all of them still hold their old lines (once ring n - 1 has
        # moved onto line n, ring n is starved and its loop would wander); then every ring HOLDS its code (its
        # thermal state, ms tails included, stays as measured) until its turn, in bus order
        self.drift_ring[b] = drift
        new = (self.target_line[b] + 1) % self.L
        for n in range(self.N):
            k = self._target_index(b, n, int(new[n]), w_below=float(f.w[b, n] - drift[n]))
            self.target_ev[b, n] = k
        self.shift[b] = (self.shift[b] + 1) % self.L
        self.target_line[b] = new
        self.assigned[b] = new
        f.state[b] = ST_HOLD
        f.locked[b] = False
        self.retarget[b] = True
        self.retried[b] = False
        self.acq_uncertain[b] = False
        self.drifts[b] = []
        self.phase[b] = PH_ACQUIRE
        self.cursor[b] = 0
        self.busy[b] = False

    # ------------------------------------------------------------------------------------------------ telemetry
    def telemetry(self) -> dict[str, np.ndarray]:
        f = self.fsm
        p_est = self._power(f.code)
        return {"state": f.state.astype(float), "phase": np.repeat(self.phase[:, None], self.N, 1).astype(float),
                "locked": f.locked.astype(float), "detuning_est_pm": f.detuning_est_pm(self.y),
                "integrator": f.u.copy(), "integrator_saturated": f.saturated.astype(float),
                "target_line": self.target_line.astype(float), "assigned_line": self.assigned.astype(float),
                "map_events": self.map_events.astype(float), "map_ok": self.map_ok.astype(float),
                "wrong_line": self.wrong.astype(float), "supply_limited": self.supply_limited.astype(float),
                "line_confident": (f.locked & self.map_ok & ~self.wrong & ~self.acq_uncertain).astype(float),
                "captured_at_start": self.captured.astype(float),
                "shift": np.repeat(self.shift[:, None], self.N, 1).astype(float),
                "p_est_mw": (np.full((self.B, self.N), np.nan) if p_est is None else 1e3 * p_est)}
