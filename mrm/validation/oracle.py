# benchmark an ideal controller that can read the true plant state.
# `oracle_target` finds the largest optical modulation amplitude, OMA, among detunings that pass the thermal stability test.
# detuning is lambda_res - lambda_line, so positive means the resonance is red of the laser.
# `reachable` checks whether the heater and shared supply can hold that point.
# a penalty against an unreachable target reflects a plant limit.
# hold the heater command fixed and require Z_opt(0) dP_abs/dT_R < 1, with any requested margin.
# the optical thermal resistance is in K/W and the absorption derivative is in W/K.
# the search uses the exact bus optics of the sampled world, then refines its grid optimum.
# the unrestricted optimum is reported too.
# in g26 nominal the optimum is +103 to +105 pm.
# this is also the unrestricted optimum, so the stability restriction costs no OMA at those nominal conditions.
# `Oracle` holds the target with true-state LQR and feed-forward.
# it uses thermal lag states, commands in the actuator pipeline, and the remaining resonance disturbance.
# the disturbance rate starts at zero and resets when the target changes.
# an integral term corrects linearization error.
# unreachable points use the nearest code instead of accumulating integral error.
# this is a reference benchmark.
# truth_fed prevents registry use as an ordinary controller.
# the runner supplies TruthState at each controller edge.
# OMA here is measured at the bus output after downstream rings.
# the engine and static replicas use their own through-port optimum calculations, so their OMA references differ slightly.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mrm.control import models as linear_models
from mrm.control.baselines.kalman_lqg import design_lqg

__all__ = ["TruthState", "OracleTarget", "oracle_target", "oma_penalty_db", "reloma_db", "reachable", "Oracle"]

_PM = 1e12                 # m -> pm
_Y_DETUNING = 1            # LinearModel output row - detuning_m (m)
_U_DISTURBANCE = 1         # LinearModel input column - ring_temperature_disturbance_k (K)


# data types
# plant Truth at a controller edge (for the oracle only).
# tick — plant tick of the edge.
# x — (B, M) thermal lag states in ThermalSystem order (the state at the edge).
# t_ring_k — (B, N) ring-mode temperatures (K).
# conditions — one `mrm.control.models.Conditions` per world (line wavelengths and chip powers after the scenario, NRZ levels, data statistics, base temperature).
# heater_applied — (B, N) latched heater codes (used at the first step).
@dataclass
class TruthState:
    tick: int
    x: np.ndarray
    t_ring_k: np.ndarray
    conditions: list
    heater_applied: np.ndarray | None = None


# the oracle's target, every array (B, N) per world and ring.
# detuning_m — the OMA-optimal stable detuning (m).
# oma_w — its OMA (W).
# oma_unrestricted_w and detuning_unrestricted_m — the optimum without the stability restriction.
# p_in_w — bus power at the ring on its line (W).
# reloma_db = 10 log10(OMA / P_in) (dB).
# loop_gain — Z_opt dP_abs/dT_R at the target (< 1).
# heater_code — the static heater code (float).
# reachable — True where the heater can hold the target (`reachable`).
# an OMA penalty against an unreachable target measures the world, not the controller.
# static — the StaticPoint of each world at its target.
@dataclass(frozen=True)
class OracleTarget:
    detuning_m: np.ndarray
    oma_w: np.ndarray
    oma_unrestricted_w: np.ndarray
    detuning_unrestricted_m: np.ndarray
    p_in_w: np.ndarray
    reloma_db: np.ndarray
    loop_gain: np.ndarray
    heater_code: np.ndarray
    reachable: np.ndarray | None = None
    static: list = field(default_factory=list, repr=False)


# static target, metric
# (N,) True where the heater can hold world `sp` (a models.StaticPoint of PlantParams `pp`).
# Three conditions must hold —
# heater power > 0 (the ring is not already red of the target with the heater off).
# the code is strictly inside the DAC range with dP/dcode > 0 (not clipped, not in the driver's compliance).
# the array supply can deliver the whole bank (`StaticPoint.supply_scale` = 1 — total power within array.heater_total_power_max_w AND total current within array.heater_supply_current_max_a).
# Above either limit the supply droops every heater, and then no ring of the world is reachable.
def reachable(pp, sp) -> np.ndarray:
    heater = pp.heater
    ok = ((np.asarray(sp.heater_power_w) > 0) & (np.asarray(sp.heater_code) > float(heater.code_min) + 0.5)
          & (np.asarray(sp.heater_code) < float(heater.code_max) - 0.5))
    if np.all(np.isfinite(np.asarray(sp.dp_dcode, float))):
        ok &= np.asarray(sp.dp_dcode) > 0
    if float(getattr(sp, "supply_scale", 1.0)) < 1.0 - 1e-9:
        ok[:] = False
    return ok


# the oracle's target in every world of the TRUE PlantParams `pp`.
# conditions — one Conditions per world, or None to take each world's nominal conditions from `cfg` (`models.conditions_from_config`).
# stable_margin — the target's loop gain must be below 1 - stable_margin.
def oracle_target(pp, conditions, *, stable_margin: float = 0.0, cfg=None) -> OracleTarget:
    B, N = int(pp.batch), int(pp.n_rings)
    out = {k: np.zeros((B, N)) for k in ("detuning_m", "oma_w", "oma_unrestricted_w", "detuning_unrestricted_m",
                                         "p_in_w", "loop_gain", "heater_code")}
    reachable_points = np.zeros((B, N), bool)
    statics = []
    for b in range(B):
        world_conditions = (linear_models.conditions_from_config(pp, b,
            cfg=cfg) if conditions is None else conditions[b])
        det_m, oma_w, oma_u_w, det_u_m = linear_models.oma_optimal_detuning(pp, b, cond=world_conditions,
                                                                           stable_margin=stable_margin)
        operating_point = linear_models.static_point(pp, b, det_m, cond=world_conditions)
        statics.append(operating_point)
        (out["detuning_m"][b], out["oma_w"][b], out["oma_unrestricted_w"][b],
         out["detuning_unrestricted_m"][b]) = det_m, oma_w, oma_u_w, det_u_m
        out["p_in_w"][b] = operating_point.p_in_w
        out["loop_gain"][b] = operating_point.loop_gain_opt
        out["heater_code"][b] = operating_point.heater_code
        reachable_points[b] = reachable(pp, operating_point)
    return OracleTarget(detuning_m=out["detuning_m"], oma_w=out["oma_w"], oma_unrestricted_w=out["oma_unrestricted_w"],
                        detuning_unrestricted_m=out["detuning_unrestricted_m"], p_in_w=out["p_in_w"],
                        reloma_db=reloma_db(out["oma_w"], out["p_in_w"]), loop_gain=out["loop_gain"],
                        heater_code=out["heater_code"], reachable=reachable_points, static=statics)


# relOMA = OMA / P_in in dB, with P_in the power of the ring's line at the ring (W).
# P_in is the reference of the printed definition ("1 mW of laser into the MRM", Bunandar, Hot Chips 37, slide 16).
# the OMA of `oracle_target` is measured at the bus output, after the downstream rings.
# so this relOMA is a little lower than the through-port relOMA of the replicas and of the engine (about 0.35 dB at g26 nominal. see docs/handoff/replicas.md, "Findings the predictions reveal", item 7).
def reloma_db(oma_w, p_in_w) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(oma_w, float), 1e-300) / np.asarray(p_in_w, float))


# oMA penalty against the oracle at the same instant — 10 log10(OMA_oracle / OMA) in dB.
# the penalty is >= 0 when the controller is worse than the oracle.
# OMA 0 gives +inf.
def oma_penalty_db(oma_w, oracle_oma_w) -> np.ndarray:
    oma = np.asarray(oma_w, float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.asarray(oracle_oma_w, float) / np.where(oma > 0, oma, 0.0))


# dynamic hold
# hold the best stable OMA point with access to true plant state.
# `stable_margin` lowers the allowed thermal loop gain.
# q_pm and r_codes set LQR error and effort scales.
# larger scales reduce their weights.
# `ki_per_step` integrates true detuning error to correct model gain error.
# when retargeting is enabled, re-solve a world's target after power, wavelength, junction levels, or ones density move beyond the tolerances.
# limit redesigns with retarget_min_ticks.
# designs and actuator delay lines are kept per world.
class Oracle:

    name = "oracle"        # controller name
    truth_fed = True       # reads Truth - mrm.control.registry refuses to register or create it
    latency_ticks = 0      # compute latency - the command of an edge latches at once (mrm/control/api.py, "Timing")
    fast_kernel = None     # no per-ADC-sample kernel - the runner calls step() at each controller edge

    def __init__(self, pp=None, cfg=None, *, stable_margin: float = 0.0, q_pm: float = 0.5, r_codes: float = 400.0,
                 ki_per_step: float = 0.3, retarget: bool = True, retarget_power_frac: float = 2e-3,
                 retarget_lambda_m: float = 0.2e-12, retarget_v: float = 2e-3, retarget_min_ticks: int = 200):
        self._pp0, self._cfg0 = pp, cfg
        self.stable_margin = float(stable_margin)
        self.q_pm = float(q_pm)
        self.r_codes = float(r_codes)
        self.ki_per_step = float(ki_per_step)
        self.retarget = bool(retarget)
        self.retarget_tol = (float(retarget_power_frac), float(retarget_lambda_m), float(retarget_v))
        self.retarget_min_ticks = int(retarget_min_ticks)
        self.spec = None
        self.n_retargets = 0

    # per-world design
    # solve the target of world b at conditions `cond` and design the LQR on its true linear model.
    def _design_world(self, b: int, cond) -> None:
        pp, spec = self.pp, self.spec
        N = int(pp.n_rings)
        optimal_detuning, oma, _, _ = linear_models.oma_optimal_detuning(pp, b, cond=cond,
            stable_margin=self.stable_margin)
        ring_models = linear_models.linearize_bank(self.cfg, linear_models.OperatingPoint(detuning_m=optimal_detuning),
                                                   spec.period_s, latency_ticks=self.latency_ticks,
                                                   measurement="none", pp=pp, b=b, cond=cond)
        design = design_lqg(ring_models, q_pm=self.q_pm, r_codes=self.r_codes, kalman=False)   # full state - no filter
        old_design = self.designs[b]
        self.designs[b] = design
        self.lag_maps[b] = [linear_models.ring_block_lags(pp, b, n) for n in range(N)]
        # c_R - ring temperature (K) per grouped thermal state = (detuning per state) / (detuning per kelvin)
        self.ring_k_per_state[b] = [ring_model.Cc[_Y_DETUNING] / ring_model.Dc[_Y_DETUNING,
            _U_DISTURBANCE] for ring_model in ring_models]
        self.det_target[b] = optimal_detuning
        self.oma_target[b] = oma
        self.reach[b] = reachable(pp, ring_models[0].op)
        self.design_conditions[b] = cond
        if old_design is None:
            self.bumpless[b] = True               # first design - start from the code in the register
        else:
            # retarget - the commands in flight change reference.
            # the integral (the linearization's gain error) carries over, and the new model's feed-forward moves the ring to the new equilibrium at once.
            self.delay_line[b] = self.delay_line[b] + (old_design.u0 - design.u0)[:, None]

    # targets from the TRUE world at the conditions of `truth`.
    # LQR + feed-forward on the true linear model.
    # `truth_pp` / `cfg` default to the constructor's.
    # with `truth` None (a runner that calls reset(spec) only) the design waits for the first step's TruthState.
    def reset(self, spec, truth_pp=None, truth: TruthState | None = None, cfg=None) -> None:
        self.spec = spec
        self.cfg = self._cfg0 if cfg is None else cfg
        self.pp = pp = self._pp0 if truth_pp is None else truth_pp
        if pp is None:
            raise ValueError("Oracle needs the TRUE PlantParams (reset(spec, truth_pp, ...) or Oracle(pp=...))")
        self._designed = False
        if truth is None:
            return
        self._design_all(truth)

    # design every world and fill the delay lines with the codes now in the registers.
    def _design_all(self, truth: TruthState) -> None:
        spec, pp = self.spec, self.pp
        B, N = int(pp.batch), int(pp.n_rings)
        if (spec.n_batch, spec.n_rings) != (B, N):
            raise ValueError(f"spec is (B, N) = {(spec.n_batch, spec.n_rings)}, Truth {(B, N)}")
        self.designs = [None] * B
        self.lag_maps = [None] * B
        self.ring_k_per_state = [None] * B
        self.design_conditions = [None] * B
        self.det_target = np.zeros((B, N))
        self.oma_target = np.zeros((B, N))
        self.reach = np.ones((B, N), bool)
        self.bumpless = np.ones(B, bool)
        # commands in flight (deviations from each world's u*), newest first - one (N, n_delay) array per world (the thermal block and the delay line may differ between worlds - sampled transport delay, ROM family)
        self.delay_line = [None] * B
        for b in range(B):
            self._design_world(b, truth.conditions[b])
        applied = (np.asarray(truth.heater_applied, float) if truth.heater_applied is not None
                   else np.stack([design.u0 for design in self.designs]))
        for b in range(B):
            design = self.designs[b]
            n_delay = design.nc - design.models[0].Ac.shape[0]
            self.delay_line[b] = np.repeat((applied[b] - design.u0)[:, None], n_delay, axis=1)
        self._d_prev_pm = np.full((B, N), np.nan)
        self.integral_pm = np.zeros((B, N))
        self.code_min = float(spec.heater_dac.code_min)
        self.code_max = float(spec.heater_dac.code_max)
        self._last_codes = np.zeros((B, N))
        self._last_retarget = np.full(B, -(1 << 62), np.int64)      # tick of the last retarget (none yet)
        self.n_retargets = 0
        self._designed = True

    # current target (detuning and OMA per world and ring. the other fields are not tracked after reset).
    @property
    def target(self) -> OracleTarget:
        nan = np.full_like(self.det_target, np.nan)
        return OracleTarget(detuning_m=self.det_target.copy(), oma_w=self.oma_target.copy(), oma_unrestricted_w=nan,
                            detuning_unrestricted_m=nan, p_in_w=nan, reloma_db=nan, loop_gain=nan,
                            heater_code=np.stack([design.u0 for design in self.designs]),
                            reachable=self.reach.copy())

    # true when the conditions moved from `old` (the design's) to `new` by more than the retarget limits.
    def _conditions_moved(self, old, new) -> bool:
        tol_power_frac, tol_lambda_m, tol_v = self.retarget_tol
        p_old_w = np.asarray(old.p_chip_w, float)
        if np.any(np.abs(np.asarray(new.p_chip_w, float) - p_old_w) > tol_power_frac * np.maximum(p_old_w, 1e-30)):
            return True
        if np.any(np.abs(np.asarray(new.lam_m) - np.asarray(old.lam_m)) > tol_lambda_m):
            return True
        if np.any(np.abs(np.asarray(new.v_lo) - np.asarray(old.v_lo)) > tol_v) or \
                np.any(np.abs(np.asarray(new.v_hi) - np.asarray(old.v_hi)) > tol_v):
            return True
        return bool(np.any(np.abs(np.asarray(new.rho) - np.asarray(old.rho)) > 1e-3))

    # one controller edge
    # the exact model state from Truth.
    # returns (z_dev, d_pm, err_pm) — z_dev, one (N, n_thermal) array per world of thermal states minus their values at the operating point.
    # d_pm (B, N), the disturbance d as pm of open-loop resonance shift.
    # err_pm (B, N), the true detuning error (pm).
    def _disturbance_pm(self, truth: TruthState):
        B, N = self._last_codes.shape
        z_dev = []
        d_pm = np.zeros((B, N))
        err_pm = np.zeros((B, N))
        for b in range(B):
            design = self.designs[b]
            op_point = design.models[0].op
            n_thermal = design.models[0].Ac.shape[0]
            z_world = np.zeros((N, n_thermal))
            detuning_m = linear_models.resonance_detuning_m(self.pp, b, truth.t_ring_k[b], truth.conditions[b])
            for n in range(N):
                lag_index, residue, source = self.lag_maps[b][n]
                z_lag = truth.x[b, lag_index] / residue
                z_op = np.where(source == 0, op_point.heater_power_w[n], op_point.p_abs_w[n])
                z_world[n, :lag_index.size] = z_lag - z_op
                if n_thermal > lag_index.size:          # heater settle-filter state (models._model_for_ring) - its
                    z_world[n, lag_index.size:] = self.delay_line[b][n, -1]  # input is the code leaving the line
                shift_m_per_k = design.models[n].Dc[_Y_DETUNING, _U_DISTURBANCE]
                disturbance_k = ((detuning_m[n] - self.det_target[b, n]) / shift_m_per_k
                       - self.ring_k_per_state[b][n] @ z_world[n])
                d_pm[b, n] = disturbance_k * shift_m_per_k * _PM
                err_pm[b, n] = (detuning_m[n] - self.det_target[b, n]) * _PM
            z_dev.append(z_world)
        return z_dev, d_pm, err_pm

    # heater codes (B, N) for this edge from the exact state in `truth` (`obs` is not used).
    def step(self, obs, truth: TruthState) -> dict[str, np.ndarray]:
        if self.spec is None:
            raise RuntimeError("Oracle: call reset(spec, truth_pp, truth) first")
        if not (hasattr(truth, "x") and hasattr(truth, "conditions")):
            raise TypeError("Oracle.step needs a mrm.validation.oracle.TruthState (thermal lag states x, ring "
                            "temperatures, per-world Conditions); got " + type(truth).__name__)
        if not self._designed:
            self._design_all(truth)
        B, N = self._last_codes.shape
        if self.retarget:
            for b in range(B):
                if (truth.tick - self._last_retarget[b] >= self.retarget_min_ticks
                        and self._conditions_moved(self.design_conditions[b], truth.conditions[b])):
                    self._design_world(b, truth.conditions[b])
                    # d changes reference with the design (target, operating point, s, c_R).
                    # the change of d across the retarget is not a rate, so the rate of this edge is 0, as on the first edge.
                    self._d_prev_pm[b] = np.nan
                    self._last_retarget[b] = truth.tick
                    self.n_retargets += 1
        z_dev, d_pm, err_pm = self._disturbance_pm(truth)
        period_ms = self.spec.period_s * 1e3
        d_rate_pm_per_ms = np.where(np.isfinite(self._d_prev_pm), (d_pm - self._d_prev_pm) / period_ms, 0.0)
        d_rate_pm_per_ms = np.where(self.bumpless[:, None], 0.0, d_rate_pm_per_ms)
        self._d_prev_pm = d_pm
        # integral of the TRUE detuning error, added to d - it removes what the linearization gets wrong (the heater gain at a code far from the linearization point, the TCR) - with the exact d the loop is otherwise open in DC
        self.integral_pm = self.integral_pm + self.ki_per_step * err_pm
        codes = np.zeros((B, N))
        for b in range(B):
            design = self.designs[b]
            if self.bumpless[b]:
                # bumpless (re)start - the integrator takes the value that reproduces the code now in the register
                z_core = np.concatenate([z_dev[b], self.delay_line[b]], axis=1)[:, :design.nc]
                k_d = design.Kf[:, design.nc]                       # feedback gain on the d state
                rest = np.einsum("nz,nz->n", design.Kf[:, :design.nc], z_core) + k_d * d_pm[b]
                applied = design.u0 + self.delay_line[b][:, 0]
                k_d_safe = np.where(np.abs(k_d) > 0, k_d, 1.0)
                self.integral_pm[b] = np.where(np.abs(k_d) > 0, (design.u0 - applied - rest) / k_d_safe, 0.0)
                self.bumpless[b] = False
            # the augmented state stacks thermal states, delay registers, d + I, and the disturbance rate.
            # trim the vector to the design size.
            z_full = np.concatenate([z_dev[b], self.delay_line[b], (d_pm[b] + self.integral_pm[b])[:, None],
                                     d_rate_pm_per_ms[b][:, None]], axis=1)[:, :design.nz]
            code_deviation = -np.einsum("nz,nz->n", design.Kf, z_full)
            # an unreachable target (reachable() - beyond the code range, no heater authority, bank over the supply) is not regulated - the oracle holds the code nearest to it (the range end) instead of winding up
            requested_code = np.where(self.reach[b], design.u0 + code_deviation, design.u0)
            codes[b] = np.clip(np.rint(np.nan_to_num(requested_code, nan=design.u0)), self.code_min, self.code_max)
            self.integral_pm[b] = np.where(self.reach[b], self.integral_pm[b], 0.0)
            self.delay_line[b] = np.concatenate([(codes[b] - design.u0)[:, None], self.delay_line[b][:, :-1]],
                                                axis=1)
        self._last_codes = codes
        return {"tx.heater": codes.astype(np.int64)}

    # telemetry — the target detuning (pm) and the integral term (pm) per world and ring.
    def telemetry(self) -> dict[str, np.ndarray]:
        return {"detuning_est_pm": self.det_target * _PM, "integrator": self.integral_pm.copy()}
