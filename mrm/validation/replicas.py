"""Replicas of the published measurement protocols (07 P0-5): one function per protocol named in the target files.

Owner: validation replicas. Contract: docs/research/07_design_critique.md ("07") P0-5 and section 4, P1-4 (the
apparent-sensitivity replica), P1-7 (bias ladder at measurement power), P1-9 (monitor), P1-11 (driver law); the
target files and their README (mrm/validation/targets). Every name in :data:`mrm.validation.targets_io.REPLICA_NAMES`
is implemented here and registered in :data:`REPLICAS`.

What a replica is
-----------------
A replica reproduces the PROTOCOL of a published measurement on the model plant: the same sweep, the same port, the
same normalization and the same read-out rule (notch minimum, half-depth width, contour crossing, ...). It never
fits anything: the plant parameters come from ``mrm.params`` as they are (calibration is a separate stage, 07 4.3).
Every replica has the signature ::

    fn(plant, target: Target | None = None, **kw) -> ReplicaResult

``plant`` is a preset name, a :class:`mrm.params.config.Config` or a realized
:class:`mrm.params.realize.PlantParams` (hypothesis sweeps such as the driver law or the thermo-optic hypothesis need
a preset or Config, because they re-load it with ``alts``). ``target.replica_args`` are applied first and keyword
arguments override them. :attr:`ReplicaResult.value` is shaped as :func:`mrm.validation.targets_io.evaluate`
expects for the target's tolerance kind and is in the target's unit; :attr:`ReplicaResult.conditions` records every
condition and assumption the replica used (07 P0-5: T, heater drive, bias, optical power, port), and
:attr:`ReplicaResult.variants` the same prediction under the alternative hypotheses the target lists.

Static replicas (quasi-static steady state; the leaf modules only)
-----------------------------------------------------------------
:class:`StaticRing` is the steady state of one ring of one world, built from PlantParams and the leaf adapters:
optics inputs from :func:`mrm.plant.phase.ring_optics_row` (R1 shift, junction, K^2 at the line, loss and eta), the
exact ring :func:`mrm.plant.optics.ring_exact`, DC thermal gains :func:`mrm.plant.thermal.dc_gain` of the assembled
ThermalSystem, the heater driver law with its TCR (:mod:`mrm.plant.electrical`), the junction swing capacitance and
data-activity heat, and the monitor PD of :class:`mrm.plant.sensing.MonitorHw`. The state variable is the
mode-weighted ring temperature T. For a given T everything is explicit:

    P_abs(T)  = P_in [rho A_1(T) + (1 - rho) A_0(T)]              absorbed (heat) power, A = eta x loss fraction
    P_h(T)    = (T - T_base - Z_or P_abs - Z_jr P_j - Z_dr P_d - dT_agg,r) / Z_hr        heater power needed
    T_H(T)    = T_base + Z_hh P_h + Z_oh P_abs + Z_jh P_j + Z_dh P_d + dT_agg,h          heater-resistor observer
    u(T)      = inverse driver law at R(T_H) = R0 (1 + alpha (T_H - T_R0)):
                voltage V/V_FS = sqrt(P_h R)/V_FS; current sqrt(P_h/R)/I_FS; power P_h/P_FS; dac_pwm P_h R/V_FS^2

with Z_xy the DC transfer resistances (K/W) of the ROM from source x (h heater, o waveguide absorption, j junction,
d driver) to observer y (r ring_mode_weighted, h heater_resistor). A fixed drive u_0 then has its steady states where
u(T) = u_0; the quasi-static dynamics dT/dt has the sign of u_0 - u(T), so a root is stable when u(T) is rising and
a slow sweep (laser wavelength, heater drive, ambient) follows the branch it is on until it vanishes at a fold and
then relaxes in the direction of u_0 - u(T) (exact quasi-static branch continuation, :func:`_relax`). Locks (peak,
fraction of peak, oracle maximum OMA) choose T directly; the heater power then follows from the same equations
(for a bank, by the linear solve with the N x N DC gain matrices, crosstalk included).

Closed-loop replicas
--------------------
``monitor_current_at_lock`` and ``stabilizer_code_vs_T`` default to the quasi-static lock (the loop is assumed settled:
valid when the loop is much faster than the disturbance, which holds for the 100-131 ms aggressor periods and any
loop above about 1 kHz). With ``method="dynamic"`` they hand a :class:`LockRunRequest` to a run function
(``run_fn``; default :func:`default_run_fn`, which uses ``mrm.sim.runner.run_lock_request`` once the engine provides
it) and read the recorded traces. ``aggressor_feasibility`` (BND-3a/b) needs controllers and the engine and is
reported ``blocked`` until both exist. ``mode_solve_published_geometry``, ``self_heating_time_constant`` and
``thermal_step_published_stack`` read the extraction artefacts of the third-party geometries
(``mrm/data/extracted/modes_decea2019.json``, ``thermal_val8_imec.json``) and are ``blocked`` while those are absent.

Conventions: SI inside; results in the target's unit. Wavelengths are vacuum wavelengths; bias is reverse bias in V
(positive = reverse, ARCHITECTURE 5.4). "P_in -> 0" (a null ``p_in_dbm``) means no optical power at all (no
self-heating) in every replica; ``p_in_dbm="preset"`` means the preset's line power at the ring (the default of the
replicas whose protocol runs at the operating power: monitor, bistability, apparent sensitivity, relOMA where the
target does not print the power). Static replicas take one ring (default ring 0) with its own laser line only,
other rings dark with their heaters off, unless they say otherwise (bank replicas, :func:`_bank_codes`, lock every
ring with every line on, filtered along the bus, and include the crosstalk).

Writing the predictions: :func:`predict_all` runs every target's replica (G25 entries on ``g25_test_vehicle``,
everything else on ``g26``) and :func:`write_predictions` stores them, with each target's value, tolerance and
:func:`targets_io.evaluate` result, in ``mrm/validation/latest_predictions.json`` for the report generator
(``python -m mrm.validation.replicas``).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize

from mrm.plant import electrical as el
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.plant import thermal as th
from mrm.units import C0, T0_K
from mrm.validation import targets_io as tio

__all__ = [
    "ReplicaResult", "ReplicaBlocked", "StaticRing", "LockPoint", "LockRunRequest", "REPLICAS", "LM_DLAMBDA_DT_M_PER_K",
    "default_run_fn", "resolve_plant", "load_config", "preset_for", "bank_gains", "bus_line_powers", "run_target",
    "predict_all", "write_predictions", "PREDICTIONS_PATH",
    "notch_er_vs_heater_v", "loaded_q_from_3db_width", "tuning_range_at_v", "relOMA_map", "monitor_current_at_lock",
    "stabilizer_code_vs_T", "apparent_shift_fixed_code", "bistability_sweep", "heater_power_per_fsr",
    "resonance_vs_heater_v", "resonance_shift_vs_bias", "cold_resonance_offset", "eo_bandwidth_at_bias",
    "pd_responsivity", "laser_line_statistics", "aggressor_trajectory", "mode_solve_published_geometry",
    "self_heating_time_constant", "thermal_step_published_stack", "adjacent_order_penalty", "order_hop_margin",
    "aggressor_feasibility", "main",
]

#: Lightmatter's printed "~15 GHz/degC" at 1310 nm, 15e9 * (1310 nm)^2 / c = 85.86 pm/K (LM26-P section 1, printed
#: aside; CAL-6). Used only where a protocol converts a resonance shift into the printed degC (07 D6, P1-4).
LM_DLAMBDA_DT_M_PER_K = 15e9 * (1310e-9) ** 2 / C0
PREDICTIONS_PATH = Path(__file__).resolve().parent / "latest_predictions.json"
#: preset each target generation is replicated on (G25 bench device vs the G26 bank)
PRESET_FOR_GENERATION = {"G25": "g25_test_vehicle", "G26": "g26", "Guide": "g26", "EVK100": "g26", "L20": "g26",
                         "M1000": "g26", "OCP": "g26", "OCI": "g26", "GF": "g26", "PAT": "g26", "none": "g26"}
_DRIVER_LAWS = ("voltage", "current", "power", "dac_pwm")
_FS_KEY = {"voltage": "ring.heater.full_scale_v", "current": "ring.heater.full_scale_a",
           "power": "ring.heater.full_scale_w", "dac_pwm": "ring.heater.full_scale_v"}


# =====================================================================================================================
# Results
# =====================================================================================================================
class ReplicaBlocked(RuntimeError):
    """A replica cannot run yet (missing artefact, controller or engine); the message says what is missing."""


@dataclasses.dataclass
class ReplicaResult:
    """Prediction of one replica for one target.

    value: the prediction in ``unit``, shaped as :func:`mrm.validation.targets_io.evaluate` expects for the target
    (None when ``status`` is ``blocked``). status: ``ok`` | ``blocked``. conditions: the conditions and assumptions
    used (JSON-serializable). variants: {hypothesis label: prediction shaped like value}. detail: diagnostics
    (intermediate numbers). notes: free-text caveats.
    """

    replica: str
    value: Any
    unit: str
    status: str = "ok"
    target_id: str | None = None
    conditions: dict = dataclasses.field(default_factory=dict)
    variants: dict = dataclasses.field(default_factory=dict)
    detail: dict = dataclasses.field(default_factory=dict)
    notes: list = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_json(self) -> dict:
        """JSON-ready dict (numpy arrays and scalars converted)."""
        return _jsonable(dataclasses.asdict(self))


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else (None if math.isnan(v) else ("inf" if v > 0 else "-inf"))
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _blocked(name: str, target, unit: str, reason: str, **conditions) -> ReplicaResult:
    return ReplicaResult(name, None, unit, status="blocked", target_id=getattr(target, "id", None),
                         conditions=dict(conditions), notes=[reason])


def _args(target, kw: Mapping[str, Any]) -> dict:
    """target.replica_args, then the caller's keyword arguments (they win; an explicit None is kept, e.g.
    ``p_in_dbm=None`` asks for P_in -> 0)."""
    a = dict(getattr(target, "replica_args", None) or {})
    a.update(kw)
    return a


def _unit(target, default: str) -> str:
    return str(getattr(target, "unit", None) or default)


# =====================================================================================================================
# Plant resolution
# =====================================================================================================================
def _is_pp(x) -> bool:
    return hasattr(x, "optics") and hasattr(x, "thermal") and hasattr(x, "batch")


def _cfg_path(cfg) -> str | Path:
    """The file a Config was loaded from (the last one of its extends chain), for re-loading it with alts."""
    f = Path(cfg.files[-1])
    if not f.is_absolute():
        f = tio.REPO_DIR / f
    return f


def load_config(plant, *, alts: Mapping[str, str] | None = None, overrides: Mapping[str, Any] | None = None):
    """A Config from a preset name, a path or a Config, with extra ``alts`` / ``overrides`` (re-loaded, keeping the
    Config's own alternatives and overrides). Artefact-fallback warnings are silenced (they are recorded in
    ``cfg.warnings``)."""
    from mrm import params
    if _is_pp(plant):
        raise TypeError("a PlantParams cannot be re-loaded; pass a preset name or a Config")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(plant, (str, Path)):
            if not alts and not overrides:
                return params.load(plant)
            return params.load(plant, overrides=dict(overrides or {}), alts=dict(alts or {}))
        if not alts and not overrides:
            return plant
        a = dict(plant.alts_applied)
        a.update(alts or {})
        o = dict(plant.overrides_applied)
        o.update(overrides or {})
        return params.load(_cfg_path(plant), overrides=o, alts=a)


def resolve_plant(plant, *, batch: int = 1, uncertainty: str = "nominal", variation: str = "off", seeds=0,
                  alts: Mapping[str, str] | None = None, overrides: Mapping[str, Any] | None = None):
    """(cfg, pp): load (preset name / path / Config) and realize; a PlantParams passes through with cfg None.

    Defaults: one world, every uncertainty entry at its nominal, no per-device variation (the median device), seeds
    0. ``alts`` pins hypotheses (e.g. ``{"uncertainty.heater.driver_mode": "current"}``)."""
    from mrm import params
    if _is_pp(plant):
        if alts or overrides:
            raise TypeError("hypothesis alternatives need a preset name or a Config, not a PlantParams")
        return None, plant
    cfg = load_config(plant, alts=alts, overrides=overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, int(batch), seeds=seeds, uncertainty=uncertainty, variation=variation)
    return cfg, pp


def _hyp_choices(cfg, entry: str) -> tuple[str, ...]:
    """Choice names of an ``uncertainty`` hypothesis entry (empty when absent)."""
    if cfg is None:
        return ()
    e = cfg.uncertainty.get(entry)
    if e is None:
        return ()
    fields = getattr(e, "fields", None) or {}
    ch = fields.get("choices") if isinstance(fields, Mapping) else None
    return tuple(ch) if ch else ()


def _pp_meta(cfg, pp) -> dict:
    return {"preset": getattr(cfg, "name", None) if cfg is not None else None,
            "alts": dict(getattr(cfg, "alts_applied", {}) or {}),
            "overrides": {k: _jsonable(v) for k, v in (getattr(cfg, "overrides_applied", {}) or {}).items()},
            "batch": int(pp.batch), "config_hash": getattr(pp, "config_hash", None),
            "artefact_fallbacks": sum(1 for w in (getattr(cfg, "warnings", []) or []) if "fallback" in w)}


# =====================================================================================================================
# The static ring
# =====================================================================================================================
@dataclasses.dataclass
class LockPoint:
    """Steady state of a locked ring (one world, one ring), SI. ``stable`` refers to the fixed-drive (open-loop)
    stability of the operating point: a slow loop can hold it only when True. ``order``: the locked resonance order
    relative to the design order. ``t_pd_k``: the monitor-PD observer temperature of this single-ring state (its own
    heater, absorption, junction and driver heat; sets the dark current). ``p_drop_w`` is the power into the drop
    monitor PD (own line, plus ``p_drop_other_w`` from the other lines when locked with ``lines='all'``); ``p_abs_w``
    includes the other lines' absorption then; ``p_bus_w``: the line power at the ring (all lines with
    ``lines='all'``), the monitor's background input."""
    mode: str
    order: int
    t_ring_k: float
    t_res_k: float
    p_abs_w: float
    p_drop_w: float
    p_thru_w: float
    reloma: float
    drop_peak_w: float
    p_j_w: float
    p_d_w: float
    p_heater_w: float
    t_heater_k: float
    u: float
    stable: bool
    detuning_m: float
    t_pd_k: float = math.nan
    p_drop_other_w: float = 0.0
    p_bus_w: float = math.nan


class StaticRing:
    """Steady state of ring ``n`` of world ``b`` of a realized plant (module docstring, "Static replicas").

    Args:
        pp: PlantParams. b, n: world and ring. topology: ``add_drop`` (the realized ring) or ``all_pass`` (the drop
            coupler removed, K^2_drop = 0, everything else unchanged; 07 P0-5 / 01 D14). driver_law: override of the
            heater law ("voltage", ..., or "bench_voltage": a voltage source with u = V in volts).
    The thermal gains are the DC transfer resistances of pp.thermal.system for this ring's own sources; the other
    rings are dark with their heaters off (single-device protocol) unless a caller adds their heat explicitly.
    """

    def __init__(self, pp, *, b: int = 0, n: int = 0, topology: str = "add_drop", driver_law: str | None = None,
                 _cache: dict | None = None):
        if topology not in ("add_drop", "all_pass"):
            raise ValueError(f"topology must be add_drop or all_pass, got {topology!r}")
        cache = _cache if _cache is not None else _plant_cache(pp)
        self.pp, self.b, self.n, self.topology = pp, int(b), int(n), topology
        P = cache["phase"]
        rp = np.array(P.ring[self.b, self.n], dtype=float)
        if topology == "all_pass":
            rp[ph.RP_K2D] = 0.0
        self.rp = rp
        r = P.table_row(self.b)
        self.grid = P.grid
        self.tab = (np.ascontiguousarray(P.dneff[r]), np.ascontiguousarray(P.alpha_fca[r]),
                    np.ascontiguousarray(P.capacitance[r]), np.ascontiguousarray(P.r_series[r]))
        rb = np.asarray(pp.optics.backscatter_rb[self.b, self.n])
        self.rb = rb if (rb.size and np.any(rb != 0)) else None
        sysm = pp.thermal.system
        G = cache["dc"][self.b]                                    # (O, S) K/W
        n = self.n

        def g(o, s):
            return float(G[o, s]) if (o >= 0 and s >= 0) else 0.0
        ro, ho = int(sysm.ring_obs[n]), int(sysm.heater_obs[n])
        po = int(sysm.pd_obs[n]) if sysm.pd_obs.size else -1
        hs, ws = int(sysm.heater_src[n]), int(sysm.wg_src[n])
        js = int(sysm.junction_src[n]) if sysm.junction_src.size else -1
        ds = int(sysm.driver_src[n]) if sysm.driver_src.size else -1
        self.z = {"hr": g(ro, hs), "hh": g(ho, hs), "or": g(ro, ws), "oh": g(ho, ws), "jr": g(ro, js),
                  "jh": g(ho, js), "dr": g(ro, ds), "dh": g(ho, ds), "hp": g(po, hs), "op": g(po, ws),
                  "jp": g(po, js), "dp": g(po, ds)}
        if not self.z["hr"] > 0:
            raise ValueError("the thermal system has no heater -> ring path for this ring")
        # heater driver (realized DAC path of this ring)
        hp = cache["heater"]
        law = int(np.asarray(hp.law).reshape(-1)[self.b if hp.law.size > 1 else 0])
        if driver_law is not None:
            law = -1 if driver_law == "bench_voltage" else el.DRIVER_LAWS[driver_law]
        self.law = law
        bi = self.b if hp.r_cold_ohm.shape[0] > 1 else 0
        ni = self.n if hp.r_cold_ohm.shape[1] > 1 else 0

        def hv(a):
            a = np.asarray(a, dtype=float)
            return float(a[bi if a.shape[0] > 1 else 0, ni if a.shape[1] > 1 else 0])
        self.r0 = hv(hp.r_cold_ohm)
        self.alpha = hv(hp.tcr_per_k)
        self.t_r0 = hv(hp.tcr_ref_k)
        self.v_fs, self.i_fs, self.p_fs = hv(hp.v_full_scale_v), hv(hp.i_full_scale_a), hv(hp.p_full_scale_w)
        # driver output compliance (07 P1-10; electrical.heater_power_row saturates as a voltage source there); a
        # bench voltage source has none
        self.v_comp = math.inf if law == -1 else hv(hp.compliance_v)
        self.bits, self.pwm_bits = int(hp.bits), int(hp.pwm_bits)
        be = np.asarray(hp.bit_err_lsb)
        self.bit_err = np.ascontiguousarray(be[bi if be.shape[0] > 1 else 0, ni if be.shape[1] > 1 else 0][None, :])
        bow = np.asarray(hp.inl_bow_lsb, dtype=float)
        self.bow = np.array([bow[bi if bow.shape[0] > 1 else 0, ni if bow.shape[1] > 1 else 0]])
        self._u_table = None
        # optics constants and the laser line of this ring
        o = pp.optics
        self.lam_res = float(rp[ph.RP_LAM_RES])
        self.t_ref = float(rp[ph.RP_T_REF])
        self.s0, self.beta, self.t0 = float(rp[ph.RP_S0]), float(rp[ph.RP_BETA]), float(rp[ph.RP_T0])
        self.q_loaded = float(np.asarray(o.q_loaded)[self.b, self.n])
        self.fsr_hz = float(np.asarray(o.fsr_hz)[self.b, self.n])
        self.line = int(np.asarray(pp.layout.ring_to_line)[self.n])
        self.lam_line = float(C0 / np.asarray(pp.laser.line_freq_hz)[self.b, self.line])
        tr = float(np.asarray(pp.laser.input_transmission).reshape(-1)[self.b if pp.batch > 1 and
                                                                         np.size(pp.laser.input_transmission) > 1
                                                                         else 0])
        self.p_line_w = float(np.asarray(pp.laser.power_w)[self.b, self.line]) * tr
        # every TX line on the bus at its chip-input power: in bank protocols the other lines leak into the drop
        # monitor and heat the ring (07 P1-9: the drop sums every line and order); chip-input powers here, the bus
        # filtering by the rings upstream is applied by the bank protocols (_bank_codes, bus_line_powers)
        tx = [i for i, d_ in enumerate(pp.laser.direction) if d_ == "tx"] if len(pp.laser.direction) else \
            list(range(int(pp.n_lines)))
        self.other_lines = np.array([i for i in tx if i != self.line], dtype=np.int64)
        self.lam_lines = C0 / np.asarray(pp.laser.line_freq_hz, dtype=float)[self.b]
        self.p_lines_w = np.asarray(pp.laser.power_w, dtype=float)[self.b] * tr
        # data path and driver heat (07 P1-8)
        d = pp.datapath
        self.swing_pp_v = float(np.asarray(d.swing_pp_v)[self.b, self.n])
        self.level_mode = int(np.asarray(d.level_convention).reshape(-1)[self.b if np.size(d.level_convention) > 1
                                                                          else 0])
        self.bitrate_hz = float(np.asarray(d.bitrate_hz).reshape(-1)[self.b if np.size(d.bitrate_hz) > 1 else 0])
        td = np.asarray(d.transition_density).reshape(-1)
        td = float(td[self.b if td.size > 1 else 0])
        self.transition_density = td if math.isfinite(td) else 0.5
        self._drv = tuple(float(np.asarray(x).reshape(-1)[self.b if np.size(x) > 1 else 0])
                          for x in (d.driver_output_resistance_ohm, d.driver_static_power_w,
                                    d.driver_energy_per_transition_j))
        # monitor
        self._mon = cache["monitor"]
        self.cache = cache

    # ------------------------------------------------------------------------------------------- optics
    def optics(self, lam, t_ring, t_heater=None, v_lo=0.0, v_hi=None) -> dict:
        """Optics of the ring at laser wavelengths ``lam`` (L,) for ring temperatures ``t_ring`` (M,), heater
        temperatures ``t_heater`` (M,) (only the loss uses it) and junction levels v_lo, v_hi (M,) or scalars
        (v_hi None: CW at v_lo). Returns per-unit-input-power fractions (M, L): thru0/thru1, drop0/drop1, abs0/abs1
        (heat: eta x intrinsic loss), loss0/loss1, plus shift (M, 2) m, cj (M, 2) F, rs (M, 2) ohm, k2t/k2d/a0/trt
        (M, L) and the junction flags."""
        lam = np.ascontiguousarray(np.atleast_1d(np.asarray(lam, dtype=float)))
        tr = np.ascontiguousarray(np.atleast_1d(np.asarray(t_ring, dtype=float)))
        M, L = tr.size, lam.size
        # real (writeable, contiguous) arrays: numba rejects read-only broadcast views in the future
        th_ = tr if t_heater is None else _full(t_heater, (M,))
        v0 = _full(v_lo, (M,))
        cw = v_hi is None
        v1 = v0 if cw else _full(v_hi, (M,))
        rp = _full(self.rp, (M, ph.N_RP))
        out = {k: np.zeros((M, L)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
        out.update(shift=np.zeros((M, 2)), cj=np.zeros((M, 2)), rs=np.zeros((M, 2)), flags=np.zeros(M, np.int8))
        dn, al, cc, rs = self.tab
        ph.ring_optics_row(rp, lam, tr, th_, v0, v1, self.grid, dn, al, cc, rs, out["phi0"], out["phi1"],
                           out["a0"], out["a1"], out["e0"], out["e1"], out["k2t"], out["k2d"], out["trt"],
                           out["shift"], out["cj"], out["rs"], out["flags"])
        r0 = op.ring_exact(out["phi0"], out["k2t"], out["k2d"], out["a0"], out["trt"], rb=self.rb)
        res = {"thru0": np.abs(r0.thru) ** 2, "drop0": np.abs(r0.drop) ** 2, "loss0": r0.p_loss}
        res["abs0"] = out["e0"] * res["loss0"]
        if cw:
            for k in ("thru", "drop", "loss", "abs"):
                res[k + "1"] = res[k + "0"]
        else:
            r1 = op.ring_exact(out["phi1"], out["k2t"], out["k2d"], out["a1"], out["trt"], rb=self.rb)
            res.update(thru1=np.abs(r1.thru) ** 2, drop1=np.abs(r1.drop) ** 2, loss1=r1.p_loss)
            res["abs1"] = out["e1"] * res["loss1"]
        res.update(shift=out["shift"], cj=out["cj"], rs=out["rs"], k2t=out["k2t"], k2d=out["k2d"], a0=out["a0"],
                   trt=out["trt"], flags=out["flags"])
        return res

    def junction_shift_m(self, v) -> np.ndarray:
        """Resonance shift (m) of the junction alone at reverse bias v (relative to the reference bias)."""
        v = np.ascontiguousarray(np.atleast_1d(np.asarray(v, dtype=float)))
        M = v.size
        rp = _full(self.rp, (M, ph.N_RP))
        out = [np.zeros(M) for _ in range(4)]
        fl = np.zeros(M, np.int8)
        dn, al, cc, rs = self.tab
        ph.resonance_shift_row(rp, np.full(M, self.t_ref), v, self.grid, dn, al, cc, rs, out[0], out[1], out[2],
                               out[3], fl)
        return out[0]

    def thermal_shift_m(self, t_k) -> np.ndarray:
        """R1 resonance shift (m) from T_ref to t_k (07 P0-3)."""
        t = np.asarray(t_k, dtype=float)
        return self.s0 * ((t - self.t_ref) + 0.5 * self.beta * ((t - self.t0) ** 2 - (self.t_ref - self.t0) ** 2))

    def s_at(self, t_k) -> np.ndarray:
        """dlambda/dT (m/K) at ring temperature t_k: s0 (1 + beta (T - T0))."""
        return self.s0 * (1.0 + self.beta * (np.asarray(t_k, dtype=float) - self.t0))

    def t_for_shift(self, shift_m) -> np.ndarray:
        """Ring temperature at which the R1 shift from T_ref equals shift_m (inverse of thermal_shift_m)."""
        from mrm.scenarios.calibrate import resonance_to_rise_k
        d = np.asarray(shift_m, dtype=float)
        return self.t_ref + resonance_to_rise_k(d, s0_m_per_k=self.s0, beta_per_k=self.beta, t_ref_k=self.t0,
                                                t_base_k=self.t_ref)

    def _phase_at(self, x) -> np.ndarray:
        """Round-trip phase (rad) of the cold phase curve (T_ref, reference bias) at wavelengths x (L,)."""
        x = np.ascontiguousarray(np.atleast_1d(np.asarray(x, dtype=float)))
        rp = np.ascontiguousarray(self.rp[None, :])
        outs = [np.zeros((1, x.size)) for _ in range(9)]
        o2 = [np.zeros((1, 2)) for _ in range(3)]
        vr = np.array([self.rp[ph.RP_V_REF]])
        ph.ring_optics_row(rp, x, np.array([self.t_ref]), np.array([self.t_ref]), vr, vr, self.grid, *self.tab,
                           *outs, *o2, np.zeros(1, np.int8))
        return outs[0][0]

    def order_offset_m(self, k: int = 0) -> float:
        """Cold position of order m + k minus the cold resonance lambda_res (m the design order; k > 0 bluer orders),
        from the phase model's cold phase curve (07 P1-6; the comb then translates rigidly)."""
        k = int(k)
        if k == 0:
            return 0.0
        cache = self.__dict__.setdefault("_orders", {})
        if k not in cache:
            m = round(float(self._phase_at(self.lam_res)[0]) / (2 * math.pi))
            fsr = _fsr_m(self)
            g = self.lam_res - k * fsr

            def f(x):
                return float(self._phase_at(x)[0]) - 2 * math.pi * (m + k)
            cache[k] = optimize.brentq(f, g - 0.4 * fsr, g + 0.4 * fsr, xtol=1e-16) - self.lam_res
        return cache[k]

    def t_res(self, lam, v=0.0, k: int = 0) -> float:
        """Ring temperature at which order m + k (m: the order of the cold resonance) sits exactly at ``lam`` with
        the junction at v."""
        return float(self.t_for_shift(lam - self.lam_res - self.order_offset_m(k) - float(self.junction_shift_m(v)[0])))

    def first_order(self, lam, v=0.0, t_floor: float | None = None, k_range=range(-3, 4)) -> int:
        """The order a heater-only acquisition locks (07 P1-12, PAT-3 ramp from code 0): the order k whose
        resonance reaches ``lam`` at the lowest ring temperature not below ``t_floor`` (the ring with the heater off;
        default T_ref)."""
        tf = self.t_ref if t_floor is None else float(t_floor)
        best, kb = math.inf, 0
        for k in k_range:
            try:
                t = self.t_res(lam, v, k)
            except ValueError:
                continue
            if t >= tf and t < best:
                best, kb = t, k
        return kb

    def linewidth_k(self, t_k=None) -> float:
        """The loaded linewidth (FWHM) in ring-temperature units, lambda/Q_L / s(T) (K)."""
        t = self.t_ref if t_k is None else t_k
        return float(self.lam_res / self.q_loaded / self.s_at(t))

    # ------------------------------------------------------------------------------------------- heater
    def _r_heater(self, t_h):
        return self.r0 * (1.0 + self.alpha * (np.asarray(t_h, dtype=float) - self.t_r0))

    def u_from_power(self, p_w, t_h) -> np.ndarray:
        """Normalized drive u that delivers heater power p_w at heater-resistor temperature t_h (inverse driver law;
        odd continuation for p < 0 so that roots stay bracketed). bench_voltage: u is the voltage in V."""
        p = np.asarray(p_w, dtype=float)
        r = self._r_heater(t_h)
        sp = np.sign(p)
        ap = np.abs(p)
        if self.law in (el.LAW_VOLTAGE, -1):
            fs = 1.0 if self.law == -1 else self.v_fs
            return sp * np.sqrt(ap * r) / fs
        if self.law == el.LAW_CURRENT:
            return sp * np.sqrt(ap / r) / self.i_fs
        if self.law == el.LAW_POWER:
            return p / self.p_fs
        return p * r / self.v_fs ** 2                                   # dac_pwm: u = <(V/V_FS)^2>

    def power_from_u(self, u, t_pre_k, g_self=None) -> np.ndarray:
        """Heater power (W) at drive u with the TCR solved implicitly (electrical.heater_steady_state): t_pre_k is
        the heater-resistor temperature without this heater's own power, g_self (default Z_hh) its own gain. The
        driver's output compliance is applied as in the plant (electrical.heater_power_row)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if self.law == -1:
            law, vfs = el.LAW_VOLTAGE, 1.0
        else:
            law, vfs = self.law, self.v_fs
        res = el.heater_steady_state(drive_frac=u, law=law, v_full_scale_v=vfs, i_full_scale_a=self.i_fs,
                                     p_full_scale_w=self.p_fs, compliance_v=self.v_comp, r_cold_ohm=self.r0,
                                     tcr_per_k=self.alpha,
                                     r_th_heater_k_per_w=self.z["hh"] if g_self is None else g_self,
                                     t_base_minus_ref_k=np.asarray(t_pre_k, dtype=float) - self.t_r0)
        return res["p_w"]

    def compliance_ok(self, p_w, t_h) -> np.ndarray:
        """True where heater power p_w at heater-resistor temperature t_h needs no more than the driver's
        compliance voltage, V = sqrt(P R(T_H)) <= V_comp (the inverse law u_from_power ignores compliance)."""
        p = np.maximum(np.asarray(p_w, dtype=float), 0.0)
        return np.sqrt(p * self._r_heater(t_h)) <= self.v_comp * (1.0 + 1e-12)

    def _table(self):
        if self._u_table is None:
            if self.law == -1:
                raise ValueError("bench_voltage has no DAC codes")
            codes = np.arange(1 << self.bits, dtype=np.int64)
            u = np.empty(codes.size)
            rip = np.empty(codes.size)
            el.heater_target_row(codes, self.law, self.bits, self.pwm_bits, self.bit_err, self.bow, 0, 0, 1, u, rip)
            self._u_table = (codes.astype(float), u, np.maximum.accumulate(u))
        return self._u_table

    def code_to_u(self, code) -> np.ndarray:
        """Normalized drive of DAC codes through the realized DAC transfer (bit errors and INL bow included)."""
        c, u, _ = self._table()
        return np.interp(np.asarray(code, dtype=float), c, u)

    def u_to_code(self, u) -> np.ndarray:
        """Inverse DAC transfer (fractional code, linear between codes; monotone envelope where DNL bends it)."""
        c, _, um = self._table()
        return np.interp(np.asarray(u, dtype=float), um, c)

    # ------------------------------------------------------------------------------------------- junction / data
    def nrz_levels(self, bias_v: float, swing_scale: float = 1.0) -> tuple[float, float]:
        """(v_lo, v_hi) of the NRZ levels at DC bias bias_v (electrical.nrz_levels_row; level 1 = v_hi)."""
        lo, hi, fl = np.zeros(1), np.zeros(1), np.zeros(1, np.int8)
        el.nrz_levels_row(np.array([float(bias_v)]), np.array([self.swing_pp_v * swing_scale]), self.level_mode,
                          el.JUNCTION_GRID_V_MIN, el.JUNCTION_GRID_V_MAX, lo, hi, fl)
        return float(lo[0]), float(hi[0])

    def data_heat(self, v_lo: float, v_hi: float, *, enable: bool = True) -> tuple[float, float]:
        """(P_j, P_d) W: junction CV^2 and driver heat of random NRZ at the levels (07 P1-8)."""
        if not enable:
            return 0.0, 0.0
        g = self.grid
        cc = self.tab[2]
        ceff = np.zeros(1)
        el.junction_swing_capacitance_row(np.array([v_lo]), np.array([v_hi]), g, cc,
                                          np.array([self.rp[ph.RP_FILL]]), np.array([self.rp[ph.RP_LEN]]), ceff)
        o = self.optics(self.lam_line, [self.t_ref], None, v_lo, v_hi)
        r_s = 0.5 * float(o["rs"][0, 0] + o["rs"][0, 1])
        pj, pd = np.zeros(1), np.zeros(1)
        r_drv, p_st, e_tr = self._drv
        el.data_activity_heat_row(ceff, np.array([v_hi - v_lo]), self.bitrate_hz if math.isfinite(self.bitrate_hz)
                                  else 0.0, np.array([self.transition_density]), np.array([1.0]),
                                  np.array([r_s]), np.array([r_drv if math.isfinite(r_drv) else 0.0]),
                                  np.array([p_st]), np.array([e_tr]), 0, pj, pd)
        return float(pj[0]), float(pd[0])

    # ------------------------------------------------------------------------------------------- monitor
    def monitor_current_a(self, p_drop_w, p_bus_w=0.0, t_pd_k=None) -> np.ndarray:
        """Monitor photocurrent (signal + dark) in A for drop power p_drop_w at the PD (sensing.MonitorHw)."""
        hw = self._mon
        p = np.atleast_1d(np.asarray(p_drop_w, dtype=float))
        out = np.empty(p.shape)
        for i, pv in enumerate(p.ravel()):
            pd_ = np.zeros((hw.B, hw.N))
            pd_[self.b, self.n] = pv
            pb = np.zeros((hw.B, hw.N))
            pb[self.b, self.n] = float(np.asarray(p_bus_w).ravel()[i if np.size(p_bus_w) > 1 else 0])
            t = None
            if t_pd_k is not None:
                t = np.full((hw.B, hw.N), float(np.asarray(t_pd_k).ravel()[i if np.size(t_pd_k) > 1 else 0]))
            i_s, i_d = hw.photocurrent_a(pd_, pb, t)
            out.ravel()[i] = i_s[self.b, self.n] + i_d[self.b, self.n]
        return out

    # ------------------------------------------------------------------------------------------- static states
    def other_lines_terms(self, t_grid, t_heater=None, v_lo: float = 0.0, v_hi: float | None = None,
                          rho: float = 0.5, p_lines_w=None) -> tuple[np.ndarray, np.ndarray]:
        """(absorbed heat, drop power) in W, (M,), from every TX line except the ring's own, on the ring-temperature
        grid (level average for NRZ). ``p_lines_w`` (L,): the line powers at this ring (default: the chip-input
        powers, i.e. no upstream filtering; :func:`_bank_codes` passes the bus-filtered ones). Every order of the
        ring counts (the exact ring is periodic), so a neighbouring order near another line (BND-1) shows up here."""
        tg = np.atleast_1d(np.asarray(t_grid, dtype=float))
        if self.other_lines.size == 0:
            return np.zeros(tg.size), np.zeros(tg.size)
        idx = self.other_lines
        o = self.optics(self.lam_lines[idx], tg, t_heater, v_lo, v_hi)
        rho_ = 0.0 if v_hi is None else float(rho)
        p = (self.p_lines_w if p_lines_w is None else np.asarray(p_lines_w, dtype=float))[idx]
        ab = (rho_ * o["abs1"] + (1.0 - rho_) * o["abs0"]) @ p
        dr = (rho_ * o["drop1"] + (1.0 - rho_) * o["drop0"]) @ p
        return ab, dr

    def state_grid(self, lam, t_grid, *, t_base: float, p_in_w: float, v_lo: float = 0.0, v_hi: float | None = None,
                   rho: float = 0.5, p_j: float = 0.0, p_d: float = 0.0, agg_r: float = 0.0, agg_h: float = 0.0,
                   heater_loss_passes: int = 2, lines: str = "own", p_lines_w=None) -> dict:
        """Everything on a ring-temperature grid t_grid (M,) at laser wavelengths lam (L,): the optics (per unit
        power) and, for the given base temperature, input power, junction levels and extra heat, the absorbed power,
        heater power and heater temperature needed, and the drive u(T) (module docstring equations). Arrays (M, L).
        ``lines='all'`` adds the other TX lines' absorption (heat) and drop leakage (``drop_other_w`` (M,), W) at
        their chip-input powers or ``p_lines_w`` (:meth:`other_lines_terms`); ``'own'`` (default): the probe line
        only.
        The optical loss depends on the heater temperature only through dln_alpha_dt_heater; when that coefficient
        is nonzero the optics are re-evaluated ``heater_loss_passes`` times at the updated T_H."""
        if lines not in ("own", "all"):
            raise ValueError(f"lines must be 'own' or 'all', got {lines!r}")
        tg = np.asarray(t_grid, dtype=float)
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        kw = dict(t_base=t_base, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, rho=rho, p_j=p_j, p_d=p_d, agg_r=agg_r,
                  agg_h=agg_h)
        if self.rp[ph.RP_CT_HEATER] == 0.0:
            ex = self.other_lines_terms(tg, None, v_lo, v_hi, rho, p_lines_w) if lines == "all" else (0.0, 0.0)
            return self.heat_balance(self.optics(lam, tg, None, v_lo, v_hi), tg, p_abs_extra=ex[0],
                                     p_drop_extra=ex[1], **kw)
        if lam.size > 1:                     # the heater temperature differs per wavelength: one column at a time
            cols = [self.state_grid(lam[i:i + 1], tg, heater_loss_passes=heater_loss_passes, lines=lines,
                                    p_lines_w=p_lines_w, **kw) for i in range(lam.size)]
            out = {}
            for k, v in cols[0].items():
                out[k] = (np.concatenate([c[k] for c in cols], axis=1) if isinstance(v, np.ndarray) and v.ndim == 2
                          and v.shape[1] == 1 else v)
            return out
        z = self.z
        t_h = tg + (z["hh"] / z["hr"] - 1.0) * (tg - t_base)          # first guess: heater-only ratio
        for _ in range(max(1, int(heater_loss_passes))):
            ex = self.other_lines_terms(tg, t_h, v_lo, v_hi, rho, p_lines_w) if lines == "all" else (0.0, 0.0)
            o = self.heat_balance(self.optics(lam, tg, t_h, v_lo, v_hi), tg, p_abs_extra=ex[0], p_drop_extra=ex[1],
                                  **kw)
            t_h = o["t_h"][:, 0]
        return o

    def heat_balance(self, o: dict, t_grid, *, t_base: float, p_in_w: float, v_lo: float = 0.0,
                     v_hi: float | None = None, rho: float = 0.5, p_j: float = 0.0, p_d: float = 0.0,
                     agg_r: float = 0.0, agg_h: float = 0.0, p_abs_extra=0.0, p_drop_extra=0.0) -> dict:
        """Add the heat balance to optics ``o`` evaluated on ``t_grid`` (see state_grid): p_abs, p_h, t_h, u, t_pd,
        the level averages drop_avg, thru_avg (per unit power of the probe line), reloma = |T1 - T0|, and
        drop_other_w (W). ``p_abs_extra`` / ``p_drop_extra`` (scalar or (M,), W): absorbed heat and drop power from
        other lines. Returns a new dict (o is not changed)."""
        tg = np.asarray(t_grid, dtype=float)
        z = self.z
        rho_ = 0.0 if v_hi is None else float(rho)
        o = dict(o)
        ex = np.broadcast_to(np.asarray(p_abs_extra, dtype=float), tg.shape)
        p_abs = p_in_w * (rho_ * o["abs1"] + (1.0 - rho_) * o["abs0"]) + ex[:, None]
        o["drop_other_w"] = np.array(np.broadcast_to(np.asarray(p_drop_extra, dtype=float), tg.shape))
        p_h = (tg[:, None] - t_base - z["or"] * p_abs - z["jr"] * p_j - z["dr"] * p_d - agg_r) / z["hr"]
        t_h = t_base + z["hh"] * p_h + z["oh"] * p_abs + z["jh"] * p_j + z["dh"] * p_d + agg_h
        o.update(p_abs=p_abs, p_h=p_h, t_h=t_h, u=self.u_from_power(p_h, t_h), rho=rho_,
                 drop_avg=rho_ * o["drop1"] + (1.0 - rho_) * o["drop0"],
                 thru_avg=rho_ * o["thru1"] + (1.0 - rho_) * o["thru0"],
                 reloma=np.abs(o["thru1"] - o["thru0"]),
                 t_pd=t_base + z["hp"] * p_h + z["op"] * p_abs + z["jp"] * p_j + z["dp"] * p_d)
        return o

    def lock(self, *, mode: str = "half_max", lam: float | None = None, t_base: float, p_in_w: float,
             bias_v: float = 0.0, modulate: bool = False, rho: float = 0.5, data_heat: bool | None = None,
             fraction: float | None = None, side: str = "hot", stable_only: bool = True, n_grid: int = 2001,
             span_linewidths: float = 6.0, agg_r: float = 0.0, agg_h: float = 0.0,
             order: int | str = 0, lines: str = "own", p_lines_w=None) -> LockPoint:
        """Steady state of the ring held at a lock point by an ideal (settled) controller.

        mode: ``peak`` (maximum drop power, CW-averaged), ``half_max`` / ``fraction`` (drop = fraction x peak on the
        ``hot`` side, i.e. the laser blue of the resonance, the side that is stable against self-heating; ``cold``
        selects the other flank), ``oracle`` (maximum |T1 - T0|, i.e. maximum relOMA; with ``stable_only`` restricted
        to fixed-drive-stable points), ``resonance`` (the resonance exactly on the line). modulate: NRZ at the
        datapath swing around bias_v (needed for ``oracle``); data_heat defaults to ``modulate``. order: the
        resonance order locked, relative to the design order (0), or ``first`` (:meth:`first_order` from the ring
        with its heater off at this base temperature). lines: ``own`` (the ring's line only) or ``all`` (every TX line
        on, as in the bank: the other lines heat the ring and leak into the drop monitor; the drop levels of
        ``peak`` / ``half_max`` / ``fraction`` are then those of the monitor's total drop power, which is what a
        fraction-of-peak controller measures). p_lines_w (L,): the other lines' powers at the ring (default: their
        chip-input powers).
        """
        lam = self.lam_line if lam is None else float(lam)
        if modulate:
            v_lo, v_hi = self.nrz_levels(bias_v)
        else:
            v_lo, v_hi = float(bias_v), None
        dh = modulate if data_heat is None else bool(data_heat)
        p_j, p_d = self.data_heat(v_lo, v_hi if v_hi is not None else v_lo, enable=dh)
        v_mid = 0.5 * (v_lo + (v_hi if v_hi is not None else v_lo))
        if order == "first":
            order = self.first_order(lam, v_mid, t_base + self.z["jr"] * p_j + self.z["dr"] * p_d + agg_r)
        t_r = self.t_res(lam, v_mid, int(order))
        w = self.linewidth_k(t_r)
        tg = np.linspace(t_r - span_linewidths * w, t_r + span_linewidths * w, int(n_grid))
        st = self.state_grid(lam, tg, t_base=t_base, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, rho=rho, p_j=p_j,
                             p_d=p_d, agg_r=agg_r, agg_h=agg_h, lines=lines, p_lines_w=p_lines_w)
        # the drop seen by the monitor (W): own line, plus the other lines' leakage with lines='all'
        drop_w = p_in_w * st["drop_avg"][:, 0] + st["drop_other_w"]
        # level search on the per-unit own drop for lines='own' (works at P_in -> 0), on the monitor power otherwise
        # (unless no light reaches the ring at all)
        drop = st["drop_avg"][:, 0] if (lines == "own" or not np.max(drop_w) > 0.0) else drop_w
        dudt = np.gradient(st["u"][:, 0], tg)
        stable = dudt > 0
        ipk = int(np.argmax(drop))
        if mode == "peak":
            i = _refine_argmax(tg, drop, ipk)
        elif mode == "resonance":
            i = float(np.interp(t_r, tg, np.arange(tg.size)))
        elif mode in ("half_max", "fraction"):
            f = 0.5 if mode == "half_max" else float(fraction)
            lev = f * drop[ipk]
            if side == "hot":
                seg = np.arange(ipk, tg.size)
            else:
                seg = np.arange(ipk, -1, -1)
            j = seg[np.flatnonzero(drop[seg] <= lev)]
            if j.size == 0:
                raise ValueError("the lock level is not reached inside the grid; widen span_linewidths")
            j = int(j[0])
            k = j - 1 if side == "hot" else j + 1
            i = k + (drop[k] - lev) / (drop[k] - drop[j]) * (j - k)
        elif mode == "oracle":
            if v_hi is None:
                raise ValueError("oracle lock needs modulate=True")
            om = st["reloma"][:, 0].copy()
            if stable_only:
                om[~stable] = -np.inf
            if not np.any(np.isfinite(om)):
                raise ValueError("no stable operating point on the grid")
            i = _refine_argmax(tg, om, int(np.argmax(om)))
        else:
            raise ValueError(f"unknown lock mode {mode!r}")
        idx = np.arange(tg.size, dtype=float)

        def at(a):
            return float(np.interp(i, idx, np.asarray(a, dtype=float)))
        t_lock = at(tg)
        pl = self.p_lines_w if p_lines_w is None else np.asarray(p_lines_w, dtype=float)
        p_bus = p_in_w + (float(np.sum(pl[self.other_lines])) if lines == "all" else 0.0)
        return LockPoint(mode=mode, order=int(order), t_ring_k=t_lock, t_res_k=t_r, p_abs_w=at(st["p_abs"][:, 0]),
                         p_drop_w=at(drop_w), p_thru_w=p_in_w * at(st["thru_avg"][:, 0]),
                         reloma=at(st["reloma"][:, 0]), drop_peak_w=float(drop_w[ipk]), p_j_w=p_j, p_d_w=p_d,
                         p_heater_w=at(st["p_h"][:, 0]), t_heater_k=at(st["t_h"][:, 0]), u=at(st["u"][:, 0]),
                         stable=bool(at(dudt) > 0),
                         detuning_m=float(lam - (self.lam_res + self.order_offset_m(int(order))
                                                 + self.thermal_shift_m(t_lock) + self.junction_shift_m(v_mid)[0])),
                         t_pd_k=at(st["t_pd"][:, 0]), p_drop_other_w=at(st["drop_other_w"]), p_bus_w=p_bus)


def _full(x, shape) -> np.ndarray:
    """A writeable C-contiguous float64 copy of x broadcast to shape."""
    return np.array(np.broadcast_to(np.asarray(x, dtype=float), shape), dtype=np.float64, order="C")


def _refine_argmax(x, y, i) -> float:
    """Fractional index of the maximum by a parabola through the three samples around i."""
    if i <= 0 or i >= len(y) - 1 or not np.all(np.isfinite(y[i - 1:i + 2])):
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2.0 * y1 + y2
    if den >= 0:
        return float(i)
    return float(i + 0.5 * (y0 - y2) / den)


def _plant_cache(pp) -> dict:
    """Per-PlantParams objects shared by every StaticRing (phase rows, DC gains, heater and monitor adapters)."""
    c = getattr(pp, "_replica_cache", None)
    if c is not None:
        return c
    from mrm.plant.sensing import MonitorHw
    c = {"phase": ph.PhaseParams.from_plant_params(pp), "dc": th.dc_gain(pp.thermal.system),
         "heater": el.HeaterDriverParams.from_plant_params(pp), "monitor": MonitorHw.from_plant_params(pp)}
    try:
        object.__setattr__(pp, "_replica_cache", c)
    except Exception:                                    # pragma: no cover - frozen without __dict__
        pass
    return c


def bank_gains(pp, b: int = 0) -> dict:
    """N x N DC gain matrices of world b (K/W): keys '<s><o>' with s in h (heater), o (waveguide absorption),
    j (junction), d (driver) and o in r (ring_mode_weighted), h (heater_resistor), p (monitor_pd; zeros when the
    system has no PD observer), with [i, j] = source of ring j -> observer of ring i."""
    sysm = pp.thermal.system
    G = _plant_cache(pp)["dc"][b]
    N = int(pp.n_rings)
    obs = {"r": sysm.ring_obs, "h": sysm.heater_obs, "p": sysm.pd_obs}
    src = {"h": sysm.heater_src, "o": sysm.wg_src, "j": sysm.junction_src, "d": sysm.driver_src}
    out = {}
    for sk, si in src.items():
        for ok, oi in obs.items():
            m = np.zeros((N, N))
            if si.size and oi.size:
                for i in range(N):
                    for j in range(N):
                        if oi[i] >= 0 and si[j] >= 0:
                            m[i, j] = G[oi[i], si[j]]
            out[sk + ok] = m
    return out


def _bus_segment_t2(pp, b: int = 0) -> float:
    """Power transmission of the bus segment between consecutive rings, exp(-bus_loss x spacing) (ENGINE_BRIEF 4.2;
    1 when the spacing is not given)."""
    sp_ = getattr(pp.layout, "ring_spacing_on_bus_m", None)
    if sp_ is None:
        return 1.0
    sp_ = float(np.asarray(sp_, dtype=float).reshape(-1)[0])
    loss = np.asarray(pp.laser.bus_loss_per_m, dtype=float).reshape(-1)
    loss = float(loss[b if loss.size > 1 else 0])
    return float(math.exp(-loss * sp_)) if math.isfinite(sp_) and math.isfinite(loss) else 1.0


def bus_line_powers(pp, b: int, t_ring_k, *, bias_v: float | None = None, v_lo=None, v_hi=None, rho: float = 0.5,
                    p_chip_w=None) -> np.ndarray:
    """Line powers (W) at the input of every ring of world b, (N, L): a single forward pass along
    ``layout.bus_order`` from the chip-input powers (``p_chip_w`` (L,) or the preset's), through each upstream ring's
    level-averaged through transmission at its temperature ``t_ring_k`` (N,) and the bus segment loss (no
    reflections, as the plant's bus pass). Junction levels: (v_lo, v_hi) per ring (N,) or the NRZ levels at
    ``bias_v``; v_hi None everywhere is CW."""
    N = int(pp.n_rings)
    t = np.asarray(t_ring_k, dtype=float).reshape(-1)
    rings = [StaticRing(pp, b=b, n=n) for n in range(N)]
    p_at = (rings[0].p_lines_w.copy() if p_chip_w is None else
            np.array(np.broadcast_to(np.asarray(p_chip_w, dtype=float), rings[0].p_lines_w.shape)))
    seg = _bus_segment_t2(pp, b)
    out = np.zeros((N, p_at.size))
    for q, n in enumerate(np.asarray(pp.layout.bus_order, dtype=np.int64).reshape(-1)):
        if q > 0:
            p_at = p_at * seg
        out[n] = p_at
        rg = rings[n]
        if bias_v is not None:
            lo, hi = rg.nrz_levels(bias_v)
        else:
            lo = 0.0 if v_lo is None else float(np.asarray(v_lo, dtype=float).reshape(-1)[n])
            hi = None if v_hi is None else float(np.asarray(v_hi, dtype=float).reshape(-1)[n])
        o = rg.optics(rg.lam_lines, [t[n]], None, lo, hi)
        r_ = 0.0 if hi is None else float(rho)
        p_at = p_at * (r_ * o["thru1"][0] + (1.0 - r_) * o["thru0"][0])
    return out


def _relax(t_grid: np.ndarray, g: np.ndarray, i0: float) -> tuple[float, float, bool]:
    """Quasi-static relaxation on a grid: from the fractional index i0, move in the direction of sign(g) (g = u_0 -
    u(T): the ring heats while the drive exceeds the drive the temperature needs) to the first sign change, and
    return (T_root, fractional index, hit_edge). Exact branch continuation for a slow sweep (module docstring)."""
    n = g.size
    i = int(min(max(round(i0), 0), n - 1))
    if g[i] == 0.0:
        return float(t_grid[i]), float(i), False
    if g[i] > 0:
        j = i
        while j + 1 < n and g[j + 1] > 0:
            j += 1
        if j + 1 >= n:
            return float(t_grid[-1]), float(n - 1), True
        f = g[j] / (g[j] - g[j + 1])
        return float(t_grid[j] + f * (t_grid[j + 1] - t_grid[j])), j + f, False
    j = i
    while j - 1 >= 0 and g[j - 1] < 0:
        j -= 1
    if j - 1 < 0:
        return float(t_grid[0]), 0.0, True
    f = g[j] / (g[j] - g[j - 1])
    return float(t_grid[j] - f * (t_grid[j] - t_grid[j - 1])), j - f, False


# =====================================================================================================================
# Protocol primitives (spectra, sweeps)
# =====================================================================================================================
def _dbm_to_w(p_dbm) -> float:
    """Optical power in W from dBm; None (not printed) means P_in -> 0."""
    return 0.0 if p_dbm is None else 1e-3 * 10.0 ** (float(p_dbm) / 10.0)


def _fsr_m(ring: StaticRing) -> float:
    """Free spectral range in wavelength at the cold resonance, lambda^2 FSR_Hz / c (m)."""
    return ring.lam_res ** 2 * ring.fsr_hz / C0


def _ring_temps_from_drive(ring: StaticRing, u: float, t_base: float) -> tuple[float, float, float]:
    """(T_ring, T_heater, P_h) at drive u without light or data heat (TCR solved; P_in -> 0)."""
    p_h = float(ring.power_from_u([u], t_base)[0])
    return t_base + ring.z["hr"] * p_h, t_base + ring.z["hh"] * p_h, p_h


def _notch(ring: StaticRing, *, t_ring: float, t_heater: float, v: float, lam_guess: float,
           half_window: float | None = None, n: int = 4001) -> dict:
    """Low-power through spectrum of one notch and its read-outs (bench protocol at P_in -> 0).

    The spectrum spans lam_guess +- half_window (default FSR/2, so the maximum is the anti-resonance). Returns
    lam_min (m, the notch minimum refined by a bounded scalar minimization), t_min, t_max (linear through
    transmission), er_db = 10 log10(t_max / t_min), fwhm_m (full width at half depth, linear: the loaded linewidth),
    fwhm_3db_m (width where T = t_max / 2, i.e. 3 dB below the off-resonance level), q (lam_min / fwhm_m)."""
    half = 0.5 * _fsr_m(ring) if half_window is None else float(half_window)
    lam = np.linspace(lam_guess - half, lam_guess + half, int(n))
    T = ring.optics(lam, [t_ring], [t_heater], v)["thru0"][0]
    i = int(np.argmin(T))

    def f(x):
        return float(ring.optics([x], [t_ring], [t_heater], v)["thru0"][0, 0])
    lo, hi = lam[max(i - 2, 0)], lam[min(i + 2, lam.size - 1)]
    r = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-17, "maxiter": 200})
    lam_min, t_min = (float(r.x), float(r.fun)) if r.fun <= T[i] else (float(lam[i]), float(T[i]))
    t_max = float(T.max())

    def width(level):
        if not t_min < level < t_max:
            return math.nan
        left = np.flatnonzero((T >= level) & (lam < lam_min))
        right = np.flatnonzero((T >= level) & (lam > lam_min))
        if left.size == 0 or right.size == 0:
            return math.nan
        a = optimize.brentq(lambda x: f(x) - level, lam[left[-1]], lam_min, xtol=1e-17)
        b = optimize.brentq(lambda x: f(x) - level, lam_min, lam[right[0]], xtol=1e-17)
        return b - a
    fwhm = width(0.5 * (t_max + t_min))
    return {"lam_min": lam_min, "t_min": t_min, "t_max": t_max,
            "er_db": 10.0 * math.log10(t_max / t_min) if t_min > 0 else math.inf,
            "fwhm_m": fwhm, "fwhm_3db_m": width(0.5 * t_max), "q": lam_min / fwhm if fwhm > 0 else math.nan}


def _sweep(ring: StaticRing, lam_grid, t_grid, *, u0: float, t_base: float, p_in_w: float, v_lo: float = 0.0,
           v_hi: float | None = None, rho: float = 0.5, direction: str = "up", start: str = "cold",
           state: dict | None = None, i_start: float | None = None) -> dict:
    """Quasi-static laser sweep at a fixed heater drive u0 with optical self-heating (exact branch continuation).

    direction ``up``: increasing wavelength (red sweep, the usual bench direction for a thermally broadened ring);
    ``down``: decreasing. start ``cold`` / ``hot``: the ring starts at the bottom / top of t_grid (or at i_start).
    Returns lam (L,) in sweep order, t_ring (L,), thru / drop (L,) per unit input power, hit_edge (L,) bool."""
    lam_grid = np.asarray(lam_grid, dtype=float)
    tg = np.asarray(t_grid, dtype=float)
    st = state if state is not None else ring.state_grid(lam_grid, tg, t_base=t_base, p_in_w=p_in_w, v_lo=v_lo,
                                                         v_hi=v_hi, rho=rho)
    g = u0 - st["u"]
    order = np.arange(lam_grid.size) if direction == "up" else np.arange(lam_grid.size)[::-1]
    i_prev = (0.0 if start == "cold" else float(tg.size - 1)) if i_start is None else float(i_start)
    out = {k: np.zeros(lam_grid.size) for k in ("t_ring", "thru", "drop", "idx")}
    edge = np.zeros(lam_grid.size, bool)
    idx = np.arange(tg.size, dtype=float)
    for j, l in enumerate(order):
        t, i_prev, e = _relax(tg, g[:, l], i_prev)
        out["t_ring"][j] = t
        out["idx"][j] = i_prev
        out["thru"][j] = np.interp(i_prev, idx, st["thru_avg"][:, l])
        out["drop"][j] = np.interp(i_prev, idx, st["drop_avg"][:, l])
        edge[j] = e
    out["lam"] = lam_grid[order]
    out["hit_edge"] = edge
    return out


def _dip_at_power(ring: StaticRing, *, u0: float, t_base: float, p_in_w: float, v: float, direction: str = "up",
                  coarse_pm: float = 4.0, fine_pm: float = 0.1) -> dict:
    """Position of the through minimum of a swept spectrum at power (thermal triangle). A coarse sweep over the whole
    dragged notch, then a fine re-sweep around the minimum that continues from the coarse state. Returns lam_min,
    t_min (through, per unit power), the coarse sweep and the edge flag."""
    t_cold, t_hcold, _ = _ring_temps_from_drive(ring, u0, t_base)
    lam0 = ring.lam_res + float(ring.thermal_shift_m(t_cold)) + float(ring.junction_shift_m(v)[0])
    fw = ring.lam_res / ring.q_loaded
    w = ring.linewidth_k(t_cold)
    dt_max = ring.z["or"] * p_in_w                       # absorbed fraction <= 1
    drag = float(ring.s_at(t_cold + dt_max)) * dt_max
    lam = np.arange(lam0 - 4 * fw, lam0 + drag + 4 * fw, coarse_pm * 1e-12)
    if direction == "down":
        lam = lam[::-1]
    step_t = w / 80.0
    tg = np.arange(t_cold - 3 * w, t_cold + dt_max + 3 * w, step_t)
    lam_sorted = np.sort(lam)
    sw = _sweep(ring, lam_sorted, tg, u0=u0, t_base=t_base, p_in_w=p_in_w, v_lo=v, direction=direction)
    j = int(np.argmin(sw["thru"]))
    j0 = max(j - 3, 0)
    lam_a, lam_b = sw["lam"][j0], sw["lam"][min(j + 3, sw["lam"].size - 1)]
    lam_f = np.arange(min(lam_a, lam_b), max(lam_a, lam_b) + 0.5 * fine_pm * 1e-12, fine_pm * 1e-12)
    t_c = sw["t_ring"][j]
    tgf = np.arange(min(t_c, sw["t_ring"][j0]) - 1.5 * w, max(t_c, sw["t_ring"][j0]) + 1.5 * w, w / 400.0)
    i_start = float(np.interp(sw["t_ring"][j0], tgf, np.arange(tgf.size)))
    swf = _sweep(ring, lam_f, tgf, u0=u0, t_base=t_base, p_in_w=p_in_w, v_lo=v, direction=direction,
                 i_start=i_start)
    k = int(np.argmin(swf["thru"]))
    return {"lam_min": float(swf["lam"][k]), "t_min": float(swf["thru"][k]), "t_ring_at_min": float(swf["t_ring"][k]),
            "coarse": sw, "hit_edge": bool(np.any(sw["hit_edge"]) or np.any(swf["hit_edge"])),
            "lam_cold": lam0}


def _resonance_position(ring: StaticRing, *, u0: float, t_base: float, v: float, p_in_w: float,
                        direction: str = "up") -> dict:
    """Measured resonance position: the through minimum of a laser sweep at P_in (P_in -> 0: the cold notch)."""
    if p_in_w <= 0.0:
        t_r, t_h, p_h = _ring_temps_from_drive(ring, u0, t_base)
        guess = ring.lam_res + float(ring.thermal_shift_m(t_r)) + float(ring.junction_shift_m(v)[0])
        nt = _notch(ring, t_ring=t_r, t_heater=t_h, v=v, lam_guess=guess)
        nt.update(t_ring=t_r, t_heater=t_h, p_heater_w=p_h)
        return nt
    return _dip_at_power(ring, u0=u0, t_base=t_base, p_in_w=p_in_w, v=v, direction=direction)


def _t_base_k(args: Mapping, ring: StaticRing) -> float:
    """Base (stage) temperature: args['t_c'] when it is a number, else the reference temperature of the cold
    resonance (optics.t_ref_k; the replicas assume the bench measured at the reference temperature)."""
    t = args.get("t_c")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t) + T0_K
    return ring.t_ref


def _topologies(args: Mapping, ring_default: str) -> list[str]:
    t = args.get("topology")
    if t is None:
        return [ring_default]
    return [t] if isinstance(t, str) else list(t)


def _plant_topology(pp, n: int = 0) -> str:
    return "add_drop" if float(np.asarray(pp.optics.kappa2_drop)[0, n]) > 0 else "all_pass"


# =====================================================================================================================
# Replicas: G25 bench protocols (laser sweeps at P_in -> 0 unless stated)
# =====================================================================================================================
def loaded_q_from_3db_width(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-1: low-power laser sweep through the notch, Q_L = lambda_min / FWHM (full width at half depth of the
    through notch, linear). Conditions as the target assumes (07 4.3 "low power, cold, 0 V"): P_in -> 0 unless
    ``p_in_dbm`` is given (then the swept triangle is read the same way), heater ``heater_v`` (default 0 V, bench
    voltage source), bias ``bias_v`` (default 0 V), stage at the reference temperature. Primary topology: the plant's
    own; ``all_pass`` (drop coupler removed) is a variant. Variant ``3db_below_max``: the width where T = T_max / 2."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    own = _plant_topology(pp, ring)
    topos = list(dict.fromkeys([own] + _topologies(a, own) + ["all_pass", "add_drop"]))
    v_h = float(a.get("heater_v") or 0.0)
    bias = float(a.get("bias_v") or 0.0)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    res = {}
    for tp in topos:
        r = StaticRing(pp, n=ring, topology=tp, driver_law="bench_voltage")
        tb = _t_base_k(a, r)
        if p_in <= 0:
            t_r, t_h, _ = _ring_temps_from_drive(r, v_h, tb)
            guess = r.lam_res + float(r.thermal_shift_m(t_r)) + float(r.junction_shift_m(bias)[0])
            nt = _notch(r, t_ring=t_r, t_heater=t_h, v=bias, lam_guess=guess)
            res[tp] = {"q": nt["q"], "q_3db": nt["lam_min"] / nt["fwhm_3db_m"], "er_db": nt["er_db"],
                       "fwhm_pm": nt["fwhm_m"] * 1e12, "lam_min_nm": nt["lam_min"] * 1e9}
        else:
            d = _dip_at_power(r, u0=v_h, t_base=tb, p_in_w=p_in, v=bias)
            sw = d["coarse"]
            tmin, tmax = float(np.min(sw["thru"])), float(np.max(sw["thru"]))
            lev = 0.5 * (tmin + tmax)
            below = sw["lam"][sw["thru"] <= lev]
            fw = float(below.max() - below.min()) if below.size else math.nan
            res[tp] = {"q": d["lam_min"] / fw, "q_3db": math.nan, "fwhm_pm": fw * 1e12,
                       "lam_min_nm": d["lam_min"] * 1e9}
    out = ReplicaResult("loaded_q_from_3db_width", float(res[own]["q"]), _unit(target, "1"),
                        target_id=getattr(target, "id", None))
    out.variants = {f"topology={tp}": float(v["q"]) for tp, v in res.items() if tp != own}
    out.variants["width=3db_below_max"] = float(res[own]["q_3db"])
    out.detail = res
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "topology": own, "heater_v": v_h, "bias_v": bias,
                      "p_in_w": p_in, "t_base_c": _t_base_k(a, StaticRing(pp, n=ring)) - T0_K,
                      "readout": "full width at half depth (linear) of the through notch; Q = lambda_min / FWHM",
                      "assumption": "P_in -> 0, heater 0 V, bias 0 V (not printed; 07 4.3 'low power, cold, 0 V')"
                      if p_in <= 0 else "swept at the stated power (thermal triangle)"}
    return out


def _heater_v_list(target, a) -> tuple[list[float], bool]:
    """Heater voltages: the curve's x of the target, else args['heater_v'] (scalar)."""
    if target is not None and getattr(target, "kind", None) == "curve" and \
            getattr(target, "value", {}).get("x_name") == "heater_v":
        return [float(x) for x in target.value["x"]], True
    v = a.get("heater_v")
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v], True
    return [float(v if v is not None else 0.0)], False


def notch_er_vs_heater_v(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-2, VAL-2a/b: low-power laser sweep at each bench heater voltage; notch depth = 10 log10(T_max / T_min) of
    the through port over one FSR around the shifted resonance (dB). The heater is a voltage source with the TCR
    solved (06 E1); P_in -> 0 unless ``p_in_dbm`` (the G25 sweep power is not printed); bias 0 V; stage at the
    reference temperature. Topologies (01 D14, 07 P0-5): ``add_drop`` is the realized ring, ``all_pass`` removes the
    drop coupler (K^2_drop = 0) and keeps every other parameter; the primary prediction uses the plant's own
    topology and the listed ones are variants."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    own = _plant_topology(pp, ring)
    topos = list(dict.fromkeys([own] + _topologies(a, own)))
    vs, is_list = _heater_v_list(target, a)
    bias = float(a.get("bias_v") or 0.0)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    res, det = {}, {}
    for tp in topos:
        r = StaticRing(pp, n=ring, topology=tp, driver_law="bench_voltage")
        tb = _t_base_k(a, r)
        ers, rows = [], []
        for v in vs:
            t_r, t_h, p_h = _ring_temps_from_drive(r, v, tb)
            guess = r.lam_res + float(r.thermal_shift_m(t_r)) + float(r.junction_shift_m(bias)[0])
            if p_in <= 0:
                nt = _notch(r, t_ring=t_r, t_heater=t_h, v=bias, lam_guess=guess)
                er = nt["er_db"]
            else:
                d = _dip_at_power(r, u0=v, t_base=tb, p_in_w=p_in, v=bias)
                er = 10.0 * math.log10(float(np.max(d["coarse"]["thru"])) / d["t_min"])
                nt = {"lam_min": d["lam_min"]}
            ers.append(float(er))
            rows.append({"heater_v": v, "p_heater_mw": p_h * 1e3, "t_ring_c": t_r - T0_K, "er_db": er,
                         "lam_min_nm": nt["lam_min"] * 1e9})
        res[tp] = ers if is_list else ers[0]
        det[tp] = rows
    out = ReplicaResult("notch_er_vs_heater_v", res[own], _unit(target, "db"), target_id=getattr(target, "id", None),
                        variants={f"topology={tp}": res[tp] for tp in topos if tp != own}, detail=det)
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "topology": own, "heater_v": vs, "bias_v": bias,
                      "p_in_w": p_in, "port": "through", "heater_drive": "bench voltage source, TCR solved",
                      "t_base_c": _t_base_k(a, StaticRing(pp, n=ring)) - T0_K,
                      "assumption": "P_in -> 0 (sweep power not printed)" if p_in <= 0 else "swept at power"}
    return out


def _resonance_shift_curve(pp, a, vs, *, ring: int, p_in: float, bias: float, rel_v: float):
    r = StaticRing(pp, n=ring, driver_law="bench_voltage")
    tb = _t_base_k(a, r)
    ref = _resonance_position(r, u0=rel_v, t_base=tb, v=bias, p_in_w=p_in)["lam_min"]
    rows, shifts = [], []
    for v in vs:
        pos = _resonance_position(r, u0=v, t_base=tb, v=bias, p_in_w=p_in)
        shifts.append(pos["lam_min"] - ref)
        rows.append({"heater_v": v, "lam_min_nm": pos["lam_min"] * 1e9, "shift_nm": (pos["lam_min"] - ref) * 1e9})
    return np.array(shifts), rows, ref, tb


def resonance_vs_heater_v(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-4: resonance (through minimum) of a low-power laser sweep at each bench heater voltage, relative to the
    position at ``relative_to_v`` (0 V), in nm. Voltage source with the TCR solved; bias 0 V; P_in -> 0 unless
    ``p_in_dbm``; stage at the reference temperature."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    vs, is_list = _heater_v_list(target, a)
    rel = float(a.get("relative_to_v") or 0.0)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    bias = float(a.get("bias_v") or 0.0)
    sh, rows, ref, tb = _resonance_shift_curve(pp, a, vs, ring=ring, p_in=p_in, bias=bias, rel_v=rel)
    val = (sh * 1e9).tolist() if is_list else float(sh[0] * 1e9)
    out = ReplicaResult("resonance_vs_heater_v", val, _unit(target, "nm"), target_id=getattr(target, "id", None),
                        detail={"rows": rows, "position_at_ref_v_nm": ref * 1e9})
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "heater_v": vs, "relative_to_v": rel, "bias_v": bias,
                      "p_in_w": p_in, "t_base_c": tb - T0_K, "heater_drive": "bench voltage source, TCR solved",
                      "readout": "through minimum of the swept spectrum"}
    return out


def tuning_range_at_v(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """VAL-1: resonance shift at ``heater_v`` (1.5 V) relative to ``relative_to_v`` (0 V), nm; same protocol as
    resonance_vs_heater_v (low-power sweep, bench voltage source with the TCR, bias 0 V)."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    v = float(a.get("heater_v", 1.5))
    rel = float(a.get("relative_to_v") or 0.0)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    bias = float(a.get("bias_v") or 0.0)
    sh, rows, ref, tb = _resonance_shift_curve(pp, a, [v], ring=ring, p_in=p_in, bias=bias, rel_v=rel)
    r = StaticRing(pp, n=ring, driver_law="bench_voltage")
    p_h = float(r.power_from_u([v], tb)[0])
    out = ReplicaResult("tuning_range_at_v", float(sh[0] * 1e9), _unit(target, "nm"),
                        target_id=getattr(target, "id", None),
                        detail={"rows": rows, "p_heater_mw": p_h * 1e3, "fsr_nm": _fsr_m(r) * 1e9})
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "heater_v": v, "relative_to_v": rel, "bias_v": bias,
                      "p_in_w": p_in, "t_base_c": tb - T0_K, "heater_drive": "bench voltage source, TCR solved"}
    return out


def cold_resonance_offset(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-7: the operating line minus the measured resonance at ``heater_v`` (0 V), nm (positive: the resonance is
    blue of the line). Low-power sweep (cold cavity) unless ``p_in_dbm``; bias 0 V; stage at the reference
    temperature (the 0 V spectrum's temperature is unknown, 07 P1-6)."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    r = StaticRing(pp, n=ring, driver_law="bench_voltage")
    tb = _t_base_k(a, r)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    v_h = float(a.get("heater_v") or 0.0)
    pos = _resonance_position(r, u0=v_h, t_base=tb, v=float(a.get("bias_v") or 0.0), p_in_w=p_in)
    off = (r.lam_line - pos["lam_min"]) * 1e9
    out = ReplicaResult("cold_resonance_offset", float(off), _unit(target, "nm"),
                        target_id=getattr(target, "id", None),
                        detail={"line_nm": r.lam_line * 1e9, "resonance_nm": pos["lam_min"] * 1e9})
    if p_in <= 0:
        d = _dip_at_power(r, u0=v_h, t_base=tb, p_in_w=r.p_line_w, v=0.0)
        out.variants["at_preset_power_red_sweep"] = float((r.lam_line - d["lam_min"]) * 1e9)
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "heater_v": v_h, "p_in_w": p_in, "t_base_c": tb - T0_K,
                      "line": r.line, "readout": "line minus through minimum"}
    return out


def heater_power_per_fsr(plant, target=None, *, ring: int | None = None, **kw) -> ReplicaResult:
    """CAL-3a/b: heater power (mW, or with ``output: voltage`` the voltage in V) that moves the resonance by one FSR,
    starting at the stage temperature (target t_c, 25 degC). Protocol: the low-power spectrum gives the cold
    resonance lambda_m and the next order red of it, lambda_{m-1}; a bench voltage sweep (TCR solved) finds the
    voltage at which the order-m minimum reaches lambda_{m-1}. ``statistic`` over the realized worlds and rings
    (median; with the default single nominal world it is that device)."""
    a = _args(target, kw)
    batch = int(a.get("batch", 1))
    unc = a.get("uncertainty", "nominal")
    var = a.get("variation", "off")
    cfg, pp = resolve_plant(plant, batch=batch, uncertainty=unc, variation=var)
    if target is not None and isinstance(getattr(target, "conditions", {}).get("t_c"), (int, float)):
        a.setdefault("t_c", target.conditions["t_c"])
    rings = range(pp.n_rings) if ring is None else [int(ring)]
    p_list, v_list, rows = [], [], []
    for b in range(pp.batch):
        for n in rings:
            r = StaticRing(pp, b=b, n=n, driver_law="bench_voltage")
            tb = _t_base_k(a, r)
            n0 = _notch(r, t_ring=tb, t_heater=tb, v=0.0, lam_guess=r.lam_res + float(r.thermal_shift_m(tb)))
            fsr = _fsr_m(r)
            n1 = _notch(r, t_ring=tb, t_heater=tb, v=0.0, lam_guess=n0["lam_min"] + fsr, half_window=0.3 * fsr)
            target_lam = n1["lam_min"]

            def pos(v):
                t_r, t_h, _ = _ring_temps_from_drive(r, v, tb)
                g = r.lam_res + float(r.thermal_shift_m(t_r))
                return _notch(r, t_ring=t_r, t_heater=t_h, v=0.0, lam_guess=g, half_window=0.3 * fsr)["lam_min"]
            v_fsr = optimize.brentq(lambda v: pos(v) - target_lam, 0.0, 5.0, xtol=1e-7)
            p_fsr = float(r.power_from_u([v_fsr], tb)[0])
            p_list.append(p_fsr)
            v_list.append(v_fsr)
            rows.append({"b": b, "n": n, "p_fsr_mw": p_fsr * 1e3, "v_fsr_v": v_fsr, "fsr_nm": (target_lam -
                         n0["lam_min"]) * 1e9, "preset_p_fsr_mw": float(np.asarray(pp.heater.p_fsr_w)[b, n]) * 1e3})
    stat = a.get("statistic", "median")
    agg = {"median": np.median, "mean": np.mean}.get(stat, np.median)
    want_v = a.get("output") == "voltage"
    val = float(agg(v_list)) if want_v else float(agg(p_list) * 1e3)
    out = ReplicaResult("heater_power_per_fsr", val, _unit(target, "v" if want_v else "mw"),
                        target_id=getattr(target, "id", None), detail={"rows": rows})
    out.variants = {"voltage_v" if not want_v else "power_mw": float(agg(p_list) * 1e3) if want_v
                    else float(agg(v_list))}
    out.conditions = {**_pp_meta(cfg, pp), "statistic": stat, "rings": list(rings),
                      "t_base_c": _t_base_k(a, StaticRing(pp)) - T0_K, "p_in_w": 0.0, "bias_v": 0.0,
                      "heater_drive": "bench voltage source, TCR solved",
                      "fsr_definition": "spacing of the cold order m and the next order red of it at the stage "
                                        "temperature (the spectrum repeats after this shift)"}
    pre = float(np.median([r_["preset_p_fsr_mw"] for r_ in rows]))
    out.notes.append(f"ring.heater.p_fsr_mw ({pre:.4g} mW) is anchored by R2 with the FSR at the LINE wavelength "
                     "(mrm.params.relations derived.optics.fsr_m); this protocol measures the FSR between the cold "
                     "order and the next order red of it, so the two differ by the lambda^2 scaling of the FSR "
                     f"({(float(agg(p_list)) * 1e3 / pre - 1) * 100:+.2f} % here)")
    return out


def resonance_shift_vs_bias(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-5: DC resonance shift against reverse bias relative to 0 V (pm), from the through minimum of a laser sweep
    at each bias, heater 0 V, stage at the reference temperature. ``p_in_dbm`` lists the optical powers (07 P1-7: the
    shift at measurement power includes the self-heating relief, the change of absorbed power with bias; P_in -> 0
    removes it). The primary prediction is at the measurement power (the first non-null entry, the G25 printed
    +4 dBm at the ring) because the protocol measured at power; P_in -> 0 is the variant without the relief. At power
    the minimum is read on a red (increasing wavelength) sweep."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    if target is not None and getattr(target, "kind", None) == "curve":
        xs, is_list = [float(x) for x in target.value["x"]], True
    else:
        bv = a.get("bias_v", 1.0)
        xs, is_list = ([float(x) for x in bv], True) if isinstance(bv, (list, tuple)) else ([float(bv)], False)
    pins = a.get("p_in_dbm", [None])
    pins = list(pins) if isinstance(pins, (list, tuple)) else [pins]
    primary = next((p for p in pins if p is not None), None)
    r = StaticRing(pp, n=ring, driver_law="bench_voltage")
    tb = _t_base_k(a, r)
    direction = a.get("direction", "up")
    res, det = {}, {}
    for p_dbm in dict.fromkeys(pins):
        p_w = _dbm_to_w(p_dbm)
        ref = _resonance_position(r, u0=0.0, t_base=tb, v=0.0, p_in_w=p_w, direction=direction)["lam_min"]
        sh = []
        for v in xs:
            sh.append((_resonance_position(r, u0=0.0, t_base=tb, v=v, p_in_w=p_w, direction=direction)["lam_min"]
                       - ref) * 1e12)
        res[p_dbm] = sh if is_list else sh[0]
        det[str(p_dbm)] = {"position_0v_nm": ref * 1e9, "shift_pm": sh}
    out = ReplicaResult("resonance_shift_vs_bias", res[primary], _unit(target, "pm"),
                        target_id=getattr(target, "id", None),
                        variants={f"p_in_dbm={p}": res[p] for p in res if p != primary}, detail=det)
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "bias_v": xs, "heater_v": 0.0, "t_base_c": tb - T0_K,
                      "p_in_dbm": primary, "sweep_direction": direction,
                      "self_heating_relief": "included (measurement power)" if primary is not None else "removed",
                      "junction_shift_only_pm": [float((r.junction_shift_m(v)[0] - r.junction_shift_m(0.0)[0])
                                                       * 1e12) for v in xs]}
    return out


# =====================================================================================================================
# Replicas: apparent thermal sensitivity at a fixed heater code (07 P1-4, D2 H86_sensor)
# =====================================================================================================================
def _values_scalar(pp, key: str, b: int = 0, default: float = 1.0) -> float:
    v = (getattr(pp, "values", {}) or {}).get(key)
    if v is None:
        return default
    v = np.asarray(v, dtype=float).reshape(-1)
    return float(v[b if v.size > 1 else 0])


def _aggressor_ratio(pp, ring: StaticRing) -> float | None:
    """DC ratio (aggressor -> heater_resistor) / (aggressor -> ring_mode_weighted) for this ring (None without an
    aggressor source)."""
    sysm = pp.thermal.system
    if not sysm.aggressor_src.size:
        return None
    G = ring.cache["dc"][ring.b]
    a = int(sysm.aggressor_src[0])
    gr = float(G[sysm.ring_obs[ring.n], a])
    gh = float(G[sysm.heater_obs[ring.n], a])
    return gh / gr if gr > 0 else None


def _fixed_drive_track(ring: StaticRing, u0: float, steps: Sequence[tuple[float, float, float]], *, p_in_w: float,
                       lam: float, v: float = 0.0, n_grid: int = 3001) -> np.ndarray:
    """Ring temperature along a slow path of (t_base, agg_r, agg_h) steps at a fixed drive u0 with the laser parked at
    ``lam`` (self-heating included; quasi-static branch continuation from the cold side). P_in -> 0: explicit."""
    out = np.zeros(len(steps))
    if p_in_w <= 0.0:
        for i, (tb, ar, ah) in enumerate(steps):
            p_h = float(ring.power_from_u([u0], tb + ah)[0])
            out[i] = tb + ar + ring.z["hr"] * p_h
        return out
    t_r = ring.t_res(lam, v)
    w = ring.linewidth_k(t_r)
    lo = min(s[0] + s[1] for s in steps) - 1.0
    hi = max(s[0] + s[1] for s in steps) + ring.z["hr"] * float(ring.power_from_u([u0], steps[0][0])[0]) * 1.2 \
        + ring.z["or"] * p_in_w + 4 * w
    tg = np.linspace(lo, hi, int(n_grid))
    o = ring.optics(lam, tg, None, v)
    i_prev = 0.0
    for i, (tb, ar, ah) in enumerate(steps):
        if ring.rp[ph.RP_CT_HEATER] != 0.0:
            st = ring.state_grid(lam, tg, t_base=tb, p_in_w=p_in_w, v_lo=v, agg_r=ar, agg_h=ah)
        else:
            st = ring.heat_balance(o, tg, t_base=tb, p_in_w=p_in_w, v_lo=v, agg_r=ar, agg_h=ah)
        out[i], i_prev, _ = _relax(tg, u0 - st["u"][:, 0], i_prev)
    return out


def apparent_shift_fixed_code(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-6 (07 P1-4, D2): the dlambda/dT a bench would report from a temperature sweep at a FIXED heater DAC code.

    For each thermo-optic hypothesis (``hypotheses``, re-loaded with ``uncertainty.thermo_optic``), driver law
    (``driver_laws``, default every choice of ``uncertainty.heater.driver_mode``) and heating path (``heating``):

    - the code is the one that puts the cold resonance on the ring's line at the first temperature (P_in -> 0), an
      integer code through the realized DAC; it is then held;
    - ``ambient``: the stage (T_base) steps from t_grid_c[0] to t_grid_c[-1]; the reported temperature is the stage's;
    - ``aggressor``: an on-die aggressor raises the ring's own temperature by the same span; the heater resistor rises
      by the ROM's DC ratio (aggressor -> heater_resistor) / (aggressor -> ring); the reported temperature is the base
      plus ``ring.thermal.reference_rise_frac`` of the ring's rise (1 for H_solver / H86, 0.8 for H86_sensor: the
      thermometer behind Lightmatter's degC sees that fraction, 07 D2);
    - the heater power follows the driver law with the TCR solved at the heater-resistor temperature (voltage drive
      lowers the apparent sensitivity, current drive raises it, power drive is neutral); with ``p_in_dbm`` (default
      the preset's line power at the ring) the laser is parked at the line and the self-heating feedback is included
      (quasi-static, cold-side start); the resonance position is that of the ring's state (a low-power probe).

    apparent = Delta lambda_res / Delta T_reported (pm/K). The primary value is the plant as given (its hypothesis
    and driver law), ambient path, at the stated power; every combination is in ``variants``."""
    a = _args(target, kw)
    t_grid = [float(x) for x in a.get("t_grid_c", [25.0, 105.0])]
    heating = a.get("heating", ["ambient", "aggressor"])
    heating = [heating] if isinstance(heating, str) else list(heating)
    n_steps = int(a.get("n_steps", 41))
    p_spec = a.get("p_in_dbm", "preset")
    cfg0, pp0 = resolve_plant(plant)
    if cfg0 is None:
        hyps, laws = [None], [None]
    else:
        hyps = [None] + [h for h in (a.get("hypotheses") or []) if h in _hyp_choices(cfg0, "thermo_optic")]
        laws = [None] + [x for x in (a.get("driver_laws") or _hyp_choices(cfg0, "heater.driver_mode"))]
    res, rows = {}, []
    for hyp in dict.fromkeys(hyps):
        for law in dict.fromkeys(laws):
            alts = {}
            if hyp is not None:
                alts["uncertainty.thermo_optic"] = hyp
            if law is not None:
                alts["uncertainty.heater.driver_mode"] = law
            cfg, pp = (cfg0, pp0) if not alts else resolve_plant(cfg0, alts=alts)
            r = StaticRing(pp, n=ring)
            law_name = {v: k for k, v in el.DRIVER_LAWS.items()}[r.law]
            p_in = r.p_line_w if p_spec == "preset" else _dbm_to_w(p_spec)
            frac = _values_scalar(pp, "ring.thermal.reference_rise_frac")
            t0, t1 = t_grid[0] + T0_K, t_grid[-1] + T0_K
            # the fixed code: cold resonance on the line at the first temperature (P_in -> 0)
            t_lock = r.t_res(r.lam_line, 0.0)
            p_h0 = (t_lock - t0) / r.z["hr"]
            u_lock = float(r.u_from_power(p_h0, t0 + r.z["hh"] * p_h0))
            code = int(np.rint(r.u_to_code(u_lock)))
            u0 = float(r.code_to_u(code))
            xs = np.linspace(0.0, t1 - t0, n_steps)
            for mode in heating:
                if mode == "ambient":
                    steps = [(t0 + x, 0.0, 0.0) for x in xs]
                    t_rep = t0 + xs
                elif mode == "aggressor":
                    k = _aggressor_ratio(pp, r)
                    if k is None:
                        continue
                    steps = [(t0, x, k * x) for x in xs]
                    t_rep = t0 + frac * xs
                else:
                    raise ValueError(f"unknown heating path {mode!r}")
                label = f"{hyp or 'plant'}/{mode}/{law_name}"
                if label in res:
                    continue
                t_ring = _fixed_drive_track(r, u0, steps, p_in_w=p_in, lam=r.lam_line)
                lam_r = r.lam_res + r.thermal_shift_m(t_ring)
                app = float((lam_r[-1] - lam_r[0]) / (t_rep[-1] - t_rep[0]) * 1e12)
                true_s = float(np.mean(r.s_at(np.linspace(t_ring[0], t_ring[-1], 101))) * 1e12)
                res[label] = app
                rows.append({"hypothesis": hyp or "plant", "heating": mode, "driver_law": law_name, "code": code,
                             "apparent_pm_per_k": app, "ring_mean_s_pm_per_k": true_s,
                             "reference_rise_frac": frac, "p_in_w": p_in,
                             "ring_rise_k": float(t_ring[-1] - t_ring[0]), "reported_rise_k": float(t_rep[-1] -
                                                                                                    t_rep[0])})
    own_law = {v: k for k, v in el.DRIVER_LAWS.items()}[StaticRing(pp0, n=ring).law]
    key = f"plant/ambient/{own_law}"
    if key not in res:
        key = next(iter(res))
    out = ReplicaResult("apparent_shift_fixed_code", res[key], _unit(target, "pm_per_k"),
                        target_id=getattr(target, "id", None), variants={k: v for k, v in res.items() if k != key},
                        detail={"rows": rows})
    out.conditions = {**_pp_meta(cfg0, pp0), "ring": ring, "t_grid_c": t_grid, "primary": key,
                      "fixed_code": "cold resonance on the line at the first temperature (P_in -> 0)",
                      "p_in": p_spec, "laser": "parked at the ring's line", "readout": "resonance of the ring state"}
    return out


# =====================================================================================================================
# Replicas: relOMA maps (07 D9: relOMA = OMA / P_in, through port)
# =====================================================================================================================
def _db(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.where(x > 0, x, 0.0))


def _lock_map(r: StaticRing, biases, temps_k, *, p_in_w: float, mode: str = "oracle", fraction: float = 0.5,
              stable_only: bool = True, n_grid: int = 1201, span: float = 6.0, rho: float = 0.5) -> dict:
    """relOMA (linear) of the ring locked at every (bias, ambient) point of a map, by an ideal settled controller.

    mode ``oracle``: the maximum |T1 - T0| over the ring temperature (stable points only when ``stable_only``, and
    only where the heater can deliver: 0 <= P_h, u <= 1 and the driver compliance); ``setpoint_lock``: drop =
    fraction x peak on the hot side (laser blue of the resonance). Returns reloma (nb, nt), feasible (nb, nt),
    t_lock_k, p_heater_w, u."""
    nb, nt = len(biases), len(temps_k)
    out = {k: np.full((nb, nt), np.nan) for k in ("reloma", "t_lock_k", "p_heater_w", "u")}
    out["feasible"] = np.zeros((nb, nt), bool)
    lam = r.lam_line
    hl = r.rp[ph.RP_CT_HEATER] != 0.0
    for i, vb in enumerate(biases):
        v_lo, v_hi = r.nrz_levels(vb)
        p_j, p_d = r.data_heat(v_lo, v_hi)
        t_r = r.t_res(lam, 0.5 * (v_lo + v_hi))
        w = r.linewidth_k(t_r)
        tg = np.linspace(t_r - span * w, t_r + span * w, int(n_grid))
        o = None if hl else r.optics(lam, tg, None, v_lo, v_hi)
        for j, tb in enumerate(temps_k):
            kw = dict(t_base=tb, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, rho=rho, p_j=p_j, p_d=p_d)
            st = r.state_grid(lam, tg, **kw) if hl else r.heat_balance(o, tg, **kw)
            u = st["u"][:, 0]
            ok = (st["p_h"][:, 0] >= 0.0) & (u <= 1.0) & r.compliance_ok(st["p_h"][:, 0], st["t_h"][:, 0])
            if stable_only:
                ok &= np.gradient(u, tg) > 0
            om = st["reloma"][:, 0]
            if mode == "oracle":
                cand = np.where(ok, om, -np.inf)
                if not np.any(np.isfinite(cand)):
                    continue
                k = _refine_argmax(tg, cand, int(np.argmax(cand)))
            elif mode == "setpoint_lock":
                drop = st["drop_avg"][:, 0]
                ipk = int(np.argmax(drop))
                lev = fraction * drop[ipk]
                jj = np.flatnonzero(drop[ipk:] <= lev)
                if jj.size == 0:
                    continue
                j1 = ipk + int(jj[0])
                k = (j1 - 1) + (drop[j1 - 1] - lev) / (drop[j1 - 1] - drop[j1])
                if not ok[int(round(k))]:
                    out["t_lock_k"][i, j] = np.interp(k, np.arange(tg.size), tg)
                    continue
            else:
                raise ValueError(f"unknown map mode {mode!r}")
            idx = np.arange(tg.size, dtype=float)
            out["reloma"][i, j] = float(np.interp(k, idx, om))
            out["t_lock_k"][i, j] = float(np.interp(k, idx, tg))
            out["p_heater_w"][i, j] = float(np.interp(k, idx, st["p_h"][:, 0]))
            out["u"][i, j] = float(np.interp(k, idx, u))
            out["feasible"][i, j] = True
    return out


def _first_crossing(x, y, level) -> float:
    """First x where y crosses ``level`` going down (linear interpolation), nan if never."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for k in range(1, x.size):
        if np.isfinite(y[k - 1]) and np.isfinite(y[k]) and y[k - 1] > level >= y[k]:
            return float(x[k - 1] + (y[k - 1] - level) / (y[k - 1] - y[k]) * (x[k] - x[k - 1]))
    return math.nan


def _contour_offsets(t_c, y_db, peak_c: float, levels) -> tuple[list[float], list[float]]:
    """Offsets (K) below and above ``peak_c`` where the curve y_db(t_c) first falls to each level, walking away from
    the peak (linear interpolation in dB between samples)."""
    t_c, y = np.asarray(t_c, float), np.asarray(y_db, float)
    below, above = [], []
    ip = int(np.argmin(np.abs(t_c - peak_c)))
    for lev in levels:
        tb = _first_crossing(t_c[:ip + 1][::-1], y[:ip + 1][::-1], lev)
        ta = _first_crossing(t_c[ip:], y[ip:], lev)
        below.append(peak_c - tb if math.isfinite(tb) else math.nan)
        above.append(ta - peak_c if math.isfinite(ta) else math.nan)
    return below, above


def _fixed_code_open_loop(r: StaticRing, *, bias_v: float, p_in_w: float, align_peak_c: float, t_lo_c: float = 0.0,
                          t_hi_c: float = 105.0, step_k: float = 0.25, direction: str = "up",
                          n_grid: int = 2401, span: float = 12.0) -> dict:
    """Loop-open relOMA against ambient at one fixed heater drive, the drive chosen so that the relOMA peak of the
    swept curve sits at ``align_peak_c`` (VAL-5: the loop-open code is unpublished). Quasi-static ambient sweep with
    branch continuation (``up``: from cold, ``down``: from hot), laser at the line, NRZ at the datapath swing."""
    lam = r.lam_line
    v_lo, v_hi = r.nrz_levels(bias_v)
    p_j, p_d = r.data_heat(v_lo, v_hi)
    t_r = r.t_res(lam, 0.5 * (v_lo + v_hi))
    w = r.linewidth_k(t_r)
    tg = np.linspace(t_r - span * w, t_r + span * w, int(n_grid))
    hl = r.rp[ph.RP_CT_HEATER] != 0.0
    o = r.optics(lam, tg, None, v_lo, v_hi)
    temps = np.arange(t_lo_c, t_hi_c + 0.5 * step_k, step_k) + T0_K
    bal = {}

    def balance(tb):
        if tb not in bal:
            kw = dict(t_base=tb, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, p_j=p_j, p_d=p_d)
            st = r.state_grid(lam, tg, **kw) if hl else r.heat_balance(o, tg, **kw)
            bal[tb] = (st["u"][:, 0], st["reloma"][:, 0])
        return bal[tb]

    def curve(u0, direction_=direction):
        y = np.zeros(temps.size)
        tr = np.zeros(temps.size)
        edge = np.zeros(temps.size, bool)
        order = np.arange(temps.size) if direction_ == "up" else np.arange(temps.size)[::-1]
        i_prev = 0.0 if direction_ == "up" else float(tg.size - 1)
        idx = np.arange(tg.size, dtype=float)
        for j in order:
            u, om = balance(temps[j])
            tr[j], i_prev, edge[j] = _relax(tg, u0 - u, i_prev)
            y[j] = 0.0 if edge[j] else float(np.interp(i_prev, idx, om))
        return y, tr, edge

    def t_peak(u0):
        y, _, _ = curve(u0)
        k = _refine_argmax(temps, y, int(np.argmax(y)))
        return float(np.interp(k, np.arange(temps.size), temps)) - T0_K

    # bracket: the drive that puts the resonance at the line at the two ends of the sweep
    def u_at(tc):
        tb = tc + T0_K
        p_h = (t_r - tb - r.z["jr"] * p_j - r.z["dr"] * p_d) / r.z["hr"]
        return float(r.u_from_power(p_h, tb + r.z["hh"] * p_h + r.z["jh"] * p_j + r.z["dh"] * p_d))
    u_a, u_b = u_at(t_hi_c + 10.0), u_at(t_lo_c - 10.0)
    fa, fb = t_peak(u_a) - align_peak_c, t_peak(u_b) - align_peak_c
    if not fa * fb < 0:
        raise ValueError(f"cannot bracket the loop-open peak at {align_peak_c} degC (peaks {fa + align_peak_c:.1f}, "
                         f"{fb + align_peak_c:.1f})")
    u0 = optimize.brentq(lambda u: t_peak(u) - align_peak_c, u_a, u_b, xtol=1e-9, rtol=1e-10)
    y, tr, edge = curve(u0)
    # hysteresis of the fixed-drive line: the opposite sweep at the same drive (where the ring states differ)
    _, tr_rev, _ = curve(u0, "down" if direction == "up" else "up")
    gap = np.abs(tr - tr_rev)
    hyst = gap > 0.05 * w
    return {"u0": u0, "temps_c": temps - T0_K, "reloma": y, "t_ring_k": tr, "edge": edge,
            "peak_c": t_peak(u0), "t_ring_reverse_k": tr_rev,
            "hysteresis_c": ([float(temps[hyst].min() - T0_K), float(temps[hyst].max() - T0_K)]
                             if np.any(hyst) else None),
            "max_branch_gap_k": float(gap.max())}


def relOMA_map(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-14, VAL-4a/b/c, VAL-5, BND-4: the G26 relOMA maps (bias 1.0-3.75 V x 0-105 degC, slide 9; relOMA = OMA /
    P_in at the through port, 07 D9), NRZ at the datapath swing around the bias, the laser at the ring's line.

    ``mode='oracle'`` (loop closed, best achievable): at every point the ring temperature that maximizes relOMA among
    fixed-drive-stable points the heater can deliver; ``'setpoint_lock'``: drop monitor held at half its peak on the
    hot side; ``'fixed_code'`` (loop open, VAL-5): one heater drive for the whole ambient sweep, chosen so that the
    swept relOMA peaks at ``align_peak_c`` (the peak is an alignment input, not a test), quasi-static up-sweep in
    ambient, sampled every ``sample_step_k`` and interpolated linearly in dB like the published map; contour offsets
    from the aligned peak. Outputs by target: CAL-14 the mean over its bias and temperature window; VAL-4a/b the bias
    of the ``contour_db`` crossing at ``t_c`` (scanning up to 4.0 V; nan if not reached); VAL-4c ``spread_db``
    (max - min over the map); BND-4 ``max_variation_over_t_at_fixed_bias_db``. Optical power: the target's
    ``p_in_dbm`` (0 dBm, "1 mW into the MRM") for the self-heating; variant at the preset's line power. An explicit
    ``p_in_dbm`` argument wins (None: P_in -> 0, no self-heating; ``"preset"``: the preset's line power at the ring);
    a target that does not print the power uses the preset's. ``alternative_mode`` gives a variant."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    r = StaticRing(pp, n=ring)
    mode = a.get("mode", "oracle")
    cond = dict(getattr(target, "conditions", {}) or {})
    if "p_in_dbm" in a:
        p_dbm = a["p_in_dbm"]
    else:
        p_dbm = cond.get("p_in_dbm")
        p_dbm = "preset" if p_dbm is None else p_dbm
    p_in = r.p_line_w if p_dbm == "preset" else _dbm_to_w(p_dbm)
    output = a.get("output")
    if output is None:
        if mode == "fixed_code":
            output = "contour_offsets"
        elif a.get("contour_db") is not None:
            output = "contour_bias_v"
        else:
            output = "level_db"
    bias_axis = np.round(np.arange(1.0, 3.75 + 1e-9, 0.05), 4)
    t_axis_c = np.arange(0.0, 105.0 + 1e-9, 5.0)

    def evaluate_output(md, p_w):
        if output == "contour_offsets":
            levels = (target.value["levels_db"] if target is not None and getattr(target, "kind", "") ==
                      "contour_offsets" else a.get("levels_db", [-16.0, -24.0]))
            step = float(a.get("sample_step_k", 5.0))
            fc = _fixed_code_open_loop(r, bias_v=float(a.get("bias_v", 2.25)), p_in_w=p_w,
                                       align_peak_c=float(a.get("align_peak_c", 69.6)),
                                       direction=a.get("direction", "up"))
            y_db = _db(fc["reloma"])
            tc = fc["temps_c"]
            on = np.isclose(np.mod(tc + 1e-9, step), 0.0, atol=1e-6)
            bs, ab = _contour_offsets(tc[on], y_db[on], fc["peak_c"], levels)
            bc, ac = _contour_offsets(tc, y_db, fc["peak_c"], levels)
            return ({"below_peak_k": bs, "above_peak_k": ab},
                    {"continuous": {"below_peak_k": bc, "above_peak_k": ac}, "u0": fc["u0"],
                     "peak_c": fc["peak_c"], "peak_db": float(np.max(y_db)),
                     "samples_c": tc[on].tolist(), "samples_db": y_db[on].tolist(),
                     "hysteresis_c": fc["hysteresis_c"], "max_branch_gap_k": fc["max_branch_gap_k"]})
        if output == "contour_bias_v":
            tc = float(a.get("t_c", cond.get("t_c", 25.0)))
            biases = np.round(np.arange(1.0, 4.0 + 1e-9, 0.025), 4)
            mp = _lock_map(r, biases, [tc + T0_K], p_in_w=p_w, mode=md)
            y = _db(mp["reloma"][:, 0])
            lev = float(a["contour_db"])
            ref_db = float(a.get("reference_level_db", -2.1))
            # the same contour measured from the map's own low-bias level (the printed map's top contour is the
            # CAL-14 level, -2.1 dB): tests the bias shape when the absolute level is not calibrated yet
            rel = _first_crossing(biases, y - float(np.mean(y[biases <= 1.15 + 1e-9])), lev - ref_db)
            return _first_crossing(biases, y, lev), {
                "t_c": tc, "reloma_db_at_1v": float(y[0]), "reloma_db_min": float(np.nanmin(y)),
                "reloma_db_at_3p75v": float(np.interp(3.75, biases, y)),
                "contour_bias_relative_to_low_bias_level_v": rel, "relative_drop_db": lev - ref_db,
                "infeasible_points": int(np.sum(~mp["feasible"]))}
        if output == "level_db":
            bw = a.get("bias_v", [1.0, 1.15])
            tw = a.get("t_c", [0.0, 105.0])
            bw = [bw, bw] if isinstance(bw, (int, float)) else bw
            tw = [tw, tw] if isinstance(tw, (int, float)) else tw
            biases = np.linspace(float(bw[0]), float(bw[1]), 4)
            temps = t_axis_c[(t_axis_c >= tw[0] - 1e-9) & (t_axis_c <= tw[1] + 1e-9)]
            mp = _lock_map(r, biases, temps + T0_K, p_in_w=p_w, mode=md)
            y = _db(mp["reloma"])
            return float(np.nanmean(y)), {"min_db": float(np.nanmin(y)), "max_db": float(np.nanmax(y)),
                                          "infeasible_points": int(np.sum(~mp["feasible"])),
                                          "biases_v": biases.tolist(), "temps_c": temps.tolist()}
        if output in ("spread_db", "max_variation_over_t_at_fixed_bias_db"):
            mp = _lock_map(r, bias_axis, t_axis_c + T0_K, p_in_w=p_w, mode=md)
            y = _db(mp["reloma"])
            info = {"max_db": float(np.nanmax(y)), "min_db": float(np.nanmin(y)),
                    "infeasible_points": int(np.sum(~mp["feasible"]))}
            if output == "spread_db":
                return float(np.nanmax(y) - np.nanmin(y)), info
            per_bias = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
            info["worst_bias_v"] = float(bias_axis[int(np.nanargmax(per_bias))])
            return float(np.nanmax(per_bias)), info
        raise ValueError(f"unknown relOMA_map output {output!r}")

    val, info = evaluate_output(mode, p_in)
    out = ReplicaResult("relOMA_map", val, _unit(target, "db" if output != "contour_bias_v" else "v"),
                        target_id=getattr(target, "id", None), detail={"primary": info})
    if output == "contour_bias_v":
        out.variants["relative_to_low_bias_level"] = info["contour_bias_relative_to_low_bias_level_v"]
        if not math.isfinite(val):
            out.notes.append(f"the {a['contour_db']} dB contour is not reached between 1.0 and 4.0 V (map from "
                             f"{info['reloma_db_at_1v']:.2f} to {info['reloma_db_min']:.2f} dB); the variant "
                             "'relative_to_low_bias_level' places it the printed 0.9 dB below the map's own "
                             "low-bias level instead")
    if output == "contour_offsets" and a.get("with_variants", True):
        other = "down" if a.get("direction", "up") == "up" else "up"
        a_dir = dict(a)
        a["direction"] = other
        v_o, i_o = evaluate_output(mode, p_in)
        a.clear()
        a.update(a_dir)
        out.variants[f"ambient_sweep={other}"] = v_o
        out.detail[f"ambient_sweep={other}"] = i_o
        same = np.allclose(np.r_[v_o["below_peak_k"], v_o["above_peak_k"]], np.r_[val["below_peak_k"],
                                                                                   val["above_peak_k"]],
                           equal_nan=True)
        hy = info.get("hysteresis_c")
        if hy is None:
            line = "At this power the fixed-code line is single-valued (no hysteresis), so both directions agree."
        else:
            line = (f"At this power the fixed-code line is hysteretic (bistable) between {hy[0]:.2f} and "
                    f"{hy[1]:.2f} degC (the two branches differ by up to {info['max_branch_gap_k']:.1f} K of ring "
                    "temperature); " + ("the 5 degC sampled offsets of both directions nevertheless agree, because "
                                        "no sample falls inside that window." if same else
                                        "the sweep direction changes the sampled offsets (see the variant)."))
        out.notes.append("the ambient sweep direction of the loop-open map is not published; the primary is the "
                         "up-sweep (from cold), chosen before any comparison. " + line + " The self-heating "
                         "steepens the cold side (laser red of the resonance, positive feedback" +
                         ("; the up-sweep jumps across it" if hy is not None else "") + ") and widens the hot side.")
    alt = a.get("alternative_mode") if a.get("with_variants", True) else None
    if alt and alt != mode and mode != "fixed_code":
        v2, i2 = evaluate_output(alt, p_in)
        out.variants[f"mode={alt}"] = v2
        out.detail[f"mode={alt}"] = i2
    if a.get("with_variants", True) and p_dbm != "preset" and abs(r.p_line_w - p_in) > 1e-9 * max(p_in, 1e-12):
        v3, i3 = evaluate_output(mode, r.p_line_w)
        out.variants["p_in=preset_line_power"] = v3
        out.detail["p_in=preset_line_power"] = i3
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "mode": mode, "output": output, "p_in_w": p_in,
                      "line_nm": r.lam_line * 1e9, "swing_pp_v": r.swing_pp_v,
                      "level_convention": {v: k for k, v in el.LEVEL_MODES.items()}[r.level_mode],
                      "data_heat": "on (random NRZ)", "rho": 0.5,
                      "reloma": "|T(level 1) - T(level 0)| at the through port (OMA / P_in)",
                      "map_axes": "bias 1.0-3.75 V step 0.05, 0-105 degC step 5 (contours: bias step 0.025 to 4.0 V)"}
    return out


# =====================================================================================================================
# Replicas: closed loop (quasi-static by default; run_fn for the dynamic path)
# =====================================================================================================================
@dataclasses.dataclass
class LockRunRequest:
    """What a closed-loop replica asks a run function to simulate (the protocol; the engine decides how).

    cfg/pp: the plant (pp realized with ``pp.batch`` worlds); scenario: a mrm.scenarios Scenario (aggressors may be
    uncalibrated: the runner calibrates them, ENGINE_BRIEF section 1); controller: a registry name or a controller
    object; setpoint_a (B, N): the monitor-current set-point of every ring (from the bank's quasi-static lock: the
    factory-calibration value a warm-started controller is given); initial_code (B, N): warm-start heater codes;
    duration_s, record_every_s; record: names of the traces the replica needs ("monitor_a", "heater_code",
    "t_ring_k", "t_base_k"). options: ``bias_v`` (tx.bias for the run), ``lock_mode``, ``lock_side``,
    ``start_branch``, ``warm_state`` ({t_ring_k, p_abs_w, p_heater_w, p_j_w, p_d_w} (B, N) and t_base_k: the
    quasi-static state, for initializing the PLANT only; it is Truth and must never reach the controller, 07 P1-1),
    and for BND-3 ``max_update_rate_hz`` / ``min_total_latency_s`` (limits the controller must respect). The run
    function returns {"t_s": (T,), name: (T, B, N) or (T, B)}, plus optionally "setpoint_a" ((T, B, N) or (B, N))
    when the controller sets its own set-point (fraction of its measured peak); the replicas then measure errors
    against it."""
    cfg: Any
    pp: Any
    scenario: Any
    controller: Any
    setpoint_a: np.ndarray
    initial_code: np.ndarray
    duration_s: float
    record_every_s: float
    record: tuple = ("monitor_a", "heater_code", "t_ring_k", "t_base_k")
    options: dict = dataclasses.field(default_factory=dict)


def default_run_fn() -> Callable[[LockRunRequest], Mapping[str, np.ndarray]]:
    """The engine's lock runner: ``mrm.sim.runner.run_lock_request(request) -> traces`` (requested from the engine
    owner in docs/handoff/replicas.md). Raises ReplicaBlocked while it does not exist."""
    try:
        from mrm.sim import runner  # noqa: PLC0415
    except Exception as e:                                              # noqa: BLE001
        raise ReplicaBlocked(f"mrm.sim.runner is not available ({e.__class__.__name__}: {e})") from None
    fn = getattr(runner, "run_lock_request", None)
    if fn is None:
        raise ReplicaBlocked("mrm.sim.runner has no run_lock_request(LockRunRequest) adapter yet "
                             "(docs/handoff/replicas.md)")
    return fn


def _setpoint_modes(a: Mapping) -> list[str]:
    sp = a.get("setpoint", "hypothesis")
    if sp in (None, "hypothesis"):
        return ["half_max", "peak", "oracle"]
    return [sp] if isinstance(sp, str) else list(sp)


def _scenario_bias(cfg, default: float = 1.05) -> float:
    try:
        v = cfg.get("scenario.bias_v") if cfg is not None else None
    except Exception:                                                   # noqa: BLE001
        v = None
    return float(v) if isinstance(v, (int, float)) else default


def monitor_current_at_lock(plant, target=None, *, ring: int = 0, method: str = "quasi_static", run_fn=None,
                            **kw) -> ReplicaResult:
    """CAL-8: drop-monitor photocurrent (uA, signal + dark, sensing.MonitorHw) held by the loop.

    Quasi-static: the ring locked at the set-point hypothesis (``setpoint``: ``half_max`` = half the peak drop on the
    hot side, the lock the g26 drop coupling was chosen for (params_content handoff 7.1); ``peak``; ``oracle`` =
    maximum relOMA) at the preset's line power (``p_in_dbm``: a power in dBm, ``"preset"`` (default) or None for
    P_in -> 0), NRZ traffic at the scenario bias, base 25 degC, every TX line of the bank on and every ring locked
    (``lines='all'``, the G26 measurement: :func:`_bank_codes`; the other lines leak into the drop PD and heat the
    ring, 07 P1-9, after the rings upstream filtered them; variant ``lines=own``: the ring alone with its own line);
    the set-point levels refer to the monitor's total drop power, as a fraction-of-peak controller measures it. The
    dark current is taken at the monitor-PD temperature of the locked state. Under a set-point lock the current does
    not follow the aggressor except through the loss temperature coefficients and the PD temperature; the value is
    also evaluated at a 105 degC base
    (``t_hot_c``; detail). ``method='dynamic'`` runs the lock of the whole bank through ``run_fn`` (LockRunRequest,
    every ring at its own set-point) and reports the median monitor current of ring ``ring`` over the second half of
    the run."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    modes = _setpoint_modes(a)
    r = StaticRing(pp, n=ring)
    bias = float(a.get("bias_v", _scenario_bias(cfg)))
    p_spec = a.get("p_in_dbm", "preset")
    p_in = r.p_line_w if p_spec == "preset" else _dbm_to_w(p_spec)
    t_base = float(a.get("t_c", 25.0)) + T0_K
    t_hot = float(a.get("t_hot_c", 105.0)) + T0_K
    lines = a.get("lines", "all")
    res, det = {}, {}

    def current(m, tb, lines_):
        if lines_ == "all" and r.other_lines.size:
            # the ring inside the locked bank: lines filtered by the rings upstream, PD heated by every source
            bc = _bank_codes(pp, 0, t_base=tb, bias=bias, mode=m, p_in_w=None if p_spec == "preset" else p_in,
                             lines="all")
            lp = bc["locks"][ring]
            return float(r.monitor_current_a(lp.p_drop_w, lp.p_bus_w, bc["t_pd_k"][0, ring])[0]), lp
        lp = r.lock(mode=m, t_base=tb, p_in_w=p_in, bias_v=bias, modulate=True, lines="own")
        return float(r.monitor_current_a(lp.p_drop_w, lp.p_bus_w, lp.t_pd_k)[0]), lp
    for m in modes:
        i_a, lp = current(m, t_base, lines)
        i_hot, _ = current(m, t_hot, lines)
        res[m] = i_a * 1e6
        det[m] = {"current_ua": i_a * 1e6, "t_pd_c": lp.t_pd_k - T0_K,
                  f"current_ua_at_base_{t_hot - T0_K:g}c": i_hot * 1e6, "drop_w": lp.p_drop_w,
                  "drop_other_lines_w": lp.p_drop_other_w, "drop_peak_w": lp.drop_peak_w,
                  "t_ring_c": lp.t_ring_k - T0_K, "heater_mw": lp.p_heater_w * 1e3, "p_abs_mw": lp.p_abs_w * 1e3,
                  "stable": lp.stable, "detuning_pm": lp.detuning_m * 1e12}
    primary = modes[0]
    other_lines = "own" if lines == "all" else "all"
    i_other = current(primary, t_base, other_lines)[0] * 1e6
    if method == "dynamic":
        dyn = _dynamic_lock(cfg, pp, primary, bias=bias, p_in=None if p_spec == "preset" else p_in, t_base=t_base,
                            run_fn=run_fn, duration_s=float(a.get("duration_s", 2e-3)))
        val = float(np.median(dyn["monitor_a"][dyn["t_s"] >= 0.5 * dyn["t_s"][-1]][:, 0, ring]) * 1e6)
        out = ReplicaResult("monitor_current_at_lock", val, _unit(target, "ua"), target_id=getattr(target, "id", None),
                            variants={"quasi_static": res[primary]},
                            detail={"dynamic": {"setpoint_ua": float(dyn["setpoint_a"][0, ring]) * 1e6},
                                    "quasi_static": det})
    else:
        out = ReplicaResult("monitor_current_at_lock", res[primary], _unit(target, "ua"),
                            target_id=getattr(target, "id", None),
                            variants={f"setpoint={m}": res[m] for m in modes if m != primary}, detail=det)
        if r.other_lines.size:
            out.variants[f"lines={other_lines}"] = i_other
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "method": method, "setpoint": primary, "bias_v": bias,
                      "lines": lines if r.other_lines.size else "own (single-line plant)",
                      "p_in_w": p_in, "t_base_c": t_base - T0_K, "traffic": "random NRZ", "port": "drop",
                      "monitor": "MonitorHw.photocurrent_a (gain_scale x responsivity x drop + dark at the PD "
                                 "temperature of the lock)"}
    return out


def _dynamic_lock(cfg, pp, mode: str, *, bias: float, t_base: float, p_in: float | None = None, run_fn=None,
                  scenario=None, duration_s: float = 2e-3, controller: str = "drop_setpoint_pi",
                  options: Mapping[str, Any] | None = None) -> dict:
    """Run the lock of every ring of every world through ``run_fn`` from the quasi-static warm start of the bank
    (:func:`_bank_codes` with every line on: each ring at its own line, set-point, order and code; heater powers from
    the bank heat balance with the crosstalk) and return the traces plus the set-points ``setpoint_a`` (B, N).
    ``p_in``: power of each ring's own line at the ring (W; None: the preset's line power)."""
    fn = run_fn or default_run_fn()
    B, N = int(pp.batch), int(pp.n_rings)
    sp = np.zeros((B, N))
    code = np.zeros((B, N), np.int64)
    warm = {k: np.zeros((B, N)) for k in ("t_ring_k", "p_abs_w", "p_heater_w", "p_j_w", "p_d_w")}
    for b in range(B):
        bc = _bank_codes(pp, b, t_base=t_base, bias=bias, mode=mode, p_in_w=p_in)
        for n in range(N):
            rg = StaticRing(pp, b=b, n=n)
            lp = bc["locks"][n]
            sp[b, n] = float(rg.monitor_current_a(lp.p_drop_w, lp.p_bus_w, bc["t_pd_k"][0, n])[0])
            code[b, n] = int(np.rint(bc["codes"][0, n]))
            warm["t_ring_k"][b, n] = lp.t_ring_k
            warm["p_abs_w"][b, n] = lp.p_abs_w
            warm["p_j_w"][b, n] = lp.p_j_w
            warm["p_d_w"][b, n] = lp.p_d_w
            warm["p_heater_w"][b, n] = bc["p_w"][0, n]
    warm["t_base_k"] = float(t_base)
    if scenario is None:
        from mrm.scenarios.generators import Scenario
        scenario = Scenario(name="static_lock", defaults={"thermal.t_base_k": float(t_base)})
    dt = float(pp.timing.dt_s)
    # the warm state of the quasi-static lock (07 P0-1 warm start), for a runner that can start from it
    opts = {"bias_v": bias, "lock_mode": mode, "start_branch": "hot", "warm_state": warm,
            "lock_side": "hot (laser blue of the resonance: more heater, less drop)"}
    opts.update(options or {})
    req = LockRunRequest(cfg=cfg, pp=pp, scenario=scenario, controller=controller, setpoint_a=sp, initial_code=code,
                         duration_s=duration_s, record_every_s=max(dt, float(duration_s) / 200_000.0), options=opts)
    tr = dict(fn(req))
    if "setpoint_a" in tr:          # a controller that sets its own set-point (fraction of its measured peak)
        tr["controller_setpoint_a"] = np.asarray(tr["setpoint_a"], dtype=float)
    tr["setpoint_a"] = sp
    tr["t_s"] = np.asarray(tr["t_s"], dtype=float)
    for k in ("monitor_a", "heater_code"):
        if k in tr:
            tr[k] = np.asarray(tr[k], dtype=float)
    return tr


# =====================================================================================================================
# Replicas: heater codes under the aggressor (CAL-9a/b/c, VAL-6)
# =====================================================================================================================
def _aggressor_site(pp) -> str | None:
    sysm = pp.thermal.system
    return str(sysm.src_names[int(sysm.aggressor_src[0])]) if sysm.aggressor_src.size else None


def _calibrated_aggressor(pp, *, which: str = "a1", victim_ring: int = 0, referencing: str = "resonance",
                          waveform: str | None = None, reading: str | None = None, ring_t0: str = "lock"):
    """The slide-10 (``a1``) or 808 degC/s (``a2``) aggressor of mrm.scenarios, calibrated on this plant's ROM as the
    engine does it (ENGINE_BRIEF section 1: Foster responses of aggressor -> victim ring, the model's R1 law, and the
    victim's lock temperature as the starting point of the resonance target when ``ring_t0='lock'``). Returns
    (scenario, CalibratedWaveform, site)."""
    from mrm.scenarios import calibrate_scenario, foster_from_system, library
    site = _aggressor_site(pp)
    if site is None:
        raise ReplicaBlocked("the plant has no aggressor site in its thermal system")
    if which == "a1":
        sc = library.a1_slide10_aggressor(site=site, victim_ring=victim_ring, referencing=referencing,
                                          waveform=waveform or "rectified_sine")
    else:
        sc = library.a2_808_aggressor(site=site, victim_ring=victim_ring, referencing=referencing,
                                      reading=reading or "symmetric_triangle")
    t = pp.thermal
    model = {"s0_m_per_k": np.asarray(t.dlambda_dt_s0_m_per_k)[:, victim_ring],
             "beta_per_k": np.asarray(t.dlambda_dt_beta_per_k)[:, 0],
             "t_ref_k": np.asarray(t.dlambda_dt_t0_k)[:, 0]}
    if ring_t0 == "lock":
        model["ring_t0_k"] = np.asarray(t.t_ring_lock_k)[:, victim_ring]
    resp = foster_from_system(t.system, victim_ring=victim_ring, site=site)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc2, cal = calibrate_scenario(sc, {site: resp}, model=model)
    return sc2, cal[site], site


def _aggressor_rises(pp, cw, *, n_periods: int = 4, n_samples: int = 256) -> dict:
    """Rise (K) of every ring_mode_weighted and heater_resistor observer under the calibrated aggressor power, over
    the last period (periodic steady state; t = 0 at the target's cusp), by an exact FOH simulation of the whole
    ThermalSystem (mrm.plant.thermal.simulate). Returns t_s (n_samples,), ring (n_samples, B, N), heater (...)."""
    sysm = pp.thermal.system
    period = float(cw.period_s)
    h = period / (2 * n_samples)
    t = np.arange(0, n_periods * 2 * n_samples + 1) * h
    u = np.zeros((t.size, pp.batch, sysm.n_src))
    a = int(sysm.aggressor_src[0])
    for b in range(pp.batch):
        u[:, b, a] = cw.power_at(t, b=b if cw.power_w.shape[0] > 1 else 0)
    obs = th.simulate(sysm, u, h, hold="foh", t_base_k=0.0)
    last = obs[-(2 * n_samples + 1):-1:2]                       # one period, every other sample
    return {"t_s": t[-(2 * n_samples + 1):-1:2] - (n_periods - 1) * period,
            "ring": last[:, :, sysm.ring_obs], "heater": last[:, :, sysm.heater_obs]}


def _bank_codes(pp, b: int, *, t_base: float, bias: float, mode: str, agg_r=None, agg_h=None,
                modulate: bool = True, t_order_base: float | None = None, p_in_w: float | None = None,
                lines: str = "all", bus_filter: bool = True) -> dict:
    """Heater codes (T, N) holding every ring of world b at its lock point while the aggressor adds agg_r / agg_h
    (T, N) K at the ring / heater-resistor observers: the bank heat balance with the N x N DC gains (crosstalk
    included); the lines reach each ring after the rings before it on the bus (single forward pass in
    ``layout.bus_order`` through their level-averaged through transmissions at the lock, plus the bus segment loss;
    ``p_in_w`` overrides every line's chip-input power; ``bus_filter=False`` gives every ring the chip-input powers);
    with ``lines='all'``, the default, every other TX line also heats each ring and leaks into its drop monitor
    (:meth:`StaticRing.other_lines_terms`); the driver law and TCR at each heater-resistor temperature, and the
    realized DAC. Negative needed power (the ring would have to be cooled), u > 1 or a voltage above the driver
    compliance is flagged (``infeasible``) and the code clipped. Each ring holds the order a heater-only acquisition
    takes (:meth:`StaticRing.first_order`) with the heater off at ``t_order_base`` (default: t_base with the first
    aggressor sample), so a die whose resonance sits red of its line locks the next bluer order. Also returns the
    single-ring lock points (``locks``) and the monitor-PD temperatures of the bank state (``t_pd_k`` (T, N),
    aggressor heat at the PD not included)."""
    N = int(pp.n_rings)
    rings = [StaticRing(pp, b=b, n=n) for n in range(N)]
    a0 = np.zeros(N) if agg_r is None else np.atleast_2d(agg_r)[0]
    orders = []
    for n, rg in enumerate(rings):
        v_lo, v_hi = rg.nrz_levels(bias) if modulate else (bias, bias)
        p_j, p_d = rg.data_heat(v_lo, v_hi, enable=modulate)
        tf = (t_base + a0[n] if t_order_base is None else float(t_order_base)) + rg.z["jr"] * p_j + rg.z["dr"] * p_d
        orders.append(rg.first_order(rg.lam_line, 0.5 * (v_lo + v_hi), tf))
    # locks in bus order: each ring sees the lines as the rings before it on the bus left them (single forward
    # pass, as the plant's bus walk; the bus segment loss included)
    lps = [None] * N
    p_ring = np.zeros((N, rings[0].p_lines_w.size))
    p_at = (rings[0].p_lines_w.copy() if p_in_w is None else np.full(rings[0].p_lines_w.size, float(p_in_w)))
    seg = _bus_segment_t2(pp, b)
    for q, n in enumerate(np.asarray(pp.layout.bus_order, dtype=np.int64).reshape(-1)):
        if q > 0 and bus_filter:
            p_at = p_at * seg
        rg = rings[n]
        p_ring[n] = p_at
        lps[n] = rg.lock(mode=mode, t_base=t_base, p_in_w=float(p_at[rg.line]), bias_v=bias, modulate=modulate,
                         order=orders[n], lines=lines, p_lines_w=p_at)
        if bus_filter:
            lo, hi = rg.nrz_levels(bias) if modulate else (bias, None)
            o = rg.optics(rg.lam_lines, [lps[n].t_ring_k], None, lo, hi)
            r_ = 0.5 if modulate else 0.0
            p_at = p_at * (r_ * o["thru1"][0] + (1.0 - r_) * o["thru0"][0])
    G = bank_gains(pp, b)
    tl = np.array([lp.t_ring_k for lp in lps])
    pa = np.array([lp.p_abs_w for lp in lps])
    pj = np.array([lp.p_j_w for lp in lps])
    pd_ = np.array([lp.p_d_w for lp in lps])
    agg_r = np.zeros((1, N)) if agg_r is None else np.atleast_2d(agg_r)
    agg_h = np.zeros_like(agg_r) if agg_h is None else np.atleast_2d(agg_h)
    rhs = tl - t_base - G["or"] @ pa - G["jr"] @ pj - G["dr"] @ pd_
    P = np.linalg.solve(G["hr"], (rhs[None, :] - agg_r).T).T
    th_ = t_base + P @ G["hh"].T + (G["oh"] @ pa + G["jh"] @ pj + G["dh"] @ pd_)[None, :] + agg_h
    tpd = t_base + P @ G["hp"].T + (G["op"] @ pa + G["jp"] @ pj + G["dp"] @ pd_)[None, :]
    codes = np.zeros_like(P)
    u = np.zeros_like(P)
    comp = np.ones(P.shape, bool)
    for n, rg in enumerate(rings):
        u[:, n] = rg.u_from_power(P[:, n], th_[:, n])
        codes[:, n] = rg.u_to_code(np.clip(u[:, n], 0.0, None))
        comp[:, n] = rg.compliance_ok(P[:, n], th_[:, n])
    return {"codes": codes, "p_w": P, "u": u, "t_lock_k": tl, "t_heater_k": th_, "t_pd_k": tpd,
            "infeasible": (P < 0) | (u > 1.0) | ~comp, "orders": orders, "locks": lps, "p_lines_at_ring_w": p_ring,
            "law": {v: k for k, v in el.DRIVER_LAWS.items()}[rings[0].law]}


def _code_stats(codes: np.ndarray) -> dict:
    """Statistics of one die's codes (T, N) over one aggressor period: per-lane swing, maxima and minima."""
    cmax, cmin = codes.max(axis=0), codes.min(axis=0)
    sw = cmax - cmin
    return {"swing": [float(sw.min()), float(sw.max())],
            "lane_offsets": {"at_code_maxima": float(cmax.max() - cmax.min()),
                             "at_code_minima": float(cmin.max() - cmin.min())},
            "code_range": [float(cmin.min()), float(cmax.max())],
            "per_lane_swing": sw.tolist(), "per_lane_max": cmax.tolist(), "per_lane_min": cmin.tolist()}


def _median_stats(stats: list[dict], output: str):
    if output == "lane_offsets":
        return {k: float(np.median([s["lane_offsets"][k] for s in stats])) for k in ("at_code_maxima",
                                                                                   "at_code_minima")}
    if output in ("swing", "code_range"):
        lo = float(np.median([s[output][0] for s in stats]))
        hi = float(np.median([s[output][1] for s in stats]))
        return [lo, hi]
    return float(np.median([s[output] for s in stats]))


def stabilizer_code_vs_T(plant, target=None, *, method: str = "quasi_static", run_fn=None, **kw) -> ReplicaResult:
    """CAL-9a/b/c (and VAL-6): heater DAC codes held by the loop over the slide-10 aggressor, per driver-law and
    full-scale hypothesis (07 4.3: the codes identify the DAC LSB only under an assumed law and full scale).

    Protocol (quasi-static): ``n_dies`` sampled dies (per-ring device variation on, every uncertainty entry nominal,
    device seeds 0..n_dies-1); the A1 aggressor power calibrated on each die's ROM (``referencing``, default
    resonance; mrm.scenarios), its periodic steady-state rise at every ring and heater-resistor observer from an
    exact FOH simulation of the ThermalSystem; every ring locked (``setpoint``, default half_max on the hot side) at
    its own line with NRZ traffic at the scenario bias; codes from the bank heat balance (crosstalk included), the
    driver law with the TCR and the realized DAC (:func:`_bank_codes`). Outputs per die, then the median over dies:
    ``swing`` (CAL-9a: [smallest, largest] per-lane swing), ``lane_offsets`` (CAL-9b: max - min over lanes of the
    per-lane code maxima and minima), ``code_range`` (CAL-9c: lowest and highest code over all lanes). With ``t_c``
    = [T_lo, T_hi] and no aggressor (VAL-6), ``excursion``: mean over lanes of code(T_lo) - code(T_hi) for an
    ambient sweep. Variants: every driver law (``driver_mode: hypothesis``), the full-scale range ends of the
    primary law (``dac_full_scale: hypothesis``) and ``aggressor_model=uniform`` (every ring sees the victim's rise).
    ``method='dynamic'`` (ambient protocol only) runs the locks through ``run_fn``."""
    a = _args(target, kw)
    output = a.get("output")
    agg = a.get("aggressor")
    ambient = agg is None and a.get("t_c") is not None
    if output is None:
        output = "excursion" if ambient else "swing"
    if output == "lane_offsets" or (target is not None and getattr(target, "kind", None) == "set"):
        output = "lane_offsets"
    n_dies = int(a.get("n_dies", 8))
    mode = a.get("setpoint", "half_max")
    mode = "half_max" if mode == "hypothesis" else mode
    cfg0, pp0 = resolve_plant(plant)
    bias = float(a.get("bias_v", _scenario_bias(cfg0)))
    laws = [None]
    if cfg0 is not None and a.get("driver_mode", "hypothesis") == "hypothesis":
        laws += list(_hyp_choices(cfg0, "heater.driver_mode"))
    fs_vals = [None]
    own_law = {v: k for k, v in el.DRIVER_LAWS.items()}[StaticRing(pp0).law]
    if cfg0 is not None and a.get("dac_full_scale") == "hypothesis":
        leaf = cfg0.leaves.get(_FS_KEY[own_law])
        if leaf is not None and isinstance(leaf.range, tuple):
            fs_vals += [float(leaf.range[0]), float(leaf.range[1])]

    def one(law, fs, uniform=False, variation="sample"):
        alts = {"uncertainty.heater.driver_mode": law} if law else {}
        ov = {_FS_KEY[law or own_law]: fs} if fs is not None else {}
        if cfg0 is None:
            pp = pp0
        else:
            _, pp = resolve_plant(cfg0, batch=n_dies if variation == "sample" else 1, uncertainty="nominal",
                                  variation=variation, seeds={"device": int(a.get("seed", 0)), "uncertainty": 0},
                                  alts=alts, overrides=ov)
        stats, info = [], {}
        if ambient:
            tl, thi = [float(x) + T0_K for x in a["t_c"]]
            for b in range(pp.batch):
                # one order per ring, acquired at the cold end and held over the sweep (an order hop would show
                # as an infeasible code at the hot end, not as a smaller excursion)
                cl = _bank_codes(pp, b, t_base=tl, bias=bias, mode=mode, t_order_base=tl)
                ch = _bank_codes(pp, b, t_base=thi, bias=bias, mode=mode, t_order_base=tl)
                exc = cl["codes"][0] - ch["codes"][0]
                stats.append({"excursion": float(np.mean(exc)), "per_lane": exc.tolist(),
                              "infeasible": bool(np.any(cl["infeasible"]) or np.any(ch["infeasible"]))})
            info["law"] = cl["law"]
        else:
            if agg not in (None, "CAL-13a", "a1"):
                raise ValueError(f"unknown aggressor {agg!r} (CAL-13a / a1)")
            sc, cw, site = _calibrated_aggressor(pp, which="a1", referencing=a.get("referencing", "resonance"))
            ris = _aggressor_rises(pp, cw)
            t_base = float(sc.defaults.get("thermal.t_base_k", 298.15))
            n_inf = 0
            for b in range(pp.batch):
                ar, ah = ris["ring"][:, b, :], ris["heater"][:, b, :]
                if uniform:
                    ar = np.repeat(ar[:, :1], pp.n_rings, axis=1)
                    ah = np.repeat(ah[:, :1], pp.n_rings, axis=1)
                bc = _bank_codes(pp, b, t_base=t_base, bias=bias, mode=mode, agg_r=ar, agg_h=ah)
                n_inf += int(np.sum(bc["infeasible"]))
                stats.append(_code_stats(bc["codes"]))
            info.update(law=bc["law"], victim_rise_k=[float(ris["ring"][:, :, 0].min()),
                                                      float(ris["ring"][:, :, 0].max())],
                        lane_rise_k=np.max(ris["ring"][:, 0, :], axis=0).tolist(), infeasible_samples=n_inf,
                        calibration_residual_rms_k=np.asarray(cw.residual_rms_k).tolist(),
                        peak_power_mw=(np.asarray(cw.peak_power_w) * 1e3).tolist(), t_base_c=t_base - T0_K)
        return _median_stats(stats, output), {"per_die": stats, **info}

    if method == "dynamic":
        if not ambient:
            raise ReplicaBlocked("the dynamic aggressor protocol needs the engine (130.7 ms periods); use the "
                                 "quasi-static method or run it through mrm.sim")
        return _stabilizer_dynamic(cfg0, pp0, a, run_fn=run_fn, target=target, bias=bias, mode=mode)
    val, info = one(None, None)
    out = ReplicaResult("stabilizer_code_vs_T", val, _unit(target, "codes"), target_id=getattr(target, "id", None),
                        detail={"primary": info})
    if not a.get("with_variants", True):
        laws, fs_vals = laws[:1], fs_vals[:1]
    for law in laws[1:]:
        if law == info.get("law"):
            continue
        v, i = one(law, None)
        out.variants[f"driver_mode={law}"] = v
        out.detail[f"driver_mode={law}"] = {k: i[k] for k in i if k != "per_die"}
    for fs in fs_vals[1:]:
        v, i = one(None, fs)
        out.variants[f"{_FS_KEY[own_law]}={fs}"] = v
    if not ambient and a.get("with_variants", True):
        v, i = one(None, None, uniform=True)
        out.variants["aggressor_model=uniform"] = v
    if cfg0 is not None and a.get("with_variants", True):
        v, i = one(None, None, variation="off")
        out.variants["variation=off (nominal die)"] = v
        out.detail["variation=off (nominal die)"] = {k: i[k] for k in i if k != "per_die"} | {"per_die": i["per_die"]}
    out.conditions = {**_pp_meta(cfg0, pp0), "method": "quasi_static", "output": output, "n_dies": n_dies,
                      "variation": "sample (device seeds 0..n_dies-1)", "uncertainty": "nominal",
                      "setpoint": mode, "bias_v": bias, "driver_law": info.get("law"),
                      "aggressor": None if ambient else "A1 slide-10 (rectified sine), calibrated on each die's ROM",
                      "referencing": a.get("referencing", "resonance"), "statistic": "median over dies",
                      "laser": "every TX line on at the preset chip-input power, filtered along the bus by the "
                               "rings upstream (single forward pass); each ring locks its own line, the others "
                               "heat it and leak into its drop monitor"}
    return out


def _stabilizer_dynamic(cfg, pp, a, *, run_fn, target, bias: float, mode: str) -> ReplicaResult:
    """Ambient-step protocol through run_fn: lock at T_lo, step the base to T_hi, record the heater code."""
    from mrm.scenarios.generators import Scenario, const, step
    tl, thi = [float(x) for x in a["t_c"]]
    dur = float(a.get("duration_s", 2e-3))
    sc = Scenario(name="ambient_step", components=(
        const("thermal.t_base_k", tl + T0_K, absolute=True), step("thermal.t_base_k", 0.5 * dur, thi - tl)))
    tr = _dynamic_lock(cfg, pp, mode, bias=bias, t_base=tl + T0_K, run_fn=run_fn, scenario=sc, duration_s=dur)
    t = tr["t_s"]
    c = tr["heater_code"]
    lo = np.median(c[(t > 0.3 * dur) & (t < 0.5 * dur)], axis=0)
    hi = np.median(c[t > 0.85 * dur], axis=0)
    exc = (lo - hi)[0]
    qs_lo = _bank_codes(pp, 0, t_base=tl + T0_K, bias=bias, mode=mode, t_order_base=tl + T0_K)["codes"][0]
    qs_hi = _bank_codes(pp, 0, t_base=thi + T0_K, bias=bias, mode=mode, t_order_base=tl + T0_K)["codes"][0]
    out = ReplicaResult("stabilizer_code_vs_T", float(np.mean(exc)), _unit(target, "codes"),
                        target_id=getattr(target, "id", None),
                        variants={"quasi_static": float(np.mean(qs_lo - qs_hi))},
                        detail={"dynamic_per_lane": exc.tolist(), "quasi_static_per_lane": (qs_lo - qs_hi).tolist()})
    out.conditions = {**_pp_meta(cfg, pp), "method": "dynamic", "t_c": [tl, thi], "duration_s": dur,
                      "setpoint": mode, "bias_v": bias}
    return out


# =====================================================================================================================
# Replica: bistability (07 P0-5 verification protocol)
# =====================================================================================================================
def bistability_sweep(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """Slow heater sweeps up and down at power (laser parked at the line, CW at ``bias_v``, base ``t_c``, default the
    reference temperature; ``p_in_dbm``: dBm, ``"preset"`` (default: the preset's line power at the ring) or None for
    P_in -> 0): the quasi-static branch is followed in both directions and the jumps (folds) located.
    value: {"jump_up_u", "jump_down_u", "hysteresis_u", "hysteresis_codes", "hysteresis_mw", "bistable"} with u the
    normalized drive of the plant's driver law (codes through the realized DAC); detail carries both traces."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    r = StaticRing(pp, n=ring)
    p_spec = a.get("p_in_dbm", "preset")
    p_in = r.p_line_w if p_spec == "preset" else _dbm_to_w(p_spec)
    v = float(a.get("bias_v", 0.0))
    tb = _t_base_k(a, r)
    lam = r.lam_line
    t_r = r.t_res(lam, v)
    w = r.linewidth_k(t_r)
    t_top = t_r + r.z["or"] * p_in + 8 * w
    n_grid = max(int(a.get("n_grid", 6001)), int((t_top - tb) / (w / 60.0)))
    tg = np.linspace(tb - 1.0, t_top, n_grid)            # from the heater-off ring to past the hot branch
    st = r.state_grid(lam, tg, t_base=tb, p_in_w=p_in, v_lo=v)
    u_t = st["u"][:, 0]
    u_hi = float(np.max(u_t))
    us = np.linspace(0.0, min(u_hi, 1.0 if r.law != -1 else u_hi), int(a.get("n_steps", 2001)))
    trace = {}
    for d, seq in (("up", us), ("down", us[::-1])):
        i_prev = 0.0 if d == "up" else float(tg.size - 1)
        tt = np.zeros(seq.size)
        for k, u0 in enumerate(seq):
            tt[k], i_prev, _ = _relax(tg, u0 - u_t, i_prev)
        trace[d] = tt
    # a jump is a step in T between consecutive drive steps larger than a quarter linewidth
    def jump(tt, seq):
        dt = np.abs(np.diff(tt))
        k = int(np.argmax(dt))
        return (float(0.5 * (seq[k] + seq[k + 1])), float(dt[k])) if dt[k] > 0.25 * w else (math.nan, float(dt[k]))
    ju, du = jump(trace["up"], us)
    jd, dd = jump(trace["down"], us[::-1])
    bistable = bool(math.isfinite(ju) and math.isfinite(jd) and ju > jd)
    hyst_u = (ju - jd) if bistable else 0.0
    hyst_codes = float(r.u_to_code(ju) - r.u_to_code(jd)) if bistable else 0.0
    hyst_mw = float((r.power_from_u([ju], tb)[0] - r.power_from_u([jd], tb)[0]) * 1e3) if bistable else 0.0
    val = {"jump_up_u": ju, "jump_down_u": jd, "hysteresis_u": hyst_u, "hysteresis_codes": hyst_codes,
           "hysteresis_mw": hyst_mw, "bistable": bistable}
    from mrm.plant.selfheat import bistability_ratio
    o = r.optics(lam, [t_r], None, v)
    ratio = float(bistability_ratio(r.s_at(t_r) * r.z["or"], p_in * float(o["abs0"][0, 0]),
                                    r.lam_res / r.q_loaded))
    out = ReplicaResult("bistability_sweep", val, "1", target_id=getattr(target, "id", None),
                        detail={"drive_u": us.tolist()[::20], "t_up_c": (trace["up"][::20] - T0_K).tolist(),
                                "t_down_c": (trace["down"][::-1][::20] - T0_K).tolist(), "jump_up_k": du,
                                "jump_down_k": dd, "bistability_ratio": ratio})
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "p_in_w": p_in, "bias_v": v, "t_base_c": tb - T0_K,
                      "laser": "parked at the ring's line, CW", "sweep": "slow heater drive up then down"}
    return out


# =====================================================================================================================
# Replica: electro-optic bandwidth (small-signal TCMT x junction RC)
# =====================================================================================================================
def _eo_response(r: StaticRing, bias_v: float, f_hz: np.ndarray, *, detuning: str = "max_slope",
                 r_source_ohm: float = 0.0) -> dict:
    """Small-signal EO response of the through port at reverse bias ``bias_v``: the TCMT of 03 section 2.2 (rates from
    optics.exact_to_tcmt at the line) linearized in the resonance frequency, times the junction RC low-pass
    1 / (1 + j 2 pi f (R_s + R_source) C_j) with C_j, R_s from the junction table at the bias.

    TCMT in the frame of the laser, A0 = -j sqrt(ke) / (j D + k/2) (input 1), through t0 = 1 - j sqrt(ke) A0; a
    modulation d omega = Re{d e^{j W t}} gives A+- = j (d/2) A0 / (+-j W + j D + k/2) and the e^{j W t} part of the
    output power P_W = 2 [conj(t0) (-j sqrt(ke)) A+ + conj(conj(t0) (-j sqrt(ke)) A-)]. D = omega_laser -
    omega_res > 0 (laser blue of the resonance). ``detuning``: ``max_slope`` (the D > 0 of maximum |dT/dD|),
    ``hwhm`` (D = k/2), ``half_depth`` (T = (T(0) + 1)/2). Returns |H(f)/H(0)| and the 3 dB bandwidth (EO S21,
    20 log10 |H/H0| = -3 dB)."""
    lam = r.lam_line
    t_r = r.t_res(lam, bias_v)
    o = r.optics(lam, [t_r], None, bias_v)
    ki, ke, kd = op.exact_to_tcmt(o["k2t"][0, 0], o["k2d"][0, 0], o["a0"][0, 0], o["trt"][0, 0])
    kap = float(ki + ke + kd)
    ke = float(ke)

    def thru(D):
        return np.abs(1.0 - ke / (1j * D + 0.5 * kap)) ** 2
    if detuning == "max_slope":
        D = optimize.minimize_scalar(lambda x: -abs((thru(x + 1e-6 * kap) - thru(x - 1e-6 * kap)) / (2e-6 * kap)),
                                     bounds=(1e-3 * kap, 2.0 * kap), method="bounded").x
    elif detuning == "hwhm":
        D = 0.5 * kap
    elif detuning == "half_depth":
        lev = 0.5 * (thru(0.0) + 1.0)
        D = optimize.brentq(lambda x: thru(x) - lev, 1e-6 * kap, 50 * kap)
    else:
        raise ValueError(f"unknown detuning rule {detuning!r}")
    W = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    A0 = -1j * math.sqrt(ke) / (1j * D + 0.5 * kap)
    t0 = 1.0 - 1j * math.sqrt(ke) * A0
    Ap = 1j * 0.5 * A0 / (1j * W + 1j * D + 0.5 * kap)
    Am = 1j * 0.5 * A0 / (-1j * W + 1j * D + 0.5 * kap)
    c = np.conj(t0) * (-1j * math.sqrt(ke))
    h_opt = 2.0 * (c * Ap + np.conj(c * Am))
    cj = float(o["cj"][0, 0])
    rs = float(o["rs"][0, 0]) + float(r_source_ohm)
    h = h_opt / (1.0 + 1j * W * rs * cj)
    mag = np.abs(h) / np.abs(h[0])
    db = 20.0 * np.log10(mag)
    k = np.flatnonzero(db <= -3.0)
    bw = math.nan
    if k.size and k[0] > 0:
        i = int(k[0])
        bw = float(np.interp(-3.0, [db[i], db[i - 1]], [f_hz[i], f_hz[i - 1]]))
    return {"bw_hz": bw, "mag": mag, "h": h, "h_opt": h_opt, "kappa_rad_per_s": kap, "ke_rad_per_s": ke,
            "detuning_rad_per_s": float(D), "kappa_hz": kap / (2 * np.pi), "detuning_hz": D / (2 * np.pi),
            "rc_hz": 1.0 / (2 * np.pi * rs * cj), "c_j_f": cj, "r_s_ohm": rs, "peaking_db": float(np.max(db))}


#: laser detuning rules of the EO protocol as x = 2 Delta / kappa (Lorentzian notch: maximum static slope at
#: 1/sqrt(3); the half width and the half depth both at x = 1)
_EO_X = {"max_slope": 1.0 / math.sqrt(3.0), "hwhm": 1.0, "half_depth": 1.0}


def _eo_waveform(pp, b: int, n: int, bias_v: float, rule: str, r_drv_ohm: float | None,
                 loss_modulation: bool = True) -> dict:
    """EO small-signal response in waveform mode (mrm.waveform: exact TCMT linearization with index and loss
    modulation, junction C_j(V) and R_s through the driver output resistance). r_drv_ohm None: the plant's
    datapath.driver_output_resistance_ohm."""
    from mrm.waveform import DriverJunction, RingTCMT, eo_bandwidth
    ring = RingTCMT.from_plant_params(pp, b, n, v_bias=float(bias_v), x_bias=_EO_X[rule])
    dj = DriverJunction.from_plant_params(pp, b, n, rise_time_s=0.0, r_drv_ohm=r_drv_ohm)
    res = eo_bandwidth(ring, dj, float(bias_v), loss_modulation=loss_modulation)
    return {"bw_hz": float(res["f3db_hz"]), "peaking_db": float(res["peaking_db"]), "fwhm_hz": float(res["fwhm_hz"]),
            "rc_pole_hz": float(res["rc_pole_hz"]), "x_bias": float(res["x_bias"]), "r_s_ohm": float(dj.r_s(bias_v)),
            "r_drv_ohm": float(dj.r_drv_ohm), "c_j_f": float(dj.c_j(bias_v))}


def eo_bandwidth_at_bias(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    """CAL-10, VAL-3a/b: small-signal electro-optic 3 dB bandwidth (EO S21, |H(f)/H(0)|^2 = 1/2) at reverse bias
    ``bias_v``, in waveform mode (README: ``mrm.waveform.eo_bandwidth``, 07 P2-6): the exact TCMT linearization with
    the index AND loss (FCA) modulation of the junction, C_j(V) and R_s from the junction table, driven through the
    driver output resistance (``r_drv_ohm``; default the plant's ``datapath.driver_output_resistance_ohm``, 50 ohm in
    the presets, as a 50 ohm VNA port would). The laser is re-centred at each bias (VAL-3b ratio at the same
    detuning rule). ``detuning: hypothesis`` -> primary ``max_slope`` (x = 2 Delta/kappa = 1/sqrt(3)), variant
    ``hwhm`` (x = 1, the same as half depth for a Lorentzian notch). Variants: ``r_drv_ohm=0``, no loss modulation,
    and the replica's own linearized index-only TCMT with R_source = 0 (:func:`_eo_response`, the primary before
    the waveform package existed). ``output: ratio`` with two biases gives BW(bias[1]) / BW(bias[0]) (VAL-3b)."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    r = StaticRing(pp, n=ring)
    bias = a.get("bias_v", 1.0)
    rule = a.get("detuning", "hypothesis")
    rules = ["max_slope", "hwhm"] if rule == "hypothesis" else [rule]
    r_drv = a.get("r_drv_ohm")
    method = a.get("method", "waveform")
    f = np.concatenate([[1e6], np.geomspace(1e8, 1e12, 4001)])
    ratio = a.get("output") == "ratio" or isinstance(bias, (list, tuple))
    biases = [float(x) for x in bias] if ratio else [float(bias)]

    def one(v, rule_, how, r_drv_, loss=True):
        if how == "waveform":
            return _eo_waveform(pp, r.b, r.n, v, rule_, r_drv_, loss_modulation=loss)
        rr = _eo_response(r, v, f, detuning=rule_, r_source_ohm=0.0 if r_drv_ is None else float(r_drv_))
        return {"bw_hz": rr["bw_hz"], "peaking_db": rr["peaking_db"], "fwhm_hz": rr["kappa_hz"],
                "rc_pole_hz": rr["rc_hz"], "x_bias": 2.0 * rr["detuning_rad_per_s"] / rr["kappa_rad_per_s"],
                "r_s_ohm": rr["r_s_ohm"], "c_j_f": rr["c_j_f"]}

    def predict(rule_, how, r_drv_, loss=True):
        rows = [one(v, rule_, how, r_drv_, loss) for v in biases]
        if ratio:
            return rows[1]["bw_hz"] / rows[0]["bw_hz"], {"bw_ghz": [x["bw_hz"] / 1e9 for x in rows], "rows": rows}
        return rows[0]["bw_hz"] / 1e9, rows[0]
    val, info = predict(rules[0], method, r_drv)
    out = ReplicaResult("eo_bandwidth_at_bias", float(val), _unit(target, "1" if ratio else "ghz"),
                        target_id=getattr(target, "id", None), detail={rules[0]: info})
    for rl in rules[1:]:
        v, i = predict(rl, method, r_drv)
        out.variants[f"detuning={rl}"] = float(v)
        out.detail[rl] = i
    if a.get("with_variants", True):
        if method == "waveform":
            if r_drv is None or float(r_drv) != 0.0:
                out.variants["r_drv_ohm=0"] = float(predict(rules[0], "waveform", 0.0)[0])
            out.variants["loss_modulation=off"] = float(predict(rules[0], "waveform", r_drv, loss=False)[0])
            v, i = predict(rules[0], "linearized", None)
            out.variants["method=linearized_index_only_r_source_0"] = float(v)
            out.detail["method=linearized_index_only_r_source_0"] = i
        else:
            out.variants["r_source_ohm=50"] = float(predict(rules[0], "linearized", 50.0)[0])
    out.conditions = {**_pp_meta(cfg, pp), "ring": ring, "bias_v": bias, "detuning": rules[0],
                      "x_bias": _EO_X[rules[0]], "method": method,
                      "r_drv_ohm": info["rows"][0].get("r_drv_ohm") if ratio else info.get("r_drv_ohm", r_drv),
                      "response": ("mrm.waveform.eo_bandwidth: TCMT with index + loss modulation, C_j(V), R_s, "
                                   "driver output resistance" if method == "waveform" else
                                   "linearized TCMT (index only) x junction RC (small signal)"),
                      "laser": "re-centred at each bias to the detuning rule (ring temperature set accordingly)"}
    if ratio:
        out.notes.append("VAL-3b: both biases at the same detuning rule (laser re-tuned at each bias)")
    if method == "waveform":
        out.notes.append("method changed at the 2026-10-09 review: the primary was the replica's own linearized "
                         "index-only TCMT with R_source = 0 (variant 'method=linearized_index_only_r_source_0') while "
                         "mrm.waveform did not exist; it is now the README's waveform mode with the plant's driver "
                         "output resistance and the loss modulation. The change was made after the earlier "
                         "predictions had been compared with CAL-10 / VAL-3 (50.1 GHz, 51.3 GHz, ratio 1.025: all "
                         "failed); report both")
    return out


# =====================================================================================================================
# Replicas: populations (responsivity, laser lines)
# =====================================================================================================================
def pd_responsivity(plant, target=None, **kw) -> ReplicaResult:
    """CAL-11: [min, max] of the sampled monitor-PD responsivity (A/W) over ``batch`` worlds x rings (uncertainty and
    device variation sampled, seeds 0)."""
    a = _args(target, kw)
    B = int(a.get("batch", 256))
    cfg, pp = resolve_plant(plant, batch=B, uncertainty="sample", variation="sample")
    x = np.asarray(pp.monitor.responsivity_a_per_w, dtype=float).ravel()
    out = ReplicaResult("pd_responsivity", [float(x.min()), float(x.max())], _unit(target, "a_per_w"),
                        target_id=getattr(target, "id", None),
                        detail={"n": int(x.size), "quantiles": np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99]).tolist()})
    out.conditions = {**_pp_meta(cfg, pp), "uncertainty": "sample", "variation": "sample", "statistic": "range"}
    return out


def laser_line_statistics(plant, target=None, **kw) -> ReplicaResult:
    """CAL-12a-d: the sampled static line error (line frequency minus grid, GHz; ``batch`` worlds with the per-die and
    per-line variation) and the 24 h drift of the ``laser_drift`` scenario (mrm.scenarios; ``drift_mode`` linear,
    the default worst case, or random_walk over ``batch`` worlds) rendered every 60 s. ``output``:
    ``max_abs_static_error_ghz`` | ``static_error_quartiles_ghz`` ({median, q25, q75}) | ``max_abs_drift_24h_nm`` |
    ``max_abs_power_drift_24h_pct``."""
    a = _args(target, kw)
    output = a.get("output", "max_abs_static_error_ghz")
    B = int(a.get("batch", 256))
    if output in ("max_abs_static_error_ghz", "static_error_quartiles_ghz"):
        cfg, pp = resolve_plant(plant, batch=B, uncertainty="sample", variation="sample")
        tx = np.array([d == "tx" for d in pp.laser.direction]) if len(pp.laser.direction) else \
            np.ones(pp.n_lines, bool)
        err = (np.asarray(pp.laser.line_freq_hz) - np.asarray(pp.laser.line_freq_grid_hz)[None, :])[:, tx] / 1e9
        e = err.ravel()
        if output == "max_abs_static_error_ghz":
            val = float(np.max(np.abs(e)))
        else:
            q = np.quantile(e, [0.25, 0.5, 0.75])
            val = {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}
        out = ReplicaResult("laser_line_statistics", val, _unit(target, "ghz"), target_id=getattr(target, "id", None),
                            detail={"n": int(e.size), "sigma_ghz": float(np.std(e)),
                                    "fraction_outside_15ghz": float(np.mean(np.abs(e) > 15.0))})
        out.conditions = {**_pp_meta(cfg, pp), "sign": "line frequency minus grid", "lines": "tx"}
        if a.get("generation") == "Guide":
            out.notes.append("the model's line error population is variation.laser.line_offset_ghz of the preset "
                             "(die and line sigma, clipped at the +-20 GHz requirement); no separate Guide model")
        return out
    from mrm.scenarios import library
    mode = a.get("drift_mode", "linear")
    sc = library.laser_drift(mode=mode)
    nb = 1 if mode == "linear" else B
    cfg, pp = resolve_plant(plant)
    L = pp.n_lines
    fr = sc.render(dt_s=60.0, n_ticks=24 * 60 + 1, n_lines=L, batch=nb, seed=0)
    if output == "max_abs_drift_24h_nm":
        df = np.asarray(fr.signals["laser.dfreq_hz"])
        lam = C0 / np.asarray(pp.laser.line_freq_grid_hz)
        dl = np.abs(df) * (lam[None, None, :] ** 2) / C0
        val = float(np.max(dl) * 1e9)
        unit = "nm"
    elif output == "max_abs_power_drift_24h_pct":
        db = np.asarray(fr.signals["laser.power_delta_db"])
        val = float(np.max(np.abs(10.0 ** (db / 10.0) - 1.0)) * 100.0)
        unit = "pct"
    else:
        raise ValueError(f"unknown output {output!r}")
    out = ReplicaResult("laser_line_statistics", val, _unit(target, unit), target_id=getattr(target, "id", None))
    out.conditions = {**_pp_meta(cfg, pp), "scenario": f"laser_drift({mode})", "window_s": 86400.0,
                      "render_step_s": 60.0, "worlds": nb}
    return out


# =====================================================================================================================
# Replica: aggressor trajectory in resonance units (CAL-13a/b)
# =====================================================================================================================
def _trajectory_c(pp, cw, target_obj, *, victim: int = 0, ring_t0: str = "lock", n_periods: int = 4,
                  n_samples: int = 512) -> dict:
    """The victim ring's aggressor-only response over one period in periodic steady state (exact FOH simulation of
    its Foster response, mrm.scenarios.calibrate.simulate_foster_foh, independent of the calibration's own response
    matrix), converted to the printed degC: resonance referencing y = T_base + shift / s_ref with the model's R1
    from the ring's starting temperature; temperature referencing y = T_base + thermometer_fraction x rise."""
    from mrm.scenarios.calibrate import foster_from_system, rise_to_resonance_m, simulate_foster_foh
    site = _aggressor_site(pp)
    resp = foster_from_system(pp.thermal.system, victim_ring=victim, site=site)[0]
    P = float(cw.period_s)
    h = P / n_samples
    t = np.arange(n_periods * n_samples + 1) * h
    rise, _ = simulate_foster_foh(resp, cw.power_at(t, b=0), h)
    tl = t[-(n_samples + 1):-1] - (n_periods - 1) * P
    x = rise[-(n_samples + 1):-1]
    tt = pp.thermal
    t_start = float(np.asarray(tt.t_ring_lock_k)[0, victim]) if ring_t0 == "lock" else target_obj.t_base_k
    if target_obj.units == "resonance":
        sh = rise_to_resonance_m(x, s0_m_per_k=float(np.asarray(tt.dlambda_dt_s0_m_per_k)[0, victim]),
                                 beta_per_k=float(np.asarray(tt.dlambda_dt_beta_per_k)[0, 0]),
                                 t_ref_k=float(np.asarray(tt.dlambda_dt_t0_k)[0, 0]), t_base_k=t_start)
        y = target_obj.t_base_k - T0_K + sh / target_obj.s_ref_m_per_k
    else:
        y = target_obj.t_base_k - T0_K + target_obj.thermometer_fraction * x
    return {"t_s": tl, "y_c": y, "rise_k": x, "t_start_k": t_start}


def aggressor_trajectory(plant, target=None, **kw) -> ReplicaResult:
    """CAL-13a/b: the aggressor of mrm.scenarios (A1 slide-10 or A2 808 degC/s) calibrated on the plant's ROM (the
    scenario's own calibration step, ENGINE_BRIEF section 1; referencing ``resonance`` with 85.86 pm/K by default),
    then simulated independently through the victim's Foster response in periodic steady state, t = 0 at the cusp.
    CAL-13a: {"t_s", "y"} over one period, y in the printed degC (resonance referencing: base + model resonance shift /
    85.86 pm/K). CAL-13b (``interpretation`` present): the printed-rate equivalent of the simulated trajectory,
    ``delta_t_per_period`` = (max - min) / period, ``one_sided_sawtooth`` = (max - min) / rise time; variants for the
    other readings and interpretations."""
    a = _args(target, kw)
    cfg, pp = resolve_plant(plant)
    ref = a.get("referencing", "resonance")
    if "interpretation" not in a:
        wf = a.get("waveform", "rectified_sine")
        sc, cw, site = _calibrated_aggressor(pp, which="a1", referencing=ref, waveform=wf)
        tgt = sc.aggressors[0].target
        tr = _trajectory_c(pp, cw, tgt)
        val = {"t_s": tr["t_s"].tolist(), "y": tr["y_c"].tolist()}
        from mrm.validation.targets_io import rectified_sine
        p = (getattr(target, "value", None) or {}).get("params", {"t_max_c": 113.6, "delta_t_k": 88.5,
                                                                  "period_s": 0.1307})
        rms = float(np.sqrt(np.mean((tr["y_c"] - rectified_sine(tr["t_s"], p["t_max_c"], p["delta_t_k"],
                                                                 p["period_s"])) ** 2)))
        out = ReplicaResult("aggressor_trajectory", val, _unit(target, "c"), target_id=getattr(target, "id", None),
                            detail={"rms_vs_target_c": rms, "victim_rise_k": [float(tr["rise_k"].min()),
                                                                              float(tr["rise_k"].max())],
                                    "calibration_residual_rms_k": np.asarray(cw.residual_rms_k).tolist(),
                                    "peak_power_mw": float(np.asarray(cw.peak_power_w)[0] * 1e3),
                                    "mean_power_mw": float(np.asarray(cw.mean_power_w)[0] * 1e3)})
        out.conditions = {**_pp_meta(cfg, pp), "aggressor": "A1 slide-10", "waveform": wf, "referencing": ref,
                          "s_ref_pm_per_k": tgt.s_ref_m_per_k * 1e12, "ring_t0": "lock temperature (ENGINE_BRIEF)",
                          "window": "one period in periodic steady state, t = 0 at the cusp", "site": site}
        out.notes.append("CAL-13a is a scenario calibration: the power waveform is fitted to this target by "
                         "mrm.scenarios, so the residual measures the calibration (and the waveform family), not a "
                         "prediction")
        return out
    interps = a["interpretation"]
    interps = [interps] if isinstance(interps, str) else list(interps)
    readings = a.get("waveform", ["rectified_sine", "symmetric_triangle"])
    readings = [readings] if isinstance(readings, str) else list(readings)
    res = {}
    for it in interps:
        rds = ["one_sided_sawtooth"] if it == "one_sided_sawtooth" else readings
        for rd in rds:
            sc, cw, _ = _calibrated_aggressor(pp, which="a2", referencing=ref, reading=rd)
            tgt = sc.aggressors[0].target
            tr = _trajectory_c(pp, cw, tgt)
            y, t = tr["y_c"], tr["t_s"]
            if it == "delta_t_per_period":
                rate = (y.max() - y.min()) / float(cw.period_s)
            else:
                i0, i1 = int(np.argmin(y)), int(np.argmax(y))
                dt_ = (t[i1] - t[i0]) % float(cw.period_s)
                rate = (y.max() - y.min()) / dt_ if dt_ > 0 else math.nan
            res[f"{it}/{rd}"] = float(rate)
    primary = f"{interps[0]}/{'one_sided_sawtooth' if interps[0] == 'one_sided_sawtooth' else 'symmetric_triangle'}"
    if primary not in res:
        primary = next(iter(res))
    out = ReplicaResult("aggressor_trajectory", res[primary], _unit(target, "k_per_s"),
                        target_id=getattr(target, "id", None), variants={k: v for k, v in res.items() if k != primary})
    out.conditions = {**_pp_meta(cfg, pp), "aggressor": "A2 808 degC/s", "primary": primary, "referencing": ref}
    out.notes.append("a scenario calibration (the A2 power is fitted to each reading); the one-sided sawtooth cannot "
                     "be followed by a heater, so its residual is large by construction")
    return out


# =====================================================================================================================
# Replicas: bounds
# =====================================================================================================================
def adjacent_order_penalty(plant, target=None, **kw) -> ReplicaResult:
    """BND-1: for every ring locked with its resonance on its own line, the distance of every other resonance order
    inside the TX span (+- 2 FWHM) to the nearest other TX line, in units of that order's FWHM (TCMT linewidth at the
    order's wavelength: coupling dispersion included). Orders from the cold phase curve of the phase model (07 P1-6,
    rigid translation), lines at their realized frequencies. Primary: the nominal world, minimum over rings and
    orders; variant: the minimum over ``batch`` sampled worlds (uncertainty and variation sampled)."""
    a = _args(target, kw)

    def min_dist(pp) -> tuple[float, list]:
        rows = []
        best = math.inf
        tx = np.array([d == "tx" for d in pp.laser.direction]) if len(pp.laser.direction) else \
            np.ones(pp.n_lines, bool)
        for b in range(pp.batch):
            f_lines = np.asarray(pp.laser.line_freq_hz)[b]
            ftx = f_lines[tx]
            for n in range(pp.n_rings):
                r = StaticRing(pp, b=b, n=n)
                own = r.line
                lam_own = C0 / f_lines[own]
                delta = lam_own - r.lam_res
                fsr = _fsr_m(r)
                lam_tx = C0 / ftx
                x_lo = lam_tx.min() - delta - 2.5 * fsr
                x_hi = lam_tx.max() - delta + 2.5 * fsr
                x = np.arange(x_lo, x_hi, 0.5e-12)
                M = x.size
                rp = np.ascontiguousarray(r.rp[None, :])
                outs = [np.zeros((1, M)) for _ in range(9)]
                o2 = [np.zeros((1, 2)) for _ in range(3)]
                fl = np.zeros(1, np.int8)
                ph.ring_optics_row(rp, x, np.array([r.t_ref]), np.array([r.t_ref]), np.array([0.0]),
                                   np.array([0.0]), r.grid, *r.tab, *outs, *o2, fl)
                phi = outs[0][0]
                q = phi / (2 * np.pi)
                m_own = int(round(float(np.interp(r.lam_res, x, q))))
                k = np.flatnonzero(np.diff(np.floor(q)) != 0)
                for i in k:
                    order = int(max(np.floor(q[i]), np.floor(q[i + 1])))
                    if order == m_own:
                        continue
                    xi = x[i] + (order - q[i]) / (q[i + 1] - q[i]) * (x[i + 1] - x[i])
                    lam_o = xi + delta
                    f_o = C0 / lam_o
                    oo = r.optics([lam_o], [r.t_ref], None, 0.0)
                    ki, ke, kd = op.exact_to_tcmt(oo["k2t"][0, 0], oo["k2d"][0, 0], oo["a0"][0, 0], oo["trt"][0, 0])
                    fwhm = float(ki + ke + kd) / (2 * np.pi)
                    if not (ftx.min() - 2 * fwhm <= f_o <= ftx.max() + 2 * fwhm):
                        continue
                    others = np.delete(f_lines, own)[np.delete(tx, own)]
                    if others.size == 0:
                        continue
                    d = float(np.min(np.abs(others - f_o))) / fwhm
                    rows.append({"b": b, "ring": n, "order_offset": order - m_own, "lam_nm": lam_o * 1e9,
                                 "distance_fwhm": d})
                    best = min(best, d)
        return best, rows
    cfg, pp = resolve_plant(plant)
    val, rows = min_dist(pp)
    out = ReplicaResult("adjacent_order_penalty", float(val) if math.isfinite(val) else 1e9, _unit(target, "1"),
                        target_id=getattr(target, "id", None),
                        detail={"rows": sorted(rows, key=lambda r_: r_["distance_fwhm"])[:16],
                                "orders_in_span": len(rows)})
    if not rows:
        out.notes.append("no other order falls inside the TX span +- 2 FWHM (value set to 1e9)")
    B = int(a.get("batch", 32))
    if cfg is not None and B > 1:
        _, pps = resolve_plant(cfg, batch=B, uncertainty="sample", variation="sample")
        vs, rs = min_dist(pps)
        out.variants["sampled_worlds_min"] = float(vs) if math.isfinite(vs) else 1e9
        per_world = {}
        for r_ in rs:
            per_world[r_["b"]] = min(per_world.get(r_["b"], math.inf), r_["distance_fwhm"])
        d = np.array(list(per_world.values()) or [math.inf])
        out.detail["sampled"] = {"batch": B, "fraction_below_2": float(np.mean(d < 2.0)),
                                 "fraction_below_1": float(np.mean(d < 1.0))}
    out.conditions = {**_pp_meta(cfg, pp), "lines": "tx bank at realized frequencies", "orders": "all inside span "
                      "+- 2 FWHM", "lock": "resonance on its own line (rigid comb)", "fwhm": "TCMT at each order"}
    return out


def order_hop_margin(plant, target=None, **kw) -> ReplicaResult:
    """BND-2: at the hot end (``t_hot_c``, base temperature) with every ring of the bank locked (``lock``: default
    ``oracle`` = maximum relOMA, the G26 operating point; variants half_max, peak), NRZ at ``bias_v``, each line at the
    preset's chip-input power: the resonance shift each ring's OWN heater still provides, R1 from T_lock - Z_hr,ii P_i
    to T_lock (nm). Negative means the heater would have to cool the ring (an order hop or a remap is needed).
    Primary: the worst lane of the nominal world; variant: the worst lane over ``n_dies`` sampled dies. The cold
    offset is the preset's at its reference temperature (T0 = 25 degC assumed, the target's condition)."""
    a = _args(target, kw)
    t_hot = float(a.get("t_hot_c", 105.0)) + T0_K
    bias = float(a.get("bias_v", 3.75))
    lock = a.get("lock", "oracle")

    t0 = float(a.get("t0_c", 25.0)) + T0_K

    def margins(pp, mode):
        out_ = []
        for b in range(pp.batch):
            bc = _bank_codes(pp, b, t_base=t_hot, bias=bias, mode=mode, t_order_base=t0)
            G = bank_gains(pp, b)
            for n in range(pp.n_rings):
                r = StaticRing(pp, b=b, n=n)
                tl = bc["t_lock_k"][n]
                dt = G["hr"][n, n] * bc["p_w"][0, n]
                out_.append(float(r.thermal_shift_m(tl) - r.thermal_shift_m(tl - dt)) * 1e9)
        return np.array(out_)
    cfg, pp = resolve_plant(plant)
    m = margins(pp, lock)
    out = ReplicaResult("order_hop_margin", float(m.min()), _unit(target, "nm"),
                        target_id=getattr(target, "id", None), detail={"per_lane_nm": m.tolist()})
    for alt in ("half_max", "peak"):
        if alt != lock:
            out.variants[f"lock={alt}"] = float(margins(pp, alt).min())
    n_dies = int(a.get("n_dies", 8))
    if cfg is not None and n_dies > 1:
        _, pps = resolve_plant(cfg, batch=n_dies, uncertainty="nominal", variation="sample")
        out.variants["sampled_dies_worst_lane"] = float(margins(pps, lock).min())
    out.conditions = {**_pp_meta(cfg, pp), "t_hot_c": t_hot - T0_K, "bias_v": bias, "lock": lock,
                      "lanes": "worst", "traffic": "random NRZ", "t0_c": t0 - T0_K,
                      "order": "the order a heater-only acquisition takes at t0 (cold start), held to the hot end",
                      "definition": "own-heater share of the shift at lock (R1), nm"}
    return out


# =====================================================================================================================
# Replicas: third-party pipeline replications (read the extraction artefacts)
# =====================================================================================================================
def _artefact(name: str) -> dict | None:
    p = tio.PACKAGE_DIR / "data" / "extracted" / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def mode_solve_published_geometry(plant=None, target=None, **kw) -> ReplicaResult:
    """VAL-7a/b: the mode solver on the de Cea et al. 2019 cross-section (mrm.extraction.modes, xsec decea2019),
    read from ``modes_decea2019.json``. ``output``: ``n_g`` -> the group index referenced to the 10 um outer radius
    (the printed n_g comes from the FSR with that radius); ``confinement_poynting_si`` -> the Si energy fraction
    (the frozen VAL-7b text asks for "an energy or power fraction in Si, not the index weight dn_eff/dn_Si"); the
    Robinson (dn_eff/dn_Si) value is a variant because the extraction owner reads the paper's Gamma as that one."""
    a = _args(target, kw)
    unit = _unit(target, "1")
    d = _artefact("modes_decea2019.json")
    if d is None or "replication" not in d:
        return _blocked("mode_solve_published_geometry", target, unit,
                        "modes_decea2019.json (mrm.extraction.modes --xsec decea2019) is not extracted yet",
                        paper=a.get("paper"))
    mn = d["replication"]["model_nominal"]
    output = a.get("output", "n_g")
    if output == "n_g":
        val, var = mn["n_g_outer"], {"n_g_centreline": mn["n_g_centreline"]}
    else:
        val, var = mn["energy_fraction_si"], {"gamma_robinson_dn_eff_dn_si": mn["gamma_robinson"]}
    out = ReplicaResult("mode_solve_published_geometry", float(val), unit, target_id=getattr(target, "id", None),
                        variants={k: float(v) for k, v in var.items()},
                        detail={"thickness_sweep": d["replication"].get("thickness_sweep")})
    out.conditions = {"artefact": "modes_decea2019.json", "paper": "DEC19", "output": output}
    if output != "n_g":
        out.notes.append("VAL-7b definition conflict: the frozen target asks for an energy/power fraction; the "
                         "extraction reads de Cea's Gamma as Robinson's dn_eff/dn_Si (variant)")
    return out


def self_heating_time_constant(plant=None, target=None, **kw) -> ReplicaResult:
    """VAL-7c: thermal_fem.py with the absorption source in the waveguide core of the de Cea 2019 device. No de Cea
    thermal stack exists in mrm/extraction/thermal_fem.py (only the imec VAL-8 stack), so this is blocked; it reads
    ``thermal_val7_decea.json`` (schema: {"tau_us": ...}) once an extraction writes it."""
    unit = _unit(target, "us")
    d = _artefact("thermal_val7_decea.json")
    if d is None:
        return _blocked("self_heating_time_constant", target, unit,
                        "no de Cea thermal stack in mrm/extraction/thermal_fem.py and no thermal_val7_decea.json",
                        paper="DEC19")
    out = ReplicaResult("self_heating_time_constant", float(d["tau_us"]), unit, target_id=getattr(target, "id", None))
    out.conditions = {"artefact": "thermal_val7_decea.json"}
    return out


_VAL8_KEYS = {("heater_resistor", None): "VAL-8a", ("ring_mode_weighted", None): "VAL-8b",
              (None, "none"): "VAL-8c", (None, "full"): "VAL-8d"}


def thermal_step_published_stack(plant=None, target=None, **kw) -> ReplicaResult:
    """VAL-8a-e: the thermal FEM pipeline on the imec stack with our materials (mrm.extraction.thermal_fem
    --stage val8), read from ``thermal_val8_imec.json`` (its ``comparison[target id].predicted``)."""
    a = _args(target, kw)
    tid = getattr(target, "id", None)
    unit = _unit(target, "us")
    d = _artefact("thermal_val8_imec.json")
    if d is None:
        return _blocked("thermal_step_published_stack", target, unit,
                        "thermal_val8_imec.json (python -m mrm.extraction.thermal_fem --stage val8) is not "
                        "extracted yet", paper=a.get("paper"))
    if tid is None:
        out_ = a.get("output")
        if out_ == "foster_poles":
            tid = "VAL-8e"
        elif out_ == "dc_efficiency_pm_per_mw":
            tid = _VAL8_KEYS[(None, a.get("undercut", "none"))]
        else:
            tid = _VAL8_KEYS[(a.get("observer", "ring_mode_weighted"), None)]
    comp = (d.get("comparison") or {}).get(tid)
    if comp is None:
        return _blocked("thermal_step_published_stack", target, unit, f"thermal_val8_imec.json has no {tid} entry")
    pred = comp["predicted"]
    if tid == "VAL-8e" and isinstance(pred, Mapping) and "tau_us" in pred:
        taus = list(pred["tau_us"])
        shares = next((pred[k] for k in ("share_frac", "share", "shares") if pred.get(k) is not None), None)
        if shares is None or len(taus) != 2 or len(shares) != 2:
            return _blocked("thermal_step_published_stack", target, unit,
                            f"thermal_val8_imec.json VAL-8e predicted has no two-pole tau_us/share pair: {pred!r}")
        pred = {"tau1_us": float(taus[0]), "tau2_us": float(taus[1]), "share1_frac": float(shares[0]),
                "share2_frac": float(shares[1])}
    out = ReplicaResult("thermal_step_published_stack", pred, unit, target_id=tid,
                        detail={"artefact_comparison": comp})
    out.conditions = {"artefact": "thermal_val8_imec.json", "materials": d.get("nominal", {}).get("materials")}
    return out


# =====================================================================================================================
# Replica: aggressor feasibility (BND-3a/b)
# =====================================================================================================================
def aggressor_feasibility(plant, target=None, *, run_fn=None, **kw) -> ReplicaResult:
    """BND-3a/b: the best of the listed causal controllers on the aggressor (A1 for BND-3a, A2 for BND-3b), at the
    stated update rate and latency (passed to the run function in ``options``), in periodic steady state (last
    period of ``n_periods``): the signed monitor error band in % of the set-point (``signed_error_pct_min_max``) or
    the monitor deviation band in uA (``monitor_deviation_ua_min_max``) of the victim ring, every ring of the bank
    locked (half maximum) from the quasi-static warm start. "Best" = the smallest miss of the target's band
    (targets_io.evaluate), then the narrowest band. It needs
    the listed controllers (mrm.control.registry) and the engine (``run_fn``, default :func:`default_run_fn`); until
    both exist it is reported ``blocked``."""
    a = _args(target, kw)
    unit = _unit(target, "pct")
    from mrm.control import registry
    names = list(a.get("controllers", []))
    have = set(registry.available())
    missing = [c for c in names if c not in have]
    if missing:
        return _blocked("aggressor_feasibility", target, unit,
                        f"controllers not registered yet: {missing} (mrm/control/baselines)",
                        controllers=names, aggressor=a.get("aggressor"))
    try:
        fn = run_fn or default_run_fn()
    except ReplicaBlocked as e:
        return _blocked("aggressor_feasibility", target, unit, str(e), controllers=names)
    cfg, pp = resolve_plant(plant)
    which = "a1" if a.get("aggressor") in ("CAL-13a", "a1") else "a2"
    bias = float(a.get("bias_v", _scenario_bias(cfg)))
    victim = int(a.get("victim_ring", 0))
    from mrm.scenarios import library
    site = _aggressor_site(pp)
    readings = a.get("interpretation", [None]) if which == "a2" else [None]
    readings = [readings] if isinstance(readings, str) or readings is None else list(readings)

    def _num(x):
        return None if x is None else float(x)          # the target file writes 1.0e6 as a string
    limits = {"max_update_rate_hz": _num(a.get("max_update_rate_hz")),
              "min_total_latency_s": _num(a.get("min_total_latency_s"))}
    best, rows = None, []
    for ctrl in names:
        for rd in readings:
            rd_ = {"delta_t_per_period": "symmetric_triangle"}.get(rd, rd)
            sc = (library.a1_slide10_aggressor(site=site, victim_ring=victim) if which == "a1" else
                  library.a2_808_aggressor(site=site, victim_ring=victim, reading=rd_ or "symmetric_triangle"))
            per = sc.aggressors[0].target.period_s
            t_base = float(sc.defaults.get("thermal.t_base_k", 298.15))
            # the run function must respect the stated controller limits (rate, total latency)
            tr = _dynamic_lock(cfg, pp, "half_max", bias=bias, t_base=t_base, run_fn=fn, scenario=sc,
                               duration_s=float(a.get("n_periods", 3)) * per, controller=ctrl, options=limits)
            m = tr["t_s"] >= tr["t_s"][-1] - per
            i = tr["monitor_a"][m][:, 0, victim]
            if "controller_setpoint_a" in tr:       # the controller's own set-point (last value), if it reports one
                csp = np.asarray(tr["controller_setpoint_a"], dtype=float)
                sp = float(csp[-1, 0, victim] if csp.ndim == 3 else csp[0, victim])
            else:
                sp = float(tr["setpoint_a"][0, victim])
            band = ([float((i.min() - sp) / sp * 100), float((i.max() - sp) / sp * 100)]
                    if a.get("output") == "signed_error_pct_min_max" else
                    [float((i.min() - sp) * 1e6), float((i.max() - sp) * 1e6)])
            # "some causal controller holds it" (07 4.5): rank by the miss of the target's band (evaluate, with its
            # slack and either_sign rule), then by the band width
            miss = 0.0
            if target is not None and getattr(target, "tolerance", None):
                try:
                    miss = float(tio.evaluate(target, band).max_abs_residual or 0.0)
                except Exception:                                   # noqa: BLE001 - malformed band: rank last
                    miss = math.inf
            rows.append({"controller": ctrl, "reading": rd, "band": band, "setpoint_ua": sp * 1e6, "miss": miss})
            if best is None or (miss, band[1] - band[0]) < (best["miss"], best["band"][1] - best["band"][0]):
                best = rows[-1]
    out = ReplicaResult("aggressor_feasibility", best["band"], unit, target_id=getattr(target, "id", None),
                        detail={"rows": rows})
    out.conditions = {**_pp_meta(cfg, pp), "best": best, **limits, "victim_ring": victim,
                      "setpoint": "half maximum of the drop on the hot side (the model's CAL-8 lock; the target's "
                                  f"setpoint_ua = {a.get('setpoint_ua')!r} names that protocol)",
                      "window": "last aggressor period of the run"}
    return out


# =====================================================================================================================
# Registry, target runner, predictions file
# =====================================================================================================================
REPLICAS: dict[str, Callable[..., ReplicaResult]] = {
    "notch_er_vs_heater_v": notch_er_vs_heater_v, "loaded_q_from_3db_width": loaded_q_from_3db_width,
    "tuning_range_at_v": tuning_range_at_v, "relOMA_map": relOMA_map,
    "monitor_current_at_lock": monitor_current_at_lock, "stabilizer_code_vs_T": stabilizer_code_vs_T,
    "apparent_shift_fixed_code": apparent_shift_fixed_code, "bistability_sweep": bistability_sweep,
    "heater_power_per_fsr": heater_power_per_fsr, "resonance_vs_heater_v": resonance_vs_heater_v,
    "resonance_shift_vs_bias": resonance_shift_vs_bias, "cold_resonance_offset": cold_resonance_offset,
    "eo_bandwidth_at_bias": eo_bandwidth_at_bias, "pd_responsivity": pd_responsivity,
    "laser_line_statistics": laser_line_statistics, "aggressor_trajectory": aggressor_trajectory,
    "mode_solve_published_geometry": mode_solve_published_geometry,
    "self_heating_time_constant": self_heating_time_constant,
    "thermal_step_published_stack": thermal_step_published_stack, "adjacent_order_penalty": adjacent_order_penalty,
    "order_hop_margin": order_hop_margin, "aggressor_feasibility": aggressor_feasibility,
}
assert set(REPLICAS) == set(tio.REPLICA_NAMES), set(tio.REPLICA_NAMES) ^ set(REPLICAS)


def preset_for(target) -> str:
    """The preset a target is replicated on: G25 bench numbers on ``g25_test_vehicle``, the rest on ``g26``."""
    return PRESET_FOR_GENERATION.get(str(getattr(target, "generation", "G26")), "g26")


def run_target(target, plant=None, **kw) -> ReplicaResult:
    """Run the replica a target names (on ``plant`` or :func:`preset_for`); a missing input gives a ``blocked``
    result, any other exception an ``error`` result with the message (never a silent pass)."""
    fn = REPLICAS[target.replica]
    plant = preset_for(target) if plant is None else plant
    try:
        res = fn(plant, target, **kw)
    except ReplicaBlocked as e:
        res = _blocked(target.replica, target, str(target.unit), str(e))
    except Exception as e:                                              # noqa: BLE001
        res = ReplicaResult(target.replica, None, str(target.unit), status="error", target_id=target.id,
                            notes=[f"{e.__class__.__name__}: {e}"])
    res.target_id = target.id
    if res.conditions is not None and "preset" not in res.conditions and isinstance(plant, str):
        res.conditions["preset"] = plant
    return res


def _nonfinite(x: Any) -> bool:
    """True when a prediction has a NaN or inf leaf (the JSON file writes NaN as null, inf as "inf")."""
    if isinstance(x, Mapping):
        return any(_nonfinite(v) for v in x.values())
    try:
        return not bool(np.all(np.isfinite(np.asarray(x, dtype=float))))
    except (TypeError, ValueError):
        return False


def _evaluation(target, res: ReplicaResult) -> dict | None:
    if res.status != "ok" or res.value is None:
        return None
    try:
        ev = tio.evaluate(target, res.value)
    except Exception as e:                                              # noqa: BLE001
        return {"error": f"{e.__class__.__name__}: {e}"}
    return dataclasses.asdict(ev)


def predict_all(roles: Sequence[str] = ("calibration", "validation", "bounds"), *, ids: Sequence[str] | None = None,
                plant_by_id: Mapping[str, Any] | None = None, verbose: bool = False) -> list[dict]:
    """Run every target's replica and return one record per target: id, role, status, scored, replica, preset, the
    target's value / unit / sigma / tolerance, the prediction (``predicted``; NaN is written as null, so
    ``predicted_nonfinite`` marks an ``ok`` replica whose value is NaN or inf, which evaluate() scores as a fail), its
    variants, conditions, notes, the replica status and the :func:`mrm.validation.targets_io.evaluate` result (for
    information; the report scores)."""
    recs = []
    for role in roles:
        for t in tio.load_targets(role):
            if ids is not None and t.id not in ids:
                continue
            t0 = time.time()
            plant = (plant_by_id or {}).get(t.id)
            res = run_target(t, plant)
            rec = {"id": t.id, "role": role, "target_status": t.status, "scored": t.scored, "replica": t.replica,
                   "preset": plant if isinstance(plant, str) else (res.conditions or {}).get("preset",
                                                                                          preset_for(t)),
                   "quantity": t.quantity, "unit": t.unit, "value": t.value, "sigma": t.sigma,
                   "tolerance": dict(t.tolerance), "replica_status": res.status, "predicted": res.value,
                   "predicted_nonfinite": res.status == "ok" and _nonfinite(res.value),
                   "variants": res.variants, "conditions": res.conditions, "notes": res.notes,
                   "detail": res.detail, "evaluation": _evaluation(t, res),
                   "runtime_s": round(time.time() - t0, 2)}
            recs.append(_jsonable(rec))
            if verbose:
                ev = rec["evaluation"] or {}
                print(f"{t.id:8s} {res.status:8s} {ev.get('level', '-'):10s} {rec['runtime_s']:7.1f}s", flush=True)
    return recs


def write_predictions(path: str | Path | None = None, records: list[dict] | None = None, **kw) -> Path:
    """Write ``latest_predictions.json`` (schema ``mrm.validation.predictions/1``) for the report generator."""
    import platform
    path = Path(path) if path is not None else PREDICTIONS_PATH
    recs = predict_all(**kw) if records is None else records
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tio.REPO_DIR, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:                                                   # noqa: BLE001
        commit = None
    doc = {"schema": "mrm.validation.predictions/1",
           "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "git_commit": commit, "python": platform.python_version(),
           "frozen_sha256": {r: tio.file_sha256(r) for r in tio.FROZEN_ROLES},
           "note": "predictions of the replica protocols on the uncalibrated presets (no parameter was tuned to a "
                   "target); 'evaluation' is targets_io.evaluate for information, the report scores",
           "counts": {s: sum(1 for r in recs if r["replica_status"] == s) for s in ("ok", "blocked", "error")},
           "records": recs}
    path.write_text(json.dumps(_jsonable(doc), indent=1, allow_nan=False) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mrm.validation.replicas [--ids CAL-1 VAL-1 ...] [--out PATH] [--list]``."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mrm.validation.replicas", description=main.__doc__)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true", help="list the replicas and exit")
    ns = ap.parse_args(argv)
    if ns.list:
        for k in sorted(REPLICAS):
            print(k)
        return 0
    p = write_predictions(ns.out, ids=ns.ids, verbose=True)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
