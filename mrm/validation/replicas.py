# replicate published measurement protocols on the model plant.
# read this first - start with run_target for one target or predict_all for all targets.
# write_predictions saves records for the report generator.
# REPLICAS maps the names in the target files to their implementations.
# the protocols use the published sweep, port, normalization and readout.
# they do not tune plant parameters.
# what does a replica accept?
# plant is a preset name, Config or realized PlantParams.
# hypothesis sweeps need a preset or Config so they can reload alternatives.
# target.replica_args supplies defaults.
# keyword arguments override them, including an explicit None.
# ReplicaResult.value uses the target's unit and tolerance shape.
# its conditions record assumptions, and its variants record alternative hypotheses.
# how are static measurements solved?
# StaticRing uses the exact ring optics, DC thermal gains, heater TCR, junction data heat and monitor PD.
# its heat_balance method defines the steady-state equations.
# slow sweeps use _relax to follow a stable branch until a fold.
# bank protocols use _bank_codes to include crosstalk and line filtering along the bus.
# other static protocols use one ring with the other rings dark and heaters off.
# when is a dynamic run needed?
# monitor_current_at_lock and stabilizer_code_vs_T assume a settled loop by default.
# this assumes the loop settles much faster than the disturbance.
# their dynamic method and aggressor_feasibility use LockRunRequest through default_run_fn or a caller-supplied runner.
# a missing runner, controller or extraction artefact can give a blocked result.
# conventions - use SI internally and vacuum wavelengths.
# positive bias means reverse bias in V.
# a null
# p_in_dbm means zero optical power and no optical self-heating.
# p_in_dbm="preset" means the preset line power
# at the ring.
# replicas at operating power use that default when their protocol does not state a power.
# predict_all uses g25_test_vehicle for G25 targets and g26 for the rest.
# write_predictions stores values, tolerances and evaluations in latest_predictions.json.
# the target files and their README are in mrm/validation/targets.
# run this module with python -m mrm.validation.replicas to write predictions.
from __future__ import annotations

import dataclasses
import datetime
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

from mrm.plant import electrical
from mrm.plant import optics
from mrm.plant import phase
from mrm.plant import thermal
from mrm.units import C0, T0_K
from mrm.validation import targets_io

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

# Lightmatter 2026 prints "~15 GHz/degC" at 1310 nm. This gives 85.86 pm/K for CAL-6 thermal sensitivity.
# export the slope in m/K for callers. Aggressor scenarios use the same conversion from resonance shift to degC
# in mrm.scenarios.calibrate.LM_DLAMBDA_DT_M_PER_K.
LM_DLAMBDA_DT_M_PER_K = 15e9 * (1310e-9) ** 2 / C0
# the predictions file that write_predictions writes and the report generator reads
PREDICTIONS_PATH = Path(__file__).resolve().parent / "latest_predictions.json"
# preset each target generation is replicated on (G25 bench device vs the G26 bank)
PRESET_FOR_GENERATION = {"G25": "g25_test_vehicle", "G26": "g26", "Guide": "g26", "EVK100": "g26", "L20": "g26",
                         "M1000": "g26", "OCP": "g26", "OCI": "g26", "GF": "g26", "PAT": "g26", "none": "g26"}
# config key of the heater DAC full scale for each driver law (V, A, W - dac_pwm uses the voltage full scale)
_FULL_SCALE_KEY = {"voltage": "ring.heater.full_scale_v", "current": "ring.heater.full_scale_a",
                   "power": "ring.heater.full_scale_w", "dac_pwm": "ring.heater.full_scale_v"}


# results
# a replica lacks an artefact, controller or runner.
# the message identifies the missing input.
class ReplicaBlocked(RuntimeError):
    pass


# prediction and diagnostics for one target.
# value uses unit and the shape that targets_io.evaluate expects.
# a non-ok result has value None.
# status is ok, blocked for a missing input, or error for an exception caught by run_target.
# conditions records JSON-compatible assumptions.
# variants maps hypothesis labels to predictions with the same shape as value.
# detail holds intermediate results.
# notes holds free-text caveats.
@dataclasses.dataclass
class ReplicaResult:

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

    # JSON-ready dict (numpy arrays and scalars converted).
    def to_json(self) -> dict:
        return _jsonable(dataclasses.asdict(self))


# convert numpy values to Python values for JSON.
# NaN becomes None.
# positive and negative infinity become the strings "inf" and "-inf".
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


# a `blocked` result of replica `name` with the reason as its note.
def _blocked(name: str, target, unit: str, reason: str, **conditions) -> ReplicaResult:
    return ReplicaResult(name, None, unit, status="blocked", target_id=getattr(target, "id", None),
                         conditions=dict(conditions), notes=[reason])


# merge target.replica_args with the caller's keyword arguments.
# caller values win.
# keep an explicit None, such as p_in_dbm=None for zero optical power.
def _replica_args(target, kw: Mapping[str, Any]) -> dict:
    a = dict(getattr(target, "replica_args", None) or {})
    a.update(kw)
    return a


# the target's unit, or `default` without a target.
def _target_unit(target, default: str) -> str:
    return str(getattr(target, "unit", None) or default)


# Plant resolution
# True for a realized PlantParams (duck-typed).
def _is_plant_params(x) -> bool:
    return hasattr(x, "optics") and hasattr(x, "thermal") and hasattr(x, "batch")


# the file a Config was loaded from (the last one of its extends chain), for re-loading it with alts.
def _config_file(cfg) -> str | Path:
    f = Path(cfg.files[-1])
    if not f.is_absolute():
        f = targets_io.REPO_DIR / f
    return f


# load a Config from a preset, path or Config, with extra alternatives and overrides.
# reload an existing Config with its original alternatives and overrides retained.
# suppress artefact-fallback warnings here because cfg.warnings already records them.
def load_config(plant, *, alts: Mapping[str, str] | None = None, overrides: Mapping[str, Any] | None = None):
    from mrm import params
    if _is_plant_params(plant):
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
        return params.load(_config_file(plant), overrides=o, alts=a)


# return (cfg, pp) from a preset, path or Config.
# a realized PlantParams returns (None, plant).
# defaults give one world, nominal uncertainty, no device variation and seed 0.
# alts pins hypotheses, for example {"uncertainty.heater.driver_mode": "current"}.
def resolve_plant(plant, *, batch: int = 1, uncertainty: str = "nominal", variation: str = "off", seeds=0,
                  alts: Mapping[str, str] | None = None, overrides: Mapping[str, Any] | None = None):
    from mrm import params
    if _is_plant_params(plant):
        if alts or overrides:
            raise TypeError("hypothesis alternatives need a preset name or a Config, not a PlantParams")
        return None, plant
    cfg = load_config(plant, alts=alts, overrides=overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, int(batch), seeds=seeds, uncertainty=uncertainty, variation=variation)
    return cfg, pp


# Choice names of an `uncertainty` hypothesis entry (empty when absent).
def _hypothesis_choices(cfg, entry: str) -> tuple[str, ...]:
    if cfg is None:
        return ()
    e = cfg.uncertainty.get(entry)
    if e is None:
        return ()
    fields = getattr(e, "fields", None) or {}
    ch = fields.get("choices") if isinstance(fields, Mapping) else None
    return tuple(ch) if ch else ()


# the plant part of `ReplicaResult.conditions` - preset, alternatives, overrides, batch, config hash and the number of artefact fallbacks.
def _plant_meta(cfg, pp) -> dict:
    return {"preset": getattr(cfg, "name", None) if cfg is not None else None,
            "alts": dict(getattr(cfg, "alts_applied", {}) or {}),
            "overrides": {k: _jsonable(v) for k, v in (getattr(cfg, "overrides_applied", {}) or {}).items()},
            "batch": int(pp.batch), "config_hash": getattr(pp, "config_hash", None),
            "artefact_fallbacks": sum(1 for w in (getattr(cfg, "warnings", []) or []) if "fallback" in w)}


# the static ring
# steady state of one locked ring in one world.
# mode names the rule in StaticRing.lock.
# order is the resonance order relative to the design order.
# t_ring_k is the lock temperature in K.
# t_res_k is the ring temperature for exact resonance on the line.
# p_abs_w is absorbed heat in W.
# with lines='all', it includes absorption from the other lines.
# p_drop_w is total power at the drop monitor in W.
# p_drop_other_w is its share from the other lines.
# p_thru_w is the own-line through power in W.
# reloma is |T1 - T0|, or OMA / P_in, at the through port.
# drop_peak_w is the peak drop power on the lock grid in W.
# p_j_w and p_d_w are junction and driver heat from data in W.
# p_heater_w is heater power in W.
# t_heater_k is the heater-resistor temperature in K.
# u is normalized drive, or V for bench_voltage.
# stable is true where required drive increases with ring temperature.
# this tests fixed-drive stability.
# detuning_m is the line wavelength minus the resonance wavelength in m.
# t_pd_k is the PD observer temperature in K from this ring's own heat sources.
# it sets the dark current.
# p_bus_w is background input power at the ring in W.
# it includes all lines with lines='all'.
@dataclasses.dataclass
class LockPoint:
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


# solve the steady state of ring n in world b of a realized PlantParams pp.
# topology selects add_drop or all_pass.
# all_pass removes the drop coupler by setting K^2_drop = 0.
# CAL-1 compares both topologies for loaded Q.
# CAL-2 and VAL-2a/b compare notch depth for the topologies listed by their targets.
# driver_law overrides the heater law.
# bench_voltage means u is voltage in V.
# z stores DC thermal gains in K/W for this ring's own heat sources.
# each key is source then observer.
# sources are h for heater, o for waveguide absorption, j for junction and d for driver.
# observers are r for mode-weighted ring temperature, h for heater resistor and p for monitor PD.
# for example, z['hr'] gives ring rise per W of heater power.
# the other rings are dark with heaters off unless the caller adds their heat.
# see heat_balance for the temperature-based solve.
class StaticRing:

    def __init__(self, pp, *, b: int = 0, n: int = 0, topology: str = "add_drop", driver_law: str | None = None,
                 _cache: dict | None = None):
        if topology not in ("add_drop", "all_pass"):
            raise ValueError(f"topology must be add_drop or all_pass, got {topology!r}")
        cache = _cache if _cache is not None else _plant_cache(pp)
        self.pp, self.b, self.n, self.topology = pp, int(b), int(n), topology
        # phase-model row of this ring (phase.RP_* columns) and the junction table of this world
        phase_params = cache["phase"]
        rp = np.array(phase_params.ring[self.b, self.n], dtype=float)
        if topology == "all_pass":
            rp[phase.RP_K2D] = 0.0
        self.rp = rp
        table_row = phase_params.table_row(self.b)
        self.grid = phase_params.grid
        # (dneff, alpha_fca, capacitance, r_series) columns of the junction table, in the order phase.*_row expect
        self.junction_tables = (np.ascontiguousarray(phase_params.dneff[table_row]),
                                np.ascontiguousarray(phase_params.alpha_fca[table_row]),
                                np.ascontiguousarray(phase_params.capacitance[table_row]),
                                np.ascontiguousarray(phase_params.r_series[table_row]))
        rb = np.asarray(pp.optics.backscatter_rb[self.b, self.n])
        self.rb = rb if (rb.size and np.any(rb != 0)) else None      # backscatter of optics.ring_exact (None - off)
        # DC thermal gains of this ring's own sources - see the StaticRing source-observer keys
        sysm = pp.thermal.system
        G = cache["dc"][self.b]                                    # (O, S) K/W
        n = self.n

        def gain(obs, src):
            return float(G[obs, src]) if (obs >= 0 and src >= 0) else 0.0
        ring_obs, heater_obs = int(sysm.ring_obs[n]), int(sysm.heater_obs[n])
        pd_obs = int(sysm.pd_obs[n]) if sysm.pd_obs.size else -1
        heater_src, wg_src = int(sysm.heater_src[n]), int(sysm.wg_src[n])
        junction_src = int(sysm.junction_src[n]) if sysm.junction_src.size else -1
        driver_src = int(sysm.driver_src[n]) if sysm.driver_src.size else -1
        self.z = {"hr": gain(ring_obs, heater_src), "hh": gain(heater_obs, heater_src),
                  "or": gain(ring_obs, wg_src), "oh": gain(heater_obs, wg_src),
                  "jr": gain(ring_obs, junction_src), "jh": gain(heater_obs, junction_src),
                  "dr": gain(ring_obs, driver_src), "dh": gain(heater_obs, driver_src),
                  "hp": gain(pd_obs, heater_src), "op": gain(pd_obs, wg_src),
                  "jp": gain(pd_obs, junction_src), "dp": gain(pd_obs, driver_src)}
        if not self.z["hr"] > 0:
            raise ValueError("the thermal system has no heater -> ring path for this ring")
        # heater driver (realized DAC path of this ring) - law -1 is the bench voltage source (u = V)
        heater_hw = cache["heater"]
        law = int(np.asarray(heater_hw.law).reshape(-1)[self.b if heater_hw.law.size > 1 else 0])
        if driver_law is not None:
            law = -1 if driver_law == "bench_voltage" else electrical.DRIVER_LAWS[driver_law]
        self.law = law
        b_idx = self.b if heater_hw.r_cold_ohm.shape[0] > 1 else 0
        n_idx = self.n if heater_hw.r_cold_ohm.shape[1] > 1 else 0

        def heater_value(a):
            a = np.asarray(a, dtype=float)
            return float(a[b_idx if a.shape[0] > 1 else 0, n_idx if a.shape[1] > 1 else 0])
        self.r_cold_ohm = heater_value(heater_hw.r_cold_ohm)
        self.tcr_per_k = heater_value(heater_hw.tcr_per_k)
        self.tcr_ref_k = heater_value(heater_hw.tcr_ref_k)
        self.v_fs, self.i_fs, self.p_fs = (heater_value(heater_hw.v_full_scale_v),
                                           heater_value(heater_hw.i_full_scale_a),
                                           heater_value(heater_hw.p_full_scale_w))
        # driver output compliance (V) - electrical.heater_power_row saturates as a voltage source there - a bench voltage source has none
        self.v_comp = math.inf if law == -1 else heater_value(heater_hw.compliance_v)
        self.bits, self.pwm_bits = int(heater_hw.bits), int(heater_hw.pwm_bits)
        bit_err = np.asarray(heater_hw.bit_err_lsb)
        self.bit_err_lsb = np.ascontiguousarray(
            bit_err[b_idx if bit_err.shape[0] > 1 else 0, n_idx if bit_err.shape[1] > 1 else 0][None, :])
        bow = np.asarray(heater_hw.inl_bow_lsb, dtype=float)
        self.inl_bow_lsb = np.array([bow[b_idx if bow.shape[0] > 1 else 0, n_idx if bow.shape[1] > 1 else 0]])
        self._dac_table = None
        # optics constants and the laser line of this ring
        opt = pp.optics
        self.lam_res = float(rp[phase.RP_LAM_RES])          # cold resonance (m)
        self.t_ref = float(rp[phase.RP_T_REF])              # its reference ring temperature (K)
        # thermo-optic slope - s(T) = s0 (1 + beta (T - T0)), s0 in m/K, beta in 1/K, T0 in K
        self.s0, self.beta, self.t0 = float(rp[phase.RP_S0]), float(rp[phase.RP_BETA]), float(rp[phase.RP_T0])
        self.q_loaded = float(np.asarray(opt.q_loaded)[self.b, self.n])
        self.fsr_hz = float(np.asarray(opt.fsr_hz)[self.b, self.n])
        self.line = int(np.asarray(pp.layout.ring_to_line)[self.n])
        self.lam_line = float(C0 / np.asarray(pp.laser.line_freq_hz)[self.b, self.line])
        input_transmission = float(np.asarray(pp.laser.input_transmission).reshape(-1)[
            self.b if pp.batch > 1 and np.size(pp.laser.input_transmission) > 1 else 0])
        self.p_line_w = float(np.asarray(pp.laser.power_w)[self.b, self.line]) * input_transmission
        # every TX line on the bus at its chip-input power.
        # in bank protocols the other lines leak into the drop monitor and heat the ring (the drop monitor sums every line and every order).
        # these are chip-input powers - the bank protocols (_bank_codes, bus_line_powers) apply the filtering by the rings upstream.
        tx_lines = [i for i, d_ in enumerate(pp.laser.direction) if d_ == "tx"] if len(pp.laser.direction) else \
            list(range(int(pp.n_lines)))
        self.other_lines = np.array([i for i in tx_lines if i != self.line], dtype=np.int64)
        self.lam_lines = C0 / np.asarray(pp.laser.line_freq_hz, dtype=float)[self.b]
        self.p_lines_w = np.asarray(pp.laser.power_w, dtype=float)[self.b] * input_transmission
        # data path - NRZ swing and level convention, and the junction and driver heat of the data
        datapath = pp.datapath
        self.swing_pp_v = float(np.asarray(datapath.swing_pp_v)[self.b, self.n])
        self.level_mode = int(np.asarray(datapath.level_convention).reshape(-1)[
            self.b if np.size(datapath.level_convention) > 1 else 0])
        self.bitrate_hz = float(np.asarray(datapath.bitrate_hz).reshape(-1)[
            self.b if np.size(datapath.bitrate_hz) > 1 else 0])
        td = np.asarray(datapath.transition_density).reshape(-1)
        td = float(td[self.b if td.size > 1 else 0])
        self.transition_density = td if math.isfinite(td) else 0.5
        # (driver output resistance ohm, driver static power W, driver energy per transition J)
        self._driver_heat_params = tuple(float(np.asarray(x).reshape(-1)[self.b if np.size(x) > 1 else 0])
                                         for x in (datapath.driver_output_resistance_ohm,
                                                   datapath.driver_static_power_w,
                                                   datapath.driver_energy_per_transition_j))
        self._monitor_hw = cache["monitor"]
        self.cache = cache

    # optics
    # evaluate ring optics at wavelengths lam (L,) in m and ring temperatures t_ring (M,) in K.
    # t_heater gives heater-resistor temperatures (M,) in K for the loss model.
    # None uses t_ring.
    # v_lo and v_hi are scalar or (M,) reverse biases in V for NRZ levels 0 and 1.
    # v_hi=None means CW at v_lo.
    # the result holds through, drop, intrinsic-loss and absorbed-heat fractions for both levels, each (M, L).
    # absorbed heat is eta times intrinsic loss.
    # shift is (M, 2) in m, cj is (M, 2) in F and rs is (M, 2) in ohm.
    # k2t and k2d are power couplings, a0 is level-0 round-trip amplitude and trt is round-trip time in s.
    # each is (M, L).
    # flags has shape (M,).
    def optics(self, lam, t_ring, t_heater=None, v_lo=0.0, v_hi=None) -> dict:
        lam = np.ascontiguousarray(np.atleast_1d(np.asarray(lam, dtype=float)))
        t_ring_k = np.ascontiguousarray(np.atleast_1d(np.asarray(t_ring, dtype=float)))
        M, L = t_ring_k.size, lam.size
        # use writable contiguous arrays because the compiled row functions require them
        t_heater_k = t_ring_k if t_heater is None else _writeable_full(t_heater, (M,))
        v0 = _writeable_full(v_lo, (M,))
        cw = v_hi is None
        v1 = v0 if cw else _writeable_full(v_hi, (M,))
        rp = _writeable_full(self.rp, (M, phase.N_RP))
        # phase.ring_optics_row returns phase, amplitude and heat fraction eta for each level.
        # it also returns couplings, round-trip time, junction shift, capacitance, series resistance and flags.
        out = {k: np.zeros((M, L)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
        out.update(shift=np.zeros((M, 2)), cj=np.zeros((M, 2)), rs=np.zeros((M, 2)), flags=np.zeros(M, np.int8))
        dn_tab, al_tab, c_tab, rs_tab = self.junction_tables
        phase.ring_optics_row(rp, lam, t_ring_k, t_heater_k, v0, v1, self.grid, dn_tab, al_tab, c_tab, rs_tab,
                              out["phi0"], out["phi1"], out["a0"], out["a1"], out["e0"], out["e1"], out["k2t"],
                              out["k2d"], out["trt"], out["shift"], out["cj"], out["rs"], out["flags"])
        exact0 = optics.ring_exact(out["phi0"], out["k2t"], out["k2d"], out["a0"], out["trt"], rb=self.rb)
        res = {"thru0": np.abs(exact0.thru) ** 2, "drop0": np.abs(exact0.drop) ** 2, "loss0": exact0.p_loss}
        res["abs0"] = out["e0"] * res["loss0"]
        if cw:
            for k in ("thru", "drop", "loss", "abs"):
                res[k + "1"] = res[k + "0"]
        else:
            exact1 = optics.ring_exact(out["phi1"], out["k2t"], out["k2d"], out["a1"], out["trt"], rb=self.rb)
            res.update(thru1=np.abs(exact1.thru) ** 2, drop1=np.abs(exact1.drop) ** 2, loss1=exact1.p_loss)
            res["abs1"] = out["e1"] * res["loss1"]
        res.update(shift=out["shift"], cj=out["cj"], rs=out["rs"], k2t=out["k2t"], k2d=out["k2d"], a0=out["a0"],
                   trt=out["trt"], flags=out["flags"])
        return res

    # resonance shift (m) of the junction alone at reverse bias v (V), relative to the reference bias.
    def junction_shift_m(self, v) -> np.ndarray:
        v = np.ascontiguousarray(np.atleast_1d(np.asarray(v, dtype=float)))
        M = v.size
        rp = _writeable_full(self.rp, (M, phase.N_RP))
        shift_m, fca_loss, c_j, r_s = (np.zeros(M) for _ in range(4))
        flags = np.zeros(M, np.int8)
        dn_tab, al_tab, c_tab, rs_tab = self.junction_tables
        phase.resonance_shift_row(rp, np.full(M, self.t_ref), v, self.grid, dn_tab, al_tab, c_tab, rs_tab, shift_m,
                                  fca_loss, c_j, r_s, flags)
        return shift_m

    # integrate the thermo-optic slope from T_ref to ring temperature t_k (K).
    # return shift in m.
    # s(T) = s0 (1 + beta (T - T0)).
    # its integral is shift = s0 [(T - T_ref) + beta/2 ((T - T0)^2 - (T_ref - T0)^2)].
    # s0 is in m/K, beta in 1/K, and T_ref and T0 in K.
    def thermal_shift_m(self, t_k) -> np.ndarray:
        t = np.asarray(t_k, dtype=float)
        return self.s0 * ((t - self.t_ref) + 0.5 * self.beta * ((t - self.t0) ** 2 - (self.t_ref - self.t0) ** 2))

    # dlambda/dT (m/K) at ring temperature t_k (K) - s0 (1 + beta (T - T0)).
    def s_at(self, t_k) -> np.ndarray:
        return self.s0 * (1.0 + self.beta * (np.asarray(t_k, dtype=float) - self.t0))

    # invert thermal_shift_m.
    # return the ring temperature in K for a shift of shift_m in m.
    def t_for_shift(self, shift_m) -> np.ndarray:
        from mrm.scenarios.calibrate import resonance_to_rise_k
        d = np.asarray(shift_m, dtype=float)
        return self.t_ref + resonance_to_rise_k(d, s0_m_per_k=self.s0, beta_per_k=self.beta, t_ref_k=self.t0,
                                                t_base_k=self.t_ref)

    # round-trip phase (rad) of the cold phase curve (T_ref, reference bias) at wavelengths x (L,) in m.
    def _phase_at(self, x) -> np.ndarray:
        x = np.ascontiguousarray(np.atleast_1d(np.asarray(x, dtype=float)))
        rp = np.ascontiguousarray(self.rp[None, :])
        level_outs = [np.zeros((1, x.size)) for _ in range(9)]       # phi0 first (see optics())
        pair_outs = [np.zeros((1, 2)) for _ in range(3)]             # shift, cj, rs
        v_ref = np.array([self.rp[phase.RP_V_REF]])
        phase.ring_optics_row(rp, x, np.array([self.t_ref]), np.array([self.t_ref]), v_ref, v_ref, self.grid,
                              *self.junction_tables, *level_outs, *pair_outs, np.zeros(1, np.int8))
        return level_outs[0][0]

    # Cold position of order m + k minus the cold resonance lambda_res (m), with m the design order.
    # k > 0 are bluer orders.
    # the positions come from the phase model's cold phase curve - the whole comb then translates rigidly with temperature and bias.
    def order_offset_m(self, k: int = 0) -> float:
        k = int(k)
        if k == 0:
            return 0.0
        cache = self.__dict__.setdefault("_orders", {})
        if k not in cache:
            m = round(float(self._phase_at(self.lam_res)[0]) / (2 * math.pi))      # design order
            fsr = _fsr_m(self)
            guess = self.lam_res - k * fsr

            def phase_error(x):
                return float(self._phase_at(x)[0]) - 2 * math.pi * (m + k)
            cache[k] = optimize.brentq(phase_error, guess - 0.4 * fsr, guess + 0.4 * fsr, xtol=1e-16) - self.lam_res
        return cache[k]

    # Ring temperature (K) at which order m + k (m - the order of the cold resonance) sits exactly at `lam` (m) with the junction at v (V).
    def t_res(self, lam, v=0.0, k: int = 0) -> float:
        return float(self.t_for_shift(lam - self.lam_res - self.order_offset_m(k)
                                      - float(self.junction_shift_m(v)[0])))

    # the order k that a heater-only acquisition locks (a ramp from code 0, as in PAT-3).
    # it is the order whose resonance reaches `lam` at the lowest ring temperature not below `t_floor` (K - the ring with the heater off - default T_ref).
    def first_order(self, lam, v=0.0, t_floor: float | None = None, k_range=range(-3, 4)) -> int:
        t_floor_k = self.t_ref if t_floor is None else float(t_floor)
        best_t, best_k = math.inf, 0
        for k in k_range:
            try:
                t = self.t_res(lam, v, k)
            except ValueError:
                continue
            if t >= t_floor_k and t < best_t:
                best_t, best_k = t, k
        return best_k

    # the loaded linewidth (FWHM) in ring-temperature units, lambda/Q_L / s(T) (K).
    def linewidth_k(self, t_k=None) -> float:
        t = self.t_ref if t_k is None else t_k
        return float(self.lam_res / self.q_loaded / self.s_at(t))

    # heater
    # Heater resistance (ohm) at heater-resistor temperature t_h (K) - R0 (1 + alpha (T_H - T_R0)).
    def _r_heater(self, t_h):
        return self.r_cold_ohm * (1.0 + self.tcr_per_k * (np.asarray(t_h, dtype=float) - self.tcr_ref_k))

    # Normalized drive u that delivers heater power p_w (W) at heater-resistor temperature t_h (K).
    # this is the inverse driver law, continued as an odd function for p < 0 so that roots stay bracketed.
    # bench_voltage - u is the voltage in V.
    def u_from_power(self, p_w, t_h) -> np.ndarray:
        p = np.asarray(p_w, dtype=float)
        r = self._r_heater(t_h)
        sign_p = np.sign(p)
        abs_p = np.abs(p)
        if self.law in (electrical.LAW_VOLTAGE, -1):
            full_scale = 1.0 if self.law == -1 else self.v_fs
            return sign_p * np.sqrt(abs_p * r) / full_scale
        if self.law == electrical.LAW_CURRENT:
            return sign_p * np.sqrt(abs_p / r) / self.i_fs
        if self.law == electrical.LAW_POWER:
            return p / self.p_fs
        return p * r / self.v_fs ** 2                                   # dac_pwm - u = <(V/V_FS)^2>

    # Heater power (W) at drive u, with the TCR solved implicitly (electrical.heater_steady_state).
    # t_pre_k - the heater-resistor temperature without this heater's own power (K).
    # g_self - the heater's own thermal gain (K/W - default Z_hh).
    # the driver's output compliance is applied as in the plant (electrical.heater_power_row).
    def power_from_u(self, u, t_pre_k, g_self=None) -> np.ndarray:
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if self.law == -1:
            law, v_full_scale = electrical.LAW_VOLTAGE, 1.0
        else:
            law, v_full_scale = self.law, self.v_fs
        res = electrical.heater_steady_state(drive_frac=u, law=law, v_full_scale_v=v_full_scale,
                                             i_full_scale_a=self.i_fs, p_full_scale_w=self.p_fs,
                                             compliance_v=self.v_comp, r_cold_ohm=self.r_cold_ohm,
                                             tcr_per_k=self.tcr_per_k,
                                             r_th_heater_k_per_w=self.z["hh"] if g_self is None else g_self,
                                             t_base_minus_ref_k=np.asarray(t_pre_k, dtype=float) - self.tcr_ref_k)
        return res["p_w"]

    # True where heater power p_w (W) at heater-resistor temperature t_h (K) needs no more than the driver's compliance voltage - V = sqrt(P R(T_H)) <= V_comp (the inverse law u_from_power ignores compliance).
    def compliance_ok(self, p_w, t_h) -> np.ndarray:
        p = np.maximum(np.asarray(p_w, dtype=float), 0.0)
        return np.sqrt(p * self._r_heater(t_h)) <= self.v_comp * (1.0 + 1e-12)

    # (codes, u, monotone envelope of u) of every DAC code through the realized DAC transfer (cached).
    def _dac_transfer(self):
        if self._dac_table is None:
            if self.law == -1:
                raise ValueError("bench_voltage has no DAC codes")
            codes = np.arange(1 << self.bits, dtype=np.int64)
            u = np.empty(codes.size)
            ripple = np.empty(codes.size)
            # pwm_ripple 0 (period average only), tick 0, dt_clk 1
            electrical.heater_target_row(codes, self.law, self.bits, self.pwm_bits, self.bit_err_lsb,
                                         self.inl_bow_lsb, 0, 0, 1, u, ripple)
            self._dac_table = (codes.astype(float), u, np.maximum.accumulate(u))
        return self._dac_table

    # Normalized drive of DAC codes through the realized DAC transfer (bit errors and INL bow included).
    def code_to_u(self, code) -> np.ndarray:
        codes, u, _ = self._dac_transfer()
        return np.interp(np.asarray(code, dtype=float), codes, u)

    # Inverse DAC transfer (fractional code, linear between codes - monotone envelope where DNL bends it).
    def u_to_code(self, u) -> np.ndarray:
        codes, _, u_monotone = self._dac_transfer()
        return np.interp(np.asarray(u, dtype=float), u_monotone, codes)

    # junction and data
    # (v_lo, v_hi) in V - the NRZ levels at DC bias bias_v (electrical.nrz_levels_row - level 1 = v_hi).
    def nrz_levels(self, bias_v: float, swing_scale: float = 1.0) -> tuple[float, float]:
        lo, hi, flags = np.zeros(1), np.zeros(1), np.zeros(1, np.int8)
        electrical.nrz_levels_row(np.array([float(bias_v)]), np.array([self.swing_pp_v * swing_scale]),
                                  self.level_mode, electrical.JUNCTION_GRID_V_MIN, electrical.JUNCTION_GRID_V_MAX,
                                  lo, hi, flags)
        return float(lo[0]), float(hi[0])

    # (P_j, P_d) in W - the junction CV^2 heat and the driver heat of random NRZ data at the levels v_lo, v_hi.
    def data_heat(self, v_lo: float, v_hi: float, *, enable: bool = True) -> tuple[float, float]:
        if not enable:
            return 0.0, 0.0
        grid = self.grid
        c_tab = self.junction_tables[2]                  # capacitance column
        c_eff_f = np.zeros(1)
        electrical.junction_swing_capacitance_row(np.array([v_lo]), np.array([v_hi]), grid, c_tab,
                                                  np.array([self.rp[phase.RP_FILL]]),
                                                  np.array([self.rp[phase.RP_LEN]]), c_eff_f)
        o = self.optics(self.lam_line, [self.t_ref], None, v_lo, v_hi)
        r_s = 0.5 * float(o["rs"][0, 0] + o["rs"][0, 1])     # series resistance, mean of the two levels (ohm)
        p_j, p_d = np.zeros(1), np.zeros(1)
        r_drv, p_static, e_transition = self._driver_heat_params
        electrical.data_activity_heat_row(c_eff_f, np.array([v_hi - v_lo]),
                                          self.bitrate_hz if math.isfinite(self.bitrate_hz) else 0.0,
                                          np.array([self.transition_density]), np.array([1.0]),
                                          np.array([r_s]), np.array([r_drv if math.isfinite(r_drv) else 0.0]),
                                          np.array([p_static]), np.array([e_transition]), 0, p_j, p_d)
        return float(p_j[0]), float(p_d[0])

    # monitor
    # Monitor photocurrent (signal + dark) in A for drop power p_drop_w (W) at the PD (sensing.MonitorHw).
    # p_bus_w - the line power at the ring (W), the monitor's background input.
    # t_pd_k - PD temperature (K) for the dark current (None - the dark-current reference temperature T0, so the dark current is I0).
    def monitor_current_a(self, p_drop_w, p_bus_w=0.0, t_pd_k=None) -> np.ndarray:
        hw = self._monitor_hw
        p = np.atleast_1d(np.asarray(p_drop_w, dtype=float))
        out = np.empty(p.shape)
        for i, p_drop in enumerate(p.ravel()):
            p_drop_bn = np.zeros((hw.B, hw.N))
            p_drop_bn[self.b, self.n] = p_drop
            p_bus_bn = np.zeros((hw.B, hw.N))
            p_bus_bn[self.b, self.n] = float(np.asarray(p_bus_w).ravel()[i if np.size(p_bus_w) > 1 else 0])
            t = None
            if t_pd_k is not None:
                t = np.full((hw.B, hw.N), float(np.asarray(t_pd_k).ravel()[i if np.size(t_pd_k) > 1 else 0]))
            i_signal, i_dark = hw.photocurrent_a(p_drop_bn, p_bus_bn, t)
            out.ravel()[i] = i_signal[self.b, self.n] + i_dark[self.b, self.n]
        return out

    # static states
    # (absorbed heat, drop power) in W, each (M,), from every TX line except the ring's own.
    # Evaluated on the ring-temperature grid t_grid (M,) in K, as the level average for NRZ.
    # `p_lines_w` (L,) - the line powers at this ring (W - default - the chip-input powers, that is no upstream filtering - `_bank_codes` passes the bus-filtered ones).
    # every order of the ring counts (the exact ring is periodic), so a neighbouring order near another line (BND-1, the adjacent-order bound) shows up here.
    def other_lines_terms(self, t_grid, t_heater=None, v_lo: float = 0.0, v_hi: float | None = None,
                          rho: float = 0.5, p_lines_w=None) -> tuple[np.ndarray, np.ndarray]:
        tg = np.atleast_1d(np.asarray(t_grid, dtype=float))
        if self.other_lines.size == 0:
            return np.zeros(tg.size), np.zeros(tg.size)
        other = self.other_lines
        o = self.optics(self.lam_lines[other], tg, t_heater, v_lo, v_hi)
        rho_ = 0.0 if v_hi is None else float(rho)
        p = (self.p_lines_w if p_lines_w is None else np.asarray(p_lines_w, dtype=float))[other]
        p_abs_w = (rho_ * o["abs1"] + (1.0 - rho_) * o["abs0"]) @ p
        p_drop_w = (rho_ * o["drop1"] + (1.0 - rho_) * o["drop0"]) @ p
        return p_abs_w, p_drop_w

    # solve optics and heat balance on ring temperatures t_grid (M,) in K and wavelengths lam (L,) in m.
    # t_base is stage temperature in K.
    # p_in_w is probe-line power at the ring in W.
    # v_lo and v_hi are junction reverse biases in V.
    # rho is the probability of NRZ level 1.
    # p_j and p_d are data heat in W.
    # agg_r and agg_h are aggressor rises at the ring and heater resistor in K.
    # optical fractions and thermal outputs have shape (M, L).
    # see optics for auxiliary output shapes.
    # lines='own' uses only the probe line.
    # lines='all' adds absorbed heat and drop leakage from the other TX
    # lines.
    # p_lines_w gives their powers in W, or None uses chip-input powers.
    # drop_other_w has shape (M,).
    # heater temperature changes optical loss only when dln_alpha_dt_heater is nonzero.
    # in that case, update the optics at the estimated heater temperature for heater_loss_passes passes, with at least one pass.
    def state_grid(self, lam, t_grid, *, t_base: float, p_in_w: float, v_lo: float = 0.0, v_hi: float | None = None,
                   rho: float = 0.5, p_j: float = 0.0, p_d: float = 0.0, agg_r: float = 0.0, agg_h: float = 0.0,
                   heater_loss_passes: int = 2, lines: str = "own", p_lines_w=None) -> dict:
        if lines not in ("own", "all"):
            raise ValueError(f"lines must be 'own' or 'all', got {lines!r}")
        tg = np.asarray(t_grid, dtype=float)
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        kw = dict(t_base=t_base, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, rho=rho, p_j=p_j, p_d=p_d, agg_r=agg_r,
                  agg_h=agg_h)
        if self.rp[phase.RP_CT_HEATER] == 0.0:
            extra = self.other_lines_terms(tg, None, v_lo, v_hi, rho, p_lines_w) if lines == "all" else (0.0, 0.0)
            return self.heat_balance(self.optics(lam, tg, None, v_lo, v_hi), tg, p_abs_extra=extra[0],
                                     p_drop_extra=extra[1], **kw)
        if lam.size > 1:                     # the heater temperature differs per wavelength - one column at a time
            cols = [self.state_grid(lam[i:i + 1], tg, heater_loss_passes=heater_loss_passes, lines=lines,
                                    p_lines_w=p_lines_w, **kw) for i in range(lam.size)]
            out = {}
            for k, v in cols[0].items():
                out[k] = (np.concatenate([c[k] for c in cols], axis=1) if isinstance(v, np.ndarray) and v.ndim == 2
                          and v.shape[1] == 1 else v)
            return out
        z = self.z
        t_h = tg + (z["hh"] / z["hr"] - 1.0) * (tg - t_base)          # first guess - heater-only ratio
        for _ in range(max(1, int(heater_loss_passes))):
            extra = self.other_lines_terms(tg, t_h, v_lo, v_hi, rho, p_lines_w) if lines == "all" else (0.0, 0.0)
            o = self.heat_balance(self.optics(lam, tg, t_h, v_lo, v_hi), tg, p_abs_extra=extra[0],
                                  p_drop_extra=extra[1], **kw)
            t_h = o["t_h"][:, 0]
        return o

    # find the heater power and drive that sustain each ring temperature in t_grid (M,) in K.
    # o contains optics from optics or state_grid.
    # return a new dict without changing o.
    # p_abs, p_h, t_h, u and t_pd have shape (M, L).
    # power is in W and temperature is in K.
    # drop_avg and thru_avg are level-averaged fractions of probe power.
    # reloma is |T1 - T0|.
    # p_abs_extra and p_drop_extra are scalar or (M,) powers in W from the other lines.
    # drop_other_w stores their drop power with shape (M,).
    # use temperature as the independent variable so the required heater power is explicit.
    # for each temperature T and wavelength, with rho the probability of level 1, P_abs(T) = P_in [rho A_1(T) + (1 - rho) A_0(T)] + P_abs_extra
    # P_h(T) = (T - T_base - Z_or P_abs - Z_jr P_j - Z_dr P_d - agg_r) / Z_hr
    # T_H(T) = T_base + Z_hh P_h + Z_oh P_abs + Z_jh P_j + Z_dh P_d + agg_h
    # A_0 and A_1 are absorbed-heat fractions.
    # P_j and P_d are junction and driver heat in W.
    # agg_r and agg_h are aggressor temperature rises in K.
    # StaticRing defines the Z source-observer keys.
    # u_from_power inverts the driver law at R(T_H) = R0 (1 + alpha (T_H - T_R0)).
    # R0 is cold resistance in ohm, alpha is TCR in 1/K and T_R0 is its reference temperature in K.
    # for voltage drive u = sqrt(P_h R) / V_FS.
    # for current drive u = sqrt(P_h / R) / I_FS.
    # for power drive u = P_h / P_FS.
    # for dac_pwm u = P_h R / V_FS^2.
    def heat_balance(self, o: dict, t_grid, *, t_base: float, p_in_w: float, v_lo: float = 0.0,
                     v_hi: float | None = None, rho: float = 0.5, p_j: float = 0.0, p_d: float = 0.0,
                     agg_r: float = 0.0, agg_h: float = 0.0, p_abs_extra=0.0, p_drop_extra=0.0) -> dict:
        tg = np.asarray(t_grid, dtype=float)
        z = self.z
        rho_ = 0.0 if v_hi is None else float(rho)
        o = dict(o)
        extra = np.broadcast_to(np.asarray(p_abs_extra, dtype=float), tg.shape)
        p_abs = p_in_w * (rho_ * o["abs1"] + (1.0 - rho_) * o["abs0"]) + extra[:, None]
        o["drop_other_w"] = np.array(np.broadcast_to(np.asarray(p_drop_extra, dtype=float), tg.shape))
        p_h = (tg[:, None] - t_base - z["or"] * p_abs - z["jr"] * p_j - z["dr"] * p_d - agg_r) / z["hr"]
        t_h = t_base + z["hh"] * p_h + z["oh"] * p_abs + z["jh"] * p_j + z["dh"] * p_d + agg_h
        o.update(p_abs=p_abs, p_h=p_h, t_h=t_h, u=self.u_from_power(p_h, t_h), rho=rho_,
                 drop_avg=rho_ * o["drop1"] + (1.0 - rho_) * o["drop0"],
                 thru_avg=rho_ * o["thru1"] + (1.0 - rho_) * o["thru0"],
                 reloma=np.abs(o["thru1"] - o["thru0"]),
                 t_pd=t_base + z["hp"] * p_h + z["op"] * p_abs + z["jp"] * p_j + z["dp"] * p_d)
        return o

    # select a steady lock point for an ideal settled controller.
    # return a LockPoint.
    # mode chooses the measured quantity - peak maximizes mean drop power, resonance puts the resonance on
    # the line, and oracle maximizes relOMA = |T1 - T0|.
    # half_max and fraction find a fraction of peak drop.
    # side='hot' selects the hotter flank, with the laser blue of resonance.
    # side='cold' selects the other.
    # stable_only restricts oracle to points stable at fixed drive.
    # it does not test heater feasibility.
    # modulate enables NRZ at the datapath swing around bias_v in V.
    # oracle requires modulate=True.
    # data_heat defaults to modulate.
    # order selects a resonance relative to design order 0.
    # 'first' uses the first order reached by heater-only acquisition from the heater-off temperature at this base temperature.
    # lines='own' uses the ring's line.
    # lines='all' includes all TX lines in heating and monitor drop power.
    # the fraction-of-peak locks then use total monitor drop power.
    # p_lines_w gives line powers (L,) in W at the ring.
    # None uses their chip-input powers.
    # t_base is stage temperature in K.
    # p_in_w is own-line power at the ring in W.
    # agg_r and agg_h are aggressor rises in K.
    # lam is laser wavelength in m, or None for the ring's line.
    # n_grid samples span_linewidths linewidths on each side of resonance in ring-temperature units.
    def lock(self, *, mode: str = "half_max", lam: float | None = None, t_base: float, p_in_w: float,
             bias_v: float = 0.0, modulate: bool = False, rho: float = 0.5, data_heat: bool | None = None,
             fraction: float | None = None, side: str = "hot", stable_only: bool = True, n_grid: int = 2001,
             span_linewidths: float = 6.0, agg_r: float = 0.0, agg_h: float = 0.0,
             order: int | str = 0, lines: str = "own", p_lines_w=None) -> LockPoint:
        lam = self.lam_line if lam is None else float(lam)
        if modulate:
            v_lo, v_hi = self.nrz_levels(bias_v)
        else:
            v_lo, v_hi = float(bias_v), None
        with_data_heat = modulate if data_heat is None else bool(data_heat)
        p_j, p_d = self.data_heat(v_lo, v_hi if v_hi is not None else v_lo, enable=with_data_heat)
        v_mid = 0.5 * (v_lo + (v_hi if v_hi is not None else v_lo))
        if order == "first":
            order = self.first_order(lam, v_mid, t_base + self.z["jr"] * p_j + self.z["dr"] * p_d + agg_r)
        t_res_k = self.t_res(lam, v_mid, int(order))
        width_k = self.linewidth_k(t_res_k)
        tg = np.linspace(t_res_k - span_linewidths * width_k, t_res_k + span_linewidths * width_k, int(n_grid))
        state = self.state_grid(lam, tg, t_base=t_base, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, rho=rho, p_j=p_j,
                                p_d=p_d, agg_r=agg_r, agg_h=agg_h, lines=lines, p_lines_w=p_lines_w)
        # the drop seen by the monitor (W) - own line, plus the other lines' leakage with lines='all'
        drop_w = p_in_w * state["drop_avg"][:, 0] + state["drop_other_w"]
        # level search on the per-unit own drop for lines='own' (works at P_in -> 0), on the monitor power otherwise
        # (unless no light reaches the ring at all)
        drop = state["drop_avg"][:, 0] if (lines == "own" or not np.max(drop_w) > 0.0) else drop_w
        du_dt = np.gradient(state["u"][:, 0], tg)
        stable = du_dt > 0
        i_peak = int(np.argmax(drop))
        # i - the fractional grid index of the lock point
        if mode == "peak":
            i = _refine_argmax(tg, drop, i_peak)
        elif mode == "resonance":
            i = float(np.interp(t_res_k, tg, np.arange(tg.size)))
        elif mode in ("half_max", "fraction"):
            f = 0.5 if mode == "half_max" else float(fraction)
            level = f * drop[i_peak]
            if side == "hot":
                walk = np.arange(i_peak, tg.size)          # from the peak towards hotter ring temperatures
            else:
                walk = np.arange(i_peak, -1, -1)
            j = walk[np.flatnonzero(drop[walk] <= level)]
            if j.size == 0:
                raise ValueError("the lock level is not reached inside the grid; widen span_linewidths")
            j = int(j[0])
            k = j - 1 if side == "hot" else j + 1
            i = k + (drop[k] - level) / (drop[k] - drop[j]) * (j - k)
        elif mode == "oracle":
            if v_hi is None:
                raise ValueError("oracle lock needs modulate=True")
            reloma = state["reloma"][:, 0].copy()
            if stable_only:
                reloma[~stable] = -np.inf
            if not np.any(np.isfinite(reloma)):
                raise ValueError("no stable operating point on the grid")
            i = _refine_argmax(tg, reloma, int(np.argmax(reloma)))
        else:
            raise ValueError(f"unknown lock mode {mode!r}")
        idx = np.arange(tg.size, dtype=float)

        def at_lock(a):
            return float(np.interp(i, idx, np.asarray(a, dtype=float)))
        t_lock = at_lock(tg)
        p_lines = self.p_lines_w if p_lines_w is None else np.asarray(p_lines_w, dtype=float)
        p_bus = p_in_w + (float(np.sum(p_lines[self.other_lines])) if lines == "all" else 0.0)
        return LockPoint(mode=mode, order=int(order), t_ring_k=t_lock, t_res_k=t_res_k,
                         p_abs_w=at_lock(state["p_abs"][:, 0]), p_drop_w=at_lock(drop_w),
                         p_thru_w=p_in_w * at_lock(state["thru_avg"][:, 0]), reloma=at_lock(state["reloma"][:, 0]),
                         drop_peak_w=float(drop_w[i_peak]), p_j_w=p_j, p_d_w=p_d,
                         p_heater_w=at_lock(state["p_h"][:, 0]), t_heater_k=at_lock(state["t_h"][:, 0]),
                         u=at_lock(state["u"][:, 0]), stable=bool(at_lock(du_dt) > 0),
                         detuning_m=float(lam - (self.lam_res + self.order_offset_m(int(order))
                                                 + self.thermal_shift_m(t_lock) + self.junction_shift_m(v_mid)[0])),
                         t_pd_k=at_lock(state["t_pd"][:, 0]), p_drop_other_w=at_lock(state["drop_other_w"]),
                         p_bus_w=p_bus)


# a writeable C-contiguous float64 copy of x broadcast to shape.
def _writeable_full(x, shape) -> np.ndarray:
    return np.array(np.broadcast_to(np.asarray(x, dtype=float), shape), dtype=np.float64, order="C")


# Fractional index of the maximum by a parabola through the three samples around i.
def _refine_argmax(x, y, i) -> float:
    if i <= 0 or i >= len(y) - 1 or not np.all(np.isfinite(y[i - 1:i + 2])):
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2.0 * y1 + y2
    if den >= 0:
        return float(i)
    return float(i + 0.5 * (y0 - y2) / den)


# cache plant objects on pp for reuse by StaticRing instances.
# phase holds PhaseParams rows.
# dc holds thermal gains (B, O, S) in K/W, with axes for world, observer and source.
# heater holds HeaterDriverParams.
# monitor holds MonitorHw.
def _plant_cache(pp) -> dict:
    cache = getattr(pp, "_replica_cache", None)
    if cache is not None:
        return cache
    from mrm.plant.sensing import MonitorHw
    cache = {"phase": phase.PhaseParams.from_plant_params(pp), "dc": thermal.dc_gain(pp.thermal.system),
             "heater": electrical.HeaterDriverParams.from_plant_params(pp), "monitor": MonitorHw.from_plant_params(pp)}
    try:
        object.__setattr__(pp, "_replica_cache", cache)
    except Exception:                                    # pragma: no cover - frozen without __dict__
        pass
    return cache


# return N x N DC thermal gain matrices for world b in K/W.
# Keys use source then observer, as in StaticRing.z.
# sources are h for heater, o for waveguide absorption, j for junction and d for driver.
# observers are r for ring, h for heater resistor and p for monitor PD.
# the PD matrices are zero if the system has no PD observer.
# Element [i, j] maps source ring j to observer ring i.
# off-diagonal elements are crosstalk.
def bank_gains(pp, b: int = 0) -> dict:
    sysm = pp.thermal.system
    G = _plant_cache(pp)["dc"][b]
    N = int(pp.n_rings)
    obs = {"r": sysm.ring_obs, "h": sysm.heater_obs, "p": sysm.pd_obs}
    src = {"h": sysm.heater_src, "o": sysm.wg_src, "j": sysm.junction_src, "d": sysm.driver_src}
    out = {}
    for src_key, src_idx in src.items():
        for obs_key, obs_idx in obs.items():
            m = np.zeros((N, N))
            if src_idx.size and obs_idx.size:
                for i in range(N):
                    for j in range(N):
                        if obs_idx[i] >= 0 and src_idx[j] >= 0:
                            m[i, j] = G[obs_idx[i], src_idx[j]]
            out[src_key + obs_key] = m
    return out


# power transmission of the bus segment between consecutive rings - exp(-bus_loss x spacing).
# bus_loss in 1/m (laser.bus_loss_per_m), spacing in m (layout.ring_spacing_on_bus_m).
# the plant's bus walk uses the same factor (phase.bus_t2_from_plant_params).
# returns 1 when the spacing is not given.
def _bus_segment_t2(pp, b: int = 0) -> float:
    spacing_m = getattr(pp.layout, "ring_spacing_on_bus_m", None)
    if spacing_m is None:
        return 1.0
    spacing_m = float(np.asarray(spacing_m, dtype=float).reshape(-1)[0])
    loss_per_m = np.asarray(pp.laser.bus_loss_per_m, dtype=float).reshape(-1)
    loss_per_m = float(loss_per_m[b if loss_per_m.size > 1 else 0])
    return (float(math.exp(-loss_per_m * spacing_m)) if math.isfinite(spacing_m) and math.isfinite(loss_per_m)
            else 1.0)


# Line powers (W) at the input of every ring of world b, (N, L).
# a single forward pass along `layout.bus_order`, with no reflections, as the plant's bus pass.
# it starts from the chip-input powers (`p_chip_w` (L,) or the preset's).
# each upstream ring multiplies the lines by its level-averaged through transmission at its temperature `t_ring_k` (N,) in K, and each bus segment by its loss.
# Junction levels - (v_lo, v_hi) per ring (N,) in V, or the NRZ levels at `bias_v` - v_hi None everywhere is CW.
# rho - probability of a one.
def bus_line_powers(pp, b: int, t_ring_k, *, bias_v: float | None = None, v_lo=None, v_hi=None, rho: float = 0.5,
                    p_chip_w=None) -> np.ndarray:
    N = int(pp.n_rings)
    t = np.asarray(t_ring_k, dtype=float).reshape(-1)
    rings = [StaticRing(pp, b=b, n=n) for n in range(N)]
    p_at = (rings[0].p_lines_w.copy() if p_chip_w is None else
            np.array(np.broadcast_to(np.asarray(p_chip_w, dtype=float), rings[0].p_lines_w.shape)))
    seg_t2 = _bus_segment_t2(pp, b)
    out = np.zeros((N, p_at.size))
    for q, n in enumerate(np.asarray(pp.layout.bus_order, dtype=np.int64).reshape(-1)):
        if q > 0:
            p_at = p_at * seg_t2
        out[n] = p_at
        ring = rings[n]
        if bias_v is not None:
            lo, hi = ring.nrz_levels(bias_v)
        else:
            lo = 0.0 if v_lo is None else float(np.asarray(v_lo, dtype=float).reshape(-1)[n])
            hi = None if v_hi is None else float(np.asarray(v_hi, dtype=float).reshape(-1)[n])
        o = ring.optics(ring.lam_lines, [t[n]], None, lo, hi)
        rho_ = 0.0 if hi is None else float(rho)
        p_at = p_at * (rho_ * o["thru1"][0] + (1.0 - rho_) * o["thru0"][0])
    return out


# follow a quasi-static thermal branch on the ring-temperature grid t_grid in K.
# g is applied drive minus required drive, u_0 - u(T), at each grid point.
# positive g heats the ring.
# starting at fractional index i0, move with the sign of g to the first sign change.
# interpolate the root.
# return (root temperature in K, fractional grid index, hit_edge).
# hit_edge is true when the walk reaches a grid edge without a root.
# a fixed-drive root is stable where required drive u(T) increases with T.
def _relax(t_grid: np.ndarray, g: np.ndarray, i0: float) -> tuple[float, float, bool]:
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


# Protocol primitives (spectra, sweeps)
# Optical power in W from dBm - None (not printed) means P_in -> 0.
def _dbm_to_w(p_dbm) -> float:
    return 0.0 if p_dbm is None else 1e-3 * 10.0 ** (float(p_dbm) / 10.0)


# Free spectral range in wavelength at the cold resonance, lambda^2 FSR_Hz / c (m).
def _fsr_m(ring: StaticRing) -> float:
    return ring.lam_res ** 2 * ring.fsr_hz / C0


# (T_ring in K, T_heater in K, P_h in W) at drive u without light or data heat (TCR solved - P_in -> 0).
# t_base - base (stage) temperature (K).
def _ring_temps_from_drive(ring: StaticRing, u: float, t_base: float) -> tuple[float, float, float]:
    p_h = float(ring.power_from_u([u], t_base)[0])
    return t_base + ring.z["hr"] * p_h, t_base + ring.z["hh"] * p_h, p_h


# measure one low-power through notch with no optical self-heating.
# t_ring and t_heater are ring and heater-resistor temperatures in K.
# v is reverse bias in V.
# sample n wavelengths across lam_guess +- half_window in m.
# the default half_window is FSR/2, which includes the anti-resonance.
# refine the notch minimum with bounded scalar minimization.
# return lam_min in m and linear transmissions t_min and t_max.
# er_db is 10 log10(t_max / t_min).
# fwhm_m is full width at half depth.
# fwhm_3db_m is width at T = t_max / 2, or 3 dB below the maximum.
# both widths are in m.
# q = lam_min / fwhm_m.
def _notch(ring: StaticRing, *, t_ring: float, t_heater: float, v: float, lam_guess: float,
           half_window: float | None = None, n: int = 4001) -> dict:
    half = 0.5 * _fsr_m(ring) if half_window is None else float(half_window)
    lam = np.linspace(lam_guess - half, lam_guess + half, int(n))
    thru = ring.optics(lam, [t_ring], [t_heater], v)["thru0"][0]
    i = int(np.argmin(thru))

    # through transmission (fraction, level 0) at laser wavelength x (m).
    def thru_at(x):
        return float(ring.optics([x], [t_ring], [t_heater], v)["thru0"][0, 0])
    lo, hi = lam[max(i - 2, 0)], lam[min(i + 2, lam.size - 1)]
    res = optimize.minimize_scalar(thru_at, bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-17, "maxiter": 200})
    lam_min, t_min = (float(res.x), float(res.fun)) if res.fun <= thru[i] else (float(lam[i]), float(thru[i]))
    t_max = float(thru.max())

    # Full width (m) of the notch at through transmission `level` - nan when the level is not crossed.
    def width(level):
        if not t_min < level < t_max:
            return math.nan
        left = np.flatnonzero((thru >= level) & (lam < lam_min))
        right = np.flatnonzero((thru >= level) & (lam > lam_min))
        if left.size == 0 or right.size == 0:
            return math.nan
        a = optimize.brentq(lambda x: thru_at(x) - level, lam[left[-1]], lam_min, xtol=1e-17)
        b = optimize.brentq(lambda x: thru_at(x) - level, lam_min, lam[right[0]], xtol=1e-17)
        return b - a
    fwhm = width(0.5 * (t_max + t_min))
    return {"lam_min": lam_min, "t_min": t_min, "t_max": t_max,
            "er_db": 10.0 * math.log10(t_max / t_min) if t_min > 0 else math.inf,
            "fwhm_m": fwhm, "fwhm_3db_m": width(0.5 * t_max), "q": lam_min / fwhm if fwhm > 0 else math.nan}


# sweep laser wavelength at fixed heater drive u0 with quasi-static optical self-heating.
# lam_grid has shape (L,) in m.
# t_grid is the ring-temperature grid in K.
# direction='up' increases
# wavelength, as in a red sweep through a thermal triangle.
# 'down' decreases wavelength.
# start='cold' or 'hot' starts at the lower or upper temperature edge.
# i_start overrides that choice
# with a fractional grid index.
# state can supply a precomputed StaticRing.state_grid result.
# return wavelength lam in sweep order, temperature t_ring in K, through and drop fractions per input power, fractional grid index idx and boolean hit_edge.
# each output has shape (L,).
def _laser_sweep(ring: StaticRing, lam_grid, t_grid, *, u0: float, t_base: float, p_in_w: float, v_lo: float = 0.0,
                 v_hi: float | None = None, rho: float = 0.5, direction: str = "up", start: str = "cold",
                 state: dict | None = None, i_start: float | None = None) -> dict:
    lam_grid = np.asarray(lam_grid, dtype=float)
    tg = np.asarray(t_grid, dtype=float)
    grid_state = state if state is not None else ring.state_grid(lam_grid, tg, t_base=t_base, p_in_w=p_in_w,
                                                                 v_lo=v_lo, v_hi=v_hi, rho=rho)
    drive_excess = u0 - grid_state["u"]                 # u_0 - u(T) - the ring heats where it is positive
    order = np.arange(lam_grid.size) if direction == "up" else np.arange(lam_grid.size)[::-1]
    i_prev = (0.0 if start == "cold" else float(tg.size - 1)) if i_start is None else float(i_start)
    out = {k: np.zeros(lam_grid.size) for k in ("t_ring", "thru", "drop", "idx")}
    edge = np.zeros(lam_grid.size, bool)
    idx = np.arange(tg.size, dtype=float)
    for j, col in enumerate(order):
        t, i_prev, edge_hit = _relax(tg, drive_excess[:, col], i_prev)
        out["t_ring"][j] = t
        out["idx"][j] = i_prev
        out["thru"][j] = np.interp(i_prev, idx, grid_state["thru_avg"][:, col])
        out["drop"][j] = np.interp(i_prev, idx, grid_state["drop_avg"][:, col])
        edge[j] = edge_hit
    out["lam"] = lam_grid[order]
    out["hit_edge"] = edge
    return out


# Position of the through minimum of a swept spectrum at power p_in_w (W) - the thermal triangle.
# a coarse sweep (steps of coarse_pm, pm) covers the whole dragged notch.
# a fine sweep (steps of fine_pm, pm) around the minimum then continues from the coarse state.
# returns lam_min (m), t_min (through per unit power), t_ring_at_min (K), the coarse sweep, the edge flag and lam_cold (the cold notch position, m).
def _dip_at_power(ring: StaticRing, *, u0: float, t_base: float, p_in_w: float, v: float, direction: str = "up",
                  coarse_pm: float = 4.0, fine_pm: float = 0.1) -> dict:
    t_cold, t_heater_cold, _ = _ring_temps_from_drive(ring, u0, t_base)
    lam0 = ring.lam_res + float(ring.thermal_shift_m(t_cold)) + float(ring.junction_shift_m(v)[0])
    fwhm_m = ring.lam_res / ring.q_loaded
    width_k = ring.linewidth_k(t_cold)
    rise_max_k = ring.z["or"] * p_in_w                       # absorbed fraction <= 1
    drag_m = float(ring.s_at(t_cold + rise_max_k)) * rise_max_k
    lam = np.arange(lam0 - 4 * fwhm_m, lam0 + drag_m + 4 * fwhm_m, coarse_pm * 1e-12)
    if direction == "down":
        lam = lam[::-1]
    step_k = width_k / 80.0
    tg = np.arange(t_cold - 3 * width_k, t_cold + rise_max_k + 3 * width_k, step_k)
    lam_sorted = np.sort(lam)
    coarse = _laser_sweep(ring, lam_sorted, tg, u0=u0, t_base=t_base, p_in_w=p_in_w, v_lo=v, direction=direction)
    j = int(np.argmin(coarse["thru"]))
    j0 = max(j - 3, 0)
    lam_a, lam_b = coarse["lam"][j0], coarse["lam"][min(j + 3, coarse["lam"].size - 1)]
    lam_fine = np.arange(min(lam_a, lam_b), max(lam_a, lam_b) + 0.5 * fine_pm * 1e-12, fine_pm * 1e-12)
    t_at_min = coarse["t_ring"][j]
    tg_fine = np.arange(min(t_at_min, coarse["t_ring"][j0]) - 1.5 * width_k,
                        max(t_at_min, coarse["t_ring"][j0]) + 1.5 * width_k, width_k / 400.0)
    i_start = float(np.interp(coarse["t_ring"][j0], tg_fine, np.arange(tg_fine.size)))
    fine = _laser_sweep(ring, lam_fine, tg_fine, u0=u0, t_base=t_base, p_in_w=p_in_w, v_lo=v, direction=direction,
                        i_start=i_start)
    k = int(np.argmin(fine["thru"]))
    return {"lam_min": float(fine["lam"][k]), "t_min": float(fine["thru"][k]),
            "t_ring_at_min": float(fine["t_ring"][k]), "coarse": coarse,
            "hit_edge": bool(np.any(coarse["hit_edge"]) or np.any(fine["hit_edge"])), "lam_cold": lam0}


# Measured resonance position - the through minimum of a laser sweep at P_in (W) - P_in -> 0 gives the cold notch (`_notch`), a positive P_in the swept thermal triangle (`_dip_at_power`).
def _resonance_position(ring: StaticRing, *, u0: float, t_base: float, v: float, p_in_w: float,
                        direction: str = "up") -> dict:
    if p_in_w <= 0.0:
        t_r, t_h, p_h = _ring_temps_from_drive(ring, u0, t_base)
        guess = ring.lam_res + float(ring.thermal_shift_m(t_r)) + float(ring.junction_shift_m(v)[0])
        notch = _notch(ring, t_ring=t_r, t_heater=t_h, v=v, lam_guess=guess)
        notch.update(t_ring=t_r, t_heater=t_h, p_heater_w=p_h)
        return notch
    return _dip_at_power(ring, u0=u0, t_base=t_base, p_in_w=p_in_w, v=v, direction=direction)


# Base (stage) temperature (K) - args['t_c'] (degC) when it is a number, else the reference temperature of the cold resonance (optics.t_ref_k - the replicas assume the bench measured at the reference temperature).
def _t_base_k(args: Mapping, ring: StaticRing) -> float:
    t = args.get("t_c")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t) + T0_K
    return ring.t_ref


# the topologies a replica evaluates - args['topology'] (one name or a list), else `ring_default`.
def _topologies(args: Mapping, ring_default: str) -> list[str]:
    t = args.get("topology")
    if t is None:
        return [ring_default]
    return [t] if isinstance(t, str) else list(t)


# the realized topology of ring n - `add_drop` when it has a drop coupler, else `all_pass`.
def _plant_topology(pp, n: int = 0) -> str:
    return "add_drop" if float(np.asarray(pp.optics.kappa2_drop)[0, n]) > 0 else "all_pass"


# Replicas - G25 bench protocols (laser sweeps at P_in -> 0 unless stated)
# CAL-1 (loaded Q) - a low-power laser sweep through the notch, Q_L = lambda_min / FWHM.
# FWHM is the full width at half depth of the through notch (linear).
# Conditions as the target assumes ("low power, cold, 0 V") - P_in -> 0 unless `p_in_dbm` is given (then the swept triangle is read the same way), heater `heater_v` (default 0 V, bench voltage source), bias `bias_v` (default 0 V), stage at the reference temperature.
# the primary topology is the plant's own - the other topology is a variant.
# Variant `3db_below_max` -
# the width where T = T_max / 2.
def loaded_q_from_3db_width(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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
    out = ReplicaResult("loaded_q_from_3db_width", float(res[own]["q"]), _target_unit(target, "1"),
                        target_id=getattr(target, "id", None))
    out.variants = {f"topology={tp}": float(v["q"]) for tp, v in res.items() if tp != own}
    out.variants["width=3db_below_max"] = float(res[own]["q_3db"])
    out.detail = res
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "topology": own, "heater_v": v_h, "bias_v": bias,
                      "p_in_w": p_in, "t_base_c": _t_base_k(a, StaticRing(pp, n=ring)) - T0_K,
                      "readout": "full width at half depth (linear) of the through notch; Q = lambda_min / FWHM",
                      "assumption": "P_in -> 0, heater 0 V, bias 0 V (not printed; the calibration target reads it as "
                                    "low power, cold, 0 V)"
                      if p_in <= 0 else "swept at the stated power (thermal triangle)"}
    return out


# Heater voltages (V) and whether the target is a curve - the curve's x of the target, else args['heater_v'].
def _heater_v_list(target, a) -> tuple[list[float], bool]:
    if target is not None and getattr(target, "kind", None) == "curve" and \
            getattr(target, "value", {}).get("x_name") == "heater_v":
        return [float(x) for x in target.value["x"]], True
    v = a.get("heater_v")
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v], True
    return [float(v if v is not None else 0.0)], False


# CAL-2, VAL-2a/b (notch depth against heater voltage) - a low-power laser sweep at each bench heater voltage.
# Notch depth = 10 log10(T_max / T_min) of the through port over one FSR around the shifted resonance (dB).
# the heater is a voltage source whose resistance follows its temperature (TCR, solved self-consistently).
# P_in -> 0 unless `p_in_dbm` is given (the G25 sweep power is not printed) - bias 0 V - stage at the reference temperature.
# Topologies - `add_drop` is the realized ring - `all_pass` removes the drop coupler (K^2_drop = 0) and keeps every other parameter.
# the G25 TX topology is ambiguous, so the targets list both topologies (`topology` in replica_args).
# the primary prediction uses the plant's own topology - the other listed ones are variants.
def notch_er_vs_heater_v(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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
    out = ReplicaResult("notch_er_vs_heater_v", res[own], _target_unit(target, "db"),
                        target_id=getattr(target, "id", None),
                        variants={f"topology={tp}": res[tp] for tp in topos if tp != own}, detail=det)
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "topology": own, "heater_v": vs, "bias_v": bias,
                      "p_in_w": p_in, "port": "through", "heater_drive": "bench voltage source, TCR solved",
                      "t_base_c": _t_base_k(a, StaticRing(pp, n=ring)) - T0_K,
                      "assumption": "P_in -> 0 (sweep power not printed)" if p_in <= 0 else "swept at power"}
    return out


# resonance positions at the heater voltages `vs` (V) relative to the position at `rel_v` (V).
# returns (shifts (m), detail rows, reference position (m), base temperature (K)).
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


# CAL-4 (resonance against heater voltage) - the through minimum of a low-power laser sweep at each bench heater voltage, relative to the position at `relative_to_v` (0 V), in nm.
# Voltage source with the TCR solved - bias 0 V - P_in -> 0 unless `p_in_dbm` - stage at the reference temperature.
def resonance_vs_heater_v(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    cfg, pp = resolve_plant(plant)
    vs, is_list = _heater_v_list(target, a)
    rel = float(a.get("relative_to_v") or 0.0)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    bias = float(a.get("bias_v") or 0.0)
    sh, rows, ref, tb = _resonance_shift_curve(pp, a, vs, ring=ring, p_in=p_in, bias=bias, rel_v=rel)
    val = (sh * 1e9).tolist() if is_list else float(sh[0] * 1e9)
    out = ReplicaResult("resonance_vs_heater_v", val, _target_unit(target, "nm"), target_id=getattr(target, "id", None),
                        detail={"rows": rows, "position_at_ref_v_nm": ref * 1e9})
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "heater_v": vs, "relative_to_v": rel, "bias_v": bias,
                      "p_in_w": p_in, "t_base_c": tb - T0_K, "heater_drive": "bench voltage source, TCR solved",
                      "readout": "through minimum of the swept spectrum"}
    return out


# VAL-1 (tuning range) - the resonance shift at `heater_v` (1.5 V) relative to `relative_to_v` (0 V), in nm.
# same protocol as resonance_vs_heater_v - low-power sweep, bench voltage source with the TCR, bias 0 V.
def tuning_range_at_v(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    cfg, pp = resolve_plant(plant)
    v = float(a.get("heater_v", 1.5))
    rel = float(a.get("relative_to_v") or 0.0)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    bias = float(a.get("bias_v") or 0.0)
    sh, rows, ref, tb = _resonance_shift_curve(pp, a, [v], ring=ring, p_in=p_in, bias=bias, rel_v=rel)
    r = StaticRing(pp, n=ring, driver_law="bench_voltage")
    p_h = float(r.power_from_u([v], tb)[0])
    out = ReplicaResult("tuning_range_at_v", float(sh[0] * 1e9), _target_unit(target, "nm"),
                        target_id=getattr(target, "id", None),
                        detail={"rows": rows, "p_heater_mw": p_h * 1e3, "fsr_nm": _fsr_m(r) * 1e9})
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "heater_v": v, "relative_to_v": rel, "bias_v": bias,
                      "p_in_w": p_in, "t_base_c": tb - T0_K, "heater_drive": "bench voltage source, TCR solved"}
    return out


# CAL-7 (cold resonance offset) - the operating line minus the measured resonance at `heater_v` (0 V), in nm.
# positive means the resonance is blue of the line.
# Low-power sweep (cold cavity) unless `p_in_dbm` - bias `bias_v` (default 0 V) - stage at the reference temperature, because the temperature of the 0 V spectrum is unknown.
# Variant (at P_in -> 0 only) - the same read-out (same heater and bias) on a red sweep at the preset's line power.
def cold_resonance_offset(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    cfg, pp = resolve_plant(plant)
    r = StaticRing(pp, n=ring, driver_law="bench_voltage")
    tb = _t_base_k(a, r)
    p_in = _dbm_to_w(a.get("p_in_dbm"))
    v_h = float(a.get("heater_v") or 0.0)
    bias = float(a.get("bias_v") or 0.0)
    pos = _resonance_position(r, u0=v_h, t_base=tb, v=bias, p_in_w=p_in)
    off = (r.lam_line - pos["lam_min"]) * 1e9
    out = ReplicaResult("cold_resonance_offset", float(off), _target_unit(target, "nm"),
                        target_id=getattr(target, "id", None),
                        detail={"line_nm": r.lam_line * 1e9, "resonance_nm": pos["lam_min"] * 1e9})
    if p_in <= 0:
        d = _dip_at_power(r, u0=v_h, t_base=tb, p_in_w=r.p_line_w, v=bias)
        out.variants["at_preset_power_red_sweep"] = float((r.lam_line - d["lam_min"]) * 1e9)
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "heater_v": v_h, "p_in_w": p_in, "t_base_c": tb - T0_K,
                      "line": r.line, "readout": "line minus through minimum"}
    return out


# CAL-3a/b (heater power per FSR) - the heater power (mW) that moves the resonance by one FSR, or with `output: voltage` the voltage (V).
# the sweep starts at the stage temperature (target t_c, 25 degC).
# Protocol - the low-power spectrum gives the cold resonance lambda_m and the next order red of it, lambda_{m-1}.
# a bench voltage sweep (TCR solved) finds the voltage at which the order-m minimum reaches lambda_{m-1}.
# `statistic` (default median) is taken over the realized worlds and rings - with the default single nominal world it is that device.
def heater_power_per_fsr(plant, target=None, *, ring: int | None = None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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
    out = ReplicaResult("heater_power_per_fsr", val, _target_unit(target, "v" if want_v else "mw"),
                        target_id=getattr(target, "id", None), detail={"rows": rows})
    out.variants = {"voltage_v" if not want_v else "power_mw": float(agg(p_list) * 1e3) if want_v
                    else float(agg(v_list))}
    out.conditions = {**_plant_meta(cfg, pp), "statistic": stat, "rings": list(rings),
                      "t_base_c": _t_base_k(a, StaticRing(pp)) - T0_K, "p_in_w": 0.0, "bias_v": 0.0,
                      "heater_drive": "bench voltage source, TCR solved",
                      "fsr_definition": "spacing of the cold order m and the next order red of it at the stage "
                                        "temperature (the spectrum repeats after this shift)"}
    pre = float(np.median([r_["preset_p_fsr_mw"] for r_ in rows]))
    where = ("ring.heater.p_fsr_ref_wavelength_nm" if "ring.heater.p_fsr_ref_wavelength_nm" in pp.values
             else "the ring's line")
    out.notes.append(f"ring.heater.p_fsr_mw ({pre:.4g} mW) is anchored by R2 with the FSR at {where} "
                     "(mrm.params.relations derived.heater.p_fsr_span_m); this protocol measures the FSR between the "
                     "cold order and the next order red of it, so the two differ by the lambda^2 scaling of the FSR "
                     f"({(float(agg(p_list)) * 1e3 / pre - 1) * 100:+.2f} % here)")
    return out


# CAL-5 (resonance shift against bias) - the DC resonance shift against reverse bias relative to 0 V, in pm.
# the shift is read from the through minimum of a laser sweep at each bias, heater 0 V, stage at the reference temperature.
# `p_in_dbm` lists the optical powers.
# at measurement power the shift includes the self-heating relief - the absorbed power changes with bias, so the ring temperature changes too.
# P_in -> 0 removes it.
# the primary prediction is at the measurement power (the first non-null entry, the G25 printed +4 dBm at the ring), because the protocol measured at power - P_in -> 0 is the variant without the relief.
# at power the minimum is read on a red (increasing wavelength) sweep.
def resonance_shift_vs_bias(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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
    out = ReplicaResult("resonance_shift_vs_bias", res[primary], _target_unit(target, "pm"),
                        target_id=getattr(target, "id", None),
                        variants={f"p_in_dbm={p}": res[p] for p in res if p != primary}, detail=det)
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "bias_v": xs, "heater_v": 0.0, "t_base_c": tb - T0_K,
                      "p_in_dbm": primary, "sweep_direction": direction,
                      "self_heating_relief": "included (measurement power)" if primary is not None else "removed",
                      "junction_shift_only_pm": [float((r.junction_shift_m(v)[0] - r.junction_shift_m(0.0)[0])
                                                       * 1e12) for v in xs]}
    return out


# Replicas - apparent thermal sensitivity at a fixed heater code (CAL-6)
# the realized value `pp.values[key]` of world b as a float (`default` when the key is absent).
def _plant_value(pp, key: str, b: int = 0, default: float = 1.0) -> float:
    v = (getattr(pp, "values", {}) or {}).get(key)
    if v is None:
        return default
    v = np.asarray(v, dtype=float).reshape(-1)
    return float(v[b if v.size > 1 else 0])


# Ring temperature (K) along a slow path at a fixed drive u0, with the laser parked at `lam` (m).
# steps - (t_base, agg_r, agg_h) per step, all K - the base temperature and the aggressor rises at the ring and at the heater resistor.
# Self-heating is included at p_in_w (W), by quasi-static branch continuation from the cold side.
# with P_in -> 0 the temperature is explicit.
# v - junction reverse bias (V).
def _fixed_drive_track(ring: StaticRing, u0: float, steps: Sequence[tuple[float, float, float]], *, p_in_w: float,
                       lam: float, v: float = 0.0, n_grid: int = 3001) -> np.ndarray:
    out = np.zeros(len(steps))
    if p_in_w <= 0.0:
        for i, (t_base_k, agg_r_k, agg_h_k) in enumerate(steps):
            p_h = float(ring.power_from_u([u0], t_base_k + agg_h_k)[0])
            out[i] = t_base_k + agg_r_k + ring.z["hr"] * p_h
        return out
    t_res_k = ring.t_res(lam, v)
    width_k = ring.linewidth_k(t_res_k)
    lo = min(s[0] + s[1] for s in steps) - 1.0
    hi = max(s[0] + s[1] for s in steps) + ring.z["hr"] * float(ring.power_from_u([u0], steps[0][0])[0]) * 1.2 \
        + ring.z["or"] * p_in_w + 4 * width_k
    tg = np.linspace(lo, hi, int(n_grid))
    o = ring.optics(lam, tg, None, v)
    i_prev = 0.0
    for i, (t_base_k, agg_r_k, agg_h_k) in enumerate(steps):
        if ring.rp[phase.RP_CT_HEATER] != 0.0:
            state = ring.state_grid(lam, tg, t_base=t_base_k, p_in_w=p_in_w, v_lo=v, agg_r=agg_r_k, agg_h=agg_h_k)
        else:
            state = ring.heat_balance(o, tg, t_base=t_base_k, p_in_w=p_in_w, v_lo=v, agg_r=agg_r_k, agg_h=agg_h_k)
        out[i], i_prev, _ = _relax(tg, u0 - state["u"][:, 0], i_prev)
    return out


# CAL-6 (apparent thermal sensitivity) - the dlambda/dT a bench would report from a temperature sweep at a FIXED heater DAC code.
# the replica loops over each thermo-optic hypothesis (`hypotheses`, re-loaded with `uncertainty.thermo_optic`), each driver law (`driver_laws`, default every choice of `uncertainty.heater.driver_mode`) and each heating path (`heating`) -
# the code is the one that puts the cold resonance on the ring's line at the first temperature (P_in -> 0), as an integer code through the realized DAC.
# it is then held.
# `ambient` - the stage (T_base) steps from t_grid_c[0] to t_grid_c[-1] (degC).
# the reported temperature is the stage's.
# `aggressor` - an on-die aggressor raises the ring's own temperature by the same span.
# the heater resistor rises by the ROM's DC ratio (aggressor -> heater_resistor) / (aggressor -> ring).
# the reported temperature is the base plus `ring.thermal.reference_rise_frac` of the ring's rise (1 for H_solver / H86, 0.8 for H86_sensor - the thermometer behind Lightmatter's degC sees that fraction of the rise).
# the heater power follows the driver law with the TCR solved at the heater-resistor temperature.
# Voltage drive lowers the apparent sensitivity, current drive raises it, power drive is neutral.
# with `p_in_dbm` (default the preset's line power at the ring) the laser is parked at the line and the self-heating feedback is included (quasi-static, cold-side start).
# the resonance position is that of the ring's state (read by a low-power probe).
# apparent = Delta lambda_res / Delta T_reported (pm/K).
# the primary value is the plant as given (its hypothesis and driver law), on the ambient path, at the stated power - every combination is in `variants`.
def apparent_shift_fixed_code(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    t_grid = [float(x) for x in a.get("t_grid_c", [25.0, 105.0])]
    heating = a.get("heating", ["ambient", "aggressor"])
    heating = [heating] if isinstance(heating, str) else list(heating)
    n_steps = int(a.get("n_steps", 41))
    p_spec = a.get("p_in_dbm", "preset")
    cfg0, pp0 = resolve_plant(plant)
    if cfg0 is None:
        hyps, laws = [None], [None]
    else:
        hyps = [None] + [h for h in (a.get("hypotheses") or []) if h in _hypothesis_choices(cfg0, "thermo_optic")]
        laws = [None] + [x for x in (a.get("driver_laws") or _hypothesis_choices(cfg0, "heater.driver_mode"))]
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
            law_name = {v: k for k, v in electrical.DRIVER_LAWS.items()}[r.law]
            p_in = r.p_line_w if p_spec == "preset" else _dbm_to_w(p_spec)
            frac = _plant_value(pp, "ring.thermal.reference_rise_frac")
            t0, t1 = t_grid[0] + T0_K, t_grid[-1] + T0_K
            # the fixed code - cold resonance on the line at the first temperature (P_in -> 0)
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
    own_law = {v: k for k, v in electrical.DRIVER_LAWS.items()}[StaticRing(pp0, n=ring).law]
    key = f"plant/ambient/{own_law}"
    if key not in res:
        key = next(iter(res))
    out = ReplicaResult("apparent_shift_fixed_code", res[key], _target_unit(target, "pm_per_k"),
                        target_id=getattr(target, "id", None), variants={k: v for k, v in res.items() if k != key},
                        detail={"rows": rows})
    out.conditions = {**_plant_meta(cfg0, pp0), "ring": ring, "t_grid_c": t_grid, "primary": key,
                      "fixed_code": "cold resonance on the line at the first temperature (P_in -> 0)",
                      "p_in": p_spec, "laser": "parked at the ring's line", "readout": "resonance of the ring state"}
    return out


# Replicas - relOMA maps (relOMA = OMA / P_in at the through port)
# 10 log10(x) in dB - x <= 0 gives -inf.
def _to_db(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.where(x > 0, x, 0.0))


# relOMA (linear) of the ring locked at every (bias, ambient) point of a map by an ideal settled controller.
# biases (nb,) in V - temps_k (nt,) base temperatures in K - p_in_w - own-line power at the ring (W) - rho - probability of a one.
# NRZ at the datapath swing around each bias, the laser at the ring's line.
# mode `oracle` - the maximum |T1 - T0| over the ring temperature.
# mode `setpoint_lock` - drop = fraction x peak on the hot side (laser blue of the resonance).
# in both modes a lock point counts only where the heater can deliver it (0 <= P_h, u <= 1 and the driver compliance) and, with `stable_only`, where it is fixed-drive stable.
# Otherwise reloma stays NaN and feasible is False (`setpoint_lock` still records t_lock_k).
# returns reloma (nb, nt), feasible (nb, nt), t_lock_k (K), p_heater_w (W) and u.
def _lock_map(r: StaticRing, biases, temps_k, *, p_in_w: float, mode: str = "oracle", fraction: float = 0.5,
              stable_only: bool = True, n_grid: int = 1201, span: float = 6.0, rho: float = 0.5) -> dict:
    nb, nt = len(biases), len(temps_k)
    out = {k: np.full((nb, nt), np.nan) for k in ("reloma", "t_lock_k", "p_heater_w", "u")}
    out["feasible"] = np.zeros((nb, nt), bool)
    lam = r.lam_line
    # d ln alpha / dT_heater != 0 - the loss depends on the heater temperature, so state_grid recomputes the optics
    loss_depends_on_heater = r.rp[phase.RP_CT_HEATER] != 0.0
    for i, vb in enumerate(biases):
        v_lo, v_hi = r.nrz_levels(vb)
        p_j, p_d = r.data_heat(v_lo, v_hi)
        t_r = r.t_res(lam, 0.5 * (v_lo + v_hi))
        w = r.linewidth_k(t_r)
        tg = np.linspace(t_r - span * w, t_r + span * w, int(n_grid))
        o = None if loss_depends_on_heater else r.optics(lam, tg, None, v_lo, v_hi)
        for j, tb in enumerate(temps_k):
            kw = dict(t_base=tb, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, rho=rho, p_j=p_j, p_d=p_d)
            st = r.state_grid(lam, tg, **kw) if loss_depends_on_heater else r.heat_balance(o, tg, **kw)
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


# first x where y crosses `level` going down (linear interpolation) - nan if it never does.
def _first_crossing(x, y, level) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    for k in range(1, x.size):
        if np.isfinite(y[k - 1]) and np.isfinite(y[k]) and y[k - 1] > level >= y[k]:
            return float(x[k - 1] + (y[k - 1] - level) / (y[k - 1] - y[k]) * (x[k] - x[k - 1]))
    return math.nan


# Offsets (K) below and above `peak_c` (degC) where the curve y_db(t_c) first falls to each level (dB).
# the walk goes away from the peak in both directions, with linear interpolation in dB between samples.
def _contour_offsets(t_c, y_db, peak_c: float, levels) -> tuple[list[float], list[float]]:
    t_c, y = np.asarray(t_c, float), np.asarray(y_db, float)
    below, above = [], []
    ip = int(np.argmin(np.abs(t_c - peak_c)))
    for lev in levels:
        tb = _first_crossing(t_c[:ip + 1][::-1], y[:ip + 1][::-1], lev)
        ta = _first_crossing(t_c[ip:], y[ip:], lev)
        below.append(peak_c - tb if math.isfinite(tb) else math.nan)
        above.append(ta - peak_c if math.isfinite(ta) else math.nan)
    return below, above


# Loop-open relOMA against ambient at one fixed heater drive (VAL-5 - the loop-open map).
# the loop-open code is unpublished, so the drive is chosen such that the relOMA peak of the swept curve sits at `align_peak_c` (degC).
# the ambient sweeps from t_lo_c to t_hi_c (degC) in steps of step_k (K), quasi-statically with branch continuation (`up` - from cold, `down` - from hot).
# Laser at the line, NRZ at the datapath swing around bias_v (V), own-line power p_in_w (W).
# returns the drive u0, the curve (temps_c, reloma, t_ring_k, edge), the peak position, the opposite-direction branch and the hysteresis window (degC) where the two branches differ.
def _fixed_code_open_loop(r: StaticRing, *, bias_v: float, p_in_w: float, align_peak_c: float, t_lo_c: float = 0.0,
                          t_hi_c: float = 105.0, step_k: float = 0.25, direction: str = "up",
                          n_grid: int = 2401, span: float = 12.0) -> dict:
    lam = r.lam_line
    v_lo, v_hi = r.nrz_levels(bias_v)
    p_j, p_d = r.data_heat(v_lo, v_hi)
    t_r = r.t_res(lam, 0.5 * (v_lo + v_hi))
    w = r.linewidth_k(t_r)
    tg = np.linspace(t_r - span * w, t_r + span * w, int(n_grid))
    # d ln alpha / dT_heater != 0 - the loss depends on the heater temperature, so state_grid recomputes the optics
    loss_depends_on_heater = r.rp[phase.RP_CT_HEATER] != 0.0
    o = r.optics(lam, tg, None, v_lo, v_hi)
    temps = np.arange(t_lo_c, t_hi_c + 0.5 * step_k, step_k) + T0_K
    bal = {}

    # (u(T), relOMA(T)) on the grid at base temperature tb (K), cached.
    def balance(tb):
        if tb not in bal:
            kw = dict(t_base=tb, p_in_w=p_in_w, v_lo=v_lo, v_hi=v_hi, p_j=p_j, p_d=p_d)
            st = r.state_grid(lam, tg, **kw) if loss_depends_on_heater else r.heat_balance(o, tg, **kw)
            bal[tb] = (st["u"][:, 0], st["reloma"][:, 0])
        return bal[tb]

    # relOMA, ring temperature and edge flag against ambient at the fixed drive u0 (one sweep direction).
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

    # Ambient (degC) of the relOMA peak of the swept curve at drive u0.
    def t_peak(u0):
        y, _, _ = curve(u0)
        k = _refine_argmax(temps, y, int(np.argmax(y)))
        return float(np.interp(k, np.arange(temps.size), temps)) - T0_K

    # bracket - the drive that puts the resonance at the line at the two ends of the sweep
    # Drive that puts the resonance on the line at ambient tc (degC), without self-heating.
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
    # hysteresis of the fixed-drive line - the opposite sweep at the same drive (where the ring states differ)
    _, tr_rev, _ = curve(u0, "down" if direction == "up" else "up")
    gap = np.abs(tr - tr_rev)
    hyst = gap > 0.05 * w
    return {"u0": u0, "temps_c": temps - T0_K, "reloma": y, "t_ring_k": tr, "edge": edge,
            "peak_c": t_peak(u0), "t_ring_reverse_k": tr_rev,
            "hysteresis_c": ([float(temps[hyst].min() - T0_K), float(temps[hyst].max() - T0_K)]
                             if np.any(hyst) else None),
            "max_branch_gap_k": float(gap.max())}


# CAL-14, VAL-4a/b/c, VAL-5, BND-4 - the G26 relOMA maps (bias 1.0-3.75 V x 0-105 degC, slide 9).
# relOMA = OMA / P_in at the through port (the printed definition, "1 mW of laser into the MRM").
# NRZ at the datapath swing around the bias, the laser at the ring's line.
# Modes -
# `oracle` (loop closed, best achievable) - at every point, the ring temperature that maximizes relOMA among the fixed-drive-stable points the heater can deliver.
# `setpoint_lock` - the drop monitor held at half its peak on the hot side.
# `fixed_code` (loop open, VAL-5) - one heater drive for the whole ambient sweep, chosen so that the swept relOMA peaks at `align_peak_c` (the peak is an alignment input, not a test).
# Quasi-static up-sweep in ambient, sampled every `sample_step_k` and interpolated linearly in dB like the published map - contour offsets from the aligned peak.
# outputs by target -
# CAL-14 (relOMA level) - the mean over its bias and temperature window (`level_db`) -
# VAL-4a/b (contour bias) - the bias of the `contour_db` crossing at `t_c` (scanning up to 4.0 V - nan if not reached - `contour_bias_v`) -
# VAL-4c (map spread) - `spread_db`, max - min over the map -
# BND-4 (oracle flatness) - `max_variation_over_t_at_fixed_bias_db`.
# Optical power - the target's `p_in_dbm` (0 dBm, "1 mW into the MRM") for the self-heating, with a variant at the preset's line power.
# an explicit `p_in_dbm` argument wins (None - P_in -> 0, no self-heating - `"preset"` - the preset's line power at the ring).
# a target that does not print the power uses the preset's.
# `alternative_mode` gives a variant.
def relOMA_map(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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

    # (value, detail) of the requested `output` for map mode md at own-line power p_w (W).
    def evaluate_output(md, p_w):
        if output == "contour_offsets":
            levels = (target.value["levels_db"] if target is not None and getattr(target, "kind", "") ==
                      "contour_offsets" else a.get("levels_db", [-16.0, -24.0]))
            step = float(a.get("sample_step_k", 5.0))
            fc = _fixed_code_open_loop(r, bias_v=float(a.get("bias_v", 2.25)), p_in_w=p_w,
                                       align_peak_c=float(a.get("align_peak_c", 69.6)),
                                       direction=a.get("direction", "up"))
            y_db = _to_db(fc["reloma"])
            tc = fc["temps_c"]
            on = np.isclose(np.mod(tc + 1e-9, step), 0.0, atol=1e-6)
            below_sampled, above_sampled = _contour_offsets(tc[on], y_db[on], fc["peak_c"], levels)
            below_continuous, above_continuous = _contour_offsets(tc, y_db, fc["peak_c"], levels)
            return ({"below_peak_k": below_sampled, "above_peak_k": above_sampled},
                    {"continuous": {"below_peak_k": below_continuous, "above_peak_k": above_continuous},
                     "u0": fc["u0"],
                     "peak_c": fc["peak_c"], "peak_db": float(np.max(y_db)),
                     "samples_c": tc[on].tolist(), "samples_db": y_db[on].tolist(),
                     "hysteresis_c": fc["hysteresis_c"], "max_branch_gap_k": fc["max_branch_gap_k"]})
        if output == "contour_bias_v":
            tc = float(a.get("t_c", cond.get("t_c", 25.0)))
            biases = np.round(np.arange(1.0, 4.0 + 1e-9, 0.025), 4)
            mp = _lock_map(r, biases, [tc + T0_K], p_in_w=p_w, mode=md)
            y = _to_db(mp["reloma"][:, 0])
            lev = float(a["contour_db"])
            ref_db = float(a.get("reference_level_db", -2.1))
            # the same contour measured from the map's own low-bias level (the printed map's top contour is the CAL-14 level, -2.1 dB) - tests the bias shape when the absolute level is not calibrated yet
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
            y = _to_db(mp["reloma"])
            return float(np.nanmean(y)), {"min_db": float(np.nanmin(y)), "max_db": float(np.nanmax(y)),
                                          "infeasible_points": int(np.sum(~mp["feasible"])),
                                          "biases_v": biases.tolist(), "temps_c": temps.tolist()}
        if output in ("spread_db", "max_variation_over_t_at_fixed_bias_db"):
            mp = _lock_map(r, bias_axis, t_axis_c + T0_K, p_in_w=p_w, mode=md)
            y = _to_db(mp["reloma"])
            info = {"max_db": float(np.nanmax(y)), "min_db": float(np.nanmin(y)),
                    "infeasible_points": int(np.sum(~mp["feasible"]))}
            if output == "spread_db":
                return float(np.nanmax(y) - np.nanmin(y)), info
            per_bias = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
            info["worst_bias_v"] = float(bias_axis[int(np.nanargmax(per_bias))])
            return float(np.nanmax(per_bias)), info
        raise ValueError(f"unknown relOMA_map output {output!r}")

    val, info = evaluate_output(mode, p_in)
    out = ReplicaResult("relOMA_map", val, _target_unit(target, "db" if output != "contour_bias_v" else "v"),
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
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "mode": mode, "output": output, "p_in_w": p_in,
                      "line_nm": r.lam_line * 1e9, "swing_pp_v": r.swing_pp_v,
                      "level_convention": {v: k for k, v in electrical.LEVEL_MODES.items()}[r.level_mode],
                      "data_heat": "on (random NRZ)", "rho": 0.5,
                      "reloma": "|T(level 1) - T(level 0)| at the through port (OMA / P_in)",
                      "map_axes": "bias 1.0-3.75 V step 0.05, 0-105 degC step 5 (contours: bias step 0.025 to 4.0 V)"}
    return out


# Replicas - closed loop (quasi-static by default - run_fn for the dynamic path)
# describe a closed-loop measurement for the engine runner.
# cfg and pp specify the plant, with B = pp.batch worlds and N rings.
# scenario may contain uncalibrated
# aggressors.
# the runner calibrates them on the plant's thermal ROM before compiling the scenario.
# controller is a registry name or controller object.
# setpoint_a (B, N) gives factory monitor-current setpoints in A from the bank's quasi-static lock.
# initial_code (B, N) gives heater codes for a warm start.
# duration_s and record_every_s are in s.
# record names the required traces, such as monitor_a, heater_code, t_ring_k and t_base_k.
# options carries bias_v in V, lock_mode, lock_side, start_branch and warm_state.
# warm_state holds t_ring_k in K and p_abs_w, p_heater_w, p_j_w and p_d_w in W, each (B, N), plus scalar t_base_k in K.
# give this true state only to plant initialization.
# it must never reach the controller.
# for BND-3 aggressor feasibility, options also gives max_update_rate_hz and min_total_latency_s.
# the runner must enforce these controller limits.
# the runner returns t_s (T,) in s and traces with shape (T, B, N) or (T, B).
# it can also return setpoint_a with shape (T, B, N) or (B, N) for the controller's measured setpoint.
# _dynamic_lock keeps that trace as controller_setpoint_a and returns the factory values as setpoint_a.
@dataclasses.dataclass
class LockRunRequest:
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


# return the engine adapter mrm.sim.runner.run_lock_request(request).
# raise ReplicaBlocked if the runner module or its adapter is unavailable.
def default_run_fn() -> Callable[[LockRunRequest], Mapping[str, np.ndarray]]:
    try:
        from mrm.sim import runner  # noqa: PLC0415
    except Exception as e:                                              # noqa: BLE001
        raise ReplicaBlocked(f"mrm.sim.runner is not available ({e.__class__.__name__}: {e})") from None
    fn = getattr(runner, "run_lock_request", None)
    if fn is None:
        raise ReplicaBlocked("mrm.sim.runner has no run_lock_request(LockRunRequest) adapter yet "
                             "(docs/handoff/replicas.md)")
    return fn


# the set-point modes to evaluate - args['setpoint'] (one name or a list) - "hypothesis" or None gives all three.
def _setpoint_modes(a: Mapping) -> list[str]:
    sp = a.get("setpoint", "hypothesis")
    if sp in (None, "hypothesis"):
        return ["half_max", "peak", "oracle"]
    return [sp] if isinstance(sp, str) else list(sp)


# the scenario's DC bias `scenario.bias_v` (V) of the Config, else `default`.
def _scenario_bias(cfg, default: float = 1.05) -> float:
    try:
        v = cfg.get("scenario.bias_v") if cfg is not None else None
    except Exception:                                                   # noqa: BLE001
        v = None
    return float(v) if isinstance(v, (int, float)) else default


# CAL-8 (monitor current at lock) - the drop-monitor photocurrent (uA, signal + dark, sensing.MonitorHw) that the loop holds.
# Quasi-static method (the default) -
# the ring is locked at the set-point hypothesis `setpoint` - `half_max` (half the peak drop on the hot side - the g26 drop coupling was chosen for this lock), `peak`, or `oracle` (maximum relOMA).
# power - the preset's line power (`p_in_dbm` - a power in dBm, `"preset"` (default) or None for P_in -> 0).
# NRZ traffic at the scenario bias, base 25 degC.
# every TX line of the bank is on and every ring is locked (`lines='all'`, the G26 measurement - `_bank_codes`).
# the other lines leak into the drop PD and heat the ring, after the rings upstream filtered them.
# Variant `lines=own` - the ring alone with its own line.
# the set-point levels refer to the monitor's total drop power, as a fraction-of-peak controller measures it.
# the dark current is taken at the monitor-PD temperature of the locked state.
# Under a set-point lock the current follows the aggressor only through the loss temperature coefficients and the PD temperature.
# the value is therefore also evaluated at a 105 degC base (`t_hot_c` - in the detail).
# `method='dynamic'` runs the lock of the whole bank through `run_fn` (LockRunRequest, every ring at its own set-point) and reports the median monitor current of ring `ring` over the second half of the run.
def monitor_current_at_lock(plant, target=None, *, ring: int = 0, method: str = "quasi_static", run_fn=None,
                            **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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

    # (monitor current in A, LockPoint) of the ring locked in mode m at base temperature tb (K).
    # lines_ `all` - the ring inside the locked bank (every line on) - `own` - the ring alone on its line.
    def current(m, tb, lines_):
        if lines_ == "all" and r.other_lines.size:
            # the ring inside the locked bank - lines filtered by the rings upstream, PD heated by every source
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
        out = ReplicaResult("monitor_current_at_lock", val, _target_unit(target, "ua"),
                            target_id=getattr(target, "id", None),
                            variants={"quasi_static": res[primary]},
                            detail={"dynamic": {"setpoint_ua": float(dyn["setpoint_a"][0, ring]) * 1e6},
                                    "quasi_static": det})
    else:
        out = ReplicaResult("monitor_current_at_lock", res[primary], _target_unit(target, "ua"),
                            target_id=getattr(target, "id", None),
                            variants={f"setpoint={m}": res[m] for m in modes if m != primary}, detail=det)
        if r.other_lines.size:
            out.variants[f"lines={other_lines}"] = i_other
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "method": method, "setpoint": primary, "bias_v": bias,
                      "lines": lines if r.other_lines.size else "own (single-line plant)",
                      "p_in_w": p_in, "t_base_c": t_base - T0_K, "traffic": "random NRZ", "port": "drop",
                      "monitor": "MonitorHw.photocurrent_a (gain_scale x responsivity x drop + dark at the PD "
                                 "temperature of the lock)"}
    return out


# run every ring from the bank's quasi-static warm start and return recorded traces.
# _bank_codes supplies each ring's order, lock point and heater code, with all lines on and crosstalk included.
# return factory setpoint_a (B, N) in A alongside the traces.
# preserve a runner-supplied setpoint_a trace as controller_setpoint_a.
# p_in is each line's power at the bus input in W, before upstream rings filter it.
# None uses preset line powers.
# t_base is stage temperature in K.
# bias is DC reverse bias in V.
# duration_s is in s.
def _dynamic_lock(cfg, pp, mode: str, *, bias: float, t_base: float, p_in: float | None = None, run_fn=None,
                  scenario=None, duration_s: float = 2e-3, controller: str = "drop_setpoint_pi",
                  options: Mapping[str, Any] | None = None) -> dict:
    fn = run_fn or default_run_fn()
    B, N = int(pp.batch), int(pp.n_rings)
    sp = np.zeros((B, N))
    code = np.zeros((B, N), np.int64)
    warm = {k: np.zeros((B, N)) for k in ("t_ring_k", "p_abs_w", "p_heater_w", "p_j_w", "p_d_w")}
    for b in range(B):
        bc = _bank_codes(pp, b, t_base=t_base, bias=bias, mode=mode, p_in_w=p_in)
        for n in range(N):
            ring = StaticRing(pp, b=b, n=n)
            lp = bc["locks"][n]
            sp[b, n] = float(ring.monitor_current_a(lp.p_drop_w, lp.p_bus_w, bc["t_pd_k"][0, n])[0])
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
    # the warm state of the quasi-static lock, for a runner that can start the plant from it (a warm start)
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


# Aggressor helpers (site, DC ratio, calibration, rises)
# Source index of the replicas' aggressor - the layout site named mrm.scenarios.library.AGGRESSOR_SITE (the default site of A1/A2) when the system has it, else the first aggressor site - None without one.
# g26 also has the stress site aggressor_pd_adjacent, which the replicas never drive.
def _aggressor_src(sysm) -> int | None:
    if not sysm.aggressor_src.size:
        return None
    from mrm.scenarios.library import AGGRESSOR_SITE
    for i in sysm.aggressor_src:
        if str(sysm.src_names[int(i)]) == AGGRESSOR_SITE:
            return int(i)
    return int(sysm.aggressor_src[0])


# Name of the replicas' aggressor site in the thermal system (`_aggressor_src`), None without one.
def _aggressor_site(pp) -> str | None:
    sysm = pp.thermal.system
    agg_src = _aggressor_src(sysm)
    return None if agg_src is None else str(sysm.src_names[agg_src])


# DC ratio (aggressor -> heater_resistor) / (aggressor -> ring_mode_weighted) for this ring.
# it says how much the heater resistor warms per kelvin of ring rise from the aggressor.
# None without an aggressor source.
def _aggressor_ratio(pp, ring: StaticRing) -> float | None:
    sysm = pp.thermal.system
    agg_src = _aggressor_src(sysm)
    if agg_src is None:
        return None
    G = ring.cache["dc"][ring.b]
    g_ring = float(G[sysm.ring_obs[ring.n], agg_src])
    g_heater = float(G[sysm.heater_obs[ring.n], agg_src])
    return g_heater / g_ring if g_ring > 0 else None


# the slide-10 (`a1`) or 808 degC/s (`a2`) aggressor of mrm.scenarios, calibrated on this plant's ROM.
# the calibration is the one the engine uses - the Foster responses aggressor -> victim ring, the model's R1 law, and (with `ring_t0='lock'`) the victim's lock temperature as the starting point of the resonance target.
# returns (scenario, CalibratedWaveform, site).
def _calibrated_aggressor(pp, *, which: str = "a1", victim_ring: int = 0, referencing: str = "resonance",
                          waveform: str | None = None, reading: str | None = None, ring_t0: str = "lock"):
    from mrm.scenarios import calibrate_scenario, foster_from_system, library
    site = _aggressor_site(pp)
    if site is None:
        raise ReplicaBlocked("the plant has no aggressor site in its thermal system")
    if which == "a1":
        scenario = library.a1_slide10_aggressor(site=site, victim_ring=victim_ring, referencing=referencing,
                                                waveform=waveform or "rectified_sine")
    else:
        scenario = library.a2_808_aggressor(site=site, victim_ring=victim_ring, referencing=referencing,
                                            reading=reading or "symmetric_triangle")
    pp_thermal = pp.thermal
    model = {"s0_m_per_k": np.asarray(pp_thermal.dlambda_dt_s0_m_per_k)[:, victim_ring],
             "beta_per_k": np.asarray(pp_thermal.dlambda_dt_beta_per_k)[:, 0],
             "t_ref_k": np.asarray(pp_thermal.dlambda_dt_t0_k)[:, 0]}
    if ring_t0 == "lock":
        model["ring_t0_k"] = np.asarray(pp_thermal.t_ring_lock_k)[:, victim_ring]
    response = foster_from_system(pp_thermal.system, victim_ring=victim_ring, site=site)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated, waveforms = calibrate_scenario(scenario, {site: response}, model=model)
    return calibrated, waveforms[site], site


# simulate aggressor temperature rise at all ring and heater-resistor observers in K.
# drive the full thermal system with calibrated waveform cw for n_periods periods.
# use first-order hold
# for the thermal solve.
# return the last period, with t = 0 at the target cusp.
# t_s has shape (n_samples,) in s.
# ring and heater each have shape (n_samples, B, N) in K.
def _aggressor_rises(pp, cw, *, n_periods: int = 4, n_samples: int = 256) -> dict:
    sysm = pp.thermal.system
    period = float(cw.period_s)
    step_s = period / (2 * n_samples)
    t = np.arange(0, n_periods * 2 * n_samples + 1) * step_s
    power_w = np.zeros((t.size, pp.batch, sysm.n_src))
    agg_src = _aggressor_src(sysm)
    for b in range(pp.batch):
        power_w[:, b, agg_src] = cw.power_at(t, b=b if cw.power_w.shape[0] > 1 else 0)
    obs_k = thermal.simulate(sysm, power_w, step_s, hold="foh", t_base_k=0.0)
    last_period = obs_k[-(2 * n_samples + 1):-1:2]                       # one period, every other sample
    return {"t_s": t[-(2 * n_samples + 1):-1:2] - (n_periods - 1) * period,
            "ring": last_period[:, :, sysm.ring_obs], "heater": last_period[:, :, sysm.heater_obs]}


# Replicas - heater codes under the aggressor (CAL-9a/b/c, VAL-6)
# compute heater codes (T, N) that hold all rings in world b at their lock points.
# t_base is stage temperature in K.
# bias is reverse bias in V.
# mode selects StaticRing.lock's rule.
# agg_r and agg_h give aggressor rises at ring and heater-resistor observers in K, with shape (T, N).
# each row is a time sample.
# a missing aggressor uses one row of zero rise.
# acquire each ring's first reachable order with its heater off at t_order_base in K.
# None uses t_base plus the first aggressor ring rise.
# hold that order for all samples.
# a resonance red of the line can therefore acquire the next bluer order.
# lock rings in layout.bus_order.
# each ring sees lines after upstream through transmission and bus loss.
# p_in_w overrides every line's bus-input power in W.
# bus_filter=False uses chip-input powers everywhere.
# lines='all' also includes the other TX lines' absorbed heat and drop leakage.
# solve the bank heat balance with N x N DC gain matrices so heater powers include crosstalk, Z_hr P_h = T_lock - T_base - Z_or P_abs - Z_jr P_j - Z_dr P_d - agg_r
# convert powers to codes through the driver law, heater TCR and realized DAC at each sample.
# flag negative power, u > 1 or exceeded voltage compliance in infeasible.
# clip codes to the DAC range.
# locks contains the single-ring lock points.
# t_pd_k (T, N) gives the bank's monitor-PD temperatures in K.
# it excludes the aggressor's direct heat at the PD.
def _bank_codes(pp, b: int, *, t_base: float, bias: float, mode: str, agg_r=None, agg_h=None,
                modulate: bool = True, t_order_base: float | None = None, p_in_w: float | None = None,
                lines: str = "all", bus_filter: bool = True) -> dict:
    N = int(pp.n_rings)
    rings = [StaticRing(pp, b=b, n=n) for n in range(N)]
    agg_r_first = np.zeros(N) if agg_r is None else np.atleast_2d(agg_r)[0]
    orders = []
    for n, ring in enumerate(rings):
        v_lo, v_hi = ring.nrz_levels(bias) if modulate else (bias, bias)
        p_j, p_d = ring.data_heat(v_lo, v_hi, enable=modulate)
        t_floor_k = ((t_base + agg_r_first[n] if t_order_base is None else float(t_order_base))
                     + ring.z["jr"] * p_j + ring.z["dr"] * p_d)
        orders.append(ring.first_order(ring.lam_line, 0.5 * (v_lo + v_hi), t_floor_k))
    # locks in bus order - each ring sees the lines as the rings before it on the bus left them (single forward pass, as the plant's bus walk - the bus segment loss included)
    locks = [None] * N
    p_lines_at_ring = np.zeros((N, rings[0].p_lines_w.size))
    p_at = (rings[0].p_lines_w.copy() if p_in_w is None else np.full(rings[0].p_lines_w.size, float(p_in_w)))
    seg_t2 = _bus_segment_t2(pp, b)
    for q, n in enumerate(np.asarray(pp.layout.bus_order, dtype=np.int64).reshape(-1)):
        if q > 0 and bus_filter:
            p_at = p_at * seg_t2
        ring = rings[n]
        p_lines_at_ring[n] = p_at
        locks[n] = ring.lock(mode=mode, t_base=t_base, p_in_w=float(p_at[ring.line]), bias_v=bias,
                             modulate=modulate, order=orders[n], lines=lines, p_lines_w=p_at)
        if bus_filter:
            lo, hi = ring.nrz_levels(bias) if modulate else (bias, None)
            o = ring.optics(ring.lam_lines, [locks[n].t_ring_k], None, lo, hi)
            rho_ = 0.5 if modulate else 0.0
            p_at = p_at * (rho_ * o["thru1"][0] + (1.0 - rho_) * o["thru0"][0])
    # solve the N x N heater gains for heater powers (T, N) at every aggressor sample
    G = bank_gains(pp, b)
    t_lock_k = np.array([lp.t_ring_k for lp in locks])
    p_abs_w = np.array([lp.p_abs_w for lp in locks])
    p_j_w = np.array([lp.p_j_w for lp in locks])
    p_d_w = np.array([lp.p_d_w for lp in locks])
    agg_r = np.zeros((1, N)) if agg_r is None else np.atleast_2d(agg_r)
    agg_h = np.zeros_like(agg_r) if agg_h is None else np.atleast_2d(agg_h)
    rhs = t_lock_k - t_base - G["or"] @ p_abs_w - G["jr"] @ p_j_w - G["dr"] @ p_d_w
    p_heater_w = np.linalg.solve(G["hr"], (rhs[None, :] - agg_r).T).T
    t_heater_k = (t_base + p_heater_w @ G["hh"].T + (G["oh"] @ p_abs_w + G["jh"] @ p_j_w + G["dh"] @ p_d_w)[None, :]
                  + agg_h)
    t_pd_k = t_base + p_heater_w @ G["hp"].T + (G["op"] @ p_abs_w + G["jp"] @ p_j_w + G["dp"] @ p_d_w)[None, :]
    codes = np.zeros_like(p_heater_w)
    u = np.zeros_like(p_heater_w)
    compliance_ok = np.ones(p_heater_w.shape, bool)
    for n, ring in enumerate(rings):
        u[:, n] = ring.u_from_power(p_heater_w[:, n], t_heater_k[:, n])
        codes[:, n] = ring.u_to_code(np.clip(u[:, n], 0.0, None))
        compliance_ok[:, n] = ring.compliance_ok(p_heater_w[:, n], t_heater_k[:, n])
    return {"codes": codes, "p_w": p_heater_w, "u": u, "t_lock_k": t_lock_k, "t_heater_k": t_heater_k,
            "t_pd_k": t_pd_k, "infeasible": (p_heater_w < 0) | (u > 1.0) | ~compliance_ok, "orders": orders,
            "locks": locks, "p_lines_at_ring_w": p_lines_at_ring,
            "law": {v: k for k, v in electrical.DRIVER_LAWS.items()}[rings[0].law]}


# Statistics of one die's codes (T, N) over one aggressor period - per-lane swing, maxima and minima.
def _code_stats(codes: np.ndarray) -> dict:
    cmax, cmin = codes.max(axis=0), codes.min(axis=0)
    sw = cmax - cmin
    return {"swing": [float(sw.min()), float(sw.max())],
            "lane_offsets": {"at_code_maxima": float(cmax.max() - cmax.min()),
                             "at_code_minima": float(cmin.max() - cmin.min())},
            "code_range": [float(cmin.min()), float(cmax.max())],
            "per_lane_swing": sw.tolist(), "per_lane_max": cmax.tolist(), "per_lane_min": cmin.tolist()}


# the median over dies of one `_code_stats` output (a number, a [lo, hi] pair or the lane-offset dict).
def _median_stats(stats: list[dict], output: str):
    if output == "lane_offsets":
        return {k: float(np.median([s["lane_offsets"][k] for s in stats])) for k in ("at_code_maxima",
                                                                                   "at_code_minima")}
    if output in ("swing", "code_range"):
        lo = float(np.median([s[output][0] for s in stats]))
        hi = float(np.median([s[output][1] for s in stats]))
        return [lo, hi]
    return float(np.median([s[output] for s in stats]))


# CAL-9a/b/c and VAL-6 (heater codes under the aggressor) - the heater DAC codes the loop holds over the slide-10 aggressor, per driver-law and full-scale hypothesis.
# the codes identify the DAC LSB only under an assumed driver law and full scale, so each hypothesis is a variant.
# Protocol (quasi-static) -
# `n_dies` sampled dies - per-ring device variation on, every uncertainty entry nominal, device seeds 0..n_dies-1.
# the A1 aggressor power is calibrated on each die's ROM (`referencing`, default resonance - mrm.scenarios).
# Its periodic steady-state rise at every ring and heater-resistor observer comes from an exact FOH simulation of the ThermalSystem.
# every ring is locked (`setpoint`, default half_max on the hot side) at its own line, with NRZ traffic at the scenario bias.
# the codes come from the bank heat balance (crosstalk included), the driver law with the TCR and the realized DAC (`_bank_codes`).
# outputs per die, then the median over dies -
# `swing` (CAL-9a) - [smallest, largest] per-lane swing -
# `lane_offsets` (CAL-9b) - max - min over lanes of the per-lane code maxima and minima -
# `code_range` (CAL-9c) - lowest and highest code over all lanes -
# `excursion` (VAL-6, with `t_c` = [T_lo, T_hi] in degC and no aggressor) - the mean over lanes of code(T_lo) - code(T_hi) for an ambient sweep.
# Variants - every driver law (`driver_mode: hypothesis`), the full-scale range ends of the primary law (`dac_full_scale: hypothesis`), `aggressor_model=uniform` (every ring sees the victim's rise) and the nominal die.
# `method='dynamic'` (ambient protocol only) runs the locks through `run_fn`.
def stabilizer_code_vs_T(plant, target=None, *, method: str = "quasi_static", run_fn=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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
        laws += list(_hypothesis_choices(cfg0, "heater.driver_mode"))
    fs_vals = [None]
    own_law = {v: k for k, v in electrical.DRIVER_LAWS.items()}[StaticRing(pp0).law]
    if cfg0 is not None and a.get("dac_full_scale") == "hypothesis":
        leaf = cfg0.leaves.get(_FULL_SCALE_KEY[own_law])
        if leaf is not None and isinstance(leaf.range, tuple):
            fs_vals += [float(leaf.range[0]), float(leaf.range[1])]

    # (value, detail) for one hypothesis - driver law `law` and DAC full scale `fs` (None - the plant's), `uniform` - the victim's aggressor rise on every ring, `variation` - die variation (`sample` or `off`).
    def one(law, fs, uniform=False, variation="sample"):
        alts = {"uncertainty.heater.driver_mode": law} if law else {}
        ov = {_FULL_SCALE_KEY[law or own_law]: fs} if fs is not None else {}
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
                # one order per ring, acquired at the cold end and held over the sweep (an order hop would show as an infeasible code at the hot end, not as a smaller excursion)
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
    out = ReplicaResult("stabilizer_code_vs_T", val, _target_unit(target, "codes"),
                        target_id=getattr(target, "id", None),
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
        out.variants[f"{_FULL_SCALE_KEY[own_law]}={fs}"] = v
    if not ambient and a.get("with_variants", True):
        v, i = one(None, None, uniform=True)
        out.variants["aggressor_model=uniform"] = v
    if cfg0 is not None and a.get("with_variants", True):
        v, i = one(None, None, variation="off")
        out.variants["variation=off (nominal die)"] = v
        out.detail["variation=off (nominal die)"] = {k: i[k] for k in i if k != "per_die"} | {"per_die": i["per_die"]}
    out.conditions = {**_plant_meta(cfg0, pp0), "method": "quasi_static", "output": output, "n_dies": n_dies,
                      "variation": "sample (device seeds 0..n_dies-1)", "uncertainty": "nominal",
                      "setpoint": mode, "bias_v": bias, "driver_law": info.get("law"),
                      "aggressor": None if ambient else "A1 slide-10 (rectified sine), calibrated on each die's ROM",
                      "referencing": a.get("referencing", "resonance"), "statistic": "median over dies",
                      "laser": "every TX line on at the preset chip-input power, filtered along the bus by the "
                               "rings upstream (single forward pass); each ring locks its own line, the others "
                               "heat it and leak into its drop monitor"}
    return out


# step ambient temperature and measure the settled change in heater code.
# a['t_c'] gives the initial and final stage temperatures in degC.
# step at half of duration_s, in s.
# use median codes before and after the step.
# return the mean change across lanes of world 0 in codes.
def _stabilizer_dynamic(cfg, pp, a, *, run_fn, target, bias: float, mode: str) -> ReplicaResult:
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
    out = ReplicaResult("stabilizer_code_vs_T", float(np.mean(exc)), _target_unit(target, "codes"),
                        target_id=getattr(target, "id", None),
                        variants={"quasi_static": float(np.mean(qs_lo - qs_hi))},
                        detail={"dynamic_per_lane": exc.tolist(), "quasi_static_per_lane": (qs_lo - qs_hi).tolist()})
    out.conditions = {**_plant_meta(cfg, pp), "method": "dynamic", "t_c": [tl, thi], "duration_s": dur,
                      "setpoint": mode, "bias_v": bias}
    return out


# Replica - bistability (slow heater sweeps at power)
# Bistability - slow heater sweeps up and down at power, with the laser parked at the line.
# no target file uses it yet.
# it is a verification protocol for the self-heating bistability (the thermal triangle and its fold jumps).
# it is listed in targets_io.REPLICA_NAMES, so it is registered.
# CW at `bias_v` (V), base `t_c` (degC - default the reference temperature), `p_in_dbm` - dBm, `"preset"` (default - the preset's line power at the ring) or None for P_in -> 0.
# the quasi-static branch is followed in both directions and the jumps (folds) are located.
# value - {"jump_up_u", "jump_down_u", "hysteresis_u", "hysteresis_codes", "hysteresis_mw", "bistable"}, with u the normalized drive of the plant's driver law (codes through the realized DAC).
# the detail carries both traces.
def bistability_sweep(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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
    # (drive at the largest step of ring temperature tt along drive sequence seq, that step in K).
    # the drive is nan when the step is not larger than a quarter linewidth (no jump).
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
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "p_in_w": p_in, "bias_v": v, "t_base_c": tb - T0_K,
                      "laser": "parked at the ring's line, CW", "sweep": "slow heater drive up then down"}
    return out


# Replica - electro-optic bandwidth (small-signal TCMT x junction RC)
# Small-signal EO response of the through port at reverse bias `bias_v` (V) - linearized TCMT x junction RC.
# the ring is described by temporal coupled-mode theory (TCMT) with the rates from optics.exact_to_tcmt at the line, linearized in the resonance frequency.
# the result is multiplied by the junction RC low-pass 1 / (1 + j 2 pi f (R_s + R_source) C_j), with C_j and R_s from the junction table at the bias.
# Symbols (rad/s) - k = ki + ke + kd the total decay rate, ke the bus (through) coupling rate, D = omega_laser - omega_res the detuning (D > 0 - laser blue of the resonance), W = 2 pi f the modulation frequency.
# TCMT in the frame of the laser, with input amplitude 1 -
# A0 = -j sqrt(ke) / (j D + k/2),   through t0 = 1 - j sqrt(ke) A0.
# a modulation d omega = Re{d e^{j W t}} gives A+- = j (d/2) A0 / (+-j W + j D + k/2), and the e^{j W t} part of the output power is P_W = 2 [conj(t0) (-j sqrt(ke)) A+ + conj(conj(t0) (-j sqrt(ke)) A-)].
# `detuning` - `max_slope` (the D > 0 of maximum |dT/dD|), `hwhm` (D = k/2) or `half_depth` (T = (T(0) + 1)/2).
# returns |H(f)/H(0)| and the 3 dB bandwidth (EO S21 - 20 log10 |H/H0| = -3 dB), in Hz.
def _eo_response(r: StaticRing, bias_v: float, f_hz: np.ndarray, *, detuning: str = "max_slope",
                 r_source_ohm: float = 0.0) -> dict:
    lam = r.lam_line
    t_r = r.t_res(lam, bias_v)
    o = r.optics(lam, [t_r], None, bias_v)
    ki, ke, kd = optics.exact_to_tcmt(o["k2t"][0, 0], o["k2d"][0, 0], o["a0"][0, 0], o["trt"][0, 0])
    kap = float(ki + ke + kd)
    ke = float(ke)

    # through transmission (fraction) of the TCMT notch at detuning D (rad/s).
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


# laser detuning rules of the EO protocol as x = 2 Delta / kappa (Lorentzian notch - maximum static slope at 1/sqrt(3) - the half width and the half depth both at x = 1)
_EO_DETUNING_X = {"max_slope": 1.0 / math.sqrt(3.0), "hwhm": 1.0, "half_depth": 1.0}


# EO small-signal response in waveform mode (mrm.waveform).
# mrm.waveform gives the exact TCMT linearization with index and loss modulation, the junction C_j(V), and R_s driven through the driver output resistance.
# rule - a key of `_EO_DETUNING_X`.
# r_drv_ohm None - the plant's datapath.driver_output_resistance_ohm.
# returns the 3 dB bandwidth (Hz) and the other read-outs.
def _eo_waveform(pp, b: int, n: int, bias_v: float, rule: str, r_drv_ohm: float | None,
                 loss_modulation: bool = True) -> dict:
    from mrm.waveform import DriverJunction, RingTCMT, eo_bandwidth
    ring = RingTCMT.from_plant_params(pp, b, n, v_bias=float(bias_v), x_bias=_EO_DETUNING_X[rule])
    dj = DriverJunction.from_plant_params(pp, b, n, rise_time_s=0.0, r_drv_ohm=r_drv_ohm)
    res = eo_bandwidth(ring, dj, float(bias_v), loss_modulation=loss_modulation)
    return {"bw_hz": float(res["f3db_hz"]), "peaking_db": float(res["peaking_db"]), "fwhm_hz": float(res["fwhm_hz"]),
            "rc_pole_hz": float(res["rc_pole_hz"]), "x_bias": float(res["x_bias"]), "r_s_ohm": float(dj.r_s(bias_v)),
            "r_drv_ohm": float(dj.r_drv_ohm), "c_j_f": float(dj.c_j(bias_v))}


# CAL-10 and VAL-3a/b - measure small-signal EO 3 dB bandwidth at reverse bias bias_v in V.
# the waveform method uses mrm.waveform.eo_bandwidth to find |H(f)/H(0)|^2 = 1/2.
# its TCMT linearization
# includes index and free-carrier-loss modulation, junction C_j(V), and series resistance R_s.
# r_drv_ohm adds driver output resistance in ohm.
# None uses datapath.driver_output_resistance_ohm, which is 50 ohm in the presets, as for a 50 ohm VNA port.
# re-center the laser at each bias with the same detuning rule.
# detuning='hypothesis' gives max_slope
# as primary, with x = 2 Delta/kappa = 1/sqrt(3), and hwhm as a variant, with x = 1.
# for a Lorentzian notch, hwhm and half depth coincide.
# further variants use zero driver resistance, no loss modulation, or the index-only _eo_response with zero source resistance.
# return bandwidth in GHz.
# output='ratio' with two biases returns BW(bias[1]) / BW(bias[0]) for VAL-3b.
def eo_bandwidth_at_bias(plant, target=None, *, ring: int = 0, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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

    # EO response row (bw_hz, peaking_db, ...) at bias v (V) with detuning rule `rule_`.
    # how `waveform` - mrm.waveform.eo_bandwidth with driver resistance r_drv_ (ohm) and loss modulation `loss` - otherwise the replica's own index-only TCMT (`_eo_response`).
    def one(v, rule_, how, r_drv_, loss=True):
        if how == "waveform":
            return _eo_waveform(pp, r.b, r.n, v, rule_, r_drv_, loss_modulation=loss)
        rr = _eo_response(r, v, f, detuning=rule_, r_source_ohm=0.0 if r_drv_ is None else float(r_drv_))
        return {"bw_hz": rr["bw_hz"], "peaking_db": rr["peaking_db"], "fwhm_hz": rr["kappa_hz"],
                "rc_pole_hz": rr["rc_hz"], "x_bias": 2.0 * rr["detuning_rad_per_s"] / rr["kappa_rad_per_s"],
                "r_s_ohm": rr["r_s_ohm"], "c_j_f": rr["c_j_f"]}

    # (value, detail) - the bandwidth (GHz) at the one bias, or the ratio BW(bias[1]) / BW(bias[0]).
    def predict(rule_, how, r_drv_, loss=True):
        rows = [one(v, rule_, how, r_drv_, loss) for v in biases]
        if ratio:
            return rows[1]["bw_hz"] / rows[0]["bw_hz"], {"bw_ghz": [x["bw_hz"] / 1e9 for x in rows], "rows": rows}
        return rows[0]["bw_hz"] / 1e9, rows[0]
    val, info = predict(rules[0], method, r_drv)
    out = ReplicaResult("eo_bandwidth_at_bias", float(val), _target_unit(target, "1" if ratio else "ghz"),
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
    out.conditions = {**_plant_meta(cfg, pp), "ring": ring, "bias_v": bias, "detuning": rules[0],
                      "x_bias": _EO_DETUNING_X[rules[0]], "method": method,
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


# Replicas - populations (responsivity, laser lines)
# CAL-11 (PD responsivity) - [min, max] of the sampled monitor-PD responsivity (A/W).
# Taken over `batch` worlds x rings, with the uncertainty and the device variation sampled (seeds 0).
def pd_responsivity(plant, target=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    B = int(a.get("batch", 256))
    cfg, pp = resolve_plant(plant, batch=B, uncertainty="sample", variation="sample")
    x = np.asarray(pp.monitor.responsivity_a_per_w, dtype=float).ravel()
    out = ReplicaResult("pd_responsivity", [float(x.min()), float(x.max())], _target_unit(target, "a_per_w"),
                        target_id=getattr(target, "id", None),
                        detail={"n": int(x.size), "quantiles": np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99]).tolist()})
    out.conditions = {**_plant_meta(cfg, pp), "uncertainty": "sample", "variation": "sample", "statistic": "range"}
    return out


# CAL-12a-d (laser line statistics) - the static line error and the 24 h drift of the laser lines.
# static error - line frequency minus grid (GHz), over `batch` worlds with the per-die and per-line variation.
# Drift - the `laser_drift` scenario of mrm.scenarios (`drift_mode` linear, the default worst case, or random_walk over `batch` worlds), rendered every 60 s over 24 h.
# `output` - `max_abs_static_error_ghz` | `static_error_quartiles_ghz` ({median, q25, q75}) | `max_abs_drift_24h_nm` | `max_abs_power_drift_24h_pct`.
def laser_line_statistics(plant, target=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
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
        out = ReplicaResult("laser_line_statistics", val, _target_unit(target, "ghz"),
                            target_id=getattr(target, "id", None),
                            detail={"n": int(e.size), "sigma_ghz": float(np.std(e)),
                                    "fraction_outside_15ghz": float(np.mean(np.abs(e) > 15.0))})
        out.conditions = {**_plant_meta(cfg, pp), "sign": "line frequency minus grid", "lines": "tx"}
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
    out = ReplicaResult("laser_line_statistics", val, _target_unit(target, unit), target_id=getattr(target, "id", None))
    out.conditions = {**_plant_meta(cfg, pp), "scenario": f"laser_drift({mode})", "window_s": 86400.0,
                      "render_step_s": 60.0, "worlds": nb}
    return out


# Replica - aggressor trajectory in resonance units (CAL-13a/b)
# the victim ring's aggressor-only response over one period, converted to the printed degC.
# the response is an exact FOH simulation of the victim's Foster response in periodic steady state (mrm.scenarios.calibrate.simulate_foster_foh), independent of the calibration's own response matrix.
# Conversion -
# resonance referencing y = T_base + shift / s_ref, with the model's R1 law from the ring's starting temperature -
# temperature referencing y = T_base + thermometer_fraction x rise.
# returns t_s, y_c (degC), rise_k and t_start_k.
def _victim_trajectory_c(pp, cw, target_obj, *, victim: int = 0, ring_t0: str = "lock", n_periods: int = 4,
                         n_samples: int = 512) -> dict:
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


# CAL-13a/b (aggressor trajectory) - the aggressor of mrm.scenarios, calibrated on the plant's ROM, then simulated independently through the victim's Foster response in periodic steady state (t = 0 at the cusp).
# the aggressor is A1 (slide 10) or A2 (808 degC/s).
# the calibration is the scenario's own step, as the engine runs it.
# the default referencing is `resonance` with 85.86 pm/K (`mrm.scenarios.calibrate.LM_DLAMBDA_DT_M_PER_K`).
# CAL-13a - {"t_s", "y"} over one period, y in the printed degC (resonance referencing - base + model resonance shift / 85.86 pm/K).
# CAL-13b (`interpretation` present) - the printed-rate equivalent of the simulated trajectory.
# `delta_t_per_period` = (max - min) / period - `one_sided_sawtooth` = (max - min) / rise time.
# the other readings and interpretations are variants.
def aggressor_trajectory(plant, target=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    cfg, pp = resolve_plant(plant)
    ref = a.get("referencing", "resonance")
    if "interpretation" not in a:
        wf = a.get("waveform", "rectified_sine")
        sc, cw, site = _calibrated_aggressor(pp, which="a1", referencing=ref, waveform=wf)
        tgt = sc.aggressors[0].target
        tr = _victim_trajectory_c(pp, cw, tgt)
        val = {"t_s": tr["t_s"].tolist(), "y": tr["y_c"].tolist()}
        from mrm.validation.targets_io import rectified_sine
        p = (getattr(target, "value", None) or {}).get("params", {"t_max_c": 113.6, "delta_t_k": 88.5,
                                                                  "period_s": 0.1307})
        rms = float(np.sqrt(np.mean((tr["y_c"] - rectified_sine(tr["t_s"], p["t_max_c"], p["delta_t_k"],
                                                                 p["period_s"])) ** 2)))
        out = ReplicaResult("aggressor_trajectory", val, _target_unit(target, "c"),
                            target_id=getattr(target, "id", None),
                            detail={"rms_vs_target_c": rms, "victim_rise_k": [float(tr["rise_k"].min()),
                                                                              float(tr["rise_k"].max())],
                                    "calibration_residual_rms_k": np.asarray(cw.residual_rms_k).tolist(),
                                    "peak_power_mw": float(np.asarray(cw.peak_power_w)[0] * 1e3),
                                    "mean_power_mw": float(np.asarray(cw.mean_power_w)[0] * 1e3)})
        out.conditions = {**_plant_meta(cfg, pp), "aggressor": "A1 slide-10", "waveform": wf, "referencing": ref,
                          "s_ref_pm_per_k": tgt.s_ref_m_per_k * 1e12,
                          "ring_t0": "lock temperature (ring.thermal.t_ring_lock_c)",
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
            tr = _victim_trajectory_c(pp, cw, tgt)
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
    out = ReplicaResult("aggressor_trajectory", res[primary], _target_unit(target, "k_per_s"),
                        target_id=getattr(target, "id", None), variants={k: v for k, v in res.items() if k != primary})
    out.conditions = {**_plant_meta(cfg, pp), "aggressor": "A2 808 degC/s", "primary": primary, "referencing": ref}
    out.notes.append("a scenario calibration (the A2 power is fitted to each reading); the one-sided sawtooth cannot "
                     "be followed by a heater, so its residual is large by construction")
    return out


# Replicas - bounds
# BND-1 (adjacent orders) - how close the other resonance orders of each ring come to the other TX lines.
# every ring is locked with its resonance on its own line.
# for every other resonance order inside the TX span (+- 2 FWHM), the replica takes the distance to the nearest other TX line, in units of that order's FWHM (the TCMT linewidth at the order's wavelength, coupling dispersion included).
# the orders come from the cold phase curve of the phase model - the comb translates rigidly.
# the lines are at their realized frequencies.
# Primary - the nominal world, minimum over rings and orders.
# Variant - the minimum over `batch` sampled worlds (uncertainty and variation sampled).
def adjacent_order_penalty(plant, target=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)

    # (smallest distance in FWHM from another order to another TX line, rows of every order in the span).
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
                level_outs = [np.zeros((1, M)) for _ in range(9)]       # phi0 first (see StaticRing.optics())
                pair_outs = [np.zeros((1, 2)) for _ in range(3)]        # shift, cj, rs
                flags = np.zeros(1, np.int8)
                phase.ring_optics_row(rp, x, np.array([r.t_ref]), np.array([r.t_ref]), np.array([0.0]),
                                      np.array([0.0]), r.grid, *r.junction_tables, *level_outs, *pair_outs, flags)
                phi = level_outs[0][0]
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
                    ki, ke, kd = optics.exact_to_tcmt(oo["k2t"][0, 0], oo["k2d"][0, 0], oo["a0"][0, 0], oo["trt"][0, 0])
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
    out = ReplicaResult("adjacent_order_penalty", float(val) if math.isfinite(val) else 1e9, _target_unit(target, "1"),
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
    out.conditions = {**_plant_meta(cfg, pp), "lines": "tx bank at realized frequencies", "orders": "all inside span "
                      "+- 2 FWHM", "lock": "resonance on its own line (rigid comb)", "fwhm": "TCMT at each order"}
    return out


# BND-2 (order-hop margin) - the resonance shift that each ring's OWN heater still provides at the hot end, in nm.
# Conditions - base temperature `t_hot_c` (degC), every ring of the bank locked (`lock` - default `oracle` = maximum relOMA, the G26 operating point - variants half_max and peak), NRZ at `bias_v`, each line at the preset's chip-input power.
# the margin is the R1 shift from T_lock - Z_hr,ii P_i to T_lock.
# negative means the heater would have to cool the ring (an order hop or a remap is needed).
# Primary - the worst lane of the nominal world.
# Variant - the worst lane over `n_dies` sampled dies.
# the cold offset is the preset's at its reference temperature (T0 = 25 degC assumed, the target's condition).
def order_hop_margin(plant, target=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    t_hot = float(a.get("t_hot_c", 105.0)) + T0_K
    bias = float(a.get("bias_v", 3.75))
    lock = a.get("lock", "oracle")

    t0 = float(a.get("t0_c", 25.0)) + T0_K

    # Own-heater share of the resonance shift at lock (nm) of every lane of every world, locked in `mode`.
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
    out = ReplicaResult("order_hop_margin", float(m.min()), _target_unit(target, "nm"),
                        target_id=getattr(target, "id", None), detail={"per_lane_nm": m.tolist()})
    for alt in ("half_max", "peak"):
        if alt != lock:
            out.variants[f"lock={alt}"] = float(margins(pp, alt).min())
    n_dies = int(a.get("n_dies", 8))
    if cfg is not None and n_dies > 1:
        _, pps = resolve_plant(cfg, batch=n_dies, uncertainty="nominal", variation="sample")
        out.variants["sampled_dies_worst_lane"] = float(margins(pps, lock).min())
    out.conditions = {**_plant_meta(cfg, pp), "t_hot_c": t_hot - T0_K, "bias_v": bias, "lock": lock,
                      "lanes": "worst", "traffic": "random NRZ", "t0_c": t0 - T0_K,
                      "order": "the order a heater-only acquisition takes at t0 (cold start), held to the hot end",
                      "definition": "own-heater share of the shift at lock (R1), nm"}
    return out


# Replicas - third-party pipeline replications (read the extraction artefacts)
# the extraction artefact `mrm/data/extracted/<name>` as a dict, or None while it is not extracted.
def _load_artefact(name: str) -> dict | None:
    p = targets_io.PACKAGE_DIR / "data" / "extracted" / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


# VAL-7a/b - read the mode solve for the de Cea et al. 2019 cross-section.
# read modes_decea2019.json from mrm.extraction.modes with xsec decea2019.
# output='n_g' returns group
# index at the 10 um outer radius used for the published FSR.
# output='confinement_poynting_si' returns
# the Si energy fraction for VAL-7b.
# the Robinson index sensitivity dn_eff/dn_Si is a variant because the extraction interprets the paper's Gamma that way.
# return a blocked result if the extraction is absent.
def mode_solve_published_geometry(plant=None, target=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    unit = _target_unit(target, "1")
    d = _load_artefact("modes_decea2019.json")
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


# VAL-7c - read the self-heating time constant of the de Cea 2019 device in us.
# thermal_val7_decea.json supplies tau_us for absorption in the waveguide core.
# return a blocked result while that extraction is absent.
# the thermal_fem pipeline currently defines only the imec VAL-8 stack.
def self_heating_time_constant(plant=None, target=None, **kw) -> ReplicaResult:
    unit = _target_unit(target, "us")
    d = _load_artefact("thermal_val7_decea.json")
    if d is None:
        return _blocked("self_heating_time_constant", target, unit,
                        "no de Cea thermal stack in mrm/extraction/thermal_fem.py and no thermal_val7_decea.json",
                        paper="DEC19")
    out = ReplicaResult("self_heating_time_constant", float(d["tau_us"]), unit, target_id=getattr(target, "id", None))
    out.conditions = {"artefact": "thermal_val7_decea.json"}
    return out


_VAL8_TARGET_IDS = {("heater_resistor", None): "VAL-8a", ("ring_mode_weighted", None): "VAL-8b",
              (None, "none"): "VAL-8c", (None, "full"): "VAL-8d"}


# VAL-8a-e - read thermal-step predictions for the published imec stack.
# read comparison[target id].predicted from thermal_val8_imec.json, produced by python -m mrm.extraction.thermal_fem --stage val8.
# Without a target, output, observer and undercut
# select the entry through _VAL8_TARGET_IDS.
# return a blocked result when an entry is absent.
def thermal_step_published_stack(plant=None, target=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    tid = getattr(target, "id", None)
    unit = _target_unit(target, "us")
    d = _load_artefact("thermal_val8_imec.json")
    if d is None:
        return _blocked("thermal_step_published_stack", target, unit,
                        "thermal_val8_imec.json (python -m mrm.extraction.thermal_fem --stage val8) is not "
                        "extracted yet", paper=a.get("paper"))
    if tid is None:
        out_ = a.get("output")
        if out_ == "foster_poles":
            tid = "VAL-8e"
        elif out_ == "dc_efficiency_pm_per_mw":
            tid = _VAL8_TARGET_IDS[(None, a.get("undercut", "none"))]
        else:
            tid = _VAL8_TARGET_IDS[(a.get("observer", "ring_mode_weighted"), None)]
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


# Replica - aggressor feasibility (BND-3a/b)
# BND-3a/b - test listed causal controllers against the A1 or A2 aggressor.
# warm-start every ring at half maximum drop.
# run n_periods at the stated rate and latency limits.
# measure only the last period.
# signed_error_pct_min_max returns the victim's signed monitor error band as a percentage of setpoint.
# monitor_deviation_ua_min_max returns its deviation band in uA.
# rank controllers by the miss of the target band from targets_io.evaluate, then by band width.
# the bound passes if any listed causal controller holds the ring.
# return blocked when no controllers are listed, a listed controller is unregistered, or the default runner is unavailable.
def aggressor_feasibility(plant, target=None, *, run_fn=None, **kw) -> ReplicaResult:
    a = _replica_args(target, kw)
    unit = _target_unit(target, "pct")
    from mrm.control import registry
    names = list(a.get("controllers") or [])
    if not names:
        return _blocked("aggressor_feasibility", target, unit, "no controllers listed in replica_args",
                        controllers=names, aggressor=a.get("aggressor"))
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

    # x as a float (None stays None).
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
            # rank by target-band miss, including tolerance slack and the either_sign rule.
            # use band width to break ties because the bound asks whether any causal controller can hold the ring.
            miss = 0.0
            if target is not None and getattr(target, "tolerance", None):
                try:
                    miss = float(targets_io.evaluate(target, band).max_abs_residual or 0.0)
                except Exception:                                   # noqa: BLE001 - malformed band: rank last
                    miss = math.inf
            rows.append({"controller": ctrl, "reading": rd, "band": band, "setpoint_ua": sp * 1e6, "miss": miss})
            if best is None or (miss, band[1] - band[0]) < (best["miss"], best["band"][1] - best["band"][0]):
                best = rows[-1]
    out = ReplicaResult("aggressor_feasibility", best["band"], unit, target_id=getattr(target, "id", None),
                        detail={"rows": rows})
    out.conditions = {**_plant_meta(cfg, pp), "best": best, **limits, "victim_ring": victim,
                      "setpoint": "half maximum of the drop on the hot side (the model's CAL-8 lock; the target's "
                                  f"setpoint_ua = {a.get('setpoint_ua')!r} names that protocol)",
                      "window": "last aggressor period of the run"}
    return out


# Registry, target runner, predictions file
# replica name (the ``replica`` field of a target) -> replica function
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
# every replica named in the target files is implemented, and no other
assert set(REPLICAS) == set(targets_io.REPLICA_NAMES), set(targets_io.REPLICA_NAMES) ^ set(REPLICAS)


# the preset a target is replicated on - G25 bench numbers on `g25_test_vehicle`, the rest on `g26`.
def preset_for(target) -> str:
    return PRESET_FOR_GENERATION.get(str(getattr(target, "generation", "G26")), "g26")


# run the target's named replica on plant, or on preset_for(target) when plant is None.
# convert ReplicaBlocked to a blocked result.
# convert other exceptions raised by the replica to error results with their messages.
# attach the target ID and preset to the result.
def run_target(target, plant=None, **kw) -> ReplicaResult:
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


# True when a prediction has a NaN or inf leaf (the JSON file writes NaN as null, inf as "inf").
def _nonfinite(x: Any) -> bool:
    if isinstance(x, Mapping):
        return any(_nonfinite(v) for v in x.values())
    try:
        return not bool(np.all(np.isfinite(np.asarray(x, dtype=float))))
    except (TypeError, ValueError):
        return False


# targets_io.evaluate of an `ok` result as a dict (None when there is no value - {"error": ...} when it fails).
def _evaluation(target, res: ReplicaResult) -> dict | None:
    if res.status != "ok" or res.value is None:
        return None
    try:
        ev = targets_io.evaluate(target, res.value)
    except Exception as e:                                              # noqa: BLE001
        return {"error": f"{e.__class__.__name__}: {e}"}
    return dataclasses.asdict(ev)


# run the selected targets and return one JSON-compatible record per target.
# each record includes target values and tolerances, prediction, variants, conditions, notes, diagnostics and runtime in s.
# evaluation records targets_io.evaluate for information.
# the report does the scoring.
# JSON stores NaN as null.
# predicted_nonfinite marks an ok result with NaN or infinity so it remains identifiable as a failed prediction.
def predict_all(roles: Sequence[str] = ("calibration", "validation", "bounds"), *, ids: Sequence[str] | None = None,
                plant_by_id: Mapping[str, Any] | None = None, verbose: bool = False) -> list[dict]:
    recs = []
    for role in roles:
        for t in targets_io.load_targets(role):
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


# write `latest_predictions.json` (schema `mrm.validation.predictions/1`) for the report generator.
def write_predictions(path: str | Path | None = None, records: list[dict] | None = None, **kw) -> Path:
    import platform
    path = Path(path) if path is not None else PREDICTIONS_PATH
    recs = predict_all(**kw) if records is None else records
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=targets_io.REPO_DIR, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:                                                   # noqa: BLE001
        commit = None
    doc = {"schema": "mrm.validation.predictions/1",
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "git_commit": commit, "python": platform.python_version(),
           "frozen_sha256": {r: targets_io.file_sha256(r) for r in targets_io.FROZEN_ROLES},
           "note": "predictions of the replica protocols on the uncalibrated presets (no parameter was tuned to a "
                   "target); 'evaluation' is targets_io.evaluate for information, the report scores",
           "counts": {s: sum(1 for r in recs if r["replica_status"] == s) for s in ("ok", "blocked", "error")},
           "records": recs}
    path.write_text(json.dumps(_jsonable(doc), indent=1, allow_nan=False) + "\n")
    return path


# `python -m mrm.validation.replicas [--ids CAL-1 VAL-1 ...] [--out PATH] [--list]`.
def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mrm.validation.replicas", description='``python -m mrm.validation.replicas [--ids CAL-1 VAL-1 ...] [--out PATH] [--list]``.')
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
