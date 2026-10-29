# build calibration problems from measured targets and explicit hypotheses.
# each `*_problems` function returns one Problem per hypothesis.
# predictions use validation.replicas with the target's generation — BENCH for G25 device data and BANK for G26 system data.
# Context supplies values fitted by earlier groups.
# this module uses calibration targets only, never held-out validation.
# coupling_er — loaded Q, intrinsic loss, and notch-change mechanisms.
# heater — tuning power, cold resistance, and the product TCR x thermal resistance.
# offset — the cold resonance position.
# junction — bias efficiency, or active fraction in the reported LUT variant.
# eo — the combined RC and photon-lifetime bandwidth limit.
# monitor — lumped readout gain for each assumed lock point.
# modulation_depth — drive swing with previously fitted efficiency and linewidth.
# dac — full-scale voltage and ring variation under an assumed heater law.
# laser — the centre and spread of line-frequency errors.
# nuisance parameters describe measurement conditions or unobserved device properties.
# they are fitted but are not written as device calibration leaves.
# non-fit targets and scenario-owned parameters are reported elsewhere.
from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mrm.calibration.core import Param, Problem
from mrm.validation import replicas

__all__ = ["Context", "BENCH", "BANK", "coupling_er_problems", "heater_problems", "offset_problems",
           "junction_problems", "eo_problems", "monitor_problems", "modulation_depth_problems", "dac_problems",
           "laser_problems", "notch_floor", "modulation_depth_scan", "dac_codes", "bench_lut_mapping",
           "CouplingConstants", "widened_child"]

BENCH = "g25_test_vehicle"     # preset of the G25 bench device (CAL-1..5, 7, 10, 11)
BANK = "g26"                   # preset of the G26 bank (CAL-8, 9, 12, 14)

# the preset leaves that the groups fit (override paths)
LEAF_ALPHA_OTHER = "ring.optics.loss_budget.alpha_other_abs_db_per_cm"     # dB/cm - sets Q_i, so the coupling branch
LEAF_Q_LOADED = "ring.optics.q_loaded"
LEAF_COUPLER_DISPERSION = "ring.optics.dln_kappa2_through_dlambda_per_nm"  # 1/nm
LEAF_DLN_ALPHA_DT_RING = "ring.optics.dln_alpha_dt_ring_per_k"             # 1/K, loss rise with ring temperature
LEAF_DLN_ALPHA_DT_HEATER = "ring.optics.dln_alpha_dt_heater_per_k"         # 1/K, loss rise with heater temperature
LEAF_P_FSR = "ring.heater.p_fsr_mw"                                         # mW of heater power per FSR of tuning
LEAF_R_COLD = "ring.heater.r_cold_ohm"
LEAF_HEATER_SELF_RATIO = "ring.thermal.heater_self_ratio"
LEAF_RESONANCE_OFFSET = "ring.optics.resonance_offset_nm"
LEAF_JUNCTION_EFF = "ring.junction.eff_at_0v_pm_per_v"
LEAF_FILL_FRACTION = "ring.junction.fill_fraction"
LEAF_SERIES_RESISTANCE = "ring.junction.series_resistance_ohm"
LEAF_CAPACITANCE_0V = "ring.junction.capacitance_at_0v_ff"
LEAF_MONITOR_GAIN = "ring.monitor.gain_scale"
LEAF_DRIVE_SWING = "driver.swing_pp_v"
LEAF_HEATER_FULL_SCALE = "ring.heater.full_scale_v"
LEAF_LINE_OFFSET = "laser.line_offset_ghz"


# Context of the earlier fits
# carry earlier fit results into later replica predictions.
# `device` overrides apply to bench and bank presets.
# `system` overrides apply to the bank.
# `ranges` widens allowed ranges in temporary child presets without changing their starting values or provenance.
# these allowed ranges are separate from the statistical priors used by the fits.
@dataclasses.dataclass
class Context:

    device: dict[str, Any] = dataclasses.field(default_factory=dict)
    system: dict[str, Any] = dataclasses.field(default_factory=dict)
    ranges: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    _scratch_dir: Any = dataclasses.field(default=None, repr=False)
    _widened_children: dict = dataclasses.field(default_factory=dict, repr=False)

    # the overrides for `preset` — the device leaves, the system leaves (except on the bench), then `extra`.
    def overrides(self, preset, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        overrides = dict(self.device)
        if str(preset) != BENCH:
            overrides.update(self.system)
        overrides.update(extra or {})
        return overrides

    # the preset to load — the preset itself, or its scratch child with the widened ranges.
    def base(self, preset):
        if not self.ranges or not isinstance(preset, str):
            return preset
        key = (preset, tuple(sorted((k, tuple(v)) for k, v in self.ranges.items())))
        if key not in self._widened_children:
            self._widened_children[key] = widened_child(preset, self.ranges, self._scratch_path())
        return self._widened_children[key]

    # the scratch directory of the widened children (made on first use, deleted with this Context).
    def _scratch_path(self):
        import tempfile
        from pathlib import Path
        if self._scratch_dir is None:
            self._scratch_dir = tempfile.TemporaryDirectory(prefix="mrm_calibration_")
        return Path(self._scratch_dir.name)

    # the Config of `preset` with the overrides of `overrides` and the alternatives `alts`.
    def config(self, preset, extra: Mapping[str, Any] | None = None, alts: Mapping[str, str] | None = None):
        return replicas.load_config(self.base(preset), overrides=self.overrides(preset, extra), alts=dict(alts or {}))


# write a scratch child preset of `preset` into `directory`.
# it restates each leaf of `ranges` with its own value, provenance, generation and source, and with the given plausible range.
# nothing else changes.
# returns the path of the file.
def widened_child(preset: str, ranges: Mapping[str, Sequence[float]], directory) -> "Path":
    from pathlib import Path

    import yaml

    from mrm.params import config
    cfg = replicas.load_config(preset)
    doc: dict = {"extends": str((Path(config.PRESET_DIR) / f"{preset}.yaml").resolve()),
                 "meta": {"name": f"{preset}_widened", "kind": "preset"}}
    for path, rng in ranges.items():
        leaf = cfg.leaf(path)
        node = doc
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = {"value": float(leaf.value), "provenance": leaf.provenance, "generation": leaf.generation,
                          "source": leaf.source, "range": [float(rng[0]), float(rng[1])]}
    out = Path(directory) / f"{preset}_widened_{len(list(Path(directory).glob('*.yaml')))}.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


# CAL-1 + CAL-2 - Q_L, the coupling branch, the extinction mechanism
# notch depth (dB) seen through an instrument floor F (dB) — the floor adds 10^(-F/10) to the through minimum.
def notch_floor(depth_db, floor_db: float):
    depth = np.asarray(depth_db, dtype=float)
    return (-10.0 * np.log10(10.0 ** (-depth / 10.0) + 10.0 ** (-float(floor_db) / 10.0))).tolist()


# the extinction mechanisms of uncertainty.er_mechanism, and the leaf each one frees.
# dispersion - the leaf declares no range, so 0-4 %/nm are fit bounds only.
# they are about 3x the femwell coupler slope of 1.3-1.5 %/nm (docs/research/05_tools.md, test t02) - the rib160 artefact gives 0.77 %/nm.
# the ~2.9 %/nm estimate of the design critique must not set them - it comes from the 17 dB notch after 13 nm of tuning, which is the held-out 1.5 V point (VAL-2).
# Without a declared prior the half-prior rule cannot apply.
ER_MECHANISMS = {
    "dispersion": Param("dln_kappa2_through", LEAF_COUPLER_DISPERSION, 0.0, 0.04, 0.014, unit="1/nm",
                        declared_prior=False,
                        prior_source="fit bounds 0-4 %/nm (no declared leaf range; ~3x femwell 1.3-1.5 %/nm, 05 "
                                     "t02; rib160 artefact 0.77 %/nm)"),
    "ring_t_loss": Param("dln_alpha_dt_ring", LEAF_DLN_ALPHA_DT_RING, 0.0, 0.005, 0.002, unit="1/K",
                         prior_source="leaf range [0, 0.005]"),
    "heater_t_loss": Param("dln_alpha_dt_heater", LEAF_DLN_ALPHA_DT_HEATER, 0.0, 0.005, 0.0016, unit="1/K",
                           prior_source="leaf range [0, 0.005]"),
}
# alpha_other (dB/cm) of the two coupling branches at Q_L 3850 in g26 (used as starts)
BRANCH_START_ALPHA_OTHER_DB_PER_CM = {"under": 13.35, "over": 4.71}
# cold coupling ratio x0 that gives the 27.9 dB CAL-2 notch at 0 V on each branch (used as starts) - in the coupled-mode picture the through-port minimum is ((1 - x0) / (1 + x0))^2, about 10^(-2.79) for both values
_X0_OF_0V_NOTCH = {"under": 0.9225, "over": 1.084}


DB_PER_CM_TO_PER_M = 100.0 * math.log(10.0) / 10.0      # power loss in dB/cm -> attenuation alpha in 1/m


# what maps (Q_L, alpha_other) to the cold 0 V coupling ratio x0 of the bench ring.
# x0 = kappa_e / (kappa_i + kappa_d) — the through-coupler decay rate kappa_e over the sum of the intrinsic loss
# rate kappa_i and the drop-coupler rate kappa_d.
# kappa = kappa_e + kappa_i + kappa_d is the total decay rate
# (1/Q_L is proportional to kappa).
# x0 > 1 is the over-coupled branch.
# with the loss budget (relation R4 of mrm/params/relations.py) —
# Q_i = 2 pi n_g / (lambda alpha_i),   alpha_i = alpha_rest + alpha_other
# Q_d = Q_L / (kappa_d / kappa)
# 1/Q_t = 1/Q_L - 1/Q_i - 1/Q_d        (Q_t — the through coupler)
# fields — n_g (group index), lambda_m (cold resonance wavelength, m), alpha_rest_per_m (intrinsic attenuation without alpha_other, 1/m), kd_over_k (kappa_d / kappa).
@dataclasses.dataclass(frozen=True)
class CouplingConstants:

    n_g: float
    lambda_m: float
    alpha_rest_per_m: float
    kd_over_k: float

    # the bench ring's n_g, cold resonance, alpha_rest and kappa_d/kappa, from the context's bench config.
    @classmethod
    def from_context(cls, ctx: Context) -> "CouplingConstants":
        cfg = ctx.config(BENCH)
        _, plant = replicas.resolve_plant(cfg)
        optics = plant.optics
        alpha_other_db_per_cm = float(cfg.get(LEAF_ALPHA_OTHER))
        return cls(n_g=float(np.asarray(optics.n_g).ravel()[0]),
                   lambda_m=float(np.asarray(optics.lambda_res_cold_m).ravel()[0]),
                   alpha_rest_per_m=float(np.asarray(optics.alpha_i_per_m).ravel()[0])
                   - alpha_other_db_per_cm * DB_PER_CM_TO_PER_M,
                   kd_over_k=float(cfg.get("ring.optics.kappa_d_over_kappa")))

    # q_i at the given alpha_other (dB/cm).
    def q_intrinsic(self, alpha_other_db_per_cm: float) -> float:
        alpha_per_m = self.alpha_rest_per_m + alpha_other_db_per_cm * DB_PER_CM_TO_PER_M
        return 2.0 * math.pi * self.n_g / (self.lambda_m * alpha_per_m)

    # the cold coupling ratio x0 = (1/Q_t) / (1/Q_i + 1/Q_d).
    def x0(self, q_loaded: float, alpha_other_db_per_cm: float) -> float:
        qi = self.q_intrinsic(alpha_other_db_per_cm)
        inv_qd = self.kd_over_k / q_loaded
        inv_qt = 1.0 / q_loaded - 1.0 / qi - inv_qd
        return inv_qt / (1.0 / qi + inv_qd)


# penalty scale of the branch constraint - one unit of residual per 0.2 % of x0 on the wrong side of critical coupling
BRANCH_WALL = 0.002


# fit loaded Q and the heater-dependent notch for each coupling hypothesis.
# free Q_L, other intrinsic loss, and one notch-change coefficient.
# hold the other mechanisms fixed.
# an optional instrument floor explains saturation of the deepest notch read-offs.
# the replica uses the plant's add-drop topology at negligible optical power.
# under- and over-coupled cold states are separate hypotheses.
# a constraint penalizes the wrong side of critical coupling.
# loss hypotheses explicitly restore solver dispersion so a previously fitted dispersion cannot enter them.
# fit bounds without a declared prior do not establish identifiability.
def coupling_er_problems(ctx: Context, targets: Mapping[str, Any], *, floor: bool = True,
                         mechanisms=("dispersion", "ring_t_loss", "heater_t_loss"),
                         regimes=("under", "over")) -> list[Problem]:
    cal1, cal2 = targets["CAL-1"], targets["CAL-2"]
    coupling = CouplingConstants.from_context(ctx)
    disp_solver = float(replicas.load_config(ctx.base(BENCH)).get(LEAF_COUPLER_DISPERSION))
    probs = []
    for mech in mechanisms:
        for regime in regimes:
            for use_floor in ((False, True) if floor else (False,)):
                mech_param = ER_MECHANISMS[mech]
                params = [Param("q_loaded", LEAF_Q_LOADED, 2600.0, 4200.0, 3850.0,
                                prior_source="leaf range [2600, 4200]"),
                          Param("alpha_other_abs", LEAF_ALPHA_OTHER, 0.0, 30.0,
                                BRANCH_START_ALPHA_OTHER_DB_PER_CM[regime], unit="db_per_cm",
                                prior_source="leaf range [0, 30]"),
                          mech_param]
                if use_floor:
                    params.append(Param("notch_floor_db", None, 28.0, 60.0, 31.0, unit="db", role="nuisance",
                                        prior_source="assumed instrument floor 28-60 dB (60 = no floor)"))
                fixed = {LEAF_DLN_ALPHA_DT_RING: 0.0, LEAF_DLN_ALPHA_DT_HEATER: 0.0,
                         LEAF_COUPLER_DISPERSION: disp_solver}
                fixed.pop(mech_param.path, None)

                def predict(th, mech_param=mech_param, fixed=fixed, use_floor=use_floor):
                    ov = dict(fixed)
                    ov.update({LEAF_Q_LOADED: th["q_loaded"], LEAF_ALPHA_OTHER: th["alpha_other_abs"],
                               mech_param.path: th[mech_param.name]})
                    cfg = ctx.config(BENCH, ov)
                    q = replicas.loaded_q_from_3db_width(cfg, cal1).value
                    er = replicas.notch_er_vs_heater_v(cfg, cal2, topology=None).value
                    if use_floor:
                        er = notch_floor(er, th["notch_floor_db"])
                    return {"CAL-1": q, "CAL-2": er}

                def constraint(th, regime=regime):
                    # a wall on log(x0) - zero on the hypothesis' own branch, rising on the other side
                    log_x0 = math.log(max(coupling.x0(th["q_loaded"], th["alpha_other_abs"]), 1e-9))
                    wrong_side = max(log_x0, 0.0) if regime == "under" else max(-log_x0, 0.0)
                    return np.array([wrong_side / BRANCH_WALL])

                def derived(th, regime=regime):
                    qi = coupling.q_intrinsic(th["alpha_other_abs"])
                    x0 = coupling.x0(th["q_loaded"], th["alpha_other_abs"])
                    return {"x0_cold_0v": x0, "branch_at_0v": "over" if x0 > 1 else "under", "q_intrinsic": qi,
                            "q_loaded_over_q_intrinsic": th["q_loaded"] / qi}
                hyp = f"{regime}/{mech}" + ("/floor" if use_floor else "")
                # starts - the g26 branch value, and the 27.9 dB notch at 0 V on this branch at Q_L 3700 and 4000
                starts = [{}] + [{"q_loaded": q, "alpha_other_abs": _alpha_for_x0(coupling, q, _X0_OF_0V_NOTCH[regime])}
                                 for q in (3700.0, 4000.0)]
                probs.append(Problem(group="coupling_er", hypothesis=hyp, params=params, targets=[cal1, cal2],
                                     predict=predict, fixed={"preset": BENCH, **fixed, "topology": "add_drop (own)"},
                                     starts=starts,
                                     constraint=constraint, derived=derived))
    return probs


# alpha_other (dB/cm, clipped to [0, 30]) that gives the cold coupling ratio x0 at Q_L (the inverse of `CouplingConstants.x0`).
def _alpha_for_x0(coupling: CouplingConstants, q_loaded: float, x0: float) -> float:
    # kappa_i + kappa_d = kappa / (1 + x0), so 1/Q_i = (1/(1 + x0) - kappa_d/kappa) / Q_L
    inv_qi = (1.0 / (1.0 + x0) - coupling.kd_over_k) / q_loaded
    alpha_i_per_m = 2.0 * math.pi * coupling.n_g * inv_qi / coupling.lambda_m
    return float(min(max((alpha_i_per_m - coupling.alpha_rest_per_m) / DB_PER_CM_TO_PER_M, 0.0), 30.0))


# CAL-3a/b + CAL-4 - heater P_FSR (the anchor), cold R, droop y = TCR x R_th,H
# thermo-optic worlds in which the heater group is fitted.
# H_solver is the nominal - H86 (a constant 86 pm/K) is a required robustness corner of the design (docs/research/07_design_critique.md, decision "Which thermo-optic quantity is independent, and its default").
# H86_sensor keeps the H_solver plant (only the reported degC differs).
# the weight-0 alternatives (bench, imec, ocp, textbook_derived) are not fitted - their worlds keep the H_solver heater leaves.
HEATER_THERMO = ("H_solver", "H86")

# CAL-4 - every shift is the difference of its own read-off and the SAME 0 V read-off (the target's verification note - "sigma = sqrt(2) x 0.03 (or 0.05)"), so the 14 shifts share one error.
# the diagonal keeps the target's sigma (0.04 / 0.06 nm, rounded).
# the shared part is set so that the independent part of the 0.04 nm points is their stated own read-off error of 0.03 nm - sqrt(0.04^2 - 0.03^2) = 0.026 nm.
# Before the rounding of 0.0424 to 0.04 it would be 0.03 nm exactly - the smaller value does not overstate the shape information that identifies the droop.
CAL4_COMMON_SIGMA_NM = math.sqrt(0.04 ** 2 - 0.03 ** 2)


# fit heater medians and the separate device's voltage-sweep shape.
# free leaves are P_FSR, cold resistance, and heater_self_ratio.
# the last sets
# y = TCR x R_th,H with the family's TCR.
# the data identify this product, not TCR and thermal resistance separately.
# the swept device gets its own P_FSR and resistance ratio.
# its log resistance prior comes from device variation.
# all 14 shifts share the same 0 V read-off error, represented by CAL4_COMMON_SIGMA_NM and whitened in the fit.
# `same_rth_heater` holds heater thermal resistance equal across the devices.
# `same_ratio` holds their ratio leaf equal.
# fit both in each thermo-optic world because temperature-dependent tuning also changes the sweep curvature.
# scratch ranges permit predictions for the swept device.
# they are not median priors.
# the resistance fit can extend to 35 ohm, but the half-prior rule uses the narrower declared range.
# the FSR anchor and bench FSR protocol also differ slightly because they refer to different wavelength positions.
def heater_problems(ctx: Context, targets: Mapping[str, Any],
                    hypotheses=("same_rth_heater", "same_ratio"), thermo=HEATER_THERMO) -> list[Problem]:
    cal3a, cal3b, cal4 = targets["CAL-3a"], targets["CAL-3b"], targets["CAL-4"]
    # Plausible ranges of the scratch child config.
    # the swept device's own R0 (the median R0 x the nuisance ratio below) and its ratio under same_rth_heater (ratio x P_FSR,dev / P_FSR) may leave the median's ranges.
    # the fitted median leaves keep their priors (the calibrated preset restates the R0 leaf with the fit prior [35, 58]).
    for path, (lo, hi) in ((LEAF_R_COLD, (25.0, 80.0)), (LEAF_HEATER_SELF_RATIO, (1.0, 8.0))):
        old = ctx.ranges.get(path, [lo, hi])
        ctx.ranges[path] = [min(float(old[0]), lo), max(float(old[1]), hi)]
    # the swept device's cold resistance is not the median's.
    # CAL-4 (a voltage sweep) identifies the droop y / R0 of that device, so y (and the median's R0 = R_hot / (1 + y P_FSR), and the ratio) depend on its R0.
    # the preset's variation entry gives the spread from device to device (lognormal, die and ring sigma).
    # it enters as a Gaussian prior on ln(R0_dev / R0) - a constraint residual, not a datum. 0.06 is the log sigma when the entry gives none.
    bench_cfg = replicas.load_config(ctx.base(BENCH))
    r0_variation = bench_cfg.variation.get(LEAF_R_COLD)
    variation_fields = dict(getattr(r0_variation, "fields", {}) or {})
    r0_log_sigma = math.hypot(float(variation_fields.get("die_sigma", 0.0)),
                              float(variation_fields.get("ring_sigma", 0.0))) or 0.06
    # the R0 leaf's declared range ([37, 58] ohm - the G25 hot range divided by 1 + y P_FSR, for every droop that the thermo-optic worlds need).
    # the fit bounds reach 35 ohm - the half-prior rule takes the declared width when it is narrower (Param.declared_range).
    # read it from the preset as written, not from bench_cfg - bench_cfg is the scratch child whose R0 range was widened to [25, 80] above for the swept device, and that range is not a prior.
    r0_lo, r0_hi = (float(x) for x in replicas.load_config(BENCH).leaves[LEAF_R_COLD].range)
    probs = []
    for thermo_h, hyp in [(t, h) for t in thermo for h in hypotheses]:
        alts = {} if thermo_h == "H_solver" else {"uncertainty.thermo_optic": thermo_h}
        name = hyp if thermo_h == "H_solver" else f"{thermo_h}/{hyp}"
        params = [Param("p_fsr_mw", LEAF_P_FSR, 24.9, 37.0, 25.45, unit="mw", prior_source="leaf range [24.9, 37.0]"),
                  Param("r_cold_ohm", LEAF_R_COLD, min(35.0, r0_lo), max(58.0, r0_hi), 45.41, unit="ohm",
                        declared_range=(r0_lo, r0_hi),
                        prior_source=f"leaf range [{r0_lo:g}, {r0_hi:g}] (R_hot / (1 + y P_FSR) for the droops of "
                                     f"the thermo-optic worlds, its own note) widened to 35 ohm, as R0 trades against "
                                     f"y; the half-prior rule is judged on the narrower declared width"),
                  Param("heater_self_ratio", LEAF_HEATER_SELF_RATIO, 1.0, 4.0, 1.158, prior_source="leaf range [1, 4]"),
                  Param("p_fsr_swept_device_mw", None, 22.0, 40.0, 29.5, unit="mw", role="nuisance",
                        prior_source="assumed [22, 40] mW around the G25 60-die range 24.9-34.1 mW"),
                  Param("r_cold_swept_device_rel", None, math.exp(-4 * r0_log_sigma), math.exp(4 * r0_log_sigma), 1.0,
                        log=True, role="nuisance",
                        prior_source=f"R0_dev / R0: Gaussian prior on its log, sigma {r0_log_sigma:.3g} (variation "
                                     f"entry of {LEAF_R_COLD}: die and ring sigma), bounds +-4 sigma")]

        def predict(th, hyp=hyp, alts=alts):
            median_leaves = {LEAF_P_FSR: th["p_fsr_mw"], LEAF_R_COLD: th["r_cold_ohm"],
                             LEAF_HEATER_SELF_RATIO: th["heater_self_ratio"]}
            cfg = ctx.config(BENCH, median_leaves, alts=alts)
            fsr_result = replicas.heater_power_per_fsr(cfg, cal3a)
            ratio_dev = th["heater_self_ratio"] * (th["p_fsr_swept_device_mw"] / th["p_fsr_mw"]
                                                   if hyp == "same_rth_heater" else 1.0)
            device_leaves = dict(median_leaves, **{LEAF_P_FSR: th["p_fsr_swept_device_mw"],
                                                   LEAF_HEATER_SELF_RATIO: ratio_dev,
                                                   LEAF_R_COLD: th["r_cold_ohm"] * th["r_cold_swept_device_rel"]})
            sweep_result = replicas.resonance_vs_heater_v(ctx.config(BENCH, device_leaves, alts=alts), cal4)
            return {"CAL-3a": fsr_result.value, "CAL-3b": fsr_result.variants["voltage_v"],
                    "CAL-4": sweep_result.value}

        def constraint(th, r0_log_sigma=r0_log_sigma):
            return np.array([math.log(th["r_cold_swept_device_rel"]) / r0_log_sigma])

        def derived(th, hyp=hyp, alts=alts, thermo_h=thermo_h):
            cfg = ctx.config(BENCH, {LEAF_P_FSR: th["p_fsr_mw"], LEAF_R_COLD: th["r_cold_ohm"],
                                     LEAF_HEATER_SELF_RATIO: th["heater_self_ratio"]},
                             alts=alts)
            _, plant = replicas.resolve_plant(cfg)
            z_k_per_w = float(np.asarray(plant.thermal.z_heater_ring_k_per_w).ravel()[0])  # heater -> ring, K/W
            tcr = float(np.asarray(plant.heater.tcr_per_k).ravel()[0])                     # 1/K
            fsr_nm = float(replicas._fsr_m(replicas.StaticRing(plant)) * 1e9)
            y_per_w = tcr * th["heater_self_ratio"] * z_k_per_w                            # y = TCR x R_th,H, 1/W
            return {"z_heater_ring_k_per_mw": z_k_per_w * 1e-3, "tcr_per_k": tcr, "y_per_mw": y_per_w * 1e-3,
                    "r_th_heater_k_per_mw": th["heater_self_ratio"] * z_k_per_w * 1e-3,
                    "efficiency_median_nm_per_mw": fsr_nm / th["p_fsr_mw"],
                    "efficiency_swept_device_nm_per_mw": fsr_nm / th["p_fsr_swept_device_mw"],
                    "r_hot_at_p_fsr_ohm": th["r_cold_ohm"] * (1.0 + y_per_w * th["p_fsr_mw"] * 1e-3),
                    "r_cold_swept_device_ohm": th["r_cold_ohm"] * th["r_cold_swept_device_rel"],
                    "swept_device_droop_hypothesis": hyp, "thermo_optic": thermo_h}
        probs.append(Problem(group="heater", hypothesis=name, params=params, targets=[cal3a, cal3b, cal4],
                             predict=predict,
                             fixed={"preset": BENCH, "thermo_optic": thermo_h,
                                    "swept_device_r_cold": f"median x nuisance (log prior sigma {r0_log_sigma:.3g})",
                                    "driver": "bench voltage source, TCR solved",
                                    "cal4_common_readoff_sigma_nm": CAL4_COMMON_SIGMA_NM},
                             starts=[{}, {"heater_self_ratio": 2.0, "p_fsr_swept_device_mw": 31.0}],
                             constraint=constraint, derived=derived,
                             common_sigma={"CAL-4": CAL4_COMMON_SIGMA_NM}))
    return probs


# CAL-7 - the cold blue offset
# cAL-7 — `ring.optics.resonance_offset_nm` (prior — the leaf range [5, 10] nm) from the low-power 0 V spectrum (the replica's primary protocol).
# the geometry relation R3 (mrm/params/relations.py) places the cold resonance by this leaf, so the fit only confirms the definition.
def offset_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    cal7 = targets["CAL-7"]
    params = [Param("resonance_offset_nm", LEAF_RESONANCE_OFFSET, 5.0, 10.0, 7.6, unit="nm",
                    prior_source="leaf range [5, 10]")]

    def predict(th):
        cfg = ctx.config(BENCH, {LEAF_RESONANCE_OFFSET: th["resonance_offset_nm"]})
        return {"CAL-7": replicas.cold_resonance_offset(cfg, cal7).value}
    return [Problem(group="offset", hypothesis="cold_cavity", params=params, targets=[cal7], predict=predict,
                    fixed={"preset": BENCH, "p_in": "-> 0 (cold cavity)", "heater_v": 0.0})]


# CAL-5 - DC junction efficiency (analytic law) - LUT-mode fill fraction as a variant
# the preset `preset` and its `extends` chain as one nested mapping (no `extends`) that `params.load` accepts.
# the merge is the loader's own (`mrm.params.config._load_chain`), so the mapping loads to the same leaves, uncertainty and variation entries as the preset name.
# for example a child's `alternatives: {name: null}` removes the parent's alternative, and each field that a child's uncertainty entry states replaces the parent's field as a whole (`corners: {}` removes every corner).
def _merged_preset_mapping(preset: str) -> dict:
    from mrm.params import config
    errors: list[str] = []
    flat, _ = config._load_chain(config._resolve_file(preset), (), errors)
    if errors:
        raise config.ConfigError(errors)
    mapping: dict = {}
    for path, node in flat.nodes.items():
        *parents, last = path.split(".")
        section = mapping
        for k in parents:
            section = section.setdefault(k, {})
        section[last] = copy.deepcopy(node.bare if node.is_bare else node.leaf)
    if flat.uncertainty:
        mapping["uncertainty"] = copy.deepcopy(flat.uncertainty)      # {dotted entry name: entry}
    variation = copy.deepcopy(flat.variation)                          # {config path: entry}
    if flat.geometry is not None:
        variation["geometry"] = copy.deepcopy(flat.geometry)
    if variation:
        mapping["variation"] = variation
    if flat.alts:
        mapping["alts"] = dict(flat.alts)
    if flat.relations:
        mapping["relations"] = copy.deepcopy(flat.relations)
    mapping["meta"] = copy.deepcopy(flat.meta)
    return mapping


# build a merged bench preset that uses the junction lookup table.
# remove analytic-junction keys, then add the LUT path.
# a child preset cannot remove inherited keys, so use the loader's merged mapping.
# `ranges` may widen allowed ranges for the reported fill-fraction variant.
def bench_lut_mapping(lut_file: str = "extracted/junction_rib160.json",
                      ranges: Mapping[str, Sequence[float]] | None = None) -> dict:
    mapping = _merged_preset_mapping(BENCH)
    mapping["meta"] = dict(mapping.get("meta", {}), name=f"{BENCH}_lut")
    mapping["fidelity"]["junction_model"] = "lut"
    junction = mapping["ring"]["junction"]
    for k in ("eff_at_0v_pm_per_v", "v_bi_v", "depletion_exponent", "dqi_dv_frac_per_v"):
        junction.pop(k, None)
    junction["lut_file"] = lut_file
    for path, rng in (ranges or {}).items():
        node = mapping
        for k in path.split("."):
            node = node[k]
        node["range"] = [float(rng[0]), float(rng[1])]
    return mapping


# measurement-power hypotheses of CAL-5 (the G25 sweep power is not printed).
# the replica's primary is the printed G25 input of +4 dBm at the ring (self-heating relief included) - the alternative is P_in -> 0 (no relief).
# {hypothesis name: P_in at the ring in dBm, or None for P_in -> 0}
CAL5_POWERS = {"p_in_4dbm": 4.0, "p_in_0": None}


# problems of the junction group — CAL-5 (DC resonance shift against reverse bias at 0.8, 1.2 and 1.6 V).
# in the analytic junction mode of g26 (`fidelity.junction_model: analytic`), the ring-level shift is `eff_at_0v_pm_per_v` times the depletion shape, and `fill_fraction` cancels.
# (The analytic table is built per fill, so the ring-level efficiency, capacitance and free-carrier absorption are the configured ones.) So CAL-5 identifies `eff_at_0v_pm_per_v` there (prior — the leaf range [50, 95] pm/V).
# the shape (V_bi 0.9 V, exponent 1/2) is held.
# in LUT mode (variant `lut/...`, the rib160 junction artefact) the shift scales with `fill_fraction` (prior — the leaf range [0.5, 0.95]), the leaf that the design plan (docs/research/07_design_critique.md) calibrates to CAL-5.
# each mode is fitted at both measurement powers (`CAL5_POWERS`).
def junction_problems(ctx: Context, targets: Mapping[str, Any], *, lut_variant: bool = True) -> list[Problem]:
    cal5 = targets["CAL-5"]
    probs = []
    for power_name, p_in_dbm in CAL5_POWERS.items():
        params = [Param("eff_at_0v_pm_per_v", LEAF_JUNCTION_EFF, 50.0, 95.0, 72.5, unit="pm_per_v",
                        prior_source="leaf range [50, 95]")]

        def predict(th, p_in_dbm=p_in_dbm):
            cfg = ctx.config(BENCH, {LEAF_JUNCTION_EFF: th["eff_at_0v_pm_per_v"]})
            return {"CAL-5": replicas.resonance_shift_vs_bias(cfg, cal5, p_in_dbm=[p_in_dbm]).value}
        probs.append(Problem(group="junction", hypothesis=f"analytic/{power_name}", params=params, targets=[cal5],
                             predict=predict, fixed={"preset": BENCH, "junction_model": "analytic",
                                                     "p_in_dbm": p_in_dbm, "v_bi_v": "preset", "exponent": "preset"}))
    if lut_variant:
        from mrm import params as mrm_params
        mapping = bench_lut_mapping(ranges=ctx.ranges)
        for power_name, p_in_dbm in CAL5_POWERS.items():
            params = [Param("fill_fraction", LEAF_FILL_FRACTION, 0.5, 0.95, 0.85,
                            prior_source="leaf range [0.5, 0.95]")]

            def predict(th, p_in_dbm=p_in_dbm):
                ov = {k: v for k, v in ctx.overrides(BENCH).items() if k not in (LEAF_JUNCTION_EFF,)}
                ov[LEAF_FILL_FRACTION] = th["fill_fraction"]
                cfg = mrm_params.load(mapping, overrides=ov)
                return {"CAL-5": replicas.resonance_shift_vs_bias(cfg, cal5, p_in_dbm=[p_in_dbm]).value}
            probs.append(Problem(group="junction", hypothesis=f"lut/{power_name}", params=params, targets=[cal5],
                                 predict=predict, fixed={"preset": f"{BENCH} (mapping, LUT mode)",
                                                         "junction_model": "lut",
                                                         "lut_file": "extracted/junction_rib160.json",
                                                         "p_in_dbm": p_in_dbm}))
    return probs


# CAL-10 - the junction R C (waveform)
# problems of the eo group — CAL-10 (EO bandwidth 30.9 GHz at 1 V reverse bias).
# one number cannot separate R_s from C_j — it fixes the product R_s x C together with the photon-lifetime limit.
# so there are three hypotheses — `rs_free` (C_j(0) held at its assumed 15 fF, R_s free in the leaf range [15, 390] ohm, log), `c_free` (R_s held at 100 ohm, C_j(0) free in [6, 30] fF, log) and `both_free` (both free, for the profile — its valley is the identified product).
# the detuning is the replica's primary (maximum slope).
# the half-width detuning is recorded as a variant prediction.
def eo_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    cal10 = targets["CAL-10"]
    series_r = Param("series_resistance_ohm", LEAF_SERIES_RESISTANCE, 15.0, 390.0, 150.0, unit="ohm", log=True,
                     prior_source="leaf range [15, 390]")
    cap_0v = Param("capacitance_at_0v_ff", LEAF_CAPACITANCE_0V, 6.0, 30.0, 15.0, unit="ff", log=True,
                   prior_source="leaf range [6, 30]")
    probs = []
    for hyp, params in (("rs_free", [series_r]), ("c_free", [cap_0v]), ("both_free", [series_r, cap_0v])):
        def predict(th):
            ov = {p.path: th[p.name] for p in (series_r, cap_0v) if p.name in th}
            r = replicas.eo_bandwidth_at_bias(ctx.config(BENCH, ov), cal10, with_variants=False)
            return {"CAL-10": r.value}

        def derived(th):
            ov = {p.path: th[p.name] for p in (series_r, cap_0v) if p.name in th}
            r = replicas.eo_bandwidth_at_bias(ctx.config(BENCH, ov), cal10)
            max_slope = r.detail.get("max_slope", {})
            return {"variants": {k: v for k, v in r.variants.items()},
                    "r_s_ohm_at_1v": max_slope.get("r_s_ohm"), "c_j_ff_at_1v": (max_slope.get("c_j_f") or 0.0) * 1e15,
                    "rc_pole_ghz": (max_slope.get("rc_pole_hz") or 0.0) / 1e9,
                    "fwhm_ghz": (max_slope.get("fwhm_hz") or 0.0) / 1e9}
        probs.append(Problem(group="eo", hypothesis=hyp, params=list(params), targets=[cal10], predict=predict,
                             fixed={"preset": BENCH, "detuning": "max_slope (primary)", "r_drv_ohm": "plant (50)"},
                             derived=derived))
    return probs


# CAL-8 - the lumped monitor gain
# lock set-point hypotheses of CAL-8 (the replica's primary first - the list was fixed before any comparison)
CAL8_SETPOINTS = ("half_max", "peak", "oracle")


# problems of the monitor group — CAL-8 (15.5 uA monitor current held at lock during the aggressor).
# the current identifies only the product gain x line power x lock factor (the line power at the ring, the drop coupling, the responsivity and the readout gain stay nuisances).
# so the fitted leaf is `ring.monitor.gain_scale`, the lumped nuisance on the monitor gain (prior — uncertainty.monitor.gain_scale, [0.5, 2], log-uniform), once per set-point hypothesis.
# the replica — the bank locked with every line on (bus-filtered), NRZ at the scenario bias, the preset line power, 25 degC base.
def monitor_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    cal8 = targets["CAL-8"]
    probs = []
    for setpoint in CAL8_SETPOINTS:
        params = [Param("gain_scale", LEAF_MONITOR_GAIN, 0.5, 2.0, 1.0, log=True,
                        prior_source="uncertainty.monitor.gain_scale [0.5, 2] (log-uniform)")]

        def predict(th, setpoint=setpoint):
            r = replicas.monitor_current_at_lock(ctx.config(BANK, {LEAF_MONITOR_GAIN: th["gain_scale"]}), cal8,
                                                 setpoint=setpoint)
            return {"CAL-8": r.value}
        probs.append(Problem(group="monitor", hypothesis=f"setpoint={setpoint}", params=params, targets=[cal8],
                             predict=predict, fixed={"preset": BANK, "setpoint": setpoint, "lines": "all",
                                                     "p_in": "preset line power"}))
    return probs


# CAL-14 - the lumped modulation depth
# DC bias (V) at which the CAL-14 scan reports the resonance swing and the NRZ levels - the middle of the CAL-14 bias window 1.0-1.15 V (the replica itself averages over the whole window)
CAL14_BIAS_V = 1.075


# problems of the modulation_depth group — CAL-14 (stabilized relOMA -2.1 dB at 1.0-1.15 V and 0-105 degC. relOMA = OMA / 1 mW into the MRM).
# the lumped depth (swing x efficiency / FWHM) is carried by `driver.swing_pp_v` (prior — the leaf range [0.9, 2.6] V), with the efficiency and Q_L of the earlier groups.
# Mode `oracle` is the replica's primary — the best achievable relOMA, so a depth fitted there is a LOWER bound if G26 locks off the optimum.
# Mode `setpoint_lock` locks at the half-maximum drop.
def modulation_depth_problems(ctx: Context, targets: Mapping[str, Any],
                              modes=("oracle", "setpoint_lock")) -> list[Problem]:
    cal14 = targets["CAL-14"]
    probs = []
    for mode in modes:
        params = [Param("swing_pp_v", LEAF_DRIVE_SWING, 0.9, 2.6, 1.3, unit="v", prior_source="leaf range [0.9, 2.6]")]

        def predict(th, mode=mode):
            r = replicas.relOMA_map(ctx.config(BANK, {LEAF_DRIVE_SWING: th["swing_pp_v"]}), cal14, mode=mode,
                                    with_variants=False)
            return {"CAL-14": r.value}
        probs.append(Problem(group="modulation_depth", hypothesis=f"mode={mode}", params=params, targets=[cal14],
                             predict=predict, fixed={"preset": BANK, "mode": mode, "p_in_dbm": 0.0,
                                                     "level_convention": "preset (centered)"}))
    return probs


# what lumped modulation depth closes CAL-14 — relOMA (the CAL-14 replica, `mode`) on a grid.
# Inside the swing prior, the preset's drive cannot reach the printed -2.1 dB (the known gap of docs/research/07_design_critique.md, decision "relOMA as an acceptance number". see docs/handoff/calibration.md).
# the grid — the swing (V, beyond the prior, to find the closing value) and a junction-efficiency scale s (eff_at_0v x s, at a 1.3 Vpp swing and at the preset's swing), for both NRZ level conventions.
# `alt_effs` ({name — eff_at_0v in pm/V}, for example the CAL-5 P_in -> 0 fit) adds a swing scan at each alternative efficiency (centered levels. the rows carry the name in `eff_base`).
# each row also gives the peak-to-peak resonance swing at the CAL-14 bias (`CAL14_BIAS_V`) and the depth D = swing_pm / FWHM_pm — the lumped modulation depth.
# the junction grid of the plant clips the levels at -0.5 V (forward) and 4.0 V.
def modulation_depth_scan(ctx: Context, targets: Mapping[str, Any], *,
                          swings=(0.9, 1.3, 1.8, 2.2, 2.6, 3.1, 3.6, 4.5, 5.0, 5.5),
                          eff_scales=(1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
                          conventions=("centered", "one_sided"), mode: str = "oracle",
                          alt_effs: Mapping[str, float] | None = None) -> dict:
    import tempfile
    cal14 = targets["CAL-14"]
    rows = []
    base_cfg = ctx.config(BANK)
    base_efficiency = float(base_cfg.get(LEAF_JUNCTION_EFF))
    # a scratch child of g26 that only widens the leaves' plausible ranges (same values and provenance), so the scan may leave the priors - it is deleted after the scan and never written to the package
    scratch = tempfile.TemporaryDirectory(prefix="mrm_cal14_scan_")
    widened_path = widened_child(BANK, {**ctx.ranges, LEAF_DRIVE_SWING: [0.1, 10.0], LEAF_JUNCTION_EFF: [1.0, 900.0]},
                                 scratch.name)

    def row(convention, swing_v, eff_scale, eff_base):
        ov = {LEAF_DRIVE_SWING: swing_v, LEAF_JUNCTION_EFF: base_efficiency * eff_scale,
            "driver.level_convention": convention}
        cfg = ctx.config(widened_path, ov)
        r = replicas.relOMA_map(cfg, cal14, with_variants=False, mode=mode)
        _, plant = replicas.resolve_plant(cfg)
        ring = replicas.StaticRing(plant)
        lo, hi = ring.nrz_levels(CAL14_BIAS_V)
        swing_pm = float((ring.junction_shift_m(hi)[0] - ring.junction_shift_m(lo)[0]) * 1e12)
        fwhm_pm = float(ring.lam_line / ring.q_loaded * 1e12)
        return {"level_convention": convention, "swing_pp_v": swing_v, "eff_scale": eff_scale, "eff_base": eff_base,
                "eff_at_0v_pm_per_v": base_efficiency * eff_scale, "levels_v": [lo, hi], "swing_pm": swing_pm,
                "fwhm_pm": fwhm_pm, "depth": swing_pm / fwhm_pm, "reloma_db": r.value}
    for convention in conventions:
        for swing_v in swings:
            for eff_scale in eff_scales:
                # the efficiency scales run only at a 1.3 Vpp swing and at the preset's swing - other swings only at scale 1
                if eff_scale != 1.0 and swing_v != 1.3 and swing_v != float(base_cfg.get(LEAF_DRIVE_SWING)):
                    continue
                rows.append(row(convention, swing_v, eff_scale, "context"))
    for name, eff in (alt_effs or {}).items():
        for swing_v in swings:
            rows.append(row("centered", swing_v, float(eff) / base_efficiency, name))
    scratch.cleanup()
    return {"mode": mode, "eff_at_0v_base_pm_per_v": base_efficiency, "alt_effs": dict(alt_effs or {}), "rows": rows}


# CAL-9 - the heater DAC LSB under the assumed law
# the heater codes of the CAL-9 protocol, from the pieces of `mrm.validation.replicas.stabilizer_code_vs_T`.
# the pieces — the A1 aggressor calibrated on the die's thermal ROM, the periodic steady-state rises, the bank locked at half maximum with every line on, and the codes through the heater law, the TCR and the realized DAC.
# with `uniform`, the aggressor rise of the victim ring is applied to every lane (the replica's variant `aggressor_model=uniform`).
# returns the statistics of each die (swing, lane offsets, code range) and their medians over the `n_dies` dies.
def dac_codes(cfg, *, n_dies: int = 1, variation: str = "off", uniform: bool = True, seed: int = 0,
              referencing: str = "resonance") -> dict:
    _, plant = replicas.resolve_plant(cfg, batch=n_dies, uncertainty="nominal", variation=variation,
                                      seeds={"device": int(seed), "uncertainty": 0})
    scenario, aggressor_waveform, _site = replicas._calibrated_aggressor(plant, which="a1", referencing=referencing)
    rises = replicas._aggressor_rises(plant, aggressor_waveform)
    t_base = float(scenario.defaults.get("thermal.t_base_k", 298.15))
    bias = replicas._scenario_bias(cfg)
    die_stats, infeasible_samples = [], 0
    for b in range(plant.batch):
        rise_ring_k, rise_heater_k = rises["ring"][:, b, :], rises["heater"][:, b, :]
        if uniform:
            rise_ring_k = np.repeat(rise_ring_k[:, :1], plant.n_rings, axis=1)
            rise_heater_k = np.repeat(rise_heater_k[:, :1], plant.n_rings, axis=1)
        bank = replicas._bank_codes(plant, b, t_base=t_base, bias=bias, mode="half_max", agg_r=rise_ring_k,
                                    agg_h=rise_heater_k)
        infeasible_samples += int(np.sum(bank["infeasible"]))
        die_stats.append(replicas._code_stats(bank["codes"]))
    return {"swing": replicas._median_stats(die_stats, "swing"),
            "lane_offsets": replicas._median_stats(die_stats, "lane_offsets"),
            "code_range": replicas._median_stats(die_stats, "code_range"), "per_die": die_stats,
            "infeasible_samples": infeasible_samples,
            "victim_rise_k": [float(rises["ring"][:, :, 0].min()), float(rises["ring"][:, :, 0].max())]}


# thermo-optic hypotheses compared on CAL-9 - the code range implies dlambda/dT of 71-88 pm/K for V_FS 1.40-1.60 V (docs/research/06_textbook_audit.md, item E2), so the fitted V_FS depends on the thermo-optic hypothesis
CAL9_THERMO = ("H_solver", "H86")


# per-ring geometry sigmas of g26 (nm of width and film - variation.geometry, provenance assumed) - a split chosen to reproduce the Berkeley 2 nm local resonance spread
RING_SIGMA_PATHS = {"variation.geometry.width.ring_sigma_nm": 2.0, "variation.geometry.thickness.ring_sigma_nm": 1.3}
DIE_SIGMA_PATHS = ("variation.geometry.width.die_sigma_nm", "variation.geometry.thickness.die_sigma_nm")


# problems of the dac group — CAL-9a (per-lane swing), CAL-9b (lane offsets) and CAL-9c (code range) of ONE G26 die.
# Protocol — the pieces of the CAL-9 replica (`dac_codes`) with a uniform aggressor — every lane sees the victim's rise, as the slide's ~20 k swings on all 8 lanes show.
# (The replica's primary geometry, an aggressor 30 um from ring 0, gives 3.6 k to 22.8 k and cannot reach the lower end with any DAC parameter.) The prediction is the median over `n_dies` dies that share the die-level state and differ only in their per-ring draws.
# the die-level geometry sigma is set to 0, so the median describes the lane spread of one die.
# Free — `ring.heater.full_scale_v` (the voltage law, the g26 nominal. prior — the leaf range [1.4, 1.8] V) and `ring_sigma_scale` (multiplies the per-ring geometry sigmas 2.0 / 1.3 nm, set to give the Berkeley 2 nm local resonance spread. prior [0, 1.5]).
# Nuisance, never written — the blue offset of the measured die (prior [5, 10] nm. the die-level spread is 6 nm, so the one plotted die need not sit at the G25 CAL-7 value).
# Hypotheses — the thermo-optic world (the code range ratio depends on the lock heater powers), and the aggressor referencing (resonance trajectory at 85.86 pm/K, the CAL-13a default, or ring temperature).
def dac_problems(ctx: Context, targets: Mapping[str, Any], thermo=CAL9_THERMO,
                 referencing=("temperature", "resonance"), n_dies: int = 8) -> list[Problem]:
    cal9a, cal9b, cal9c = targets["CAL-9a"], targets["CAL-9b"], targets["CAL-9c"]
    probs = []
    for ref in referencing:
        for hyp in thermo:
            params = [Param("full_scale_v", LEAF_HEATER_FULL_SCALE, 1.4, 1.8, 1.45, unit="v",
                            prior_source="leaf range [1.4, 1.8]"),
                      Param("ring_sigma_scale", "variation.geometry.ring_sigma_scale", 0.0, 1.5, 0.3,
                            declared_prior=False,
                            prior_source="fit bounds [0, 1.5] x the Berkeley local spread (no declared range)"),
                      Param("die_resonance_offset_nm", LEAF_RESONANCE_OFFSET, 5.0, 10.0, 9.0, unit="nm",
                            role="nuisance",
                            prior_source="leaf range [5, 10] (the plotted die's offset is unpublished)")]

            def bank_config_for(th, hyp=hyp):
                ov = {LEAF_HEATER_FULL_SCALE: th["full_scale_v"], LEAF_RESONANCE_OFFSET: th["die_resonance_offset_nm"]}
                ov.update({k: v * th["ring_sigma_scale"] for k, v in RING_SIGMA_PATHS.items()})
                ov.update({k: 0.0 for k in DIE_SIGMA_PATHS})
                return ctx.config(BANK, ov, alts={"uncertainty.thermo_optic": hyp})

            def predict(th, ref=ref, bank_config_for=bank_config_for):
                d = dac_codes(bank_config_for(th), referencing=ref, n_dies=n_dies, variation="sample")
                return {"CAL-9a": d["swing"], "CAL-9b": d["lane_offsets"], "CAL-9c": d["code_range"]}

            def derived(th, ref=ref, bank_config_for=bank_config_for):
                d = dac_codes(bank_config_for(th), referencing=ref, n_dies=n_dies, variation="sample")
                return {"median_of_dies": {k: d[k] for k in ("swing", "lane_offsets", "code_range",
                                                              "infeasible_samples", "victim_rise_k")},
                        "per_die": d["per_die"]}
            probs.append(Problem(group="dac", hypothesis=f"voltage/{hyp}/{ref}", params=params,
                                 targets=[cal9a, cal9b, cal9c], predict=predict,
                                 fixed={"preset": BANK, "driver_mode": "voltage (g26 nominal)", "thermo_optic": hyp,
                                        "aggressor_referencing": ref, "aggressor_model": "uniform",
                                        "dies": f"{n_dies} draws of the per-ring variation, die sigma 0",
                                        "setpoint": "half_max"},
                                 starts=[{}, {"full_scale_v": 1.6, "ring_sigma_scale": 0.6,
                                              "die_resonance_offset_nm": 8.0}],
                                 derived=derived))
    return probs


# CAL-12b - the static line-error population (the preset's variation entry)
# die and line sigmas (GHz) of variation.laser.line_offset_ghz in g26
LINE_SIGMA_PATHS = {"variation.laser.line_offset_ghz.die_sigma": 4.0, "variation.laser.line_offset_ghz.line_sigma": 6.0}


# problems of the laser group — CAL-12b (median and quartiles of the Guide line-error population).
# Free — `laser.line_offset_ghz` (the centre. prior — the leaf range [-20, 20] GHz) and a scale on the die and line sigmas of variation.laser.line_offset_ghz (4 and 6 GHz. prior [0.25, 2], log. the die —line split is held).
# the replica samples 256 worlds (seed 0) and clips at the +-20 GHz requirement.
def laser_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    cal12b = targets["CAL-12b"]
    params = [Param("line_offset_ghz", LEAF_LINE_OFFSET, -20.0, 20.0, 0.0, unit="ghz",
                    prior_source="leaf range [-20, 20]"),
              Param("line_sigma_scale", "variation.laser.line_offset_ghz.sigma_scale", 0.25, 2.0, 1.0, log=True,
                    declared_prior=False,
                    prior_source="fit bounds [0.25, 2] x the preset's 4 / 6 GHz die / line sigma (the variation "
                                 "entry declares no range)")]

    def predict(th):
        ov = {LEAF_LINE_OFFSET: th["line_offset_ghz"]}
        ov.update({k: v * th["line_sigma_scale"] for k, v in LINE_SIGMA_PATHS.items()})
        r = replicas.laser_line_statistics(ctx.config(BANK, ov), cal12b)
        return {"CAL-12b": r.value}
    return [Problem(group="laser", hypothesis="gaussian_die_plus_line", params=params, targets=[cal12b],
                    predict=predict, fixed={"preset": BANK, "worlds": 256, "clip_ghz": 20.0})]
