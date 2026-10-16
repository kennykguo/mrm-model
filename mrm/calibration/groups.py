"""The fit groups of 07 section 4.3: which calibration targets identify which preset leaves, under which hypotheses.

Owner: calibration. Every prediction goes through a replica of :mod:`mrm.validation.replicas` on the preset the
target's generation runs on (G25 targets on ``g25_test_vehicle``, G26 targets on ``g26``), with the parameters of
the fit applied as overrides on top of the leaves already calibrated by earlier groups (:class:`Context`). Nothing
here reads or predicts any target other than the calibration targets.

Groups (07 4.3 "fit in groups that the targets can identify"):

=================  =================================  ==========================================================
group              targets                            free parameters (prior range)
=================  =================================  ==========================================================
coupling_er        CAL-1, CAL-2                       Q_L, alpha_other (sets Q_i, hence the coupling branch), the
                                                      extinction-mechanism coefficient of each hypothesis;
                                                      nuisance: an instrument notch floor (hypothesis ``floor``)
heater             CAL-3a, CAL-3b, CAL-4              P_FSR (anchor), cold R, heater_self_ratio (y = TCR R_th,H);
                                                      nuisance: the swept device's P_FSR (06 E4)
offset             CAL-7                              resonance_offset_nm
junction           CAL-5                              eff_at_0v_pm_per_v (analytic law) per measurement power;
                                                      the LUT-mode fill_fraction is a reported variant
eo                 CAL-10                             series resistance (and capacitance: only R C is identified)
monitor            CAL-8                              ring.monitor.gain_scale per lock set-point hypothesis
modulation_depth   CAL-14                             driver.swing_pp_v (lumped depth swing x eff / FWHM)
dac                CAL-9a, CAL-9b, CAL-9c             ring.heater.full_scale_v under the assumed voltage law, the
                                                      per-ring geometry sigma scale; nuisance: the die's offset
laser              CAL-12b                            laser.line_offset_ghz and the line-error sigma scale
=================  =================================  ==========================================================

Not fitted here: CAL-6 (the H86 hypothesis value, ``fit: false``), CAL-11 (a population range the preset already
spans), CAL-12a/c/d and CAL-13a/b (scenario bounds and scenario calibrations: their residuals are reported, the
scenario layer owns the parameters).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mrm.calibration.core import Param, Problem
from mrm.validation import replicas as R

__all__ = ["Context", "BENCH", "BANK", "coupling_er_problems", "heater_problems", "offset_problems",
           "junction_problems", "eo_problems", "monitor_problems", "modulation_depth_problems", "dac_problems",
           "laser_problems", "notch_floor", "modulation_depth_scan", "dac_codes", "bench_lut_mapping",
           "CouplingConstants", "widened_child"]

BENCH = "g25_test_vehicle"     # the G25 bench device (CAL-1..5, 7, 10, 11)
BANK = "g26"                   # the G26 bank (CAL-8, 9, 12, 14)

P_COUPLING = "ring.optics.loss_budget.alpha_other_abs_db_per_cm"
P_Q = "ring.optics.q_loaded"
P_DISP = "ring.optics.dln_kappa2_through_dlambda_per_nm"
P_CT_RING = "ring.optics.dln_alpha_dt_ring_per_k"
P_CT_HEATER = "ring.optics.dln_alpha_dt_heater_per_k"
P_PFSR = "ring.heater.p_fsr_mw"
P_R0 = "ring.heater.r_cold_ohm"
P_RATIO = "ring.thermal.heater_self_ratio"
P_OFFSET = "ring.optics.resonance_offset_nm"
P_EFF = "ring.junction.eff_at_0v_pm_per_v"
P_FILL = "ring.junction.fill_fraction"
P_RS = "ring.junction.series_resistance_ohm"
P_C0 = "ring.junction.capacitance_at_0v_ff"
P_GAIN = "ring.monitor.gain_scale"
P_SWING = "driver.swing_pp_v"
P_VFS = "ring.heater.full_scale_v"
P_LINE = "laser.line_offset_ghz"


@dataclasses.dataclass
class Context:
    """Leaves calibrated by earlier groups, applied as overrides to every later prediction.

    device: leaves of the device physics (G26 takes them from G25): applied on both presets. system: G26 system
    leaves (monitor gain, DAC, laser population, drive): applied on the bank preset only."""

    device: dict[str, Any] = dataclasses.field(default_factory=dict)
    system: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: leaves whose plausible range a fit prior widens (path -> [lo, hi]); config() then loads a scratch child of
    #: the preset that restates those leaves (same value and provenance) with the wider range, because an override
    #: must lie inside the leaf's own range
    ranges: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    _tmp: Any = dataclasses.field(default=None, repr=False)
    _files: dict = dataclasses.field(default_factory=dict, repr=False)

    def overrides(self, preset, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        o = dict(self.device)
        if str(preset) != BENCH:
            o.update(self.system)
        o.update(extra or {})
        return o

    def base(self, preset):
        """The preset to load: the preset itself, or its scratch child with the widened ranges."""
        if not self.ranges or not isinstance(preset, str):
            return preset
        key = (preset, tuple(sorted((k, tuple(v)) for k, v in self.ranges.items())))
        if key not in self._files:
            self._files[key] = widened_child(preset, self.ranges, self._scratch())
        return self._files[key]

    def _scratch(self):
        import tempfile
        from pathlib import Path
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="mrm_calibration_")
        return Path(self._tmp.name)

    def config(self, preset, extra: Mapping[str, Any] | None = None, alts: Mapping[str, str] | None = None):
        return R.load_config(self.base(preset), overrides=self.overrides(preset, extra), alts=dict(alts or {}))


def widened_child(preset: str, ranges: Mapping[str, Sequence[float]], directory) -> "Path":
    """Write a scratch child preset of ``preset`` that restates each leaf of ``ranges`` with its own value,
    provenance, generation and source and the given plausible range (nothing else changes). Returns its path."""
    from pathlib import Path

    import yaml

    from mrm.params import config as C
    cfg = R.load_config(preset)
    doc: dict = {"extends": str((Path(C.PRESET_DIR) / f"{preset}.yaml").resolve()),
                 "meta": {"name": f"{preset}_widened", "kind": "preset"}}
    for path, rng in ranges.items():
        lf = cfg.leaf(path)
        node = doc
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = {"value": float(lf.value), "provenance": lf.provenance, "generation": lf.generation,
                          "source": lf.source, "range": [float(rng[0]), float(rng[1])]}
    out = Path(directory) / f"{preset}_widened_{len(list(Path(directory).glob('*.yaml')))}.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


# =====================================================================================================================
# CAL-1 + CAL-2: Q_L, the coupling branch and the extinction mechanism (07 P1-5)
# =====================================================================================================================
def notch_floor(depth_db, floor_db: float):
    """Notch depth seen through an instrument floor: the floor adds 10^(-F/10) to the through minimum."""
    d = np.asarray(depth_db, dtype=float)
    return (-10.0 * np.log10(10.0 ** (-d / 10.0) + 10.0 ** (-float(floor_db) / 10.0))).tolist()


#: the extinction mechanisms of uncertainty.er_mechanism and the leaf each one frees (07 P1-5)
ER_MECHANISMS = {
    # the leaf declares no range: 0-4 %/nm are fit bounds only (about 3x the femwell slope 1.3-1.5 %/nm, 05 t02;
    # rib160 artefact 0.77 %/nm). 07 P1-5's "~2.9 %/nm" must not justify them: it is derived from the 17 dB notch at
    # 13 nm of tuning, the held-out 1.5 V point (VAL-2). Without a declared prior the half-prior rule cannot apply.
    "dispersion": Param("dln_kappa2_through", P_DISP, 0.0, 0.04, 0.014, unit="1/nm", declared_prior=False,
                        prior_source="fit bounds 0-4 %/nm (no declared leaf range; ~3x femwell 1.3-1.5 %/nm, 05 "
                                     "t02; rib160 artefact 0.77 %/nm)"),
    "ring_t_loss": Param("dln_alpha_dt_ring", P_CT_RING, 0.0, 0.005, 0.002, unit="1/K",
                         prior_source="leaf range [0, 0.005]"),
    "heater_t_loss": Param("dln_alpha_dt_heater", P_CT_HEATER, 0.0, 0.005, 0.0016, unit="1/K",
                           prior_source="leaf range [0, 0.005]"),
}
#: alpha_other of the two coupling branches at Q_L 3850 (g26: under 13.35, over 4.71 dB/cm), used as starts
BRANCH_START = {"under": 13.35, "over": 4.71}


DB_PER_CM = 100.0 * math.log(10.0) / 10.0      # dB/cm of power loss -> alpha in 1/m


@dataclasses.dataclass(frozen=True)
class CouplingConstants:
    """What maps (Q_L, alpha_other) to the cold 0 V coupling ratio x0 = kappa_e / (kappa_i + kappa_d) of the bench
    ring (R4 and the loss budget): Q_i = 2 pi n_g / (lambda alpha_i), alpha_i = alpha_rest + alpha_other,
    Q_d = Q_L / (kappa_d/kappa), 1/Q_t = 1/Q_L - 1/Q_i - 1/Q_d. x0 > 1 is the over-coupled branch."""

    n_g: float
    lambda_m: float
    alpha_rest_per_m: float
    kd_over_k: float

    @classmethod
    def from_context(cls, ctx: Context) -> "CouplingConstants":
        cfg = ctx.config(BENCH)
        _, pp = R.resolve_plant(cfg)
        o = pp.optics
        ao = float(cfg.get(P_COUPLING))
        return cls(n_g=float(np.asarray(o.n_g).ravel()[0]), lambda_m=float(np.asarray(o.lambda_res_cold_m).ravel()[0]),
                   alpha_rest_per_m=float(np.asarray(o.alpha_i_per_m).ravel()[0]) - ao * DB_PER_CM,
                   kd_over_k=float(cfg.get("ring.optics.kappa_d_over_kappa")))

    def q_intrinsic(self, alpha_other_db_per_cm: float) -> float:
        a = self.alpha_rest_per_m + alpha_other_db_per_cm * DB_PER_CM
        return 2.0 * math.pi * self.n_g / (self.lambda_m * a)

    def x0(self, q_loaded: float, alpha_other_db_per_cm: float) -> float:
        qi = self.q_intrinsic(alpha_other_db_per_cm)
        inv_qd = self.kd_over_k / q_loaded
        inv_qt = 1.0 / q_loaded - 1.0 / qi - inv_qd
        return inv_qt / (1.0 / qi + inv_qd)


#: penalty scale of the branch constraint (one unit of residual per 0.2 % of x0 on the wrong side)
BRANCH_WALL = 0.002


def coupling_er_problems(ctx: Context, targets: Mapping[str, Any], *, floor: bool = True,
                         mechanisms=("dispersion", "ring_t_loss", "heater_t_loss"),
                         regimes=("under", "over")) -> list[Problem]:
    """One Problem per (coupling branch, extinction mechanism[, instrument floor]).

    Free: Q_L (prior: the leaf range [2600, 4200]), alpha_other_abs (sets Q_i; [0, 30] dB/cm, the leaf range; the
    branch is which side of critical coupling the 0 V notch sits on), the mechanism's coefficient (the other two
    held at 0, the dispersion at the solver value when not free). Hypothesis ``floor``: a notch-depth instrument
    floor F in [28, 60] dB as a nuisance of the measurement (the two deepest read-offs are equal, 29.8 dB, and
    the targets' verification warns that depths near 30 dB are floor-sensitive); without it the replica's depth is
    compared directly. The replica is the CAL-2 protocol on the plant's own (add-drop) topology at P_in -> 0.
    The branch is enforced on the cold 0 V state (x0 < 1 under, > 1 over; :class:`CouplingConstants`), so the two
    mirror solutions are separate hypotheses rather than two starts of one fit. A loss mechanism holds the coupler
    dispersion at the preset's own (solver) value explicitly, so a context that later carries the dispersion fitted
    under the dispersion mechanism (the nominal world) cannot leak into the loss worlds."""
    t1, t2 = targets["CAL-1"], targets["CAL-2"]
    cc = CouplingConstants.from_context(ctx)
    disp_solver = float(R.load_config(ctx.base(BENCH)).get(P_DISP))
    probs = []
    for mech in mechanisms:
        for regime in regimes:
            for use_floor in ((False, True) if floor else (False,)):
                mp = ER_MECHANISMS[mech]
                params = [Param("q_loaded", P_Q, 2600.0, 4200.0, 3850.0, prior_source="leaf range [2600, 4200]"),
                          Param("alpha_other_abs", P_COUPLING, 0.0, 30.0, BRANCH_START[regime], unit="db_per_cm",
                                prior_source="leaf range [0, 30]"),
                          mp]
                if use_floor:
                    params.append(Param("notch_floor_db", None, 28.0, 60.0, 31.0, unit="db", role="nuisance",
                                        prior_source="assumed instrument floor 28-60 dB (60 = no floor)"))
                fixed = {P_CT_RING: 0.0, P_CT_HEATER: 0.0, P_DISP: disp_solver}
                fixed.pop(mp.path, None)

                def predict(th, mp=mp, fixed=fixed, use_floor=use_floor):
                    ov = dict(fixed)
                    ov.update({P_Q: th["q_loaded"], P_COUPLING: th["alpha_other_abs"], mp.path: th[mp.name]})
                    cfg = ctx.config(BENCH, ov)
                    q = R.loaded_q_from_3db_width(cfg, t1).value
                    er = R.notch_er_vs_heater_v(cfg, t2, topology=None).value
                    if use_floor:
                        er = notch_floor(er, th["notch_floor_db"])
                    return {"CAL-1": q, "CAL-2": er}
                def constraint(th, regime=regime):
                    lx = math.log(max(cc.x0(th["q_loaded"], th["alpha_other_abs"]), 1e-9))
                    bad = max(lx, 0.0) if regime == "under" else max(-lx, 0.0)
                    return np.array([bad / BRANCH_WALL])

                def derived(th, regime=regime):
                    qi = cc.q_intrinsic(th["alpha_other_abs"])
                    x0 = cc.x0(th["q_loaded"], th["alpha_other_abs"])
                    return {"x0_cold_0v": x0, "branch_at_0v": "over" if x0 > 1 else "under", "q_intrinsic": qi,
                            "q_loaded_over_q_intrinsic": th["q_loaded"] / qi}
                hyp = f"{regime}/{mech}" + ("/floor" if use_floor else "")
                # starts: the g26 branch value, and the 27.9 dB notch at 0 V on this branch at Q_L 3700 and 4000
                starts = [{}] + [{"q_loaded": q, "alpha_other_abs": _alpha_for_x0(
                    cc, q, 0.9225 if regime == "under" else 1.084)} for q in (3700.0, 4000.0)]
                probs.append(Problem(group="coupling_er", hypothesis=hyp, params=params, targets=[t1, t2],
                                     predict=predict, fixed={"preset": BENCH, **fixed, "topology": "add_drop (own)"},
                                     starts=starts,
                                     constraint=constraint, derived=derived))
    return probs


def _alpha_for_x0(cc: CouplingConstants, q_loaded: float, x0: float) -> float:
    """alpha_other (dB/cm) that gives the cold coupling ratio x0 at Q_L (inverse of CouplingConstants.x0)."""
    # kappa_i + kappa_d = kappa / (1 + x0) -> 1/Q_i = (1/(1 + x0) - kd/k) / Q_L
    inv_qi = (1.0 / (1.0 + x0) - cc.kd_over_k) / q_loaded
    a = 2.0 * math.pi * cc.n_g * inv_qi / cc.lambda_m
    return float(min(max((a - cc.alpha_rest_per_m) / DB_PER_CM, 0.0), 30.0))


# =====================================================================================================================
# CAL-3a/b + CAL-4: heater efficiency (P_FSR anchor), cold resistance and the TCR droop y = TCR x R_th,H
# =====================================================================================================================
def heater_problems(ctx: Context, targets: Mapping[str, Any],
                    hypotheses=("same_rth_heater", "same_ratio")) -> list[Problem]:
    """CAL-3a (P_FSR median), CAL-3b (V_FSR median) and CAL-4 (shift against heater voltage of the swept device).

    Free leaves: ``ring.heater.p_fsr_mw`` (prior: leaf range [24.9, 37.0]; R2 anchors it with the FSR at the line,
    the bench protocol measures the FSR between the cold order and the next red order, 0.2-0.4 % apart),
    ``ring.heater.r_cold_ohm`` ([35, 58]: the leaf range [42, 58] widened, see the Param) and ``ring.thermal.heater_self_ratio`` ([1, 4]; with the family's TCR it
    sets y = TCR R_th,H, the droop CAL-4 identifies; TCR and R_th,H are not separable, 06 E1). Nuisance: the swept
    device's own P_FSR (CAL-4 is not the median device: 0.348 against 0.405 nm/mW, 06 E4; prior [22, 40] mW around
    the 60-die range 24.9-34.1), with the median cold resistance (assumed: one device sweep cannot separate its
    efficiency from its resistance). Hypotheses for the swept device's droop: ``same_rth_heater`` (the heater's own
    thermal resistance R_th,H is a heater property, equal on both devices: ratio_dev = ratio P_FSR,dev / P_FSR, since
    Z = dT_FSR / P_FSR under the fsr_integral anchor) and ``same_ratio`` (the interim model's ratio leaf as is)."""
    t3a, t3b, t4 = targets["CAL-3a"], targets["CAL-3b"], targets["CAL-4"]
    ctx.ranges.setdefault(P_R0, [35.0, 58.0])
    # the swept device's ratio under same_rth_heater (ratio x P_FSR,dev / P_FSR) may leave the median's [1, 4]; the
    # scratch configs allow it (the written leaf keeps its range)
    ctx.ranges.setdefault(P_RATIO, [1.0, 8.0])
    probs = []
    for hyp in hypotheses:
        params = [Param("p_fsr_mw", P_PFSR, 24.9, 37.0, 25.45, unit="mw", prior_source="leaf range [24.9, 37.0]"),
                  Param("r_cold_ohm", P_R0, 35.0, 58.0, 45.41, unit="ohm",
                        prior_source="leaf range [42, 58] widened to 35 ohm: the leaf range was set as R_hot / "
                                     "(1 + y P_FSR) with the earlier y = 0.00904 /mW, and R0 trades against y"),
                  Param("heater_self_ratio", P_RATIO, 1.0, 4.0, 1.158, prior_source="leaf range [1, 4]"),
                  Param("p_fsr_swept_device_mw", None, 22.0, 40.0, 29.5, unit="mw", role="nuisance",
                        prior_source="assumed [22, 40] mW around the G25 60-die range 24.9-34.1 mW")]

        def predict(th, hyp=hyp):
            med = {P_PFSR: th["p_fsr_mw"], P_R0: th["r_cold_ohm"], P_RATIO: th["heater_self_ratio"]}
            cfg = ctx.config(BENCH, med)
            r3 = R.heater_power_per_fsr(cfg, t3a)
            ratio_dev = th["heater_self_ratio"] * (th["p_fsr_swept_device_mw"] / th["p_fsr_mw"]
                                                   if hyp == "same_rth_heater" else 1.0)
            dev = dict(med, **{P_PFSR: th["p_fsr_swept_device_mw"], P_RATIO: ratio_dev})
            r4 = R.resonance_vs_heater_v(ctx.config(BENCH, dev), t4)
            return {"CAL-3a": r3.value, "CAL-3b": r3.variants["voltage_v"], "CAL-4": r4.value}

        def derived(th, hyp=hyp):
            cfg = ctx.config(BENCH, {P_PFSR: th["p_fsr_mw"], P_R0: th["r_cold_ohm"], P_RATIO: th["heater_self_ratio"]})
            _, pp = R.resolve_plant(cfg)
            z = float(np.asarray(pp.thermal.z_heater_ring_k_per_w).ravel()[0])
            tcr = float(np.asarray(pp.heater.tcr_per_k).ravel()[0])
            fsr_nm = float(R._fsr_m(R.StaticRing(pp)) * 1e9)
            y = tcr * th["heater_self_ratio"] * z
            return {"z_heater_ring_k_per_mw": z * 1e-3, "tcr_per_k": tcr, "y_per_mw": y * 1e-3,
                    "r_th_heater_k_per_mw": th["heater_self_ratio"] * z * 1e-3,
                    "efficiency_median_nm_per_mw": fsr_nm / th["p_fsr_mw"],
                    "efficiency_swept_device_nm_per_mw": fsr_nm / th["p_fsr_swept_device_mw"],
                    "r_hot_at_p_fsr_ohm": th["r_cold_ohm"] * (1.0 + y * th["p_fsr_mw"] * 1e-3),
                    "swept_device_droop_hypothesis": hyp}
        probs.append(Problem(group="heater", hypothesis=hyp, params=params, targets=[t3a, t3b, t4], predict=predict,
                             fixed={"preset": BENCH, "swept_device_r_cold": "median (assumed)",
                                    "driver": "bench voltage source, TCR solved"},
                             starts=[{}, {"heater_self_ratio": 2.0, "p_fsr_swept_device_mw": 31.0}],
                             derived=derived))
    return probs


# =====================================================================================================================
# CAL-7: the cold blue offset
# =====================================================================================================================
def offset_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    """CAL-7: ``ring.optics.resonance_offset_nm`` (prior: leaf range [5, 10]) from the low-power 0 V spectrum (the
    replica's primary; R3 closes it by construction, so the fit only confirms the definition)."""
    t7 = targets["CAL-7"]
    params = [Param("resonance_offset_nm", P_OFFSET, 5.0, 10.0, 7.6, unit="nm", prior_source="leaf range [5, 10]")]

    def predict(th):
        return {"CAL-7": R.cold_resonance_offset(ctx.config(BENCH, {P_OFFSET: th["resonance_offset_nm"]}), t7).value}
    return [Problem(group="offset", hypothesis="cold_cavity", params=params, targets=[t7], predict=predict,
                    fixed={"preset": BENCH, "p_in": "-> 0 (cold cavity)", "heater_v": 0.0})]


# =====================================================================================================================
# CAL-5: the DC junction efficiency (analytic law) and, as a reported variant, the LUT-mode fill fraction
# =====================================================================================================================
def _is_leaf(d: Any) -> bool:
    return isinstance(d, dict) and ("value" in d or "from" in d)


def _deep_merge(parent: dict, child: dict) -> dict:
    out = dict(parent)
    for k, v in child.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            if _is_leaf(out[k]) and _is_leaf(v):
                m = dict(out[k])
                if "value" in v or "from" in v:
                    m.pop("value", None)
                    m.pop("from", None)
                    m.pop("at", None)
                m.update(v)
                out[k] = m
            else:
                out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def bench_lut_mapping(lut_file: str = "extracted/junction_rib160.json",
                      ranges: Mapping[str, Sequence[float]] | None = None) -> dict:
    """The bench preset (g25_test_vehicle over g26) as one mapping, switched to the junction LUT
    (``fidelity.junction_model: lut``; the analytic-only keys removed, as tests/verification/
    test_extraction_junction.py does for g26). A child preset cannot unset keys, so the LUT variant is built here
    as a mapping. Used only for the reported CAL-5 fill-fraction variant."""
    from mrm.params import config as C
    base = C.yaml_load((C.PRESET_DIR / f"{BANK}.yaml").read_text())
    bench = C.yaml_load((C.PRESET_DIR / f"{BENCH}.yaml").read_text())
    bench.pop("extends", None)
    alts = dict(base.pop("alts", {}) or {})
    alts.update(bench.pop("alts", {}) or {})
    d = _deep_merge(base, bench)
    if alts:
        d["alts"] = alts
    d["meta"] = dict(d.get("meta", {}), name=f"{BENCH}_lut")
    d["fidelity"]["junction_model"] = "lut"
    j = d["ring"]["junction"]
    for k in ("eff_at_0v_pm_per_v", "v_bi_v", "depletion_exponent", "dqi_dv_frac_per_v"):
        j.pop(k, None)
    j["lut_file"] = lut_file
    for path, rng in (ranges or {}).items():
        node = d
        for k in path.split("."):
            node = node[k]
        node["range"] = [float(rng[0]), float(rng[1])]
    return d


#: measurement-power hypotheses of CAL-5 (07 P1-7; the G25 sweep power is not printed): the replica's primary is the
#: printed G25 input +4 dBm at the ring (self-heating relief included), the alternative P_in -> 0 (no relief)
CAL5_POWERS = {"p_in_4dbm": 4.0, "p_in_0": None}


def junction_problems(ctx: Context, targets: Mapping[str, Any], *, lut_variant: bool = True) -> list[Problem]:
    """CAL-5 (DC shift against reverse bias at 0.8, 1.2, 1.6 V).

    In the analytic junction mode of g26 (``fidelity.junction_model: analytic``) the ring-level shift is
    ``eff_at_0v_pm_per_v`` times the depletion shape, and ``fill_fraction`` cancels (the analytic table is built
    per fill so the ring-level efficiency, capacitance and FCA are the configured ones), so CAL-5 identifies
    ``eff_at_0v_pm_per_v`` there (prior: leaf range [50, 95]); the shape (V_bi 0.9 V, exponent 1/2) is held. In
    LUT mode (variant ``lut/...``, the rib160 junction artefact) the shift scales with ``fill_fraction``
    (prior: leaf range [0.5, 0.95]), the quantity 07 P1-7 names. Each at both measurement powers."""
    t5 = targets["CAL-5"]
    probs = []
    for hname, p_dbm in CAL5_POWERS.items():
        params = [Param("eff_at_0v_pm_per_v", P_EFF, 50.0, 95.0, 72.5, unit="pm_per_v",
                        prior_source="leaf range [50, 95]")]

        def predict(th, p_dbm=p_dbm):
            cfg = ctx.config(BENCH, {P_EFF: th["eff_at_0v_pm_per_v"]})
            return {"CAL-5": R.resonance_shift_vs_bias(cfg, t5, p_in_dbm=[p_dbm]).value}
        probs.append(Problem(group="junction", hypothesis=f"analytic/{hname}", params=params, targets=[t5],
                             predict=predict, fixed={"preset": BENCH, "junction_model": "analytic",
                                                     "p_in_dbm": p_dbm, "v_bi_v": "preset", "exponent": "preset"}))
    if lut_variant:
        from mrm import params as mp
        mapping = bench_lut_mapping(ranges=ctx.ranges)
        for hname, p_dbm in CAL5_POWERS.items():
            params = [Param("fill_fraction", P_FILL, 0.5, 0.95, 0.85, prior_source="leaf range [0.5, 0.95]")]

            def predict(th, p_dbm=p_dbm):
                ov = {k: v for k, v in ctx.overrides(BENCH).items() if k not in (P_EFF,)}
                ov[P_FILL] = th["fill_fraction"]
                cfg = mp.load(mapping, overrides=ov)
                return {"CAL-5": R.resonance_shift_vs_bias(cfg, t5, p_in_dbm=[p_dbm]).value}
            probs.append(Problem(group="junction", hypothesis=f"lut/{hname}", params=params, targets=[t5],
                                 predict=predict, fixed={"preset": f"{BENCH} (mapping, LUT mode)",
                                                         "junction_model": "lut",
                                                         "lut_file": "extracted/junction_rib160.json",
                                                         "p_in_dbm": p_dbm}))
    return probs


# =====================================================================================================================
# CAL-10: the junction R C (waveform mode)
# =====================================================================================================================
def eo_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    """CAL-10 (30.9 GHz at 1 V reverse). One number cannot separate R_s from C_j (07 P1-7: "R_s * C together with
    the photon-lifetime limit"), so three hypotheses: ``rs_free`` (C_j(0) held at its assumed 15 fF, R_s free in
    the leaf range [15, 390] ohm, log), ``c_free`` (R_s held at 100 ohm, C_j(0) free in [6, 30] fF, log) and
    ``both_free`` (for the profile: both free, the valley is the identified product). Detuning: the replica's primary
    (maximum slope); the half-width detuning is recorded as a variant prediction."""
    t10 = targets["CAL-10"]
    rs = Param("series_resistance_ohm", P_RS, 15.0, 390.0, 150.0, unit="ohm", log=True,
               prior_source="leaf range [15, 390]")
    c0 = Param("capacitance_at_0v_ff", P_C0, 6.0, 30.0, 15.0, unit="ff", log=True,
               prior_source="leaf range [6, 30]")
    probs = []
    for hyp, params in (("rs_free", [rs]), ("c_free", [c0]), ("both_free", [rs, c0])):
        def predict(th):
            ov = {p.path: th[p.name] for p in (rs, c0) if p.name in th}
            r = R.eo_bandwidth_at_bias(ctx.config(BENCH, ov), t10, with_variants=False)
            return {"CAL-10": r.value}

        def derived(th):
            ov = {p.path: th[p.name] for p in (rs, c0) if p.name in th}
            r = R.eo_bandwidth_at_bias(ctx.config(BENCH, ov), t10)
            d = r.detail.get("max_slope", {})
            return {"variants": {k: v for k, v in r.variants.items()},
                    "r_s_ohm_at_1v": d.get("r_s_ohm"), "c_j_ff_at_1v": (d.get("c_j_f") or 0.0) * 1e15,
                    "rc_pole_ghz": (d.get("rc_pole_hz") or 0.0) / 1e9, "fwhm_ghz": (d.get("fwhm_hz") or 0.0) / 1e9}
        probs.append(Problem(group="eo", hypothesis=hyp, params=list(params), targets=[t10], predict=predict,
                             fixed={"preset": BENCH, "detuning": "max_slope (primary)", "r_drv_ohm": "plant (50)"},
                             derived=derived))
    return probs


# =====================================================================================================================
# CAL-8: the lumped monitor gain
# =====================================================================================================================
#: lock set-point hypotheses of CAL-8 (the replica's primary first; fixed by the replica owner before comparisons)
CAL8_SETPOINTS = ("half_max", "peak", "oracle")


def monitor_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    """CAL-8 (15.5 uA held at lock during the aggressor). Only the product gain x line power x lock factor is
    identified (07 D7, P1-9); the fitted leaf is ``ring.monitor.gain_scale`` (the lumped nuisance on the monitor
    gain; prior: uncertainty.monitor.gain_scale, [0.5, 2], log-uniform), once per set-point hypothesis. The replica:
    bank locked with every line on (bus-filtered), NRZ at the scenario bias, the preset line power, 25 degC base."""
    t8 = targets["CAL-8"]
    probs = []
    for sp in CAL8_SETPOINTS:
        params = [Param("gain_scale", P_GAIN, 0.5, 2.0, 1.0, log=True,
                        prior_source="uncertainty.monitor.gain_scale [0.5, 2] (log-uniform)")]

        def predict(th, sp=sp):
            r = R.monitor_current_at_lock(ctx.config(BANK, {P_GAIN: th["gain_scale"]}), t8, setpoint=sp)
            return {"CAL-8": r.value}
        probs.append(Problem(group="monitor", hypothesis=f"setpoint={sp}", params=params, targets=[t8],
                             predict=predict, fixed={"preset": BANK, "setpoint": sp, "lines": "all",
                                                     "p_in": "preset line power"}))
    return probs


# =====================================================================================================================
# CAL-14: the lumped modulation depth
# =====================================================================================================================
def modulation_depth_problems(ctx: Context, targets: Mapping[str, Any],
                              modes=("oracle", "setpoint_lock")) -> list[Problem]:
    """CAL-14 (stabilized relOMA -2.1 dB at 1.0-1.15 V, 0-105 degC, relOMA = OMA / 1 mW into the MRM, 07 D9). The
    lumped depth (swing x efficiency / FWHM) is carried by ``driver.swing_pp_v`` (prior: leaf range [0.9, 2.6] V)
    with the efficiency and Q_L of the earlier groups. Mode ``oracle`` (the replica's primary: best achievable, so a
    depth fitted there is a LOWER bound if G26 locks off-optimum) and ``setpoint_lock`` (half-maximum drop lock)."""
    t14 = targets["CAL-14"]
    probs = []
    for md in modes:
        params = [Param("swing_pp_v", P_SWING, 0.9, 2.6, 1.3, unit="v", prior_source="leaf range [0.9, 2.6]")]

        def predict(th, md=md):
            r = R.relOMA_map(ctx.config(BANK, {P_SWING: th["swing_pp_v"]}), t14, mode=md, with_variants=False)
            return {"CAL-14": r.value}
        probs.append(Problem(group="modulation_depth", hypothesis=f"mode={md}", params=params, targets=[t14],
                             predict=predict, fixed={"preset": BANK, "mode": md, "p_in_dbm": 0.0,
                                                     "level_convention": "preset (centered)"}))
    return probs


def modulation_depth_scan(ctx: Context, targets: Mapping[str, Any], *,
                          swings=(0.9, 1.3, 1.8, 2.2, 2.6, 3.1, 3.6, 4.5),
                          eff_scales=(1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
                          conventions=("centered", "one_sided"), mode: str = "oracle") -> dict:
    """What lumped modulation depth closes CAL-14 (the known gap, 07 D9): relOMA (the CAL-14 replica, oracle) on a
    grid of swing (beyond the prior, to find the closing value) and of a junction-efficiency scale (eff_at_0v x s
    at the swing of the preset), for both NRZ level conventions. Each row also gives the peak-to-peak resonance
    swing at the CAL-14 bias (1.075 V) and the depth D = swing_pm / FWHM_pm, the quantity 07 D9 calls the lumped
    modulation depth. The junction grid of the plant clips the levels at -0.5 V (forward) and 4.0 V."""
    import tempfile
    t14 = targets["CAL-14"]
    rows = []
    base_cfg = ctx.config(BANK)
    eff0 = float(base_cfg.get(P_EFF))
    # a scratch child of g26 that only widens the leaves' plausible ranges (same values and provenance), so the scan
    # may leave the priors; it is deleted after the scan and never written to the package
    tmp = tempfile.TemporaryDirectory(prefix="mrm_cal14_scan_")
    path = widened_child(BANK, {**ctx.ranges, P_SWING: [0.1, 10.0], P_EFF: [1.0, 900.0]}, tmp.name)
    for conv in conventions:
        for sw in swings:
            for es in eff_scales:
                if es != 1.0 and sw != 1.3 and sw != float(base_cfg.get(P_SWING)):
                    continue
                ov = {P_SWING: sw, P_EFF: eff0 * es, "driver.level_convention": conv}
                cfg = ctx.config(path, ov)
                r = R.relOMA_map(cfg, t14, with_variants=False, mode=mode)
                _, pp = R.resolve_plant(cfg)
                ring = R.StaticRing(pp)
                lo, hi = ring.nrz_levels(1.075)
                d_pm = float((ring.junction_shift_m(hi)[0] - ring.junction_shift_m(lo)[0]) * 1e12)
                fwhm_pm = float(ring.lam_line / ring.q_loaded * 1e12)
                rows.append({"level_convention": conv, "swing_pp_v": sw, "eff_scale": es,
                             "eff_at_0v_pm_per_v": eff0 * es, "levels_v": [lo, hi], "swing_pm": d_pm,
                             "fwhm_pm": fwhm_pm, "depth": d_pm / fwhm_pm, "reloma_db": r.value})
    tmp.cleanup()
    return {"mode": mode, "eff_at_0v_base_pm_per_v": eff0, "rows": rows}


# =====================================================================================================================
# CAL-9: the heater DAC LSB under the assumed driver law
# =====================================================================================================================
def dac_codes(cfg, *, n_dies: int = 1, variation: str = "off", uniform: bool = True, seed: int = 0,
              referencing: str = "resonance") -> dict:
    """The CAL-9 protocol pieces of :func:`mrm.validation.replicas.stabilizer_code_vs_T` (A1 calibrated on the
    die's ROM, periodic steady-state rises, bank locked at half maximum with every line on, codes through the law,
    the TCR and the realized DAC), with the aggressor rise of the victim ring applied to every lane when
    ``uniform`` (the replica's variant ``aggressor_model=uniform``). Returns the per-die statistics (swing, lane
    offsets, code range) and their medians."""
    _, pp = R.resolve_plant(cfg, batch=n_dies, uncertainty="nominal", variation=variation,
                            seeds={"device": int(seed), "uncertainty": 0})
    sc, cw, _site = R._calibrated_aggressor(pp, which="a1", referencing=referencing)
    ris = R._aggressor_rises(pp, cw)
    t_base = float(sc.defaults.get("thermal.t_base_k", 298.15))
    bias = R._scenario_bias(cfg)
    stats, n_inf = [], 0
    for b in range(pp.batch):
        ar, ah = ris["ring"][:, b, :], ris["heater"][:, b, :]
        if uniform:
            ar = np.repeat(ar[:, :1], pp.n_rings, axis=1)
            ah = np.repeat(ah[:, :1], pp.n_rings, axis=1)
        bc = R._bank_codes(pp, b, t_base=t_base, bias=bias, mode="half_max", agg_r=ar, agg_h=ah)
        n_inf += int(np.sum(bc["infeasible"]))
        stats.append(R._code_stats(bc["codes"]))
    return {"swing": R._median_stats(stats, "swing"), "lane_offsets": R._median_stats(stats, "lane_offsets"),
            "code_range": R._median_stats(stats, "code_range"), "per_die": stats, "infeasible_samples": n_inf,
            "victim_rise_k": [float(ris["ring"][:, :, 0].min()), float(ris["ring"][:, :, 0].max())]}


#: thermo-optic hypotheses compared on CAL-9 (06 E2: the code range implies 71-88 pm/K across V_FS 1.40-1.60 V)
CAL9_THERMO = ("H_solver", "H86")


#: per-ring geometry sigmas of g26 (variation.geometry, Berkeley local spread; 07 P1-16)
RING_SIGMA_PATHS = {"variation.geometry.width.ring_sigma_nm": 2.0, "variation.geometry.thickness.ring_sigma_nm": 1.3}
DIE_SIGMA_PATHS = ("variation.geometry.width.die_sigma_nm", "variation.geometry.thickness.die_sigma_nm")


def dac_problems(ctx: Context, targets: Mapping[str, Any], thermo=CAL9_THERMO,
                 referencing=("temperature", "resonance"), n_dies: int = 8) -> list[Problem]:
    """CAL-9a (per-lane swing), CAL-9b (lane offsets) and CAL-9c (code range) of ONE G26 die.

    Protocol: the pieces of the CAL-9 replica (:func:`dac_codes`) with a uniform aggressor (every lane sees the
    victim's rise, as the slide's ~20 k swings on all 8 lanes show; the replica's primary geometry, an aggressor
    30 um from ring 0, gives 3.6 k to 22.8 k and cannot reach the lower end with any DAC parameter), the median
    over ``n_dies`` dies that share the die-level state and differ only in their per-ring draws (die-level geometry
    sigma set to 0, so the median describes the lane spread of one die). Free: ``ring.heater.full_scale_v``
    (voltage law, the g26 nominal; prior: leaf range [1.4, 1.8] V), ``ring_sigma_scale`` (multiplies the per-ring
    geometry sigmas 2.0 / 1.3 nm, the Berkeley local spread; prior [0, 1.5]) and, as a nuisance never written, the
    measured die's blue offset (prior [5, 10] nm: the die-level spread is 6 nm, so the one plotted die need not sit
    at the G25 CAL-7 value). Hypotheses: thermo-optic (the code range ratio depends on the lock heater powers) and
    aggressor referencing (07 D6: resonance trajectory at 85.86 pm/K, the CAL-13a default, or ring temperature)."""
    t9a, t9b, t9c = targets["CAL-9a"], targets["CAL-9b"], targets["CAL-9c"]
    probs = []
    for ref in referencing:
        for hyp in thermo:
            params = [Param("full_scale_v", P_VFS, 1.4, 1.8, 1.45, unit="v", prior_source="leaf range [1.4, 1.8]"),
                      Param("ring_sigma_scale", "variation.geometry.ring_sigma_scale", 0.0, 1.5, 0.3,
                            declared_prior=False,
                            prior_source="fit bounds [0, 1.5] x the Berkeley local spread (07 P1-16; no declared "
                                         "range)"),
                      Param("die_resonance_offset_nm", P_OFFSET, 5.0, 10.0, 9.0, unit="nm", role="nuisance",
                            prior_source="leaf range [5, 10] (the plotted die's offset is unpublished)")]

            def _cfg(th, hyp=hyp):
                ov = {P_VFS: th["full_scale_v"], P_OFFSET: th["die_resonance_offset_nm"]}
                ov.update({k: v * th["ring_sigma_scale"] for k, v in RING_SIGMA_PATHS.items()})
                ov.update({k: 0.0 for k in DIE_SIGMA_PATHS})
                return ctx.config(BANK, ov, alts={"uncertainty.thermo_optic": hyp})

            def predict(th, ref=ref, _cfg=_cfg):
                d = dac_codes(_cfg(th), referencing=ref, n_dies=n_dies, variation="sample")
                return {"CAL-9a": d["swing"], "CAL-9b": d["lane_offsets"], "CAL-9c": d["code_range"]}

            def derived(th, ref=ref, _cfg=_cfg):
                d = dac_codes(_cfg(th), referencing=ref, n_dies=n_dies, variation="sample")
                return {"median_of_dies": {k: d[k] for k in ("swing", "lane_offsets", "code_range",
                                                              "infeasible_samples", "victim_rise_k")},
                        "per_die": d["per_die"]}
            probs.append(Problem(group="dac", hypothesis=f"voltage/{hyp}/{ref}", params=params,
                                 targets=[t9a, t9b, t9c], predict=predict,
                                 fixed={"preset": BANK, "driver_mode": "voltage (g26 nominal)", "thermo_optic": hyp,
                                        "aggressor_referencing": ref, "aggressor_model": "uniform",
                                        "dies": f"{n_dies} draws of the per-ring variation, die sigma 0",
                                        "setpoint": "half_max"},
                                 starts=[{}, {"full_scale_v": 1.6, "ring_sigma_scale": 0.6,
                                              "die_resonance_offset_nm": 8.0}],
                                 derived=derived))
    return probs


# =====================================================================================================================
# CAL-12b: the static line-error population (a scenario-level population, fitted on the preset's variation entry)
# =====================================================================================================================
LINE_SIGMA_PATHS = {"variation.laser.line_offset_ghz.die_sigma": 4.0, "variation.laser.line_offset_ghz.line_sigma": 6.0}


def laser_problems(ctx: Context, targets: Mapping[str, Any]) -> list[Problem]:
    """CAL-12b (median and quartiles of the Guide line-error population): ``laser.line_offset_ghz`` (the centre;
    prior: leaf range [-20, 20] GHz) and a scale on the die and line sigmas of variation.laser.line_offset_ghz (4 and
    6 GHz; prior [0.25, 2], log; the die:line split is held). The replica samples 256 worlds (seed 0) and clips at
    the +-20 GHz requirement."""
    t12 = targets["CAL-12b"]
    params = [Param("line_offset_ghz", P_LINE, -20.0, 20.0, 0.0, unit="ghz", prior_source="leaf range [-20, 20]"),
              Param("line_sigma_scale", "variation.laser.line_offset_ghz.sigma_scale", 0.25, 2.0, 1.0, log=True,
                    declared_prior=False,
                    prior_source="fit bounds [0.25, 2] x the preset's 4 / 6 GHz die / line sigma (the variation "
                                 "entry declares no range)")]

    def predict(th):
        ov = {P_LINE: th["line_offset_ghz"]}
        ov.update({k: v * th["line_sigma_scale"] for k, v in LINE_SIGMA_PATHS.items()})
        r = R.laser_line_statistics(ctx.config(BANK, ov), t12)
        return {"CAL-12b": r.value}
    return [Problem(group="laser", hypothesis="gaussian_die_plus_line", params=params, targets=[t12],
                    predict=predict, fixed={"preset": BANK, "worlds": 256, "clip_ghz": 20.0})]
