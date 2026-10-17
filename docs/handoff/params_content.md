# Handoff: params content (presets, sources, schema key list, PARAMETERS.md)

Owner files: `mrm/params/presets/*.yaml` (g26, g25_test_vehicle, teaching_basic and the 12 corners of 07 4.8),
`mrm/params/sources.yaml`, `mrm/params/schema.py` (key-list additions only, format unchanged), `docs/PARAMETERS.md`
(generated), `tests/closure/test_presets.py`. Also regenerated, on the machinery owner's request (their handoff
section 3): the generated key block of `docs/PARAMETERS_FORMAT.md` between `<!-- keys:begin -->` and
`<!-- keys:end -->` (no other line of that file was touched).

Tests: `tests/closure/test_presets.py`: 89 fast tests (about 20-25 s, also with `NUMBA_DISABLE_JIT=1`) plus 17
`@pytest.mark.slow` ones (every hypothesis choice, every interval corner, a 512-world Monte Carlo realize of every
preset) in about 35 s more (90 s without JIT). All pass. Section 7 lists what the adversarial review changed. The params,
closure, control-api and isolation suites (391 tests after the review) pass except
`test_params_cli.py::test_parameters_md_is_current`, which needs item 1 below.

## 1. Request to params_machinery (blocking): fall back to a `fallback` alternative when an artefact is missing

The presets take every solver-derived leaf from an extraction artefact with the from-form, plus an alternative
named `fallback` that holds a literature or inferred value:

```yaml
n_g:
  value: {from: "extracted/modes_rib160.json:n_g"}
  provenance: solver(assumed-geometry)
  ...
  alternatives:
    fallback: {value: 4.0, provenance: inferred, generation: none, source: BU-45CLO, note: "..."}
```

`modes_rib160.json` and `coupling_rib160.json` do not exist yet, so today `load("g26")` raises `ConfigError`
(artefact not found). Please add to `mrm/params/config.py`: when the artefact FILE is missing (not a bad sha256, a
missing sidecar or a missing key: those stay errors) and the leaf has a `fallback` alternative with a `value`, use it
(value, provenance, generation, source), set `selected_alt = "fallback"` (the PARAMETERS table then shows
**fallback** in bold), append a warning, and export `ARTEFACT_FALLBACK_ALT = "fallback"`. The exact patch, verified
end to end on a scratch copy of the package (the staleness test and all preset tests pass with it, and the tests then
use the loader instead of their adapter):

```diff
--- mrm/params/config.py
+++ mrm/params/config.py
@@ -61,6 +61,8 @@
                          "textbook", "disagrees", "targets", "residual", "identifiable"})
 _ALT_FIELDS = frozenset({"value", "from", "at", "provenance", "generation", "source", "note"})
 _STRICT_KINDS = ("preset", "corner", "private", "test")
+#: name of the leaf alternative a missing ``from:`` artefact falls back to (docs/handoff/params_content.md 1)
+ARTEFACT_FALLBACK_ALT = "fallback"
 _PROV_RE = re.compile(r"^([a-z-]+)(?:\((.*)\))?$")
@@ -876,6 +878,7 @@
     val = d.get("value")
     spec = d.get("from")
     at = d.get("at")
+    fell_back = False
     if isinstance(val, dict) and "from" in val:
         spec = val["from"]
         at = val.get("at", at)
@@ -883,13 +886,26 @@
         try:
             val, art = resolve_from(spec, at, cfg.data_root)
         except ArtefactError as e:
-            errs.append(f"{path}: {e}")
-            val = None
-        if "provenance" not in d:
-            d["provenance"] = art["provenance"] if art is not None else "solver"
-        elif art is not None and not str(d.get("provenance", "")).startswith("solver"):
-            errs.append(f"{path}: a value taken from a solver artefact must have provenance 'solver' "
-                        f"(got {d.get('provenance')!r})")
+            # a missing artefact FILE (not a bad sha256, sidecar or key) falls back to the leaf's 'fallback'
+            # alternative, with a warning, until the extraction has produced it
+            fb = (d.get("alternatives") or {}).get(ARTEFACT_FALLBACK_ALT)
+            if "not found under" in str(e) and isinstance(fb, dict) and "value" in fb:
+                val = fb["value"]
+                for f in ("provenance", "generation", "source"):
+                    if fb.get(f) is not None:
+                        d[f] = fb[f]
+                fell_back = True
+                cfg.warnings.append(f"{path}: {e}; using its '{ARTEFACT_FALLBACK_ALT}' alternative {val!r} "
+                                    f"({d.get('provenance')}) until the artefact exists")
+            else:
+                errs.append(f"{path}: {e}")
+                val = None
+        if not fell_back:
+            if "provenance" not in d:
+                d["provenance"] = art["provenance"] if art is not None else "solver"
+            elif art is not None and not str(d.get("provenance", "")).startswith("solver"):
+                errs.append(f"{path}: a value taken from a solver artefact must have provenance 'solver' "
+                            f"(got {d.get('provenance')!r})")
     prov = d.get("provenance")
@@ -907,10 +923,14 @@
         alts["default"] = {"value": val, "provenance": prov, "generation": d.get("generation"),
                            "source": d.get("source"), "_resolved_artefact": art}
     rng = d.get("range")
-    return Leaf(path=path, value=val, provenance=prov, generation=d.get("generation"), source=d.get("source"),
+    leaf = Leaf(path=path, value=val, provenance=prov, generation=d.get("generation"), source=d.get("source"),
                 range=tuple(rng) if isinstance(rng, (list, tuple)) else rng, alternatives=alts, note=d.get("note"),
                 textbook=d.get("textbook"), disagrees=d.get("disagrees"), targets=d.get("targets"),
                 residual=d.get("residual"), identifiable=d.get("identifiable"), origin=list(origin), artefact=art)
+    if fell_back:
+        leaf.selected_alt = ARTEFACT_FALLBACK_ALT
+        leaf.origin.append(f"artefact {spec!r} missing: alternative {ARTEFACT_FALLBACK_ALT!r}")
+    return leaf
```

Notes on the semantics: the implicit `default` alternative is built after the fallback, so `default` is then the
fallback value (a hypothesis that selects `default` gets it). The nominal choice of every hypothesis in g26 leaves the
solver leaves alone, so `selected_alt = "fallback"` passes the "nominal must be the config as written" check. A
from-form ALTERNATIVE (for example `strip400x200`) is resolved only when drawn; every one in the presets points at
`modes_strip400x200.json`, which exists.

Until the patch lands: `tests/closure/test_presets.py` installs the same function as a test-local adapter
(`patched_make_leaf`, removed after each test, skipped once `config.ARTEFACT_FALLBACK_ALT` exists), and
`docs/PARAMETERS.md` was generated by the real CLI (`mrm params table g26 --out docs/PARAMETERS.md`) with that adapter
installed. `test_params_cli.py::test_parameters_md_is_current` fails with the ConfigError above until the patch is
applied; with the patch it passes against the committed file (checked).

Every time an artefact lands (modes_rib160, coupling_rib160) the solver value replaces the fallback, so
`docs/PARAMETERS.md` must be regenerated (`mrm params table g26 --out docs/PARAMETERS.md`) and `pytest
tests/closure/test_presets.py -m ""` re-run: see section 4 for what to check.

## 2. Request to params_machinery (table.py): render the inventory and the generation mix in PARAMETERS.md

`docs/research/00_prior_notes_checklist.md` asks that every item be modelled, excluded with a reason, or listed as
unknown "in docs/PARAMETERS.md". The inventory lives in `g26.yaml` under `meta.inventory` (schema key added, an open
subtree validated by `test_inventory_covers_the_checklist`: all 188 items of sections A-K and of the OFC 2026
"does not give" list). `table()` does not print `cfg.meta` yet. Patch (tested on the scratch copy; regenerate
PARAMETERS.md after applying it):

```diff
@@ -199,6 +199,19 @@ def table(...)
         if material_table_markdown is not None:
             out += ["", "## Materials (mrm.materials)", "", material_table_markdown().rstrip("\n")]
+    gm = (cfg.meta or {}).get("generation_mix")
+    if gm:
+        out += ["", "## Generation mix (07 4.9 item 6)", "", _md(gm)]
+    inv = (cfg.meta or {}).get("inventory")
+    if isinstance(inv, dict) and inv:
+        out += ["", "## Inventory of docs/research/00_prior_notes_checklist.md", "",
+                "Every item of the prior-notes checklist is a key, excluded with a reason, or unknown (with what stands "
+                "in for it).", "", "| section | item | status | keys | reason |", "|---|---|---|---|---|"]
+        for sec, items in inv.items():
+            for item, e in (items or {}).items():
+                e = e or {}
+                keys = ", ".join(f"`{k}`" for k in (e.get("keys") or []))
+                out.append(f"| {_md(sec)} | {_md(item)} | {_md(e.get('status', ''))} | {keys} | {_md(e.get('reason', ''))} |")
     dis = [(p, lf) for p, lf in sorted(cfg.leaves.items()) if lf.textbook or lf.disagrees]
```

## 3. Other requests, by owner

### params_machinery (features the presets would use; none blocking)
1. **Tied draws.** One interval draw that moves several leaves (`targets: [{path, mode, coef}]`). The presets need
   it twice: (a) `q_loaded` and the loss budget must scale together to keep the G25 notch (27.9 dB at 0 V) in every
   world; today Q_L is drawn alone over 3500-4200 with Q_i fixed, so the regime label is wrong in about 28 % of the
   'under' worlds (over-coupled below Q_L 3699) and 27 % of the 'over' worlds (under-coupled above 4014), and the
   0 V notch spans 17.5 dB to arbitrarily deep (it crosses critical coupling); in 256 LHS worlds of g25_test_vehicle
   only 15 % sit within 1.5 dB of 27.9 dB. This is the largest remaining calibration inconsistency of the presets.
   (b) a continuous `self_heating.eta` (07 4.8: 0.5-0.95) at a fixed alpha_i needs alpha_fca up and alpha_scatter
   down together; today it is a 5-point hypothesis (0.5 / 0.65 / 0.75 / 0.848 / 0.95, triangular weights, stress
   0.3).
   An alternative for (a): an R4_q member `kappa_t_over_kappa` (like `kappa_d_over_kappa`), so Q_L, the coupling
   ratios and the regime are given and Q_i is derived; the budget would then be given as `heat_fraction`.
2. **Removing parent keys in a child preset.** A `g26_rom` (FEM thermal ROM) or `g26_lut` (junction LUT) child
   cannot be written: the interim keys and the analytic junction keys are forbidden in those modes, and a strict
   child cannot null a parent leaf. Either an `unset: [paths]` field in a preset, or let the integrator switch g26
   itself when the ROM / LUT pass the calibration split (07 build order step 3).
3. `ring.thermal.xt_dist_ref_um`, `xt_far_um`, `site_ref_um` reach `InterimRom` through `cfg.get` (the config value),
   not the per-world arrays, so a hypothesis that changes them per world is silently ignored in sampled worlds. The
   only place the presets change one is the weight-0 `textbook` family choice, reached by pinning (load time), so the
   corner is correct; please either pass the per-world values or reject hypotheses that set them.
4. `fidelity.heater_anchor` stays `lock_slope` (07 as written). The numbers support your recommendation: under
   H_solver, lock_slope gives Z = 5.33 K/mW at T_lock 132 degC and fsr_integral about 5.5 K/mW; under lock_slope Z
   follows each ring's sampled offset (geometry spread 6 nm die / 2 nm ring).
5. `pyproject.toml` (integrator): add `"params/*.yaml"` to package-data so `sources.yaml` ships in a wheel.

### plant_thermal
- `ring.thermal.crosstalk_dynamics: self` (the textbook's crosstalk with the self-path shape) is a global setting, so
  the thin corner `g26_textbook_thermal` cannot select it. The corner uses one 25 us crosstalk pole instead (the
  minimax single pole: within 0.16 of the DC at every time against the textbook's 10 us 75 % + 300 us 25 %; the
  review moved it from 20 us, see section 7 item 9). At dt = 1 us a 10 us pole breaks the 07 P0-1 lag bound
  already at +4 dBm (lagged cross-ring gain 0.08 > 0.05), and even the 25 us pole reaches 0.11 in +8 dBm, eta 0.95,
  z 2 worlds, so the corner also pins the optical power and the self-heating strength at their nominal. Making `crosstalk_dynamics` a per-world switch would let the corner be
  exact (it would then need dt = 0.25 us).
- The FEM ROM of the in-ring family gives Z_heater->ring = 2.85 K/mW; the measured P_FSR with H_solver needs 5.33
  K/mW (lock_slope), so rom_dc_scale = 1.88, just inside the [0.5, 2] warning band. Its heater-resistor ratio is 2.65
  where the G25 droop (y = TCR x R_th,H = 0.00904 /mW) with a doped-Si TCR of 1.4e-3 /K implies 1.21. Both say the
  assumed in-ring geometry is less efficient than the G25 heater; worth a look before the ROM becomes the default.
- **2026-10-16 (params review), g26 changes and choices for the content owner** (docs/CHANGELOG.md has the reasons
  and numbers): (a) `ring.geometry.jacobian_coupling` (coupling_rib160) moves each device's couplers with its
  width, film and gaps; the gaps share one assumed local sigma, 1.5 nm, and a width error closes both gaps
  (`variation.geometry.width_closes_gaps`); the q_loaded (rel 0.015) and kappa_d_over_kappa (rel 0.1) stand-in
  draws are gone. Revisit the 1.5 nm when a gap spread is published. (b) With sampled geometry the radius is the
  nominal device's (a mask dimension) and the FSR follows n_g (~11 GHz sd at 1.8 THz). (c) The film-200 world is the
  rib200 artefact; its Jacobian gives a 4.2 nm die resonance spread with the same fabrication sigmas (6.2 nm on
  rib160): the sigmas are kept as fabrication errors, so the thicker film is less sensitive; if the Berkeley 6 nm is
  meant per world, the sigmas need a per-film value. (d) `ring.heater.p_fsr_ref_wavelength_nm` 1309.5 (inferred:
  the G25 operating line of CAL-7) makes Z and the efficiency line-independent. (e) beta is the secant of the solver
  curve through the nominal lock (1.2377e-3); the modes table ends at 125 degC while the P_FSR sweep reaches
  ~170 degC (extraction request, extract_modes.md). (f) heater_self_ratio 1.132 holds y = 0.00904 /mW at the new Z.
  (g) Ranges now hold every selectable value (chip_input_loss_db [1, 10], depletion_exponent [0.334, 0.666],
  r_cold_ohm [37, 58]); the loader enforces it.
- **2026-10-16 (physics_thermal review), g26 changes for the content owner:** (a) a third choice of
  `uncertainty.thermal.stack`, `side_by_side_tec_held` (weight 0, appended last so no sampled or LHS draw changes: the
  in-ring FEM ROM's side-by-side stack in its TEC-held package; it selects the `tec_held` alternatives of xt_nearest /
  xt_next / xt_floor / xt_tau_ref / xt_tau_min and of the new key `ring.thermal.site_floor_frac`; the default and
  stacked_eic keep imec's natural-convection numbers), and the corner `g26_tec_package` (in NAMED_CORNERS).
  Weighting side_by_side_tec_held in the sampled set is yours to decide: it changes every sampled-world result; the
  07 P0-1 lag bound stays far below 0.05 (6e-4 at the nominal). (b) `site_floor_frac` (0.015, the far floor of the site law: die sensor
  and far sites) follows xt_floor_frac in every other choice (stacked_eic and the textbook family set it too). (c)
  `heater_node_tau_s` is tagged `assumed` (it is the pole of the excess (h - 1) Z0 only, not the FEM's Z_HH t63);
  its metal / undercut alternatives are imec's whole-node time constants used the same way. (d) the `selfheat_tau_s`
  note quotes the ROM's step fractions (the 65 % was an older fit's pole share). (e) The crosstalk tau law now takes
  the heat-source-to-victim distance (centre distance less the ring radii): g26's nearest neighbour is 1.52 ms
  instead of 2.85 ms with unchanged keys.

### extraction (modes, coupling, junction, thermal)
- The presets read: `modes_rib160.json:{n_eff, n_g, dn_g_dlambda_per_nm, dlambda_dt_pm_per_k (at t_c 25),
  dlambda_dt_beta_per_k (at t0_c 25), alpha_rad_db_per_cm, jacobian}` and `coupling_rib160.json:{through,
  drop}.dln_kappa2_dlambda_per_nm`, the same layout as `modes_strip400x200.json` and `mrm.extraction.coupling`.
  Please keep these keys.
- Please record the geometry provenance in the sidecar as a string (`inputs.geometry.provenance: "assumed"`); the
  loader then qualifies the leaves itself. The presets state `solver(assumed-geometry)` explicitly meanwhile.
- `ring.optics.index_weight_oxide` is a sum (BOX + cladding) that a from-form cannot take from `index_weights.box`
  and `.clad`; an `index_weights.oxide` key would allow it (it is a diagnostic input only).
- The geometry variation uses width and film; if the rib Jacobian carries `slab_nm`, add a slab sigma.
- When `modes_rib200.json` exists, the `film_thickness: "200"` choice can move from the SMTAI strip
  (`strip400x200`, the render the 200 nm hypothesis comes from) to the rib. [done 2026-10-16, params review:
  rib200 with rib200_q_anchor 2.03 dB/cm; docs/CHANGELOG.md]
- The preset's nominal drop share is kappa_d/kappa = 0.007 (07 D7 log-centre; it reproduces the 15.5 uA monitor
  current at a half-maximum lock, see section 7); the coupling artefact's design drop gap uses 0.03. The drop
  dispersion is read at your gap: if it changes much with the gap, report it at 0.007 too.

### calibration
- `ring.thermal.heater_self_ratio` (1.21 in-ring) holds y = 0.00904 /mW, the textbook's fit with a CONSTANT
  dlambda/dT. With beta = 1.4e-3 /K the gain rises about 10 % over the G25 sweep, so a CAL-4 refit with the order-2
  model will raise y (rough estimate 0.012-0.013 /mW) and the ratio. A silicide heater (TCR 2-4e-3) needs
  R_th,H < Z to keep the droop, which is impossible: the G25 droop bounds TCR R_th,H, and R_th,H >= Z bounds the TCR
  at about 1.7e-3 /K (06 E1). The silicide family uses 2e-3 with ratio 1.0 (y 18 % high) and says so.
- The ER-mechanism alternatives (`dln_alpha_dt_ring_per_k` 2.0e-3, `..._heater_per_k` 1.6e-3) were set with the
  1.4 %/nm fallback dispersion and the 1.2 V point only (CAL-2; VAL-2 untouched). Refit on CAL-2 when the coupling
  artefact gives the solver dispersion.

### validation targets
- The source keys you asked for are all in `sources.yaml` (tested).

## 4. When the artefacts land: what to check

1. `alpha_rad_db_per_cm` of the 160 nm rib: if it is more than about 5 dB/cm, rebalance `alpha_scatter_db_per_cm`
   so alpha_i stays about 111.7 dB/cm (Q_i is fixed by Q_L and the notch, 07 D1); otherwise eta and the coupling
   regime move. Keep alpha_fca + alpha_scatter of every eta_* alternative equal to the nominal sum.
2. `n_g`: Q_i = 2 pi n_g/(lambda alpha_i), so a solver n_g of 3.9 or 4.1 moves the nominal notch from 27.9 dB to
   ~23.6 dB (under) or to near-critical. Rescale the whole budget by n_g/4.0 to keep Q_i (the CAL-2 test
   `test_g25_ring_reproduces_the_cal2_notch_in_its_labelled_regime` fails otherwise), and recompute
   `alpha_bend_radiation_db_per_cm.alternatives.strip400x200_q_anchor` = (97 + other) x n_g,200/n_g,160 - 96 - other
   (balanced between the under and over values of `other`). [Since 2026-10-16 the 200 nm world uses
   `rib200_q_anchor` = alpha_i,160 (n_g,200/n_g,160 - 1) + alpha_rad,160 per regime, then the mean; recomputed by
   tests/closure/test_params_notes.py.]
3. `dlambda_dt_pm_per_k`: the 07 expectation for H_solver is 66-69 pm/K at 25 degC; `implied_index_weight` must
   stay below `index_weight_ceiling` (derive diagnostic).
4. Re-run `pytest tests/closure/test_presets.py -m ""` and regenerate `docs/PARAMETERS.md`.

## 5. Deviations from the contract (07 is normative; each is also noted in the preset)

1. **07 4.8 entries, renamed or reshaped:** `waveguide_thickness_nm` -> `film_thickness` (entries apply in sorted
   name order; it must apply before `thermo_optic`, and a hypothesis has no unit); `power_at_ring_dbm` ->
   `chip_input_loss` (it targets the loss key: low loss = high power, corners `ring_plus8dbm`, `ring_0dbm`,
   `ocp_ring_plus9dbm`); `thermal.package_tail` -> `package_tail_tau_s` + `package_tail_share_frac` (the
   ThermalScales fields); `thermal.crosstalk` -> `crosstalk_scale` (one factor on every ring-to-ring residue: floor
   0.5-2.6 % fully, nearest 1-5.2 %, 10 % in the corner `nearest_10pct`); `heater.tcr` -> part of `thermal.family`
   (07 says "tied to the family"; the heater_self_ratio moves with it to keep the G25 droop); `self_heating.eta` ->
   5 points with triangular weights (section 3 item 1); `aggressor.referencing/.waveform` are untargeted hypotheses (world variables for the
   scenario); the waveform choice is `rectified_sine` (01 D7 correction), not "parabolic".
2. **Added entries** beyond 07 4.8: `thermal.scaling` (07 P1-2 lists it under uncertainty.thermal; `independent` is
   left out because it equals `capacitance` numerically), `driver.levels` (centered vs one-sided NRZ levels,
   plant_electrical note 4), plain alternatives with weight 0 in `thermo_optic` (bench, imec, ocp, textbook_derived)
   and in `fsr` (stress hypotheses).
3. **FSR window A** = 1.47 THz, not 07's 1.32-1.48 THz (its interior puts a second order within 2 FWHM of a TX line;
   validation_targets review of BND-1). `g26_fsr_edge` is the G25-cluster stress FSR 1.53 THz; `pdk_size` 1.60 THz
   is one `--alt` away. Note for the content owner (2026-10-16, physics_optics review): 1.47 THz was checked with
   derive's former idealized comb (f_line + k FSR, cold FWHM: 2.16-2.19 FWHM). On the plant's comb (the lock shift
   squeezes the order spacing, plus dispersion and each order's FWHM) ring 0's order +2 sits 1.74 FWHM and ring 7's
   order -2 1.68 FWHM from a TX line (the BND-1 soft fail of `g26_fsr_window_a` in docs/VALIDATION.md); derive now
   reports these values. Re-check the window A value on the plant's comb if window A is meant to pass BND-1.
4. **H86_sensor** needed a key: `ring.thermal.reference_rise_frac` (1 nominal, 0.8 under H86_sensor). The plant does
   not use it; replicas and degC scenario targets read it from `PlantParams.values`.
5. **monitor.adc_bits** (16 vs about 10, core_control_api) is an alternative only: it is a global int, which a
   hypothesis cannot vary per world.
6. **Thermal model and junction model stay interim / analytic** in g26 (ARCHITECTURE 5.3; 07 build order: each
   artefact replaces assumed values through the calibration split). The in-ring FEM ROM exists, but see section 3
   (plant_thermal) and request 2 of params_machinery.
7. **Ranges include every selectable alternative** (a selected alternative is range-checked): for example
   `fsr_thz` [1.4, 1.9], `q_loaded` [2600, 4200], `p_fsr_mw` [24.9, 37]. The plausible spans are in the notes.
8. **Schema doc fixes on request:** `depletion_exponent` doc now says 1/2-2/3 by charge control (07 wrote 1/3-1/2;
   bounds unchanged); `backscatter_orders` and `tcmt_orders` got bounds >= 1 (plant_optics request).

## 6. How to use

```python
from mrm import params
cfg = params.load("g26")                                        # default: G26 TX bank
cfg = params.load("g26_h86")                                    # a corner
cfg = params.load("g26", alts={"uncertainty.thermo_optic": "bench"})          # any hypothesis choice
cfg = params.load("g26", alts={"uncertainty.thermal.family": "metal_above"})  # heater family
cfg = params.load("g26", overrides=["ring.optics.q_loaded=4000"])             # a plain override
pp = params.realize(cfg, batch=256, seeds=0, uncertainty="lhs")               # 07 4.8 worlds x devices
nominal = params.derive(cfg)                                    # B = 1 nominal view + diagnostics
params.load("teaching_basic")                                   # deterministic, no self-heating, no noise
params.load("g25_test_vehicle")                                 # one ring, one line: the calibration replicas
```

`mrm params validate g26 g26_* g25_test_vehicle teaching_basic` checks them all; `mrm params explain g26
ring.thermal.dlambda_dt_pm_per_k` prints the chain and the alternatives.

## 7. Adversarial review (changes made by the reviewer)

Each item has a regression test in `tests/closure/test_presets.py` unless stated.

1. **Monitor current vs CAL-8 and the ADC (fixed).** kappa_d/kappa = 0.01 gave a drop peak of 46 uA at the nominal
   +4 dBm (0.95 A/W), above the 40 uA ADC full scale, and 34 uA at a max-slope lock against the ~15.5 uA of
   LM26-P Fig. 3b. Now 0.007 (the log-centre of 07 D7's 0.4-1.2 %): peak 32 uA, 16 uA at a half-maximum lock.
   `alpha_other_abs` was re-solved on the exact G25 ring: 14.74 (under) and 5.97 (over) give 27.90 dB each;
   the eta alternatives keep alpha_fca + alpha_scatter = 96 dB/cm. Alternatives `phy_low` / `phy_high` added; the
   `gf_pdk` note now says it needs a rebalanced budget (alone it gives an ~8.3 dB notch).
   Test: `test_monitor_current_is_consistent_with_cal8_and_does_not_clip`.
2. **The 200 nm film world broke CAL-1/CAL-2 (fixed).** With the budget fixed in dB/cm, the strip's n_g 4.349 raised
   Q_i 8.7 %, so both regime choices were over-coupled (27.5 and 18.6 dB) in every film-200 world. The '200' choice
   now selects `alpha_bend_radiation_db_per_cm: strip400x200_q_anchor` (10.37 dB/cm: the strip radiation plus the
   non-heating loss that keeps the measured Q_i): 28.6 dB in both labelled regimes. Cost: eta x0.92 in those worlds
   (which component carries the extra loss is unknown; only a non-heating term composes with the eta and regime
   hypotheses, because entries apply in sorted order). Test: `test_g25_ring_reproduces_the_cal2_notch_...`.
   [Superseded 2026-10-16 (params review): the '200' choice is the 400 x 200 nm rib (modes_rib200) with
   `rib200_q_anchor` 2.03 dB/cm, eta x0.98; the strip stays a plain alternative.]
3. **Reset ramp too short (fixed).** `sim.reset_ramp_s` 0.5 ms was below 07 P0-1 item 6 (>= 10 tau_slow) for the
   undercut family (2 x 45.7 us x 10 = 0.91 ms) and the textbook corner (3 ms). Now 3 ms.
   Test: `test_reset_ramp_covers_ten_slow_thermal_poles`.
4. **G25 bench heater (fixed).** g25_test_vehicle left `uncertainty.heater.driver_mode` open, so its worlds used
   current, power and DAC+PWM drive although the G25 sweep is printed against a bench voltage. Pinned to voltage
   with `alts`. Its `laser.power_per_line_dbm` also inherited the M1000 textbook field and disagreement; now
   `REF.inputPowerDbm`, no disagreement. Test: `test_g25_bench_heater_is_voltage_driven_in_every_world`.
5. **Eta corners did not hold their label (fixed).** In over-coupled or 200 nm worlds `g26_low_eta` gave 0.24-0.28
   and `g26_hot_selfheat` 0.876-0.946 (the alternatives are exact only for the under-coupled 160 nm world). Both
   corners now also pin `coupling_regime: under` and `film_thickness: "160"`.
   Test: `test_corner_pins_what_it_says_in_every_world` (every corner, 8 LHS worlds).
6. **Eta discretization (improved).** 3 points with weights 2:1:1 over-weighted the extremes; now 5 points (0.5,
   0.65, 0.75, 0.848, 0.95) with the masses of a triangular density on [0.5, 0.95] with its mode at the nominal
   (1 : 6 : 9 : 10 : 2). Test: `test_eta_choices_keep_q_intrinsic_and_hit_their_label`.
7. **Through-gap variation missing (added).** 07 P1-16 asks for a gap_through sigma; the Jacobian cannot move Q_t
   (free under the Q_L anchor). Added `variation.ring.optics.q_loaded` rel_sigma 0.015 (d ln K^2/d gap -2 %/nm x
   ~1.5 nm, assumed), so the per-ring notch spreads around 27.9 dB as near-critical rings do. [Superseded 2026-10-16
   (params review): the coupling Jacobian now moves the per-pass couplings of every device whichever R4 member is
   free; g26 draws gap_through and gap_drop (one 1.5 nm local sigma) through coupling_rib160 and this stand-in and
   the kappa_d_over_kappa one are removed.]
8. **Notes corrected.** q_loaded and coupling_regime notes now state the regime flip (section 3 item 1); the P_FSR
   and R_cold variation notes state that a symmetric lognormal cannot reproduce the skewed G25 wafer maps (1/3 of
   the drawn rings fall below the observed 24.9 mW minimum; the +34 % tail is missing); the heater_self_ratio note
   states that y follows Z across worlds (below); the inventory's DAC-noise and P2-5 entries were corrected.
9. **Textbook corner failed realize for Monte Carlo batches (fixed).** With its 20 us crosstalk pole the lagged
   cross-ring optical gain reached 0.0500-0.0504 in batches of 256-512 LHS worlds with device variation, so
   `realize()` raised (the builder's 16-world test missed it; it fails without this review's other changes too).
   The stand-in pole is now 25 us, which is also the minimax fit of the textbook shape (0.161 vs 0.174 max error):
   max gain 0.044 over 5 x 512 worlds. Test: `test_preset_realizes_a_monte_carlo_batch` (slow, every preset,
   512 worlds).
10. **Tests added:** CAL-2 notch per regime (incl. 200 nm), eta choices, CAL-8 and ADC clipping, the reset ramp, the
   G25 driver law, the Berkeley 6 nm / 2 nm spreads, a per-corner pin check in sampled worlds (and a check that every
   corner has one), alternative source keys, a drift guard that the test-local adapter equals `config._make_leaf`
   except for the fallback, a PARAMETERS.md staleness check under the adapter, and a 512-world Monte Carlo realize of
   every preset (slow).

Still open after the review (requests):
- **params_machinery, BLOCKING:** section 1 (artefact fallback). Without it `mrm.params.load("g26")` raises for every
  user, and `test_params_cli.py::test_parameters_md_is_current` fails; with the adapter installed the committed
  `docs/PARAMETERS.md` is current (checked).
- **params_machinery:** tied draws (section 3 item 1) are now the main accuracy gap (Q_L vs regime).
- **plant_thermal / params_machinery:** the interim model takes `heater_self_ratio` (R_th,H / Z_heater->ring), so the
  G25 droop y = TCR x R_th,H follows Z across the FSR and thermo-optic worlds although the measured droop does not:
  y = 0.0073 /mW under fsr window_a (Z 4.31 K/mW), 0.0079 under H86, 0.0076 under g25_cluster (nominal 0.00904).
  An optional absolute `ring.thermal.heater_self_k_per_mw` (or y itself) would hold it.
- **params_machinery + plant_thermal:** 07 P2-5 (temperature-dependent conductivity) is in the v1 scope but has no
  schema switch; `mrm.plant.thermal.k_temp_residue_scale` and the FEM sidecar `k_temp_scaling` exist. A key such as
  `ring.thermal.k_temp_model: off | film_fit` needs a realize path; I did not add a key nothing reads.
- **calibration:** with kappa_d/kappa 0.007 the ring_t_loss alternative (2.0e-3 /K, exponential) gives ~20.2 dB at
  1.2 V (CAL-2 19.8 dB, within its sigma); the dispersion-only mechanism crosses critical coupling (~43 dB near
  1.0 V) and cannot fit CAL-2's 24.4/22.1/19.8 dB, as 07 P1-5 says. The plant does not consume these keys yet, so
  whether the coefficient multiplies all of alpha_i or only alpha_FCA (07 P0-2 writes alpha_FCA (1 + c_T dT)) must be
  fixed before the CAL-2 refit.
- **g25_test_vehicle:** its chip_input_loss interval 0-2 dB gives 2-4 dBm at the ring, one-sided around the
  printed "about 4 dBm" (a negative loss is not allowed); acceptable for the replicas, which use the nominal.
