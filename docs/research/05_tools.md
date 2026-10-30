> the t11 Gymnasium and t14 PPO research smoke scripts now live in [archive/rl/research](../../archive/rl/research). commands below record their historical locations.

# 05 - Open-source tool landscape, verified by installation

Status: research note for the `mrm-model` plant. Written 2026-10-09 on Kenny's Apple M2 (8 cores, 16 GB, macOS
Darwin 25.1). Scope: pick the most accurate **free, open-source, locally runnable** tool for each sub-problem of
the microring-modulator (MRM) model, and prove on this Mac that it installs and gives sane numbers on a tiny case.
Every test script is in `docs/research/05_tools_smoke/` (index in section 9); every number below came from a run
in this session unless it is tagged otherwise.

## 0. How to read this file

**Provenance words** (textbook vocabulary, `docs/SOURCES_AND_GENERATIONS.md`): **printed** (verbatim in the
source), **read-off** (digitized from a figure), **inferred** (arithmetic on printed values), **derived** (computed
by us; every smoke-test output here is *derived*), **assumed** (no source; every geometry or material value that a
smoke test needed and that has no Lightmatter source is *assumed* and says so).

**Shorthand used in the command column.** All commands run from `/Users/kenny/mrm-model`.

- `MG` = `.venv/bin/python tools/memguard.py --limit-gb 3 --` (memory watchdog; no run needed `--heavy`; the
  largest peak was 2.22 GB).
- `UVR` = `~/.local/bin/uv run --isolated --no-project --python 3.12` (uv 0.9.13). It builds a throw-away
  environment in uv's cache, so nothing was installed into the shared `.venv` (other agents use it).
- `NONET` = `env HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9`
  (points every HTTP client at a dead port, so a solver that needs the network fails).
- `MEEP` = `/Users/kenny/photonics-simulations/.meep/bin/python` (existing micromamba env: pymeep 1.34.0,
  mpb 1.12.0, harminv 1.4.3, numpy 2.5.3, all conda-forge, serial build).

**Test geometry.** Lightmatter's dimensioned strip, 400 x 200 nm (printed, SMTA Oct 2025, `REF.waveguideLmNm`;
single-source per the textbook), Si core in SiO2, lambda = 1310 nm, indices from the textbook's Sellmeier fits
`nSilicon`/`nSilica` (n_Si = 3.50053, n_SiO2 = 1.44680 at 1.31 um), thermo-optic `REF.dnSi_dT = 1.86e-4 /K`,
`REF.dnSiO2_dT = 1e-5 /K`. Ring radius 7.5 um is **assumed** from the "about 15 um diameter" that `REF` says
Lightmatter slides print; gaps, doping, stack and heater geometry are **assumed**.

## 1. Bottom line

| Sub-problem | Chosen tool (licence) | Verified here | Second opinion / cross-check | Main accuracy limit found |
|---|---|---|---|---|
| Straight waveguide modes: n_eff, n_g, Gamma, dn_eff/dT | **femwell 0.1.12** (GPL-3.0) on scikit-fem + gmsh | yes (C2) | Tidy3D 2.12 local mode solver (LGPLv2+ per package metadata, solver core MIT) yes (C1); MPB 1.12 yes (C5) | three solvers agree on n_eff to 0.04 %; femwell and Tidy3D agree on dn_eff/dT to 0.09 % [verified: an independent MPB script by the checker gives n_eff 2.48495, dn_eff/dT 2.0428e-4 /K, n_g 4.3346 at 128 px/um, so the three-solver dn_eff/dT spread is 0.13 %. Note: the Si index itself is less certain than the solvers: Frey 2006 Table 2 prints 3.50616 at 1.3 um, 295 K, about 3.5048 at 1.31 um, i.e. 0.004 above the Li-form 3.50053 used here; that moves n_eff by about 0.004 (derived, x 1.086), 4x the solver spread] |
| Bent modes | Tidy3D local (exact radial transform, PML) **and** femwell (conformal eps(1+x/R)^2), always both | yes (C1b, C4) | Meep cylindrical FDTD yes (C3) | bend correction at R = 7.5 um (<= 0.002 in n_eff) is smaller than solver-to-solver spread; Tidy3D gives a +/-0.0015 sign asymmetry on a 10 nm grid [corrected: the checker's rerun shows the Tidy3D +R/-R mean shift is +0.00038 on 20, 10 and 5 nm grids and on both plane sizes (spread < 3e-6), so it is resolved; the asymmetry scales as 1/R, which marks a coordinate-offset artifact, not physics. The real disagreement is femwell +0.0014 vs Tidy3D +0.00038 (3.7x), see section 5.4] |
| Ring-bus coupling and its dispersion | femwell supermodes (or overlap CMT) + point-coupler integral | yes (C2) | Meep 3-D FDTD of the coupler (heavy, see lab visuals/03) | coupled-mode approximation; 3-D FDTD not converged at 20-25 px/um in the lab |
| FDTD ring spectrum | **Meep 1.34** (GPL-2.0+) via conda-forge | yes (C3) | Tidy3D FDTD is cloud-only, paid | Meep converges only ~1st order here (p = 0.94) [corrected: p = 0.94 comes from the three finest resolutions (40/60/80) only; a least-squares fit of all five gives p = 0.79, and the three-point order rises from 0.69 to 0.94 as coarse points are dropped, so the runs are pre-asymptotic, see section 3.2]; 3-D ring+bus does not fit the 3 GB budget |
| Transient heat, layered stack | **scikit-fem 12.0.2** (BSD-3) axisymmetric r-z and 3-D | yes (C6) | FEniCSx 0.11 (LGPL-3.0) yes (C13) for >1e5 DOF 3-D | material data and geometry are the error, not the solver |
| Thermal model-order reduction | **pyMOR 2026.1.1** (BSD-2/3) balanced truncation | yes (C6) | IRKA (pyMOR), Krylov moment matching (scipy), python-control 0.10.2 + slycot 0.7.0 `balred` | BT 6 states: 0.39 % worst-case transfer error, DC gain 0.39 % low (use SPA or `matchdc` if DC matters) |
| Semiconductor junction (carriers vs bias) | **DEVSIM 2.11.0** (Apache-2.0) | yes (C7a, C7b, C7c) | femwell `pn_analytical` (depletion approximation + Nedeljkovic 1310 nm coefficients); Tidy3D Charge is cloud-only | 1-D cut verified; a 2-D rib with slab contacts is not yet built |
| WDM bus S-matrix, many rings | **SAX 0.18.2** (Apache-2.0, JAX 0.9.2) for netlists; closed-form numpy/numba inside the loop | yes (C8) | scikit-rf 2.1.0 (BSD-3) for electrical parts yes (C10) | none (SAX equals the numpy cascade to 4.4e-16) |
| Closed-loop time-domain simulation | **numba 0.68** (BSD-2) kernels; python-control for linear design | yes (C11) | JAX (Apache-2.0) when gradients are needed | toy plant only; 146x faster than pure Python |
| RL interface | **gymnasium 1.4.0** (MIT) | yes (C11) | Stable-Baselines3 2.9.0 (MIT) + torch 2.14.1 trains on it, yes (C14) | none found |
| Fixed point / FPGA path | plain-integer Python golden model + **cocotb 2.1.0** (BSD-3) | yes (C12) | Icarus Verilog 12.0 (GPL-2.0+, already in Homebrew) and Verilator 5.052 (LGPL-3.0 or Artistic-2.0, conda-forge) both pass; fxpmath 0.4.10 (MIT), apytypes 0.5.1 (MIT) | the golden model caught a real Verilog signedness bug (54/2000 cycles) |

Rejected or not usable: **Photontorch 0.4.1** (AGPL-3.0, last release 2021-02-13) fails with torch 2.14.1 (Python
3.12) and torch 1.12.1 (Python 3.10) and runs only with Python 3.9 + torch 1.11.0 (C9b); **Simphony 0.7.3** (MIT)
works but pins `sax==0.14.7` [corrected: PyPI metadata declares `sax<0.15.0`, which resolves to 0.14.7, the newest 0.14.x; the conclusion stands], so it cannot share an environment with SAX 0.18 (C9a); **EMode** is proprietary
(free "EMode2D" seat at "USD 0 per year", EMode3D "USD 5000 per year", printed on emodephotonix.com/get-emode;
needs an account, not tested); **Elmer FEM** (GPL) has no Homebrew formula and no conda-forge osx-arm64 build, so
it is a source build with gfortran (not tested); **Tidy3D FDTD, EME, Heat and Charge** run only on Flexcompute's
servers and cost FlexCredits.

## 2. Environment plan (what to install where)

The shared `.venv` (uv, CPython 3.12.8) already holds femwell 0.1.12, scikit-fem 12.0.2, gmsh 4.15.2, meshio,
meshwell 2.3.6, shapely, pyamg 5.3.0, tidy3d 2.12.0, devsim 2.11.0, numba 0.68.0, gymnasium 1.4.0, casadi 3.8.1,
nlopt 2.11.0, numpy 2.4.6, scipy 1.18.1. `uv pip install --dry-run --python .venv/bin/python <pkg>` (run for each
package below, results quoted) shows what each addition would do:

| Package | Dry-run effect on `.venv` | Recommendation |
|---|---|---|
| `pymor` | adds 4 (pymor 2026.1.1, diskcache, qtpy, threadpoolctl), changes nothing | add (extract group) |
| `control slycot` | adds control 0.10.2, slycot 0.7.0 | add (slycot only where `balred` is needed: it is GPL-2.0) |
| `sax` | adds 18 incl. jax 0.9.2, jaxlib 0.9.2, scikit-rf 1.13.0, sympy, networkx, optax, klujax; changes nothing | add if netlists are needed |
| `jax` alone | would install jax 0.11.2 | let SAX pin it instead |
| `cocotb fxpmath apytypes` | adds 6, changes nothing | add (hdl group) |
| `gdsfactory` | adds ~67 (jupyter stack, kfactory 3.0.4, klayout 0.30.12) and **downgrades pydantic 2.14.0 -> 2.13.5, cachetools 7.2.1 -> 7.1.8** | separate env (tidy3d accepts pydantic >=2.10.3,<3, so it would probably work, but not worth the risk) |
| `simphony` | pulls `sax==0.14.7` (from its `sax<0.15.0` requirement), flax, orbax, tensorstore | do not install |
| `photontorch` | declares no dependencies at all (not even numpy) | do not install |

Separate environments: Meep/MPB/Harminv have no PyPI wheels, so conda-forge is the practical source on macOS arm64
(`pip install meep` gets an unrelated 2018 PyPI package called "meep", version 1.0.6), so keep a micromamba env (reuse `/Users/kenny/photonics-simulations/.meep`
or recreate it with `micromamba create -p .meep -c conda-forge python=3.12 pymeep=1.34.0`). FEniCSx and Verilator,
if needed, also come from conda-forge (C13, C12). Note: tidy3d 2.12 requires `numpy<2.5.0`, so the Meep env
(numpy 2.5.3) and the main env cannot simply be merged.

The repo's `pyproject.toml` (written by another agent) already separates a GPL-heavy `extract` group (femwell,
gmsh, devsim, tidy3d) from the runtime (`numpy, scipy, matplotlib, pyyaml, numba`). That split matches section 7:
the runtime plant should load precomputed matrices and tables and not import GPL code. Suggested additions:
`extract += pymor, control`, `rl += stable-baselines3`, a new `hdl = cocotb, fxpmath`.

## 3. Sub-problem tables

Column "Verified" gives the command ID (section 9 has the exact command) and the result. "Accuracy limits" lists
what was *observed*, then what is *known* from the tool's documentation.

### 3.1 Waveguide eigenmodes: n_eff, n_g, confinement, dn_eff/dT (straight)

| Tool | Role | Version, licence, arm64 | Verified | Accuracy limits |
|---|---|---|---|---|
| femwell | **chosen** | 0.1.12 (2025-01-06), GPL-3.0, pure Python on scikit-fem (BSD-3) + gmsh 4.15.2 wheel (GPL-2.0+, arm64 wheel) | yes, C2: TE0 n_eff = 2.48560 with 10 nm core elements and 2.48560 with 5 nm (change 7e-6); n_g = 4.3359 (Sellmeier materials, +/-5 nm central difference); power fraction in core 0.807; dn_eff/dn_Si = 1.086; dn_eff/dT = 2.0436e-4 /K; 5.8 s per solve, 4322 triangles, peak 1.46 GB for the whole script | conforming triangles, 2nd-order Nedelec, so no staircasing; converged to 1e-5. No PML (modes near cut-off feel the box). `calculate_confinement_factor` is dn_eff/dn, not a power fraction (lab TOOLS.md) |
| Tidy3D `ModeSolver.solve()` | cross-check | 2.12.0 (2026-07-27), client LGPLv2+ per package metadata, mode-solver core MIT (`tidy3d/components/mode/LICENSE`, derived from ceviche), pure Python | yes, C1 with the network blocked (`NONET`) and no API key on the machine (`~/.tidy3d` absent, no TIDY3D env var): n_eff = 2.48738 (10 nm grid, 2.61 x 2.01 um plane), 2.48602 (5 nm grid, 1.605 x 1.205 um plane; the same small plane at 10 nm gives 2.48736, so the plane size is not the cause); built-in n_group = 4.1425; TE fraction 0.983; TM0 n_eff 1.8937; dn_eff/dT = 2.0454e-4 /K; 24 s per 2-mode solve, import alone 12 to 17 s | no subpixel smoothing locally (keep the plane an odd multiple of the pitch, lab trick); grid error 0.0014 in n_eff at 10 nm; a uniform 5 nm grid on the 2.6 x 2.0 um plane exceeded 3 GB and was killed (C1 first attempt), use a small plane or a non-uniform grid. Built-in n_group uses the media as given, so with constant-permittivity media it excludes material dispersion (4.1425 vs 4.3380 by finite difference with Sellmeier media, C1b) |
| MPB (in the Meep env) | cross-check | 1.12.0, GPL-2.0+, conda-forge osx-arm64 | yes, C5: n_eff 2.47262 / 2.48372 / 2.48495 at 32 / 64 / 128 px/um (constant indices), waveguide n_g 4.1393 at 128 px/um; 1 s | plane-wave, periodic supercell; converges fast here (differences 0.0111 then 0.0012) |
| EMode | not tested | proprietary; "EMode2D ... USD 0 per year" free seat (printed, vendor page) | no: needs an account and licence activation, and a university guide (Aarhus) lists Windows 10 as the requirement | n/a |

### 3.2 Bent modes (ring waveguide)

| Tool | Role | Verified | Accuracy limits |
|---|---|---|---|
| Tidy3D local, `ModeSpec(bend_radius=R, bend_axis=1, num_pml=(12,12))` | chosen (with femwell) | yes, C1b: same 3.61 x 2.01 um plane and PML for straight and bent: straight 2.48738, R = +7.5 um 2.48629, R = +5 um 2.48603, **R = -7.5 um 2.48923**; k_eff negative and below 1e-9 in magnitude (radiation loss below solver resolution) | full transformation with Jacobians on both E and H (`components/mode/transforms.py`), PML gives radiation loss. **Observed: the two mirror-image bends differ by 0.0029 in n_eff on a 10 nm grid**, so the sign-averaged bend shift (+0.0004) is below the discretization error [corrected: the checker reran +/-R on a 1.6 x 1.2 um plane without PML at 20 / 10 / 5 nm: the mean shift is +0.000380 / +0.000378 / +0.000378, the same as on the 3.61 um plane, so it is grid-converged and is the usable Tidy3D bend value. The asymmetry is 0.0024 / 0.0012 / 0.0015 (not monotonic in the grid step; 0.0029 on the 3.61 um plane) and drops by 6.7x at R = +/-50 um, i.e. it scales as 1/R. That is the signature of a radius-reference offset of 2 to 4 nm, not a bend effect. A candidate cause, read in `tidy3d/components/mode/transforms.py` but not proven: the E-field Jacobian `dwdz_e = radius / u[:-1]` is evaluated at the cell-edge coordinate for all E components. Average +R and -R] |
| femwell `compute_modes(..., radius=R)` | chosen (with Tidy3D) | yes, C2/C4: straight 2.48561, R = 7.5 um 2.48702 (+0.0014), R = 5 um 2.48878 (+0.0032) | isotropic conformal map eps*(1 + x/R)^2 applied to the whole form (read in `femwell/maxwell/waveguide.py`); an approximation, no PML, so no bend loss |
| Meep cylindrical FDTD, `dimensions=mp.CYLINDRICAL, m=89` | arbiter | yes, C3: isolated ring R = 7.5 um, constant indices. Resonance of m = 89: 1.36911 / 1.35271 / 1.34357 / 1.33395 / 1.32903 um at 20 / 30 / 40 / 60 / 80 px/um (18 s to 639 s, peak 0.19 GB). Fitted convergence order p = 0.94; extrapolated 1.3132 um (p fitted) or 1.3143 um (p = 1 from 60/80) [corrected: p = 0.94 / 1.3132 um is the exact three-point solution through 40/60/80 only. A least-squares fit of a + b h^p to all five points gives p = 0.79 and 1.3089 um; the other three-point sets give p = 0.69 to 0.91 and 1.3019 to 1.3126 um. The order rises as coarse points are dropped, so the runs are pre-asymptotic and the extrapolated resonance is uncertain by about 5 nm (1.309 to 1.314 um)]. Harminv Q_rad > 1e10 at every resolution (some negative: unresolvably large) | exact 3-D vectorial ring for one azimuthal order, but **only first-order convergence** here (Meep's own docs: corners and discontinuous fields give "error intermediate between first and second order"); at 80 px/um the resonance is still 15 nm (1.1 %) red. Needs Richardson extrapolation; cannot arbitrate a 0.001 n_eff difference |
| prediction of the same m = 89 resonance from mode solvers | C4 | femwell bend 1.31409 um, femwell straight 1.31365, Tidy3D straight 1.31421, Tidy3D +R 1.31386, Tidy3D -R 1.31480 | spread 1.1 nm, which contains the extrapolated Meep value [corrected: the range 1.31365 to 1.31480 um contains only the p = 1 extrapolation (1.31429 um). The p = 0.94 value (1.31320 um) is 0.45 nm below it, and the five-point fit (1.3089 um) is 4.7 nm below. Meep is consistent with the mode solvers within its own ~5 nm extrapolation uncertainty, which does not confirm them to 1 nm] |

**Consequence for the model (derived).** At R >= 5 um the bend changes n_eff by at most 0.003 and adds no
measurable radiation loss; solver-to-solver spread on the absolute resonance is about 1 nm, and the lab computed
346 pm of resonance shift per nm of width for the 500 x 220 strip (lab experiment 08). So the plant model should
take n_g and dn_eff/dT from the straight-guide solvers and treat each ring's absolute resonance as a calibrated
(random) parameter, not a solver output. [verified, and strengthened: the Si index source alone (Li-form 3.5005 vs Frey
about 3.5048 at 1.31 um, 295 K) moves the resonance by 1.3 to 1.4 nm (derived: 0.0039 to 0.0043 in n_Si, depending on whether the 2 K between
293 and 295 K is corrected, x 1.086 x 1310 / 4.34), which is more
than the solver spread. The 346 pm/nm figure is in the lab's `experiments/08_soi_strip_mode_solvers/out/results.txt`.]

### 3.3 Ring-bus coupling coefficient and its wavelength dispersion

| Tool | Role | Verified | Accuracy limits |
|---|---|---|---|
| femwell supermodes of two parallel 400 x 200 nm guides, kappa = pi (n_even - n_odd) / lambda | **chosen** | yes, C2 (12 solves, 15 nm core mesh): kappa at gap 0.15 / 0.20 / 0.25 / 0.30 um and 1310 nm = 0.0646 / 0.0377 / 0.0222 / 0.0132 rad/um; decay constant 10.9 / 10.6 / 10.3 /um at 1290 / 1310 / 1330 nm. Point-coupler integral for R = 7.5 um and g0 = 0.20 um (both assumed): power coupling K^2 = 0.49 % / 0.63 % / 0.81 % at 1290 / 1310 / 1330 nm, i.e. dK^2/dlambda = 8.2e-5 per nm (+1.3 % of K^2 per nm) [verified: kappa = pi (n_e - n_o)/lambda = Delta-beta/2 is the standard supermode form; recomputed K^2 = sin^2(kappa0 sqrt(2 pi R / gamma)) = sin^2(0.0377 x 2.109) = 0.630 % at 1310 nm, and (0.814 - 0.486) %/40 nm = 8.2e-5 per nm] | coupled-mode approximation: ignores the phase mismatch between straight bus and curved ring and the bend-mode shape; gap is the dominant sensitivity (kappa changes about 10 % per 10 nm of gap here, derived). femwell also offers `calculate_coupling_coefficient` (overlap form) |
| Meep 3-D FDTD of the coupler region | cross-check, heavy | not rerun; the lab's `visuals/03_coupled_modes/fdtd3d.py` did it for 500 x 220 nm: L_c 65.0 um at 20 px/um, 59.6 um at 25 px/um vs 65.6 um from converged CMT, "not grid-converged", 35 to 80 min (lab README) | consistent with the slow convergence in C3 |
| Tidy3D EME / FDTD | not free | cloud, FlexCredits | n/a |

### 3.4 2-D / 3-D FDTD validation of a ring spectrum

| Tool | Role | Verified | Accuracy limits |
|---|---|---|---|
| Meep 1.34.0 (conda-forge) | **chosen** | yes, C3 (cylindrical, above); the lab's experiment 12 has a working 2-D ring + bus with loss calibration, Harminv and CW build-up (`ring_meep.py`, 408 s) | 2-D rings are not the 3-D ring (lab: n_g 3.59 instead of 4.2). Full 3-D ring + bus at 40 px/um is about 800 x 800 x 120 = 77 M voxels, of order 10 GB (derived from ~150 bytes/voxel), so it does not fit this machine's budget; use cylindrical (isolated ring: resonance, Q_rad) + 3-D coupler section + circuit model instead |
| Tidy3D FDTD | not free | cloud only | n/a |
| `fdtd` (PyPI, 0.3.5) | rejected | not rerun (lab installed it, unused) | teaching code, no eigenmode source |

### 3.5 Transient heat conduction in a layered stack (axisymmetric and 3-D)

Stack for C6 (all **assumed**): Si substrate truncated at -22 um with fixed bottom temperature, 2 um BOX, ring
core 400 x 200 nm at R = 7.5 um that also carries the heat (1 mW uniform), 3 um top oxide, adiabatic top and side,
k_Si,film 100, k_Si,sub 148, k_SiO2 1.4 W/(m K) (`REF` uses 1.4), rho c 1.66e6 / 1.62e6 J/(m^3 K).

| Tool | Role | Version, licence | Verified | Accuracy limits |
|---|---|---|---|---|
| scikit-fem | **chosen** | 12.0.2, BSD-3, pure Python | yes, C6. Axisymmetric (weight 2 pi r) on a graded tensor mesh aligned to every layer edge: 3465 nodes, 3402 states after the Dirichlet bottom; steady R_th = 8.19 K/mW, four slowest time constants 13.7 / 10.9 / 7.9 / 5.5 us, -3 dB at 60 kHz (all derived for the assumed stack) [verified as plausible, not re-run: a 1-D BOX estimate (2 um, k = 1.4) gives 8.19 K/mW for a 3.7 um effective conduction width under the 0.4 um ring, and a -3 dB of 90 kHz with no lateral spreading. The 3-D straight-heater value 31.7 K/mW agrees with 8.19 x (2 pi 7.5 um / 12 um) = 32.2 K/mW]. 3-D block (straight 12 um heater line): 22 360 nodes, 117 936 tets, steady solve 2.6 s, R_th 31.7 K/mW. Whole script 82 s, peak 0.70 GB | P1 elements; the matrices M, K, b, c are plain scipy sparse, which is what MOR needs. Direct LU is fine to ~1e5 3-D nodes; beyond that use pyamg (installed) or FEniCSx |
| femwell `thermal`, `thermal_transient` | helper | in femwell 0.1.12 | not run (read the source) | 2-D cross-section only, Joule heating from current densities |
| FEniCSx (dolfinx) | scale-up | 0.11.0, LGPL-3.0, conda-forge osx-arm64 (227 MB download, 125 packages) | yes, C13: 3-D box 96 x 24 x 64 cells, 884 736 tets, 157 625 DOFs, CG + hypre, steady + 20 backward-Euler steps in 92 s, peak 1.28 GB; R_th 23.8 K/mW (coarse 0.25 um cells widen the heater, so not comparable to the scikit-fem number) | PETSc/MPI; form compilation at first run |
| Elmer FEM | not tested | GPL; no Homebrew formula, not on conda-forge osx-arm64 | no | source build with gfortran + cmake |
| Tidy3D HeatChargeSimulation | not free | cloud (`web.run`), FlexCredits, "beta" | no | n/a |

### 3.6 Thermal model-order reduction to a small state space

Same axisymmetric model (3402 states), input = heater power, output = mean core temperature, error = max over 36
log-spaced frequencies (10 Hz to 100 MHz) of |H_r - H| / max |H| (C6, all derived):

| Method (tool) | 2 states | 4 states | 6 states | 8 states | Time |
|---|---|---|---|---|---|
| one-sided Krylov, moment matching at s = 0 plus real shifts (scipy) | 25.9 % | 15.8 % | 8.0 % | 6.1 % | < 0.3 s; DC gain exact (8.188 K/mW) |
| balanced truncation (pyMOR `BTReductor`, low-rank ADI) | 15.1 % | 2.05 % | **0.39 %** | **0.081 %** | 7.8 s first call (Gramians), then 0.01 s |
| IRKA (pyMOR `IRKAReductor`, H2-optimal) | 41.7 % | 14.1 % | 1.6 % | 0.49 % | 7 to 27 s |

| Tool | Role | Version, licence | Verified | Accuracy limits |
|---|---|---|---|---|
| pyMOR | **chosen** | 2026.1.1, BSD-2-Clause AND BSD-3-Clause, pure Python | yes, C6 | BT does not preserve DC gain (6 states: 8.156 vs 8.188 K/mW, -0.39 %); my Krylov shift choice was naive, not a verdict on Krylov |
| python-control | analysis, small dense balred | 0.10.2, BSD-3 | yes, C6: 6-state ROM time constants 9.55 / 1.33 / 0.30 / 0.071 / 0.015 / 0.0024 us; `balred(6 -> 3, method="truncate")` DC 7.738 K/mW (-5.5 %) | `balred` needs slycot (GPL-2.0, arm64 wheel 0.7.0 installs fine); dense only |
| scipy (custom Krylov / PRIMA) | fallback | BSD-3 | yes, C6 | fine for SISO; choose shifts by IRKA or by the loop bandwidth |

### 3.7 Semiconductor junction: carrier density vs bias in a lateral PN rib

| Tool | Role | Version, licence | Verified | Accuracy limits |
|---|---|---|---|---|
| DEVSIM | **chosen** | 2.11.0, Apache-2.0, arm64 wheel, already in `.venv` (examples in `.venv/devsim_data/examples`) | yes. C7a: the shipped `diode_1d.py` runs in 1.8 s and conserves current to 4e-6 (top 0.018155, bottom -0.018155 A/cm^2 at 0.5 V). C7b: shipped `gmsh_diode2d.py` (2-D gmsh mesh) runs in 0.6 s. C7c (`t04`, 1-D lateral cut, N_A = N_D = 5e17 cm^-3 **assumed**, contacts at +/-0.5 um, 1281 nodes, 9 biases in 2.7 s): depletion width 63.8 nm at 0 V and 117.1 nm at -2 V vs abrupt-depletion formula 68.9 / 122.8 nm [corrected: the formula values are right for eps_r = 11.7 (V_bi = 0.9166 V; checker recomputed 68.86 / 122.83 nm), but DEVSIM's shipped `simple_physics.py` sets `eps_si = 11.1` (and eps_0 = 8.85e-14 F/cm), so the comparison is not like-for-like: at 11.1 the formula gives 67.1 / 119.6 nm, and the 2kT/q Debye correction brings it to 65.2 / 118.6 nm. For the plant, override Permittivity to 11.7 eps_0, because both C_j and W scale as sqrt(eps_r) (2.7 % here)]; electron loss in the 400 nm core 1.34e12 cm^-2 at -2 V, equal to the hole loss (neutrality) | 1-D cut ignores the slab and 2-D field spreading; a 2-D rib + slab mesh (gmsh) with P++/N++ contacts is the next build. `simple_physics` uses constant mobilities (mu_n 400, mu_p 200 cm^2/Vs) and n_i = 1e10 cm^-3, no bandgap narrowing; the DC charge-vs-bias result does not depend on mobility, but the RC/bandwidth does (DEVSIM ships `Klaassen.py` for doping-dependent mobility, not tested) |
| femwell `pn_analytical`, `tcad` | helper | GPL-3.0 | read the source | depletion approximation; plasma-dispersion coefficients at 1.31 um coded as dn = -2.98e-22 dN^1.016 - 1.25e-18 dP^0.835 (femwell transcription citing Nedeljkovic et al., IEEE Photonics J. 2011; not checked against the paper here) [verified against an independent transcription: Tidy3D 2.12's `NedeljkovicSorefMashanovich` default table (`components/parameter_perturbation.py`, also citing doi 10.1109/JPHOT.2011.2171930), row lambda = 1.3 um, has the same eight numbers: dn = -(2.98e-22 dN^1.016 + 1.25e-18 dP^0.835), dalpha = 3.48e-22 dN^1.229 + 1.02e-19 dP^1.089 cm^-1. The paper's row is 1.3 um; femwell keys it as 1.31. The paper's full text was not opened (Southampton eprint 337781 returned HTTP 403)] |
| Tidy3D Charge | not free | cloud | no | n/a |

### 3.8 Circuit-level S-matrix of a WDM bus with many rings

| Tool | Role | Version, licence | Verified | Accuracy limits |
|---|---|---|---|---|
| SAX | **chosen for netlists** | 0.18.2, Apache-2.0 (pins jax 0.9.2) | yes, C8: 8 add-drop rings (toy values, assumed) on one bus, 30 001 wavelengths: first call 2.76 s, second 0.44 s; max difference from the closed-form numpy cascade 4.4e-16 (float64 on); jax.grad of through power vs heater phase -4.889234 vs finite difference -4.889234 (3e-9 relative); peak 1.42 GB | float32 unless `jax_enable_x64` (lab saw 3e-6 residuals in float32). For the control loop at a few laser wavelengths per step a closed-form numba function is faster |
| scikit-rf | electrical side (driver, pads, bond wires) | 2.1.0, BSD-3 | yes, C10: line-shunt C-line cascade, S21 = -0.627 dB at 50 GHz, passive | not wavelength-native |
| Simphony | rejected | 0.7.3, MIT | runs, C9a: ring through-port min 2.1e-4 | now a thin layer over **sax 0.14.7**; conflicts with SAX 0.18 |
| Photontorch | rejected | 0.4.1 (2021), AGPL-3.0 | fails with torch 2.14.1 and with torch 1.12.1 (`torch.solve` removed); runs only on Python 3.9 + torch 1.11.0 + numpy<1.24 + scipy<1.10 (C9b) | unmaintained; only option with built-in time-domain circuits, replace with own TCMT in numba |
| gdsfactory | layout (not simulation) | 9.51.0, MIT | yes, C9c: `ring_double(radius=7.5, gap=0.2)` written to GDS, 4 ports, 12.7 s | downgrades pydantic in `.venv` (section 2) |

### 3.9 Time-domain closed loop, RL interface, fixed point

| Tool | Role | Version, licence | Verified | Accuracy limits |
|---|---|---|---|---|
| numba | **chosen** for the loop kernel | 0.68.0, BSD-2, arm64 wheel | yes, C11: 8 rings, 2-pole thermal plant + crosstalk, 1 MHz PI with 12-bit DAC, aggressor ramp 25 -> 105 C at 808 C/s with 100 ms period (G26, printed; the instant-reset sawtooth shape is assumed) [verified in the G26 PDF, section 3.2: "ramped up and down between 25oC and >105oC with a period of 100ms" at 808 °C/s. Note (inferred): 80 K at 808 °C/s takes 99 ms, so if both printed numbers hold, the down-ramp lasts 1 ms or less. Its speed is not printed, and it sets the max error below]: 200 000 steps in 0.057 s (compile 2.2 s), pure Python estimate 8.3 s (146x); lock holds with median error 0.13 pm (RMS 62 pm, max 6.9 nm, both set by the instantaneous 80 K reset of the toy sawtooth) | toy plant, every gain and time constant assumed |
| JAX | differentiable alternative | 0.9.2 (via SAX) / 0.11.2 latest, Apache-2.0 | yes (inside C8: x64, jit, grad) | first-call compile; CPU only here |
| python-control | linear loop design | 0.10.2, BSD-3 | yes (C6) | Pade for delays (lab cross-checks exact margins) |
| CasADi | MPC / optimal control | 3.8.1, LGPL-3.0, in `.venv` | import only (already installed) | n/a |
| gymnasium | **chosen** RL API | 1.4.0, MIT | yes, C11: `check_env` passed; 11 104 env steps/s (each = 10 plant steps) | none |
| Stable-Baselines3 | RL algorithms | 2.9.0, MIT (+ torch 2.14.1) | yes, C14: PPO trained 4096 steps in 5.7 s | CPU only |
| cocotb | **chosen** HDL test bench | 2.1.0, BSD-3, arm64 wheel | yes, C12: Verilog PI (16-bit error, 32-bit saturating integrator, 12-bit DAC) vs a plain-integer Python golden model, 2000 random cycles bit-exact on **Icarus 12.0** (0.17 s) and **Verilator 5.052** (0.04 s, conda-forge, scratch env) | the first Verilator-clean rewrite failed 54/2000 cycles: a `{...}` concatenation is unsigned in Verilog, which turned `>>>` into a logical shift. The golden model is what caught it |
| fxpmath, apytypes | Q-format helpers | 0.4.10 MIT, 0.5.1 MIT | yes, C12: Kp = 0.37512 in Q7.8 -> 0.375 (raw 96) in both; 40000 saturates to 32767 | slower than plain integer numpy; use plain ints in numba for speed |

## 4. Cross-solver numbers for the 400 x 200 nm strip at 1310 nm (all derived)

| Quantity | femwell | Tidy3D local | MPB | Notes |
|---|---|---|---|---|
| n_eff, TE0 (Sellmeier materials) | 2.48560 | 2.48602 (5 nm grid) | 2.48495 (128 px/um, constant indices = same values) | spread 0.04 % |
| n_g with material dispersion | 4.3359 | 4.3380 | n/a | `REF.ng = 4.2` is assumed |
| n_g of the waveguide alone (constant indices) | 4.143 (at 1.314 um, from the C4 n_eff at 1.312 and 1.316 um) | 4.1425 (built-in, 1.31 um); 4.145 from C4 at 1.314 um | 4.1393 (128 px/um) | the local solver does return n_group (section 5.3) |
| power fraction in core | 0.807 | n/a | n/a | `REF.confinement = 0.85` is assumed |
| dn_eff/dn_Si | 1.086 | n/a | n/a | > 1 (waveguide dispersion), as the lab found 1.077 for 500 x 220 |
| dn_eff/dT (REF dn/dT values) | 2.0436e-4 /K | 2.0454e-4 /K | n/a | 0.09 % apart |
| d lambda_r / dT = lambda (dn_eff/dT) / n_g | **61.7 pm/K** | 61.7 pm/K (with its dispersive n_g) | 61.7 pm/K (checker's independent MPB run, 128 px/um: dn_eff/dT 2.0428e-4, n_g 4.3346) | compare section 5. [corrected: this uses REF.dnSi_dT = 1.86e-4 /K, which is a ~1.5 um value. Frey 2006 Table 4 prints 1.87e-4 at 1.5 um and **1.94e-4 /K at 1.3 um, 295 K**. With 1.94e-4 and the same solver sensitivity (dn_eff/dn_Si = 1.086, oxide term 2.4e-6 /K) the result is **64.4 pm/K** (derived)] |

## 5. Disagreements found (reported, not resolved)

1. **Thermo-optic shift.** The solvers give 61.7 pm/K (derived) for the ideal 400 x 200 nm strip at room-temperature
   dn/dT. Lightmatter prints "~15 GHz/C" (G26, `REF.dLambdaDT_pmK` = 86 pm/K, inferred conversion). The derived
   value is within 1.3 % of Lightmatter's own bench equivalence 62.5 pm/K (`REF.dLambdaDT_benchPmK`, inferred from
   two printed numbers), 23 % below the OCP planning figure 80 pm/K (printed, `REF.dLambdaDT_ocpPmK`), and 23 %
   above the textbook's own 50 pm/K (`REF.dLambdaDT_derivedPmK`, derived with Gamma = 0.85 and n_g = 4.2, both
   assumed). Reaching 86 pm/K at n_g = 4.336 needs dn_eff/dT = 2.85e-4 /K, 1.39x the solver value. [verified: the arithmetic
   (15 GHz -> 85.9 pm at 1310 nm; -1.2 %, -22.8 %, +23.5 %; 2.846e-4, 1.393x) and the printed "~15 GHz/°C" (G26 PDF, abstract) are correct.
   corrected: the 61.7 pm/K input is REF.dnSi_dT = 1.86e-4 /K. `02_materials.md` attributes this to Cocorullo 1999 at 1523 nm, and
   Frey 2006 Table 4 prints 1.87e-4 at 1.5 um. At 1.3 um, 295 K, Frey prints 1.94e-4 /K, which gives 64.4 pm/K (derived). That is
   +3.0 % above the 62.5 pm/K bench equivalence and 19.5 % below the OCP 80 pm/K, and 86 pm/K would need 1.34x the solver
   dn_eff/dT. The 62.5 pm/K figure is 50 nm/s divided by the 800 °C/s of the optically emulated test (`REF` comment), so agreement
   with it is weak evidence.] Not included in the solver runs: temperature dependence of dn_Si/dT (it
   rises with T), thermal expansion, stress-optic effects, the real (possibly rib, doped) cross-section.
2. **Textbook "Gamma = 1 caps at 58 pm/K" (REF comment).** The full-vector sensitivity dn_eff/dn_Si is 1.086 > 1
   for this strip, so confinement-weighted bulk thermo-optics is not capped by Gamma = 1; with the solver's
   dn_eff/dT and the textbook's n_g = 4.2 one gets 63.7 pm/K. The textbook's main conclusion (an ideal strip does
   not reach 86 pm/K) still holds. [verified: 1310 x 1.86e-4 / 4.2 = 58.0 pm/K and 1310 x 2.0436e-4 / 4.2 = 63.7 pm/K.
   `calculate_confinement_factor` is c eps0 int sqrt(eps)|E|^2 over the core (femwell source), i.e. dn_eff/dn_Si for a
   power-normalized mode. An independent check: the sum rule n_Si S_Si + n_ox S_ox = n_g,wg gives S_ox = 0.236 from
   4.143, and the oxide part of dn_eff/dT gives 0.236 from 2.36e-6 / 1e-5]
3. **Tidy3D documentation vs observed behaviour.** The Tidy3D FAQ ("How do I run the mode solver locally?", latest
   docs) says "the local version will not include the group index calculation or subpixel-smoothing". On
   tidy3d 2.12.0 the local solver *did* return `n_group` (4.1425), and it matches finite differences of femwell
   (4.143) and MPB (4.1393). Subpixel smoothing is indeed absent locally (needs `tidy3d-extras` and a free API key,
   per the same docs; not tested). [corrected: the FAQ quote is verbatim, re-read 2026-10-09. But the FAQ page does not
   mention `tidy3d-extras` or an API key. That requirement comes from the tidy3d 2.12 source: `components/mode/simulation.py`
   uses `tidy3d_extras` for local subpixel, and `packaging.py` reports an init failure "likely due to an invalid API key".
   Whether that key is free was not checked.]
4. **Tidy3D bend sign asymmetry.** n_eff(R = +7.5 um) = 2.48629 and n_eff(R = -7.5 um) = 2.48923 on the same grid;
   a mirror-symmetric problem must give equal values. femwell's conformal map eps(1 + x/R)^2 is mirror-symmetric by
   construction (only R = +7.5 um was run: +0.0014). [corrected and extended by the checker's rerun (`05_tools_smoke/verify/v01_tidy3d_bend_asym.py`,
   1.6 x 1.2 um plane, no PML, peak 1.61 GB):
   - The +/-R mean shift is +0.000378 at R = 7.5 um on 20, 10 and 5 nm grids (spread 2e-6), and +8e-6 at R = 50 um.
     That is 1/R^2 scaling, as a real bend shift should have.
   - The asymmetry is 0.0024, 0.0012 and 0.0015 at 20, 10 and 5 nm, and 1/6.7 of that at R = 50 um. That is 1/R
     scaling, i.e. an offset of 2 to 4 nm in the reference radius. So the artifact cancels in the +/-R mean.
   - femwell's isotropic map gives +0.0014 at 7.5 um and +0.0032 at 5 um (also about 1/R^2), which is 3.7x Tidy3D's
     anisotropic (exact-transform) mean. This is a real formulation disagreement; neither the 5 nm Meep cylindrical
     uncertainty nor a third exact solver resolves it here. It is 0.3 nm of resonance (derived), small next to
     fabrication scatter.]
5. **Meep convergence.** First order (p = 0.94) for this thin high-contrast ring, so Meep at practical
   resolutions over-estimates n_eff by 1 to 4 % (20 to 80 px/um); always extrapolate. [corrected: the 1 to 4 % range
   is right (+4.2 % at 20, +1.1 % at 80 px/um). But p = 0.94 is the 40/60/80 three-point value; all five points give
   p = 0.79, and the order is still rising, so the extrapolation is uncertain by about 5 nm. The Meep docs quote
   ("error intermediate between first and second order" for sharp corners) is verbatim.]
6. **The lab's TOOLS.md describes another machine.** It documents `/Users/kennyg/photonics-simulations` with SAX,
   JAX, gdsfactory, python-control, Manim, Tidy3D in `.venv`. On this machine
   `/Users/kenny/photonics-simulations/.venv` (Python 3.12.8) has femwell, scikit-fem, gmsh, numba, casadi, pyvista
   but **not** control, sax, jax, gdsfactory, tidy3d, manim or nbformat (checked by import). The Meep env works.

## 6. What to reuse from `/Users/kenny/photonics-simulations`

| File | Reuse for | Caveat |
|---|---|---|
| `experiments/08_soi_strip_mode_solvers/solvers.py` | `solve_tidy3d` (grid-alignment trick, trapezoid core weights), `femwell_mesh`, `solve_femwell` (Gamma_P, Gamma_E, S_core, field gridding), `group_index_fd` | written for 500 x 220 nm; pass 400 x 200 |
| `experiments/08_soi_strip_mode_solvers/materials.py`, `visuals/indices.py` | Sellmeier fits, slab solver, strip effective-index method | check the Si fit equals the textbook `nSilicon` |
| `experiments/11_directional_coupler/slab_coupler.py`, `visuals/03_coupled_modes/{cmt,solve,fdtd3d}.py` | supermode/CMT kappa, point-coupler length sqrt(2 pi R / gamma), 3-D Meep coupler | 3-D FDTD not grid-converged (its README) |
| `experiments/12_round_trip_to_ring/ring_sax.py` | SAX coupler/waveguide models with explicit n_g dispersion, `analytic_T`, `analytic_numbers`, `dips`, `fwhm_of_dip` | needs SAX + JAX, absent from that venv here |
| `experiments/12_round_trip_to_ring/ring_meep.py` | 2-D ring + bus, loss calibration through `D_conductivity`, Harminv family filter, `ta_from_Q`, `t_from_Qc` | 2-D ring is not the 3-D ring |
| `experiments/13_capstone_ring_lock/ringlock.py` | `Plant`, `Sensor`, `PIController`, `crosstalk_matrix`, `simulate`, `design_pi`, `loop_tf`, `exact_margins` | uses the lab's own parameter set |
| `common/units.py` | dB, alpha, GHz <-> nm helpers | none |
| `env/check_env.py` | pattern for an environment self-test | paths and package list differ here |

Do **not** import `common/params.py` as truth: it is an older parameter set (500 x 220 nm, 50 pm/K, R = 6.3 um,
200 GHz grid, tags [LM]/[P]/[T]/[N]/[C]) that predates the textbook `REF` with its G25/G26 generation tags.

## 7. Licence notes

GPL code in the chain: femwell and meshwell (GPL-3.0), gmsh (GPL-2.0+), Meep/MPB (GPL-2.0+), slycot (GPL-2.0),
Icarus (GPL-2.0+); Photontorch is AGPL-3.0. Permissive: scikit-fem, scipy, numba, pyMOR, python-control, SAX, JAX,
DEVSIM (Apache-2.0), gymnasium, SB3, cocotb, fxpmath, apytypes, scikit-rf; Tidy3D client LGPLv2+ (package metadata) with an MIT
mode-solver core; FEniCSx and CasADi LGPL-3.0. [verified from the installed `.dist-info` files and the PyPI JSON on
2026-10-09: femwell and meshwell are GPL v3 (licence text), gmsh GPLv2+, DEVSIM Apache 2.0, scikit-fem BSD, tidy3d "LGPLv2+"
with an MIT `components/mode/LICENSE` (ceviche-derived), pyMOR "BSD-2-Clause AND BSD-3-Clause", control and cocotb
BSD-3-Clause, SAX Apache, Photontorch AGPL v3 (classifier; last upload 2021-02-13). One addition: slycot 0.7.0 declares
"GPL-2.0 AND BSD-3-Clause AND GPL-3.0-with-GCC-exception AND BSD-2-Clause", so treat it as GPL-2.0 as stated] Practical rule (not legal advice): run GPL tools only in offline
`extract` scripts that write data files (JSON/NPZ/YAML), so the runtime plant that Lightmatter might take in-house
imports only permissive code.

## 8. Not verified, open questions

- EMode (account needed), Elmer (source build), Tidy3D cloud solvers (paid), `tidy3d-extras` local subpixel
  smoothing (needs a free API key; creating one is the user's decision).
- A 2-D DEVSIM rib with slab and P++/N++ contacts, coupled into femwell through Delta-epsilon from the
  Nedeljkovic coefficients: all pieces run, the coupled chain has not been built.
- A converged 3-D coupler (Meep) to check the CMT K^2(lambda) of section 3.3.
- Why Tidy3D's bend is sign-asymmetric (grid staggering vs a definition issue): try 5 nm grid, or ask Flexcompute.
  [partly answered by the checker (section 5.4): the 5 nm grid does not remove it, it scales as 1/R, and the +/-R mean
  is grid-independent. The open question is now femwell's +0.0014 vs Tidy3D's +0.00038 at R = 7.5 um.]
- The 61.7 vs 86 pm/K gap (section 5.1) is a physics question for the model, not a tool question.

## 9. Command log (run from `/Users/kenny/mrm-model`) and script index

| ID | Command | Result summary |
|---|---|---|
| C0 | `.venv/bin/python docs/research/05_tools_smoke/pypi_meta.py femwell scikit-fem gmsh tidy3d devsim pymor control slycot sax simphony gdsfactory photontorch scikit-rf numba jax jaxlib gymnasium cocotb fxpmath apytypes casadi meshwell ngsolve fenics-dolfinx emode emodeconnection torch diffrax pyvista`, then the same script with `stable-baselines3 meep mpb pyamg petsc4py` | versions and licences quoted in section 3 (`fenics-dolfinx`, `emode` are not on PyPI; PyPI `meep` is an unrelated package) |
| C1 | `NONET MG .venv/bin/python -u docs/research/05_tools_smoke/t01_tidy3d_mode.py` | 94.5 s, peak 2.22 GB, `out/t01_tidy3d_mode.json` (first version with a full-plane 5 nm grid: killed by memguard at 3.00 GB) |
| C1b | `NONET MG .venv/bin/python -u docs/research/05_tools_smoke/t01b_tidy3d_bend_ng.py` | 109 s, peak 2.13 GB, `out/t01b_tidy3d_bend_ng.json` |
| C2 | `MG .venv/bin/python -u docs/research/05_tools_smoke/t02_femwell_modes.py` | 241 s, peak 1.46 GB, `out/t02_femwell_modes.json` |
| C3 | `T_AFTER=150 MG $MEEP -u docs/research/05_tools_smoke/t03_meep_cyl_ring.py <res>` for res 20, 30, 60, 80 (res 40 with the default `T_AFTER=400`) | `out/t03_meep_cyl_res*_m89.json` |
| C4 | `NONET MG .venv/bin/python -u docs/research/05_tools_smoke/t03b_bend_resonance_predict.py` | peak 1.99 GB, `out/t03b_bend_resonance_predict.json` |
| C5 | `MG $MEEP docs/research/05_tools_smoke/t03c_mpb_strip.py` | 2 s, `out/t03c_mpb_strip.json` |
| C7a | copy `diode_1d.py`, `diode_common.py` from `.venv/devsim_data/examples/diode/` to a scratch dir, then `MG .venv/bin/python diode_1d.py` there | 1.8 s, 6 Newton iterations at 0.5 V |
| C7b | same for `gmsh_diode2d.py` + `gmsh_diode2d.msh` | 0.6 s, top/bottom current 1.7976e-7 / -1.7976e-7 (DEVSIM 2-D units, per unit depth) at 0.5 V |
| C7c | `MG .venv/bin/python docs/research/05_tools_smoke/t04_devsim_lateral_pn.py` | 2.7 s, `out/t04_devsim_lateral_pn.json` |
| C6 | `MG UVR --with pymor==2026.1.1 --with scikit-fem==12.0.2 --with control==0.10.2 --with slycot==0.7.0 python docs/research/05_tools_smoke/t05_heat_mor.py` | 82 s, peak 0.70 GB, `out/t05_heat_mor.json` |
| C8 | `MG UVR --with sax==0.18.2 python docs/research/05_tools_smoke/t06_sax_wdm_bus.py` | 5.3 s, `out/t06_sax_wdm_bus.json` |
| C9a | `MG UVR --with simphony==0.7.3 python docs/research/05_tools_smoke/t07_simphony.py` | runs, sax 0.14.7 |
| C9b | `MG UVR --with photontorch==0.4.1 --with torch --with numpy --with scipy python .../t08_photontorch.py` (fails, `torch.solve` removed); `--python 3.10 ... --with torch==1.12.1` (fails); `--python 3.9 --with photontorch==0.4.1 --with torch==1.11.0 --with "numpy<1.24" --with "scipy<1.10"` | only the last runs (6.7 s) |
| C9c | `MG UVR --with gdsfactory==9.51.0 python docs/research/05_tools_smoke/t09_gdsfactory.py` | 12.7 s, 7388-byte GDS |
| C10 | `MG UVR --with scikit-rf==2.1.0 python docs/research/05_tools_smoke/t10_skrf.py` | S21 -0.627 dB at 50 GHz |
| C11 | `NUMBA_CACHE_DIR=<scratch> MG .venv/bin/python docs/research/05_tools_smoke/t11_closed_loop_numba_gym.py` | `out/t11_closed_loop_numba_gym.json` |
| C12 | `[SIM=verilator] PATH=<verilator env>/bin:/opt/homebrew/bin:$PATH MG UVR --with cocotb==2.1.0 --with fxpmath==0.4.10 --with apytypes==0.5.1 python docs/research/05_tools_smoke/t12_fixed_point/run_cocotb.py`; Verilator env: `micromamba create -p <scratch>/verilator_probe -c conda-forge verilator=5.052` (24 s) | PASS on both simulators |
| C13 | `micromamba create -p <scratch>/fenicsx_probe -c conda-forge python=3.12 fenics-dolfinx=0.11.0` (49 s), then `MG <scratch>/fenicsx_probe/bin/python docs/research/05_tools_smoke/t13_fenicsx_heat3d.py` | 92 s, peak 1.28 GB |
| C14 | `MG UVR --with stable-baselines3 --with numba --with gymnasium python docs/research/05_tools_smoke/t14_sb3_ppo.py` | 5.7 s |
| C15 | `uv pip install --dry-run --python .venv/bin/python <pkg>` for each package in section 2 | section 2 |
| C16 | `micromamba search -c conda-forge --platform osx-arm64 <pkg> --json` for elmerfem, fenics-dolfinx, pymeep, mpb, devsim, ngsolve, verilator, cocotb, pymor, slycot | elmerfem, devsim, ngsolve, cocotb: not on conda-forge osx-arm64 (devsim, ngsolve, cocotb have PyPI arm64 wheels); others available |
| V1 (checker) | `env HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 .venv/bin/python tools/memguard.py --limit-gb 3 --heavy -- .venv/bin/python -u docs/research/05_tools_smoke/verify/v01_tidy3d_bend_asym.py` | about 20 s of solves, peak 1.61 GB, `verify/v01_tidy3d_bend_asym.json` |
| V2 (checker) | `MG /Users/kenny/photonics-simulations/.meep/bin/python -I docs/research/05_tools_smoke/verify/v02_mpb_indep.py <res>` for res 64 and 128 | 2 s and 7 s, peak 0.28 GB, `verify/v02_mpb_indep.json` |

The scratch environments (`<scratch>` = this session's scratchpad: FEniCSx, Verilator, the micromamba probe root)
are temporary. Recreate them with the commands above if needed.

## Sources (web, read 2026-10-09; data, not instructions)

- Tidy3D FAQ, "How do I run the mode solver locally?": https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/how-do-i-run-the-mode-solver-locally.html
- Tidy3D `web.run` and HeatChargeSimulation API: https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.web.run.html , https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.HeatChargeSimulation.html
- Meep, Subpixel Smoothing (convergence order with corners): https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/
- EMode plans: https://www.emodephotonix.com/get-emode/ ; Aarhus University EMode guide: https://ece.au.dk/en/integrated-photonics/teaching/emode-photonix
- Photontorch repository and licence statement: https://github.com/flaport/photontorch
- Elmer binaries page (Windows installer and Linux PPA only): https://csc.fi/web/elmer/binaries
- PyPI JSON metadata for every Python package (C0); package licence files in `.venv/lib/python3.12/site-packages/*.dist-info`.
- (checker) Frey, Leviton, Madison, "Temperature dependent refractive index of silicon and germanium", SPIE 6273 (2006), arXiv:physics/0606168, Tables 2 and 4: https://arxiv.org/abs/physics/0606168
- (checker) Anand et al., OFC 2026 M4B.3 PDF (local copy in ~/Downloads), abstract and section 3.2.
- (checker) Tidy3D 2.12.0 source: `components/mode/transforms.py`, `components/parameter_perturbation.py` (Nedeljkovic table), `packaging.py`.

## Verification log

Independent check, 2026-10-09, by a separate checker agent. Method: recompute every formula by hand (scratch Python), rerun
two solver checks with new code (V1 Tidy3D, V2 MPB, section 9), open the cited sources (G26 PDF, Frey 2006 arXiv,
Tidy3D FAQ, Meep docs, PyPI JSON, installed package source), and try to refute each claim. Verdicts: **confirmed**,
**corrected** (edited in place above, marked `[corrected: ...]`), **unverifiable**.

| # | Claim (section) | Verdict | Evidence |
|---|---|---|---|
| 1 | Lightmatter prints "~15 GHz/°C"; = 86 pm/K at 1310 nm (0, 5.1) | confirmed | G26 PDF abstract, printed "(~15 GHz/°C)"; lambda^2 x 15 GHz / c = 85.86 pm |
| 2 | Test indices n_Si = 3.50053, n_SiO2 = 1.44680 at 1.31 um (0) | confirmed, with a caveat added | `nSilicon` is Li 1980 (293 K form): 3.500529; Malitson: 1.446804. Frey 2006 Table 2 prints 3.50616 at 1.3 um, 295 K, i.e. about 0.004 higher at 1.31 um. That moves n_eff by about 0.004, 4x the solver spread |
| 3 | n_eff 2.4856 / n_g 4.336 / dn_eff/dT 2.044e-4 / 61.7 pm/K, three solvers agree (1, 3.1, 4) | confirmed | new MPB script (V2), 128 px/um: 2.48495 / 4.3346 / 2.0428e-4 / 61.74 pm/K. JSON outputs match every quoted digit |
| 4 | 61.7 pm/K is the solver answer for the strip at 1310 nm (4, 5.1) | corrected | arithmetic right, but the REF input 1.86e-4 /K is a ~1.5 um value (Frey Table 4: 1.87e-4 at 1.5 um; 1.94e-4 at 1.3 um, 295 K, printed). With 1.94e-4: 64.4 pm/K (derived), +3.0 % vs 62.5, -19.5 % vs 80, 86 needs 1.34x |
| 5 | Percentages vs 62.5 / 80 / 50 pm/K, and 2.85e-4 = 1.39x for 86 pm/K (5.1) | confirmed | -1.2 %, -22.8 %, +23.5 %; 2.846e-4 /K, 1.393x. Also noted: 62.5 = 50 nm/s / 800 °C/s, the optically emulated test, not a device measurement |
| 6 | "Gamma = 1 caps at 58 pm/K" is not a cap because dn_eff/dn_Si = 1.086 > 1; 63.7 pm/K at n_g 4.2 (5.2) | confirmed | 58.01 and 63.74 pm/K recomputed. `calculate_confinement_factor` source read. Sum rule n_Si S_Si + n_ox S_ox = n_g,wg gives S_ox = 0.236, the same as the oxide share of dn_eff/dT (0.236) |
| 7 | Tidy3D FAQ: local solver "will not include the group index calculation or subpixel-smoothing" (5.3) | confirmed (quote) / corrected (attribution) | quote verbatim on the live FAQ. The FAQ does not mention tidy3d-extras or an API key; that comes from tidy3d 2.12 source (`packaging.py`). Local n_group 4.1425 confirmed in the JSON |
| 8 | Tidy3D bend: +/-R asymmetry 0.0029; "sign-averaged shift below the discretization error" (3.2, 5.4) | corrected | V1: the mean shift is +0.000378 at 20 / 10 / 5 nm (spread 2e-6) and scales as 1/R^2; the asymmetry scales exactly as 1/R (6.67x from R 7.5 to 50 um) and does not converge with the grid. So it is an artifact, and the mean is resolved. Real open disagreement: femwell +0.0014 vs Tidy3D +0.00038 |
| 9 | femwell bend uses isotropic eps(1 + x/R)^2 on the whole form (3.2) | confirmed | `femwell/maxwell/waveguide.py`: `epsilon = w.epsilon * (1 + w.x[0] / radius) ** 2` in `aform` |
| 10 | Meep cylindrical: "fitted convergence order p = 0.94", extrapolated 1.3132 um, inside the 1.1 nm mode-solver spread (3.2, 5.5) | corrected | 0.94 is the 40/60/80 three-point solution. All five points (least squares): p = 0.79, 1.3089 um. Other triples: p 0.69 to 0.91. Only the p = 1 value (1.31429 um) lies inside 1.31365 to 1.31480 um; 1.3132 is 0.45 nm below it. "1 to 4 % over-estimate" is confirmed (4.2 % / 1.1 %) |
| 11 | Meep docs: corners give "error intermediate between first and second order" (3.2) | confirmed | Meep "Subpixel Smoothing" page, verbatim |
| 12 | kappa = pi (n_e - n_o)/lambda; point coupler K^2 = 0.63 %, dK^2/dlambda = 8.2e-5 /nm (3.3) | confirmed | sin^2(0.03768 x sqrt(2 pi 7.5 / 10.59)) = 0.630 %; local decay constants 10.8 / 10.6 / 10.5 /um from the kappa(g) table; slope 8.21e-5 /nm |
| 13 | Nedeljkovic 1.31 um coefficients dn = -2.98e-22 dN^1.016 - 1.25e-18 dP^0.835 (3.7) | confirmed against a secondary source; primary not opened | Tidy3D 2.12's independent table (row 1.3 um) gives identical n and alpha coefficients and cites the same DOI. The paper row is 1.3 um, not 1.31. Southampton eprint 403, IEEE paywalled |
| 14 | DEVSIM 1-D cut: W 63.8 / 117.1 nm vs formula 68.9 / 122.8 nm; core charge 1.34e12 cm^-2 (3.7) | corrected | formula values confirmed (V_bi 0.9166 V, eps 11.7). But DEVSIM `simple_physics.py` uses eps_si = 11.1, so the comparison mixes permittivities (11.1: 67.1 / 119.6 nm; with the Debye correction 65.2 / 118.6). Charge 1.336e12 vs 1.349e12 (abrupt) confirmed. mu_n 400, mu_p 200, n_i 1e10 confirmed in source |
| 15 | Thermal: R_th 8.19 K/mW, f_3dB 60 kHz, 3-D line 31.7 K/mW; BT 6 states 0.39 %, DC -0.39 %; balred DC -5.5 % (3.5, 3.6) | confirmed (consistency; not re-run) | JSON matches every number. 1-D BOX bound: 90 kHz, 3.7 um effective width for 8.19 K/mW. Ring-to-12 um scaling gives 32.2 vs 31.7 K/mW. 8.156 / 8.188 = -0.39 %; 7.738 / 8.188 = -5.5 % |
| 16 | G26 aggressor: 808 °C/s, 25 to >105 °C, 100 ms period (3.9) | confirmed, with a note | G26 section 3.2 prints "ramped up and down ... with a period of 100ms" at 808 °C/s. 80 K / 808 °C/s = 99 ms, so the down-ramp (not printed) is 1 ms or less; the toy's instant reset is labelled assumed |
| 17 | Licences (7) | confirmed | dist-info and PyPI: femwell / meshwell GPL v3, gmsh GPLv2+, tidy3d LGPLv2+ with MIT mode core, DEVSIM Apache-2.0, pyMOR BSD-2/3, control / cocotb BSD-3, SAX Apache, Photontorch AGPL v3 (2021-02-13). slycot's expression also lists BSD and GPL-3.0-with-GCC-exception parts; GPL-2.0 governs |
| 18 | Simphony "pins sax==0.14.7" (1, 2) | corrected | PyPI requires_dist `sax<0.15.0`, which resolves to 0.14.7; the conclusion (conflicts with SAX 0.18) stands |
| 19 | tidy3d 2.12 requires numpy<2.5.0, Meep env has numpy 2.5.3; PyPI `meep` is an unrelated 2018 1.0.6 package (2) | confirmed | tidy3d METADATA `numpy<2.5.0,>=2.2.6`; Meep env import prints numpy 2.5.3, meep 1.34.0; PyPI meep 1.0.6 uploaded 2018-09-21 |

Not checked (low impact on the plant): EMode prices, the gdsfactory dependency count, the cocotb Verilog signedness detail,
the SB3/gymnasium throughput figures, and the FEniCSx run. Elmer: the anaconda.org API returns "could not be found" for
`conda-forge/elmerfem`, which agrees with "no conda-forge osx-arm64 build" (it has no conda-forge build at all under that name).
