# Handoff: extract_junction (`mrm/extraction/junction.py`, `junction_<xsec>[_lateral].json`)

Status: DRAFT, numbers filled in after the production run (see the bottom section).

## What was built

- `python -m mrm.extraction.junction [--quick] [--xsec rib160 rib200] [--topology lshape lateral] [--doping-sweep]
  [--out DIR]`, always through `tools/memguard.py --limit-gb 3 --heavy`. `--quick` writes to a temp dir.
- Artefacts (schema `junction/1`, ARCHITECTURE 5.4), each with `<name>.sidecar.json` (sha256, inputs with
  provenance, tool versions, meshes, convergence, validation, uncertainty):
  - `junction_rib160.json`, `junction_rib200.json`: the DEFAULT junction hypothesis, an **L-shaped** PN junction
    (horizontal junction across the rib from a shallow n implant, plus a vertical leg on the n side).
  - `junction_rib160_lateral.json`, `junction_rib200_lateral.json`: the lateral PN junction the task asked for.
- Contract keys (read by `mrm.plant.electrical.load_junction_table`): `v_reverse_v` (-0.5 .. 4.0 V, 0.05 V,
  forward negative), `dneff` (bent TE0 at r_ref, relative to 0 V), `alpha_fca_per_m` (absolute, intensity, full
  circumference), `capacitance_f_per_m` (low-frequency dQ/dV), `r_series_ohm_m` (R_s = r/(fill*L)).
- Extra keys: `aux` (straight-mode dn_eff, analytic model, Soref-Bennett alternative, AC values, q(V), carriers,
  depletion width), `overlap` (n_eff, n_g, index weights, r_ref and its source), `reports` (CAL-5, bandwidth, loss
  budget, doping sweep), `jacobian`, `temperature_sensitivity`, `plasma_model_alternative`, `summary`.

## Method (short)

DEVSIM 2-D drift-diffusion on a structured triangular mesh of rib + slab + SiO2 cladding (Poisson in the oxide for
the fringe field), Arora mobility, n_i and V_bi from `mrm.materials`, permittivity 11.7 eps0 (05 C7c), ohmic
contacts at the slab ends. TE0 from femwell (order 2, conformal bend at r_ref). Index-weight density
w = c eps0 n (1 + x/R)^2 |E|^2 / P evaluated at the DEVSIM nodes; dn_eff = sum w dn A, alpha = sum w alpha A with
Nedeljkovic 2011 (Soref-Bennett kept as an alternative). C from dQ/dV of the 0.05 V DC sweep, checked against
DEVSIM small-signal AC at ten biases; R_s from AC Re Z (20 GHz) plus a transfer-length contact term.

## Numerical notes for anyone re-running DEVSIM on this machine

- DEVSIM's relative-error test never passes at nodes where the potential crosses 0 V; both contacts carry a +10 V
  offset after the equilibrium solve. In deeply depleted nodes the carrier equations stall at a 1.1e-10 relative
  update, so DC solves use 1e-8.
- DEVSIM 2.11 keeps one complex matrix allocated per AC `solve()` (about 37 MB per call on a 15k-node mesh); long
  sweeps therefore run DC only and AC in separate small processes. Peak RSS is about 50 kB per mesh node and stays
  bounded during DC sweeps (physical footprint checked equal to RSS).
- `reset_devsim()` also drops the UMFPACK solver registered at import; `_devsim_reset()` re-installs it.
- Every solver job runs in a fresh spawned process (`_isolated`), so femwell/DEVSIM memory is returned to the OS.
