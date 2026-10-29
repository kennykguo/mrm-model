# one optical calculation

## objective

use one general two-direction ring field calculation for every ring and trial temperature.
remove switches between Airy, polynomial, and far-line implementations.
keep the existing physics and parameter values.

## scope

- production trial heating and commit use the general field calculation for zero and nonzero backscatter.
- the Airy cache and its helper functions move to an explicit numerical-reference archive.
- far-line hold and reference-engine Airy selection are removed.
- exact reuse of an identical final trial remains; it introduces no different model or approximation.
- reserved packed-layout slots remain in place.
- derivatives remain available where the thermal root solve needs them.
- a singular uncoupled lossless ring raises an error rather than producing invalid numbers.

## verification

compare absorbed power, derivatives, and committed outputs with the archived implementation.
retain independent dense-field, energy-balance, binary-average, and finite-difference checks.
run engine parity, thermal integration, and non-soak golden checks.
measure the performance cost of removing the specialized zero-backscatter calculation.
record results in docs/READABILITY_PASS.md.

## result - 28 october 2026

complete. 136 Python checks and 13 variable-map checks pass. one existing expected failure remains.
all 12 non-soak golden checks pass; the two long soaks were not rerun.
two golden records were regenerated for rounding changes and recorded in docs/CHANGELOG.md.
integer metrics are unchanged. maximum observed detuning-metric change is about 2.2e-10 pm.
core field and derivative equations are unchanged by executable AST comparison.
four comparisons cover 1,800 values. nonzero-backscatter values match exactly.
zero-backscatter absorbed-power differences are below 1.9e-16 W.
the old singular lossless-uncoupled-ring test now raises the expected error through the general solver.

performance measurement - one warmed NRZ heating evaluation over eight wavelengths takes about
0.25 us in the archived specialization and 1.70 us in the general calculation, including Python call overhead.
this sampled inner operation is about 6.7 times slower; it is not a full-simulation slowdown measurement.
removing equivalent specialization simplifies the implementation; it does not establish higher zero-backscatter numerical accuracy.

snapshots, comparison values, benchmark details, and test logs are in
`/tmp/mrm-optics-shortcut-retirement-20261028/`.
