#!/usr/bin/env bash
# Run the TLC experiment set and its comparators, then write the report (docs/TLC_PORT_SPEC_AND_PLAN.md 4).
#   tools/tlc_run_all.sh [--out docs/tlc/results.md]
# Steady state: tlc (faithful core), tlc_pi (its BaselinePI), pi_lock_c, and the stub-core skeleton check.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${2:-docs/tlc/results.md}

EXPERIMENTS=(
  experiments/tlc/steady_tlc_faithful.yaml
  experiments/tlc/steady_tlc_pi.yaml
  experiments/examples/steady_pi_lock.yaml
  experiments/tlc/steady_tlc.yaml
)

RUNS=()
for exp in "${EXPERIMENTS[@]}"; do
  echo "== $exp" >&2
  RUNS+=("$($PY tools/memguard.py -- $PY tools/tlc_run.py "$exp" | tail -n 1)")
done
$PY tools/tlc_report.py "${RUNS[@]}" --out "$OUT"
