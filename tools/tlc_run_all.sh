#!/usr/bin/env bash
# Run the TLC experiment set and its comparators, then write the report (docs/TLC_PORT_SPEC_AND_PLAN.md 4).
#   tools/tlc_run_all.sh [--out docs/tlc/results.md]
# Phase 1: steady state only, tlc (stub core) against pi_lock_c. Later phases add experiments here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${2:-docs/tlc/results.md}

EXPERIMENTS=(
  experiments/tlc/steady_tlc.yaml
  experiments/examples/steady_pi_lock.yaml
)

RUNS=()
for exp in "${EXPERIMENTS[@]}"; do
  echo "== $exp" >&2
  RUNS+=("$($PY tools/memguard.py -- $PY tools/tlc_run.py "$exp" | tail -n 1)")
done
$PY tools/tlc_report.py "${RUNS[@]}" --out "$OUT"
