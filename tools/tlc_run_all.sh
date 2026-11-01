#!/usr/bin/env bash
# One command for the whole TLC evaluation (docs/TLC_PORT_SPEC_AND_PLAN.md 0): the scenario matrix of
# experiments/tlc/matrix.yaml (every experiment x controller, Monte Carlo campaign, model-trust corners), then the
# report docs/tlc/results.md (+ .json). About 10 min on a laptop after the one-off kernel compile.
#   tools/tlc_run_all.sh [extra tools/tlc_matrix.py arguments, e.g. --only tlc,pi_lock_c --no-corners]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
exec $PY tools/memguard.py -- $PY tools/tlc_matrix.py experiments/tlc/matrix.yaml "$@"
