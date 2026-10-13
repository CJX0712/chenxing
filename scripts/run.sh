#!/usr/bin/env bash
# One-click reproducible verification for Chenxing (晨星).
# Works with any Python that has numpy (no venv required).
set -e
cd "$(dirname "$0")/.." || exit 1
python3 scripts/run_verification.py --report "$@"
