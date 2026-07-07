#!/usr/bin/env bash
# Pilot go/no-go (EXPERIMENT_PLAN.md §9.2): sample → run E1-E4 → check signature.
# Env: MODEL (default internvl3-8b), BACKEND (default hf), DEVICE (default cuda).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-internvl3-8b}"
BACKEND="${BACKEND:-hf}"
DEVICE="${DEVICE:-cuda}"

echo "== Building pilot sample + configs =="
python scripts/make_pilot_sample.py --model "$MODEL" --backend "$BACKEND"

for COND in E1 E2 E3 E4; do
  CFG="configs/pilot__${COND}__${MODEL}.yaml"
  echo "== Pilot $COND =="
  python scripts/run_condition.py --config "$CFG" --backend "$BACKEND" --device "$DEVICE"
done

echo "== Go/No-Go =="
python scripts/check_pilot_gonogo.py --model "$MODEL"
