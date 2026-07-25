#!/usr/bin/env bash
# Ablations + CircularEval (EXPERIMENT_PLAN.md §8, §5.5): runs the "Ablations" and
# "CircularEval" sections of configs/RUN_ORDER.txt, then regenerates tables.
# Env: BACKEND (default hf), DEVICE (default cuda).
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-hf}"
DEVICE="${DEVICE:-cuda}"
ORDER="configs/RUN_ORDER.txt"

if [[ ! -f "$ORDER" ]]; then
  echo "Missing $ORDER — run: python scripts/gen_configs.py" >&2
  exit 1
fi

active=0
while IFS= read -r line; do
  case "$line" in
    "# Ablations"*|"# CircularEval"*) active=1; continue ;;
    "# Main sweep"*) active=0; continue ;;
    "#"*) continue ;;
    "") continue ;;
  esac
  if [[ $active -eq 1 ]]; then
    CFG="configs/${line}.yaml"
    echo "== $line =="
    python scripts/run_condition.py --config "$CFG" --backend "$BACKEND" --device "$DEVICE"
  fi
done < "$ORDER"

echo "== Building tables =="
python scripts/make_tables.py
echo "Ablations complete."
