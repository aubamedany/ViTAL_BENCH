#!/usr/bin/env bash
# Main sweep (EXPERIMENT_PLAN.md §9.3): E1/E4 full ladder + E2/E3 leakage subset.
# Runs every "main" config (basename with exactly 3 __-segments, no variant),
# smallest models first via the ordering in configs/RUN_ORDER.txt.
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

# read run ids in the "Main sweep" section (until the first blank line after it)
in_main=0
while IFS= read -r line; do
  case "$line" in
    "# Main sweep"*) in_main=1; continue ;;
    "#"*) in_main=0; continue ;;
    "") [[ $in_main -eq 1 ]] && in_main=0; continue ;;
  esac
  if [[ $in_main -eq 1 ]]; then
    CFG="configs/${line}.yaml"
    echo "== $line =="
    python scripts/run_condition.py --config "$CFG" --backend "$BACKEND" --device "$DEVICE"
  fi
done < "$ORDER"

echo "Main sweep complete."
