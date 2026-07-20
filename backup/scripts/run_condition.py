#!/usr/bin/env python3
"""Run a single condition from a config file (EXPERIMENT_PLAN.md §11).

Usage:
    python scripts/run_condition.py --config configs/law__E4__internvl3-8b.yaml
    python scripts/run_condition.py --config <cfg> --backend mock   # dry run
    python scripts/run_condition.py --config <cfg> --limit 5        # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backup.vntraffic.config import RunConfig
from backup.vntraffic.runner import run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", default=None, help="override backend (hf|openai|mock)")
    ap.add_argument("--limit", type=int, default=None, help="override item cap")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = RunConfig.load(args.config)
    if args.backend:
        cfg.backend = args.backend
    if args.limit is not None:
        cfg.limit = args.limit
    if args.device:
        cfg.device = args.device

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
