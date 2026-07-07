#!/usr/bin/env python3
"""Stratified 200-item law pilot sample + pilot configs (EXPERIMENT_PLAN.md §9.2).

Stratifies the law tier by category × gradeability and samples proportionally
with a fixed seed, then writes:
    artifacts/pilot_sample.json          the sampled ids + strata sizes
    configs/pilot__E{1..4}__<model>.yaml  the 4 pilot runs on the pilot model

Usage:
    python scripts/make_pilot_sample.py [--n 200] [--model internvl3-8b]
                                        [--seed N] [--backend hf]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vntraffic import constants as C
from vntraffic.config import RunConfig
from vntraffic.dataset import filter_tier, load_items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(C.DATASET_V1_DIR, "data.json"))
    ap.add_argument("--dataset-dir", default=C.DATASET_V1_DIR)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="internvl3-8b")
    ap.add_argument("--seed", type=int, default=C.GLOBAL_SEED)
    ap.add_argument("--backend", default="hf")
    ap.add_argument("--configs", default=C.CONFIGS_DIR)
    args = ap.parse_args()

    data_path = args.data if os.path.exists(args.data) else C.DATA_JSON
    law = filter_tier(load_items(data_path), C.TIER_LAW)

    grade_path = os.path.join(args.dataset_dir, "gradeability.json")
    gradeable = set()
    if os.path.exists(grade_path):
        with open(grade_path, "r", encoding="utf-8") as f:
            gradeable = set(json.load(f).get("f1_gradeable_ids", []))

    # strata: (category, gradeable?)
    strata = defaultdict(list)
    for it in law:
        strata[(it.category, it.id in gradeable)].append(it.id)

    rng = random.Random(args.seed)
    total = len(law)
    sample = []
    strata_info = {}
    for key, ids in sorted(strata.items()):
        take = max(1, round(args.n * len(ids) / total))
        take = min(take, len(ids))
        chosen = rng.sample(sorted(ids), take)
        sample.extend(chosen)
        strata_info[f"{key[0]}|{'gradeable' if key[1] else 'ungradeable'}"] = {
            "available": len(ids), "sampled": take}

    # trim/pad to exactly n (deterministic)
    sample = sorted(set(sample))
    if len(sample) > args.n:
        sample = rng.sample(sample, args.n)
        sample.sort()

    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    out = {"n_requested": args.n, "n_sampled": len(sample), "seed": args.seed,
           "strata": strata_info, "ids": sample}
    with open(os.path.join(C.ARTIFACTS_DIR, "pilot_sample.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Sampled {len(sample)} law items across {len(strata_info)} strata "
          f"→ artifacts/pilot_sample.json")

    os.makedirs(args.configs, exist_ok=True)
    for cond in C.CONDITIONS:
        cfg = RunConfig(run_id=f"pilot__{cond}__{args.model}", condition=cond,
                        model_key=args.model, tier=C.TIER_LAW, backend=args.backend,
                        item_ids=sample, data=data_path)
        cfg.save(os.path.join(args.configs, cfg.run_id + ".yaml"))
    print(f"Wrote 4 pilot configs (E1-E4) for {args.model} → {args.configs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
