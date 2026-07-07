#!/usr/bin/env python3
"""Generate the full run matrix as config files (EXPERIMENT_PLAN.md §3, §4, §8).

Main sweep (§3 tier assignment):
  * Law tier:   E1 + E4 on the full ladder; E2 + E3 on the leakage subset.
  * Detection:  E1 (headline) + E4 (control) on the full ladder. No E2/E3.
Ablations (§8, best model, law tier):
  * A1 budget B ∈ {1,5} (B=3 is the main E4 run)
  * A2 top-k ∈ {3,10}   (k=5 is the main run) on E2/E3/E4
  * A4 embedder = Vietnamese-specific alternative on E2/E3/E4
  (A3 resolve on/off is a metric-time toggle, handled in make_tables.py.)
CircularEval (§5.5): E1 detection, E1 law, E4 law — best model.

Naming: {tier}__{condition}__{model}[__variant].yaml
Also writes configs/RUN_ORDER.txt (smallest-model-first within each phase, §11).

Usage:
    python scripts/gen_configs.py [--out configs] [--best-model internvl3-14b]
                                  [--alt-embedder <hf-id>]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vntraffic import constants as C
from vntraffic.config import RunConfig
from vntraffic.models.registry import (DEFAULT_BEST_MODEL, ladder_keys,
                                       leakage_subset_keys)

DEFAULT_ALT_EMBEDDER = "bkai-foundation-models/vietnamese-bi-encoder"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=C.CONFIGS_DIR)
    ap.add_argument("--best-model", default=DEFAULT_BEST_MODEL)
    ap.add_argument("--alt-embedder", default=DEFAULT_ALT_EMBEDDER)
    ap.add_argument("--backend", default="hf")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ladder = ladder_keys()                 # smallest-first
    leak = leakage_subset_keys()
    best = args.best_model
    order_main, order_ablation, order_circular = [], [], []

    def emit(cfg: RunConfig, bucket):
        path = os.path.join(args.out, cfg.run_id + ".yaml")
        cfg.save(path)
        bucket.append(cfg.run_id)

    def mk(run_id, condition, model, tier, **kw) -> RunConfig:
        return RunConfig(run_id=run_id, condition=condition, model_key=model,
                         tier=tier, backend=args.backend, **kw)

    # --- Main sweep: law tier E1+E4 full ladder ------------------------- #
    for model in ladder:
        emit(mk(f"law__E1__{model}", C.COND_E1, model, C.TIER_LAW), order_main)
        emit(mk(f"law__E4__{model}", C.COND_E4, model, C.TIER_LAW), order_main)

    # --- Main sweep: law tier E2+E3 leakage subset ---------------------- #
    for model in leak:
        emit(mk(f"law__E2__{model}", C.COND_E2, model, C.TIER_LAW), order_main)
        emit(mk(f"law__E3__{model}", C.COND_E3, model, C.TIER_LAW), order_main)

    # --- Main sweep: detection tier E1 (headline) + E4 (control) -------- #
    for model in ladder:
        emit(mk(f"det__E1__{model}", C.COND_E1, model, C.TIER_DETECTION), order_main)
        emit(mk(f"det__E4__{model}", C.COND_E4, model, C.TIER_DETECTION), order_main)

    # --- A1 budget ablation (E4, best model, law) ----------------------- #
    for b in (1, 5):
        emit(mk(f"law__E4__{best}__b{b}", C.COND_E4, best, C.TIER_LAW, budget=b),
             order_ablation)

    # --- A2 top-k ablation (E2/E3/E4, best model, law) ------------------ #
    for k in (3, 10):
        for cond in (C.COND_E2, C.COND_E3, C.COND_E4):
            emit(mk(f"law__{cond}__{best}__k{k}", cond, best, C.TIER_LAW, top_k=k),
                 order_ablation)

    # --- A4 embedder ablation (E2/E3/E4, best model, law) --------------- #
    emb_tag = args.alt_embedder.split("/")[-1]
    for cond in (C.COND_E2, C.COND_E3, C.COND_E4):
        emit(mk(f"law__{cond}__{best}__emb-{emb_tag}", cond, best, C.TIER_LAW,
                embedder=args.alt_embedder), order_ablation)

    # --- CircularEval (§5.5) -------------------------------------------- #
    emit(mk(f"det__E1__{best}__circular", C.COND_E1, best, C.TIER_DETECTION,
            circular=True), order_circular)
    emit(mk(f"law__E1__{best}__circular", C.COND_E1, best, C.TIER_LAW,
            circular=True), order_circular)
    emit(mk(f"law__E4__{best}__circular", C.COND_E4, best, C.TIER_LAW,
            circular=True), order_circular)

    with open(os.path.join(args.out, "RUN_ORDER.txt"), "w", encoding="utf-8") as f:
        f.write("# Main sweep (run smallest models first)\n")
        f.write("\n".join(order_main) + "\n\n")
        f.write("# Ablations (best model)\n")
        f.write("\n".join(order_ablation) + "\n\n")
        f.write("# CircularEval robustness\n")
        f.write("\n".join(order_circular) + "\n")

    total = len(order_main) + len(order_ablation) + len(order_circular)
    print(f"Wrote {total} configs → {args.out}")
    print(f"  main={len(order_main)} ablation={len(order_ablation)} "
          f"circular={len(order_circular)}")
    print(f"  RUN_ORDER.txt written; best model = {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
