#!/usr/bin/env python3
"""Aggregate runs/ into paper tables T1-T6 + stats (EXPERIMENT_PLAN.md §10, §5.6).

Writes paper_tables/{T1..T6}.md and .tex, plus stats.json with bootstrap CIs on
headline accuracies and paired McNemar tests (E3 vs E4 answer accuracy; E4 vs E2
grounded accuracy).

Usage:
    python scripts/make_tables.py [--runs runs] [--out paper_tables]
                                  [--dataset-dir artifacts/dataset_v1]
                                  [--best-model internvl3-14b] [--no-ci]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backup.vntraffic import constants as C
from backup.vntraffic import metrics as M
from backup.vntraffic import stats as S
from backup.vntraffic import tables as T
from backup.vntraffic.models.registry import DEFAULT_BEST_MODEL, ladder_keys, leakage_subset_keys


def _write(out_dir, name, md, tex):
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(os.path.join(out_dir, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    print(f"  wrote {name}.md / {name}.tex")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=C.RUNS_DIR)
    ap.add_argument("--out", default=C.TABLES_DIR)
    ap.add_argument("--dataset-dir", default=C.DATASET_V1_DIR)
    ap.add_argument("--corpus", default=C.CORPUS_JSON)
    ap.add_argument("--corpus-aliases", default=None)
    ap.add_argument("--best-model", default=DEFAULT_BEST_MODEL)
    ap.add_argument("--no-ci", action="store_true", help="skip bootstrap CIs (faster)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ctx = T.Context.load(args.dataset_dir, args.corpus, args.corpus_aliases)
    runs = T.discover_runs(args.runs)
    if not runs:
        print(f"No runs found under {args.runs}. Nothing to do.")
        return 1
    ladder = ladder_keys()
    leak = leakage_subset_keys()

    print("Building tables ...")
    _write(args.out, "T1", *T.table_t1(runs, ctx, ladder))
    _write(args.out, "T2", *T.table_t2(runs, ctx, leak))
    _write(args.out, "T3", *T.table_t3(runs, ctx, leak))
    _write(args.out, "T4", *T.table_t4(runs, ctx, ladder))
    _write(args.out, "T5", *T.table_t5(runs, ctx, args.best_model))
    _write(args.out, "T6a_categories", *T.table_t6_categories(runs, ctx, args.best_model))
    _write(args.out, "T6b_abstention", *T.table_t6_abstention(runs, ctx, leak))
    sub = T.table_t6_subtypes(runs, ctx, args.best_model)
    if sub:
        _write(args.out, "T6c_subtypes", *sub)
    else:
        print("  (subtype table dropped — no law_subtype tags in dataset)")

    # ---- stats: CIs + McNemar ------------------------------------------ #
    stats_out = {"bootstrap_resamples": (0 if args.no_ci else args.bootstrap),
                 "ci": {}, "mcnemar": {}}
    if not args.no_ci:
        print("Bootstrap CIs on headline numbers ...")
        for m in ladder:
            for tier in (C.TIER_DETECTION, C.TIER_LAW):
                rv = T._find(runs, tier, C.COND_E1, m)
                if rv:
                    a = M.accuracy(rv.records())
                    stats_out["ci"][f"E1_{tier}_{m}_acc"] = \
                        S.bootstrap_ci(a.values, args.bootstrap).to_dict()
            e4 = T._find(runs, C.TIER_LAW, C.COND_E4, m)
            if e4:
                lk = M.leakage_decomposition(e4.records(), ctx.items_by_id,
                                             ctx.corpus, ctx.f1_gradeable_ids)
                stats_out["ci"][f"E4_law_{m}_grounded_acc"] = \
                    S.bootstrap_ci(lk.grounded_values, args.bootstrap).to_dict()

    print("Paired McNemar tests ...")
    for m in leak:
        e2 = T._find(runs, C.TIER_LAW, C.COND_E2, m)
        e3 = T._find(runs, C.TIER_LAW, C.COND_E3, m)
        e4 = T._find(runs, C.TIER_LAW, C.COND_E4, m)
        if e3 and e4:  # E3 vs E4 answer accuracy
            a3, a4 = M.accuracy(e3.records()), M.accuracy(e4.records())
            stats_out["mcnemar"][f"E3_vs_E4_acc_{m}"] = \
                S.mcnemar_by_ids(a3.ids, a3.values, a4.ids, a4.values).to_dict()
        if e2 and e4:  # E4 vs E2 grounded accuracy
            l2 = M.leakage_decomposition(e2.records(), ctx.items_by_id, ctx.corpus,
                                         ctx.f1_gradeable_ids)
            l4 = M.leakage_decomposition(e4.records(), ctx.items_by_id, ctx.corpus,
                                         ctx.f1_gradeable_ids)
            stats_out["mcnemar"][f"E4_vs_E2_grounded_{m}"] = \
                S.mcnemar_by_ids(l4.ids, l4.grounded_values,
                                 l2.ids, l2.grounded_values).to_dict()

    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)
    print(f"  wrote stats.json")
    print(f"\nTables → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
