#!/usr/bin/env python3
"""Run all blocking data gates and freeze dataset v1 (EXPERIMENT_PLAN.md §7, §9.1).

Order (as specified in §9.1):
  dedup (§7.5) → resolver coverage (§7.1) → gradeability / Path B (§7.2)
  → detection rebalance (§7.3) → optional subtype tags (§7.4) → freeze + manifest.

The original data.json / corpus are never mutated. Outputs land under
artifacts/dataset_v1/:
    data.json            frozen, deduped, rebalanced (subtype-tagged if reviewed)
    manifest.json        versions, seeds, coverage, gradeability, rebalance provenance
    coverage.json        full resolver coverage report
    gradeability.json    full gradeability report (incl. frozen gradeable id sets)

Usage:
    python scripts/run_gates.py [--reviewed-subtypes path/to/reviewed.csv]
                                [--no-rebalance] [--seed N] [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vntraffic import constants as C
from vntraffic.corpus import load_corpus
from vntraffic.dataset import filter_tier, load_items, save_items
from vntraffic.gates import (dataset_version, dedup_items, rebalance_detection)
from vntraffic.gradeability import build_gradeability_report
from vntraffic.resolver import build_coverage_report, gold_citations_from_items
from vntraffic.subtags import apply_reviewed, read_reviewed_csv


def _check(label, got, expected):
    flag = "OK " if got == expected else "!! "
    print(f"  [{flag}] {label}: {got} (expected {expected})")
    return got == expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=C.DATA_JSON)
    ap.add_argument("--corpus", default=C.CORPUS_JSON)
    ap.add_argument("--out", default=C.DATASET_V1_DIR)
    ap.add_argument("--seed", type=int, default=C.GLOBAL_SEED)
    ap.add_argument("--no-rebalance", action="store_true",
                    help="skip detection rebalance (rely on CircularEval instead)")
    ap.add_argument("--reviewed-subtypes", default=None,
                    help="reviewed subtype CSV to merge (§7.4); omit to skip tagging")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.data} and {args.corpus} ...")
    items = load_items(args.data)
    corpus = load_corpus(args.corpus)
    print(f"  {len(items)} items, {len(corpus)} chunks (corpus v{corpus.version})")

    # --- pre-registration snapshot: verify against EXPERIMENT_PLAN §2 ----- #
    # These are computed on the *shipped* (pre-dedup) data so they reproduce the
    # frozen numbers in the plan exactly. The frozen artifact below re-reports
    # coverage/gradeability on the deduped set (which drops the 1 duplicate).
    print("\n[0/5] Pre-registration snapshot (verify EXPERIMENT_PLAN §2)")
    law0 = filter_tier(items, C.TIER_LAW)
    coverage0 = build_coverage_report(gold_citations_from_items(law0), corpus)
    print("       " + coverage0.summary_line())
    ok = True
    for k, exp in C.EXPECTED_COVERAGE.items():
        ok &= _check(f"coverage {k}", coverage0.counts.get(k, 0), exp)
    grade0 = build_gradeability_report(law0, corpus)
    print("       " + grade0.summary_line())
    ok &= _check("gradeable law items", grade0.n_gradeable, C.EXPECTED_GRADEABLE_LAW_ITEMS)
    if not ok:
        print("  WARNING: pre-registration numbers drifted from the plan snapshot.")

    # --- §7.5 dedup ------------------------------------------------------- #
    print("\n[1/5] Dedup (§7.5)")
    items, dd = dedup_items(items)
    _check("removed duplicates", len(dd.removed_ids), 1)
    print(f"       kept {dd.n_after} items (removed ids: {dd.removed_ids})")

    # --- §7.1 resolver coverage (frozen, post-dedup) --------------------- #
    print("\n[2/5] Resolver coverage (§7.1, frozen set)")
    law = filter_tier(items, C.TIER_LAW)
    coverage = build_coverage_report(gold_citations_from_items(law), corpus)
    print("       " + coverage.summary_line())

    # --- §7.2 gradeability (Path B, frozen post-dedup) ------------------- #
    print("\n[3/5] Gradeability / Path B (§7.2, frozen set)")
    grade = build_gradeability_report(law, corpus)
    print("       " + grade.summary_line())
    print(f"       (pre-dedup gradeable was {grade0.n_gradeable}; "
          f"dedup removed {grade0.n_gradeable - grade.n_gradeable} gradeable law item)")

    # --- §7.3 detection rebalance ---------------------------------------- #
    print("\n[4/5] Detection rebalance (§7.3)")
    if args.no_rebalance:
        print("       skipped (--no-rebalance); relying on CircularEval")
        rebal = None
    else:
        items, rebal = rebalance_detection(items, seed=args.seed)
        print(f"       GT before: {rebal.gt_before}")
        print(f"       GT after:  {rebal.gt_after}")

    # --- §7.4 subtype tags (optional) ------------------------------------ #
    print("\n[5/5] Subtype tags (§7.4)")
    subtype_info = {"applied": False}
    if args.reviewed_subtypes:
        reviewed = read_reviewed_csv(args.reviewed_subtypes)
        n = apply_reviewed(items, reviewed)
        subtype_info = {"applied": True, "n_tagged": n,
                        "source": os.path.abspath(args.reviewed_subtypes)}
        print(f"       applied {n} reviewed subtype tags")
    else:
        print("       skipped (no --reviewed-subtypes); subtype table will be dropped")

    # --- freeze ----------------------------------------------------------- #
    ds_ver = dataset_version(items)
    out_data = os.path.join(args.out, "data.json")
    save_items(items, out_data)

    with open(os.path.join(args.out, "coverage.json"), "w", encoding="utf-8") as f:
        json.dump(coverage.to_dict(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "gradeability.json"), "w", encoding="utf-8") as f:
        json.dump(grade.to_dict(), f, ensure_ascii=False, indent=2)

    manifest = {
        "created": _dt.datetime.utcnow().isoformat() + "Z",
        "dataset_version": ds_ver,
        "corpus_version": corpus.version,
        "seed": args.seed,
        "n_items": len(items),
        "n_detection": len(filter_tier(items, C.TIER_DETECTION)),
        "n_law": len(filter_tier(items, C.TIER_LAW)),
        "dedup": {"n_before": dd.n_before, "n_after": dd.n_after,
                  "removed_ids": dd.removed_ids, "groups": dd.groups},
        "pre_registration": {
            "note": "computed on shipped pre-dedup data; matches EXPERIMENT_PLAN §2",
            "coverage": coverage0.counts,
            "n_gradeable": grade0.n_gradeable,
            "n_law": len(law0),
        },
        "coverage": coverage.counts,
        "gradeability": {k: v for k, v in grade.to_dict().items()
                         if k not in ("gradeable_ids", "f1_gradeable_ids")},
        "gradeable_ids": grade.gradeable_ids,
        "f1_gradeable_ids": grade.f1_gradeable_ids,
        "rebalance": (None if rebal is None else {
            "gt_before": rebal.gt_before, "gt_after": rebal.gt_after,
            "provenance": {str(k): v for k, v in rebal.provenance.items()}}),
        "subtypes": subtype_info,
        "source_data": os.path.abspath(args.data),
        "source_corpus": os.path.abspath(args.corpus),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nFrozen dataset v1 → {args.out}")
    print(f"  dataset_version={ds_ver}  corpus_version={corpus.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
