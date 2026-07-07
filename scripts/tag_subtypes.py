#!/usr/bin/env python3
"""Emit / inspect the law_subtype review CSV (EXPERIMENT_PLAN.md §7.4).

The tagger never writes tags into the dataset directly. It proposes tags with
rule evidence into a CSV for human review; run_gates.py later merges the
reviewed file. This script has two subcommands:

    propose   scan law items → write review CSV (with rule evidence)
    summary   print how many items each proposed subtype covers

Usage:
    python scripts/tag_subtypes.py propose --out artifacts/subtype_review.csv
    python scripts/tag_subtypes.py summary
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vntraffic import constants as C
from vntraffic.dataset import filter_tier, load_items
from vntraffic.subtags import (build_proposals, coverage_summary, write_review_csv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prop = sub.add_parser("propose", help="write the review CSV")
    p_prop.add_argument("--data", default=C.DATA_JSON)
    p_prop.add_argument("--out", default=os.path.join(C.ARTIFACTS_DIR, "subtype_review.csv"))

    p_sum = sub.add_parser("summary", help="print proposed-subtype coverage")
    p_sum.add_argument("--data", default=C.DATA_JSON)

    args = ap.parse_args()
    law = filter_tier(load_items(args.data), C.TIER_LAW)
    proposals = build_proposals(law)

    if args.cmd == "propose":
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_review_csv(proposals, args.out)
        cov = coverage_summary(proposals)
        print(f"Wrote {len(proposals)} proposals → {args.out}")
        print("Proposed coverage:")
        for k, v in cov.items():
            print(f"  {k}: {v}")
        print("\nEdit the 'reviewed_subtype' column, then pass the file to "
              "run_gates.py --reviewed-subtypes.")
    elif args.cmd == "summary":
        for k, v in coverage_summary(proposals).items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
