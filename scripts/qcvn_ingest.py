#!/usr/bin/env python3
"""QCVN 41:2024 ingestion CLI (EXPERIMENT_PLAN.md §7.2).

Two subcommands:

    skeleton   write a fill-in YAML of every unresolvable gold citation
    merge      merge a filled skeleton → augmented corpus + alias map, print
               the coverage delta

Usage:
    python scripts/qcvn_ingest.py skeleton --out artifacts/qcvn_skeleton.yaml
    # ...fill 'content' fields from the official QCVN 41:2024 text...
    python scripts/qcvn_ingest.py merge \
        --skeleton artifacts/qcvn_skeleton.yaml \
        --out-corpus artifacts/legal_traffic_chunks_v2.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vntraffic import constants as C
from vntraffic.corpus import load_corpus
from vntraffic.dataset import filter_tier, load_items
from vntraffic.qcvn_ingest import build_skeleton, merge_skeleton


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sk = sub.add_parser("skeleton", help="emit fill-in skeleton")
    p_sk.add_argument("--data", default=C.DATA_JSON)
    p_sk.add_argument("--corpus", default=C.CORPUS_JSON)
    p_sk.add_argument("--out", default=os.path.join(C.ARTIFACTS_DIR, "qcvn_skeleton.yaml"))

    p_mg = sub.add_parser("merge", help="merge filled skeleton")
    p_mg.add_argument("--data", default=C.DATA_JSON)
    p_mg.add_argument("--corpus", default=C.CORPUS_JSON)
    p_mg.add_argument("--skeleton", required=True)
    p_mg.add_argument("--out-corpus",
                      default=os.path.join(C.ARTIFACTS_DIR, "legal_traffic_chunks_v2.json"))

    args = ap.parse_args()
    items = load_items(args.data)
    corpus = load_corpus(args.corpus)

    if args.cmd == "skeleton":
        n = build_skeleton(items, corpus, args.out)
        print(f"Wrote {n} skeleton entries → {args.out}")
        print("Fill the 'content' fields from the official QCVN 41:2024 text, "
              "then run: qcvn_ingest.py merge --skeleton <file>")
    elif args.cmd == "merge":
        res = merge_skeleton(items, corpus, args.skeleton, args.out_corpus)
        print(f"Entries: {res.n_entries} | filled: {res.n_filled} | "
              f"skipped-empty: {res.n_skipped_empty}")
        print(f"New chunks: {res.n_new_chunks} | aliases: {res.n_aliases}")
        print(f"Coverage before: {res.coverage_before}")
        print(f"Coverage after:  {res.coverage_after}")
        print(f"Augmented corpus → {res.out_corpus_path}")
        print(f"Alias map        → {res.out_alias_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
