#!/usr/bin/env python3
"""Automatic pilot go/no-go gate (EXPERIMENT_PLAN.md §9.2).

Go criteria:
  (a) pipeline runs end-to-end with full audit logs (E4 records carry an audit
      trail);
  (b) leakage signature visible: E3 accuracy > E2 accuracy AND E3 grounding F1
      ≈ E2 grounding F1 (i.e. E3 does not buy grounding, only answers);
  (c) E4 grounding F1 > E3 grounding F1.

If leakage does not appear, STOP and debug E3 query construction before spending
main-sweep compute.

Usage:
    python scripts/check_pilot_gonogo.py [--runs runs] [--model internvl3-8b]
                                         [--dataset-dir artifacts/dataset_v1]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backup.vntraffic import constants as C
from backup.vntraffic import metrics as M
from backup.vntraffic import tables as T


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=C.RUNS_DIR)
    ap.add_argument("--model", default="internvl3-8b")
    ap.add_argument("--dataset-dir", default=C.DATASET_V1_DIR)
    ap.add_argument("--corpus", default=C.CORPUS_JSON)
    ap.add_argument("--f1-eps", type=float, default=0.02,
                    help="tolerance for 'E3 F1 ≈ E2 F1' (F1_E3 <= F1_E2 + eps)")
    args = ap.parse_args()

    ctx = T.Context.load(args.dataset_dir, args.corpus)
    runs = T.discover_runs(args.runs)

    def get(cond):
        # prefer a pilot__ run; fall back to any law run for the model+cond
        for r in runs:
            if r.run_id == f"pilot__{cond}__{args.model}":
                return r
        return T._find(runs, C.TIER_LAW, cond, args.model)

    e2, e3, e4 = get(C.COND_E2), get(C.COND_E3), get(C.COND_E4)
    if not all([e2, e3, e4]):
        print("Missing pilot runs (need E2, E3, E4). Found: "
              f"E2={bool(e2)} E3={bool(e3)} E4={bool(e4)}")
        return 2

    acc2 = M.accuracy(e2.records()).accuracy
    acc3 = M.accuracy(e3.records()).accuracy
    f1_2 = M.grounding(e2.records(), ctx.items_by_id, ctx.corpus, ctx.f1_gradeable_ids).f1
    f1_3 = M.grounding(e3.records(), ctx.items_by_id, ctx.corpus, ctx.f1_gradeable_ids).f1
    f1_4 = M.grounding(e4.records(), ctx.items_by_id, ctx.corpus, ctx.f1_gradeable_ids).f1

    # (a) audit logs present
    audit_ok = any(r.get("audit") for r in M.only_identity(e4.records()))
    # (b) leakage signature
    leak_acc_ok = acc3 > acc2
    leak_f1_ok = f1_3 <= f1_2 + args.f1_eps      # E3 F1 not meaningfully higher
    leak_ok = leak_acc_ok and leak_f1_ok
    # (c) agent grounds better than leaky RAG
    e4_ok = f1_4 > f1_3

    print("Pilot metrics:")
    print(f"  E2 acc={acc2:.3f}  E3 acc={acc3:.3f}   (E3>E2: {_flag(leak_acc_ok)})")
    print(f"  E2 F1 ={f1_2:.3f}  E3 F1 ={f1_3:.3f}   (E3≈E2: {_flag(leak_f1_ok)})")
    print(f"  E4 F1 ={f1_4:.3f}                    (E4>E3: {_flag(e4_ok)})")
    print("\nCriteria:")
    print(f"  (a) audit logs present : {_flag(audit_ok)}")
    print(f"  (b) leakage signature  : {_flag(leak_ok)}")
    print(f"  (c) E4 grounds > E3    : {_flag(e4_ok)}")

    go = audit_ok and leak_ok and e4_ok
    print(f"\n==> {'GO' if go else 'NO-GO'}")
    if not go and not leak_ok:
        print("Leakage signature absent — debug E3 query construction before the "
              "main sweep (§9.2, §12).")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
