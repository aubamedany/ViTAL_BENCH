"""Metrics (EXPERIMENT_PLAN.md §5). All metrics recompute from per-item JSONL.

§5.1 answer accuracy      exact-match A/B/C/D + random/majority baselines;
                          balanced accuracy (defuses the detection B-skew)
§5.2 citation grounding   precision/recall/F1 at clause granularity, after
                          resolve_citation, on the gradeable subset; empty-gold
                          items are excluded from F1
§5.3 leakage 2x2          answer-correct × citations-correct decomposition;
                          grounded-accuracy = P(correct answer ∧ citations correct)
                          — the headline number for Claim 4; Jaccard as a softer
                          secondary criterion
§5.4 abstention           on no-violation / indeterminate images

Every function returns per-item values (0/1 or floats) alongside the aggregate so
stats.py can bootstrap CIs and run McNemar on identical items.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import constants as C
from .corpus import Corpus
from .resolver import resolve_citation


# --------------------------------------------------------------------------- #
# IO / joins
# --------------------------------------------------------------------------- #
def load_records(jsonl_path: str) -> List[dict]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def only_identity(records: Sequence[dict]) -> List[dict]:
    """Keep the un-permuted (perm_shift 0 / absent) records — the primary run."""
    return [r for r in records if r.get("perm_shift", 0) == 0]


def resolve_set(citations: Iterable[str], corpus: Corpus) -> Set[str]:
    out: Set[str] = set()
    for c in citations:
        r = resolve_citation(c, corpus)
        if r.chunk_id:
            out.add(r.chunk_id)
    return out


# --------------------------------------------------------------------------- #
# §5.1 accuracy
# --------------------------------------------------------------------------- #
@dataclass
class AccuracyResult:
    n: int
    correct: int
    accuracy: float
    balanced_accuracy: float
    per_letter_recall: Dict[str, float]
    values: List[int] = field(default_factory=list)  # per-item 0/1 for bootstrap
    ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n": self.n, "correct": self.correct,
                "accuracy": round(self.accuracy, 4),
                "balanced_accuracy": round(self.balanced_accuracy, 4),
                "per_letter_recall": {k: round(v, 4) for k, v in self.per_letter_recall.items()}}


def accuracy(records: Sequence[dict]) -> AccuracyResult:
    recs = only_identity(records)
    values = [1 if r.get("correct") else 0 for r in recs]
    ids = [r["id"] for r in recs]
    n = len(recs)
    correct = sum(values)
    # balanced accuracy: mean per-GT-letter recall
    by_letter_total: Dict[str, int] = defaultdict(int)
    by_letter_hit: Dict[str, int] = defaultdict(int)
    for r in recs:
        gt = r.get("gt")
        by_letter_total[gt] += 1
        if r.get("correct"):
            by_letter_hit[gt] += 1
    per_letter = {L: (by_letter_hit[L] / by_letter_total[L]) if by_letter_total[L] else 0.0
                  for L in C.OPTION_LETTERS}
    present = [L for L in C.OPTION_LETTERS if by_letter_total[L] > 0]
    bal = sum(per_letter[L] for L in present) / len(present) if present else 0.0
    return AccuracyResult(
        n=n, correct=correct, accuracy=(correct / n if n else 0.0),
        balanced_accuracy=bal, per_letter_recall=per_letter, values=values, ids=ids,
    )


def baselines(tier: str) -> Dict[str, float]:
    return {"random": C.RANDOM_BASELINE,
            "majority": C.MAJORITY_BASELINE.get(tier, C.RANDOM_BASELINE),
            "majority_label": C.MAJORITY_LABEL.get(tier, "?")}


# --------------------------------------------------------------------------- #
# §5.2 grounding P/R/F1  and  §5.3 leakage 2x2 / grounded accuracy
# --------------------------------------------------------------------------- #
@dataclass
class GroundingResult:
    n: int                                   # items graded (non-empty gold, gradeable)
    precision: float
    recall: float
    f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    mean_jaccard: float
    per_item_f1: List[float] = field(default_factory=list)
    ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n": self.n,
                "precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "micro_precision": round(self.micro_precision, 4),
                "micro_recall": round(self.micro_recall, 4),
                "micro_f1": round(self.micro_f1, 4),
                "mean_jaccard": round(self.mean_jaccard, 4)}


def _prf(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def _jaccard(pred: Set[str], gold: Set[str]) -> float:
    if not pred and not gold:
        return 1.0
    union = pred | gold
    return len(pred & gold) / len(union) if union else 0.0


def grounding(records: Sequence[dict], items_by_id: Dict[int, object],
              corpus: Corpus, graded_ids: Set[int]) -> GroundingResult:
    recs = only_identity(records)
    ps, rs, fs, js, ids = [], [], [], [], []
    micro_tp = micro_pred = micro_gold = 0
    for r in recs:
        if r["id"] not in graded_ids:
            continue
        item = items_by_id.get(r["id"])
        if item is None or not getattr(item, "law", None):
            continue  # empty-gold excluded from F1
        gold = resolve_set(item.law, corpus)
        pred = resolve_set(r.get("pred_citations", []), corpus)
        p, rr, f = _prf(pred, gold)
        ps.append(p); rs.append(rr); fs.append(f); js.append(_jaccard(pred, gold))
        ids.append(r["id"])
        micro_tp += len(pred & gold); micro_pred += len(pred); micro_gold += len(gold)
    n = len(fs)
    mp = micro_tp / micro_pred if micro_pred else 0.0
    mr = micro_tp / micro_gold if micro_gold else 0.0
    mf = (2 * mp * mr / (mp + mr)) if (mp + mr) else 0.0
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return GroundingResult(
        n=n, precision=mean(ps), recall=mean(rs), f1=mean(fs),
        micro_precision=mp, micro_recall=mr, micro_f1=mf, mean_jaccard=mean(js),
        per_item_f1=fs, ids=ids,
    )


@dataclass
class LeakageResult:
    n: int
    # 2x2 cells
    grounded_correct: int      # ✅ answer correct  & citations correct
    leaked_correct: int        # ⚠️ answer correct  & citations wrong/absent
    grounded_misselect: int    # 🔎 answer wrong    & citations correct
    fully_wrong: int           # ❌ answer wrong    & citations wrong/absent
    grounded_accuracy: float   # P(✅)  — headline for Claim 4
    answer_accuracy: float     # over the same graded subset (for reference)
    citation_correct_rate: float
    # per-item values for bootstrapping / McNemar
    grounded_values: List[int] = field(default_factory=list)  # 1 if ✅
    ids: List[int] = field(default_factory=list)

    def cells(self) -> dict:
        return {"grounded_correct": self.grounded_correct,
                "leaked_correct": self.leaked_correct,
                "grounded_misselect": self.grounded_misselect,
                "fully_wrong": self.fully_wrong}

    def to_dict(self) -> dict:
        return {"n": self.n, "cells": self.cells(),
                "grounded_accuracy": round(self.grounded_accuracy, 4),
                "answer_accuracy": round(self.answer_accuracy, 4),
                "citation_correct_rate": round(self.citation_correct_rate, 4)}


def citations_correct(pred: Set[str], gold: Set[str]) -> bool:
    """Definition from §5.3: gold ⊆ resolved predicted set (gold non-empty)."""
    return bool(gold) and gold.issubset(pred)


def leakage_decomposition(records: Sequence[dict], items_by_id: Dict[int, object],
                          corpus: Corpus, graded_ids: Set[int]) -> LeakageResult:
    recs = only_identity(records)
    gc = lc = gm = fw = 0
    grounded_values, ids = [], []
    for r in recs:
        if r["id"] not in graded_ids:
            continue
        item = items_by_id.get(r["id"])
        if item is None or not getattr(item, "law", None):
            continue
        gold = resolve_set(item.law, corpus)
        pred = resolve_set(r.get("pred_citations", []), corpus)
        ans_ok = bool(r.get("correct"))
        cit_ok = citations_correct(pred, gold)
        if ans_ok and cit_ok:
            gc += 1
        elif ans_ok and not cit_ok:
            lc += 1
        elif (not ans_ok) and cit_ok:
            gm += 1
        else:
            fw += 1
        grounded_values.append(1 if (ans_ok and cit_ok) else 0)
        ids.append(r["id"])
    n = gc + lc + gm + fw
    return LeakageResult(
        n=n, grounded_correct=gc, leaked_correct=lc, grounded_misselect=gm,
        fully_wrong=fw,
        grounded_accuracy=(gc / n if n else 0.0),
        answer_accuracy=((gc + lc) / n if n else 0.0),
        citation_correct_rate=((gc + gm) / n if n else 0.0),
        grounded_values=grounded_values, ids=ids,
    )


# --------------------------------------------------------------------------- #
# §5.4 abstention / calibration
# --------------------------------------------------------------------------- #
@dataclass
class AbstentionResult:
    category: str
    n: int
    abstain_when_gt: float          # picks the GT "cannot determine/no violation" option
    n_gt_abstain: int
    false_violation_rate: float     # hallucinated citations where GT is no-violation
    n_false_violation: int

    def to_dict(self) -> dict:
        return {"category": self.category, "n": self.n,
                "abstain_when_gt": round(self.abstain_when_gt, 4),
                "n_gt_abstain": self.n_gt_abstain,
                "false_violation_rate": round(self.false_violation_rate, 4),
                "n_false_violation": self.n_false_violation}


def abstention(records: Sequence[dict], items_by_id: Dict[int, object],
               category: str) -> AbstentionResult:
    """Abstention rate + false-violation (hallucinated citation) rate.

    'abstain_when_gt' = accuracy on items of this category (the GT option here is
    the no-violation / indeterminate option, so selecting GT == abstaining
    correctly). 'false_violation_rate' = fraction of this category's items where
    the model emitted ≥1 citation (claimed a violation) despite the category.
    """
    recs = [r for r in only_identity(records) if r.get("category") == category]
    n = len(recs)
    n_abstain = sum(1 for r in recs if r.get("correct"))
    n_false = sum(1 for r in recs if r.get("pred_citations"))
    return AbstentionResult(
        category=category, n=n,
        abstain_when_gt=(n_abstain / n if n else 0.0), n_gt_abstain=n_abstain,
        false_violation_rate=(n_false / n if n else 0.0), n_false_violation=n_false,
    )
