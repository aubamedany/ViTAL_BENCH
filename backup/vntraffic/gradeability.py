"""Citation-gradeability policy — Path B pre-registration (EXPERIMENT_PLAN.md §7.2).

Policy (frozen before any headline number is computed):

  A law item is **citation-gradeable** iff every one of its gold citations
  resolves to a corpus chunk (``resolve_citation`` returns a chunk_id). Items
  with an empty ``law`` list are gradeable *vacuously* (there is nothing to
  fail), but they are excluded from grounding precision/recall/F1 because there
  is no gold set — they are reported separately.

Consequences, held constant for the whole sweep:
  * grounding metrics (§5.2, §5.3) are computed **only** on the gradeable subset;
  * answer accuracy (§5.1) is always computed on the **full** law tier;
  * per-category gradeability is reported in the appendix.

With the shipped corpus + resolver this yields 974 gradeable law items
(955 with a non-empty, fully-resolvable gold set + 19 with empty gold).
Running the QCVN ingestion tool (qcvn_ingest.py) grows the corpus, which flips
currently-unresolvable items to gradeable and is picked up automatically the
next time this report is regenerated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from . import constants as C
from .corpus import Corpus
from .resolver import resolve_citation


@dataclass
class GradeabilityReport:
    n_law_items: int
    n_gradeable: int                 # all gold citations resolve (incl. empty gold)
    n_empty_gold: int                # gradeable-but-no-gold (excluded from F1)
    n_f1_gradeable: int              # gradeable AND non-empty gold
    n_partial: int                   # some but not all citations resolve
    n_none_resolvable: int           # non-empty gold, nothing resolves
    per_category: Dict[str, Dict[str, int]]  # category -> {n_law, n_gradeable, pct}
    gradeable_ids: List[int] = field(default_factory=list)
    f1_gradeable_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_law_items": self.n_law_items,
            "n_gradeable": self.n_gradeable,
            "n_empty_gold": self.n_empty_gold,
            "n_f1_gradeable": self.n_f1_gradeable,
            "n_partial": self.n_partial,
            "n_none_resolvable": self.n_none_resolvable,
            "per_category": self.per_category,
            "gradeable_ids": sorted(self.gradeable_ids),
            "f1_gradeable_ids": sorted(self.f1_gradeable_ids),
        }

    def summary_line(self) -> str:
        return (f"law items: {self.n_law_items} | gradeable={self.n_gradeable} "
                f"(non-empty gold={self.n_f1_gradeable}, empty gold={self.n_empty_gold}) "
                f"| partial={self.n_partial} none-resolvable={self.n_none_resolvable}")


def item_gradeable(item, corpus: Corpus) -> bool:
    """True iff every gold citation on *item* resolves (empty gold is vacuously True)."""
    for cit in item.law:
        if not resolve_citation(cit, corpus).resolved:
            return False
    return True


def build_gradeability_report(law_items: Sequence, corpus: Corpus) -> GradeabilityReport:
    n_gradeable = n_empty = n_f1 = n_partial = n_none = 0
    gradeable_ids: List[int] = []
    f1_ids: List[int] = []
    per_cat_total: Counter = Counter()
    per_cat_gradeable: Counter = Counter()

    for it in law_items:
        per_cat_total[it.category] += 1
        resolved_flags = [resolve_citation(c, corpus).resolved for c in it.law]
        n_res = sum(resolved_flags)
        n_cit = len(it.law)

        gradeable = (n_res == n_cit)  # empty gold → 0 == 0 → True
        if gradeable:
            n_gradeable += 1
            gradeable_ids.append(it.id)
            per_cat_gradeable[it.category] += 1
            if n_cit == 0:
                n_empty += 1
            else:
                n_f1 += 1
                f1_ids.append(it.id)
        else:
            if n_res == 0:
                n_none += 1
            else:
                n_partial += 1

    per_category: Dict[str, Dict[str, int]] = {}
    for cat in sorted(per_cat_total):
        tot = per_cat_total[cat]
        gr = per_cat_gradeable[cat]
        per_category[cat] = {
            "n_law": tot,
            "n_gradeable": gr,
            "pct_gradeable": round(100.0 * gr / tot, 1) if tot else 0.0,
        }

    return GradeabilityReport(
        n_law_items=len(law_items),
        n_gradeable=n_gradeable,
        n_empty_gold=n_empty,
        n_f1_gradeable=n_f1,
        n_partial=n_partial,
        n_none_resolvable=n_none,
        per_category=per_category,
        gradeable_ids=gradeable_ids,
        f1_gradeable_ids=f1_ids,
    )


def load_gradeable_ids(report_dict: dict, f1_only: bool = False) -> Set[int]:
    """Read a frozen gradeability report dict back into an id set."""
    key = "f1_gradeable_ids" if f1_only else "gradeable_ids"
    return set(report_dict.get(key, []))
