"""CircularEval aggregation (EXPERIMENT_PLAN.md §5.5, MMBench-style).

Each item is evaluated under all 4 cyclic option permutations; it counts as
correct only if it is correct under *all* permutations. This kills the
"your accuracy is option-position bias" critique. Requires a run produced with
``circular: true`` (4 records per id, distinguished by ``perm_shift``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from . import constants as C


@dataclass
class CircularResult:
    n_items: int
    n_perms_expected: int
    circular_correct: int
    circular_accuracy: float          # correct iff correct under all perms
    vanilla_accuracy: float           # identity permutation only (for contrast)
    values: List[int] = field(default_factory=list)
    ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_items": self.n_items, "n_perms": self.n_perms_expected,
                "circular_correct": self.circular_correct,
                "circular_accuracy": round(self.circular_accuracy, 4),
                "vanilla_accuracy": round(self.vanilla_accuracy, 4)}


def circular_eval(records: Sequence[dict], n_perms: int = 4) -> CircularResult:
    by_id: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_id[r["id"]].append(r)

    values, ids = [], []
    vanilla_hits = vanilla_n = 0
    for iid, recs in by_id.items():
        # all-permutation correctness (require the full set present)
        all_correct = len(recs) >= n_perms and all(r.get("correct") for r in recs)
        values.append(1 if all_correct else 0)
        ids.append(iid)
        # vanilla = identity permutation record
        ident = next((r for r in recs if r.get("perm_shift", 0) == 0), None)
        if ident is not None:
            vanilla_n += 1
            vanilla_hits += 1 if ident.get("correct") else 0

    n = len(values)
    return CircularResult(
        n_items=n, n_perms_expected=n_perms, circular_correct=sum(values),
        circular_accuracy=(sum(values) / n if n else 0.0),
        vanilla_accuracy=(vanilla_hits / vanilla_n if vanilla_n else 0.0),
        values=values, ids=ids,
    )
