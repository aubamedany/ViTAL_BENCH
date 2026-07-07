"""Data-gate operations: dedup, detection rebalance, and dataset freeze.

Implements EXPERIMENT_PLAN.md §7.3 (rebalance) and §7.5 (dedup) plus the freeze
that stamps a dataset version. The original ``data.json`` is never mutated — the
frozen artifact is written under ``artifacts/dataset_v1/``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import constants as C
from .dataset import (QAItem, apply_permutation, permutation_to_target_gt)


# --------------------------------------------------------------------------- #
# Dedup (§7.5)
# --------------------------------------------------------------------------- #
@dataclass
class DedupResult:
    n_before: int
    n_after: int
    removed_ids: List[int] = field(default_factory=list)
    groups: List[List[int]] = field(default_factory=list)  # duplicate id groups


def dedup_items(items: Sequence[QAItem]) -> Tuple[List[QAItem], DedupResult]:
    """Remove exact (image, question) duplicates, keeping the lowest id."""
    by_key: Dict[Tuple[str, str], List[QAItem]] = {}
    for it in items:
        by_key.setdefault((it.image, it.question), []).append(it)

    keep: List[QAItem] = []
    removed: List[int] = []
    groups: List[List[int]] = []
    for key, group in by_key.items():
        if len(group) > 1:
            group_sorted = sorted(group, key=lambda x: x.id)
            keep.append(group_sorted[0])
            removed.extend(g.id for g in group_sorted[1:])
            groups.append([g.id for g in group_sorted])
        else:
            keep.append(group[0])

    keep.sort(key=lambda x: x.id)
    return keep, DedupResult(
        n_before=len(items), n_after=len(keep),
        removed_ids=sorted(removed), groups=groups,
    )


# --------------------------------------------------------------------------- #
# Detection rebalance (§7.3)
# --------------------------------------------------------------------------- #
@dataclass
class RebalanceResult:
    n_items: int
    gt_before: Dict[str, int]
    gt_after: Dict[str, int]
    # provenance: id -> {orig_GT, new_GT, perm}; kept in the manifest, not items
    provenance: Dict[int, dict] = field(default_factory=dict)


def _gt_distribution(items: Sequence[QAItem]) -> Dict[str, int]:
    d = {L: 0 for L in C.OPTION_LETTERS}
    for it in items:
        d[it.gt] += 1
    return d


def rebalance_detection(items: Sequence[QAItem],
                        seed: int = C.GLOBAL_SEED) -> Tuple[List[QAItem], RebalanceResult]:
    """Permute detection-item options so GT is ~uniform over A/B/C/D.

    Target letters are assigned as evenly as possible (n//4 each, remainder
    spread) and shuffled with a fixed seed; each item's options are then permuted
    so its correct answer lands on the assigned letter. Only detection items are
    touched — the law tier is returned unchanged. Deterministic given *seed*.
    """
    det = [it for it in items if it.is_detection]
    other = [it for it in items if not it.is_detection]

    n = len(det)
    letters = list(C.OPTION_LETTERS)
    base = n // len(letters)
    rem = n % len(letters)
    targets: List[str] = []
    for i, L in enumerate(letters):
        targets.extend([L] * (base + (1 if i < rem else 0)))

    rng = random.Random(seed)
    rng.shuffle(targets)
    # assign targets to items in a fixed (id-sorted) order for reproducibility
    det_sorted = sorted(det, key=lambda x: x.id)

    gt_before = _gt_distribution(det)
    provenance: Dict[int, dict] = {}
    rebalanced: List[QAItem] = []
    for it, target in zip(det_sorted, targets):
        perm = permutation_to_target_gt(it, target)
        new_it = apply_permutation(it, perm)
        rebalanced.append(new_it)
        provenance[it.id] = {"orig_GT": it.gt, "new_GT": new_it.gt, "perm": perm}

    gt_after = _gt_distribution(rebalanced)
    merged = sorted(rebalanced + other, key=lambda x: x.id)
    return merged, RebalanceResult(
        n_items=n, gt_before=gt_before, gt_after=gt_after, provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Freeze
# --------------------------------------------------------------------------- #
def dataset_version(items: Sequence[QAItem]) -> str:
    """Content hash of the (ordered) frozen items → dataset version string."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.id):
        h.update(repr(it.to_dict()).encode("utf-8"))
    return h.hexdigest()[:12]
