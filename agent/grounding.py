"""Grounding-correctness metric — FROZEN per AGENT_PLAN.md §6b, before any
headline number exists, so it can't be tuned to the result.

Two numbers, always reported together: strict grounding (primary, T3) and
citation F1 (secondary, partial credit). Both operate on already-resolved
chunk_id sets (via agent.resolve.CitationIndex.resolve_set), so a điểm-level
gold citation matched by its khoản-level chunk counts as correct.
"""

from __future__ import annotations


def strict_grounding_correct(gold_resolved: set[str], pred_resolved: set[str]) -> bool:
    """Correct iff G ⊆ P — every citation the item requires was cited.
    Extra retrieved/cited chunks never penalize (over-retrieval is not the
    sin; missing law is). No-violation items (G empty) are scored on whether
    P is also empty — a hallucinated citation on a no-violation image is a
    false-violation error, not a grounding pass."""
    if not gold_resolved:
        return len(pred_resolved) == 0
    return gold_resolved <= pred_resolved


def citation_prf1(gold_resolved: set[str], pred_resolved: set[str]) -> tuple[float, float, float]:
    """Per-item precision/recall/F1 of pred against gold; macro-average
    across items at the caller. Exists because mean citations/item ~= 2 and
    24% of items need >=3 — strict all-or-nothing would score a 2-of-3 agent
    as fully ungrounded and understate real competence."""
    if not gold_resolved and not pred_resolved:
        return 1.0, 1.0, 1.0
    if not gold_resolved or not pred_resolved:
        return 0.0, 0.0, 0.0
    tp = len(gold_resolved & pred_resolved)
    precision = tp / len(pred_resolved)
    recall = tp / len(gold_resolved)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
