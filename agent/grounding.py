"""Grounding-correctness metrics (AGENT_PLAN_v2.md §6) — Grounded-Acc is the
headline metric: answer correct AND the cited chunk resolves to a gold
citation at khoản level. Divergence = P(citation wrong | answer correct) is
the signature to watch on E3 (the leakage probe): accuracy goes up while
citation accuracy stays flat means the system found the answer without
finding the law.

Both operate on an already-resolved citation (via agent.resolve.CitationIndex)
against the resolved gold set, so a điểm-level gold citation matched by its
khoản-level chunk counts as correct.
"""

from __future__ import annotations


def citation_wrong(citation_resolved: str | None, gold_resolved: set[str]) -> bool:
    """True iff the model's single citation does not support the gold
    citation set. No-violation items (gold empty) require citation to also
    be None — a hallucinated citation on a no-violation image is wrong."""
    if not gold_resolved:
        return citation_resolved is not None
    return citation_resolved is None or citation_resolved not in gold_resolved


def grounded_correct(answer_correct: bool, citation_resolved: str | None, gold_resolved: set[str]) -> bool:
    """Grounded-Acc (§6, headline): answer correct AND citation resolves to
    gold at khoản level."""
    return answer_correct and not citation_wrong(citation_resolved, gold_resolved)
