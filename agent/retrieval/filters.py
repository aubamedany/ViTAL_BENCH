"""Hard filters — free precision from metadata (AGENT_PLAN.md §3.3).

Every filter here fails open by design: an unmappable/missing value means "no
filter", never a silent exclusion. That is the load-bearing property (a wrong
forced exclusion drops the correct statute; a missed filter just costs a bit
of pruning) — see agent/vehicle.py and the §3.5 note on why law_db.json's
`vehicle_type`/`effective_date` are still empty today.
"""

from __future__ import annotations

from agent.corpus import CorpusChunk
from agent.schemas import RetrievalQuery

# Score boost, not a hard exclusion (§3.3): 32% of items need >=2 documents,
# so hard-excluding by doc_boost would break cross-document reasoning.
DOC_BOOST_AMOUNT = 0.15


def passes_vehicle_filter(chunk: CorpusChunk, vehicle_filter: list[str]) -> bool:
    if not vehicle_filter:
        return True  # no vehicle recognized -> fail open, no filter applied
    if not chunk.vehicle_type:
        return True  # empty = generic (55% of chunks); never filtered out
    return chunk.vehicle_type in vehicle_filter


def passes_effective_date(chunk: CorpusChunk, as_of: str | None) -> bool:
    if as_of is None or chunk.effective_date is None:
        return True  # fail open; metadata not yet ingested (§3.3/§3.5)
    return chunk.effective_date <= as_of  # ISO 8601 strings sort lexicographically


def doc_boost_score(chunk: CorpusChunk, doc_boost: list[str]) -> float:
    return DOC_BOOST_AMOUNT if chunk.document_id in doc_boost else 0.0


def apply_hard_filters(
    chunks: list[CorpusChunk], query: RetrievalQuery, as_of: str | None
) -> list[CorpusChunk]:
    """Prune the candidate set before scoring. Never raises on an
    unmappable/missing filter value — see module docstring."""
    return [
        c
        for c in chunks
        if passes_vehicle_filter(c, query.vehicle_filter)
        and passes_effective_date(c, as_of)
    ]
