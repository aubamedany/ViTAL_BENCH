"""Retriever interface (AGENT_PLAN_v2.md §3.3): one swappable component behind
`retrieve(query, corpus, k, as_of) -> list[RetrievedChunk]`. The pipeline only
ever talks to this Protocol.
"""

from __future__ import annotations

from typing import Protocol

from agent.corpus import Corpus
from agent.schemas import RetrievalQuery, RetrievedChunk


class Retriever(Protocol):
    def retrieve(
        self,
        query: RetrievalQuery,
        corpus: Corpus,
        k: int,
        as_of: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to `k` chunks for `query`, scored after hard filters
        (agent/retrieval/filters.py) have been applied."""
        ...
