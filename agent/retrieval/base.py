"""Retriever interface (AGENT_PLAN.md §3.2): one swappable component behind
`retrieve(query, k, filters) -> list[RetrievedChunk]`. Retriever choice (BM25 /
BGE-M3 / e5 / hybrid / rerank) is an empirical question (experiment R, M1),
never a design assumption — the pipeline only ever talks to this Protocol.
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
        iteration: int,
        as_of: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to `k` chunks for `query`, tagged with `iteration` (§7
        logging) and scored after hard filters (agent/retrieval/filters.py)
        have been applied."""
        ...
