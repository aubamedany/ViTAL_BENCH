"""Retriever interface (AGENT_PLAN_v2.md §3.3): one swappable component behind
`retrieve(query, corpus, k, as_of) -> list[RetrievedChunk]`. The pipeline only
ever talks to this Protocol.
"""

from __future__ import annotations

from typing import Literal, Protocol

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


Engine = Literal["lexical", "dense", "hybrid"]


def build_retriever(
    engine: Engine,
    corpus: Corpus,
    bm25_kwargs: dict | None = None,
    dense_kwargs: dict | None = None,
) -> Retriever:
    """QUERY_GEN_SPEC.md v3 Step 5: "One flag... lexical: BM25 over full
    text | dense: BGE-M3 over heading + body | hybrid: both, fused by RRF."
    Heavy imports (pyserini, FlagEmbedding) are deferred to inside each
    branch — constructing a "lexical"-engine retriever should never require
    torch/FlagEmbedding to even be importable, and vice versa."""
    if engine == "lexical":
        from agent.retrieval.bm25 import BM25Retriever

        return BM25Retriever(**(bm25_kwargs or {}))
    if engine == "dense":
        from agent.retrieval.dense import DenseRetriever

        return DenseRetriever(corpus, **(dense_kwargs or {}))
    if engine == "hybrid":
        from agent.retrieval.bm25 import BM25Retriever
        from agent.retrieval.dense import DenseRetriever
        from agent.retrieval.hybrid import HybridRetriever

        lexical = BM25Retriever(**(bm25_kwargs or {}))
        dense = DenseRetriever(corpus, **(dense_kwargs or {}))
        return HybridRetriever(lexical, dense)
    raise ValueError(f"unknown engine: {engine!r} (expected one of 'lexical', 'dense', 'hybrid')")
