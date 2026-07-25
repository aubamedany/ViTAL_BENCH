"""HybridRetriever — RRF fusion of a lexical and a dense engine behind the
same `Retriever` protocol (QUERY_GEN_SPEC.md v3 Step 5's "hybrid" row: "both,
fused by RRF").
"""

from __future__ import annotations

from agent.corpus import Corpus
from agent.retrieval.base import Retriever
from agent.schemas import RetrievalQuery, RetrievedChunk

# Same overfetch idiom as BM25Retriever: fuse from a wider pool than `k` so
# RRF has enough of each side's ranking to combine meaningfully. Fusing two
# already-truncated top-k lists would bias toward whichever side happened to
# rank an item inside the truncation window on its own.
OVERFETCH_MULTIPLIER = 4
OVERFETCH_MIN = 50


class HybridRetriever:
    def __init__(self, lexical: Retriever, dense: Retriever, rrf_k: int = 60):
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: RetrievalQuery,
        corpus: Corpus,
        k: int,
        as_of: str | None = None,
    ) -> list[RetrievedChunk]:
        fetch_n = max(k * OVERFETCH_MULTIPLIER, OVERFETCH_MIN)
        lex_hits = self.lexical.retrieve(query, corpus, fetch_n, as_of)
        dense_hits = self.dense.retrieve(query, corpus, fetch_n, as_of)

        rrf_scores: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}
        for ranked in (lex_hits, dense_hits):
            for rank, chunk in enumerate(ranked, start=1):
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                if chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score:
                    best[chunk.chunk_id] = chunk

        ordered_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [best[cid].model_copy(update={"score": rrf_scores[cid]}) for cid in ordered_ids[:k]]
