"""MockRetriever — deterministic lexical-overlap stand-in for R-BM25/R-BGE/etc.

AGENT_PLAN.md §3.2 makes retriever choice an empirical question answered by
experiment R, comparing BM25/BGE-M3/e5/Vietnamese-encoder/hybrid behind one
`Retriever` interface. This module is *not* one of those candidates — it is a
zero-dependency, fully deterministic scorer whose only job is to let the rest
of the pipeline (FORMULATE -> RETRIEVE -> ASSESS -> REFINE loop, logging,
citation closure) be built and tested against the real law_db.json corpus
before any embedding model or BM25 index is wired in. Swap it for a real
retriever without touching pipeline.py — that swappability is the point.
"""

from __future__ import annotations

import re
import unicodedata

from agent.corpus import Corpus, CorpusChunk
from agent.retrieval.filters import apply_hard_filters, doc_boost_score
from agent.schemas import RetrievalQuery, RetrievedChunk

_WORD_RE = re.compile(r"[a-zA-Z0-9À-ỹ]+")


def _tokenize(text: str) -> set[str]:
    folded = unicodedata.normalize("NFC", text).lower()
    return set(_WORD_RE.findall(folded))


class MockRetriever:
    """Scores chunks by Jaccard word overlap between the query text and the
    chunk's title+content, then adds the §3.3 document-boost. Ties break on
    chunk_id for full determinism."""

    def __init__(self, corpus: Corpus):
        self._tokens: dict[str, set[str]] = {
            c.chunk_id: _tokenize(f"{c.title} {c.text}") for c in corpus.chunks
        }

    def retrieve(
        self,
        query: RetrievalQuery,
        corpus: Corpus,
        k: int,
        iteration: int,
        as_of: str | None = None,
    ) -> list[RetrievedChunk]:
        candidates = apply_hard_filters(corpus.chunks, query, as_of)
        q_tokens = _tokenize(query.text)

        scored: list[tuple[float, str, CorpusChunk]] = []
        for chunk in candidates:
            c_tokens = self._tokens[chunk.chunk_id]
            if not q_tokens or not c_tokens:
                overlap = 0.0
            else:
                overlap = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
            score = overlap + doc_boost_score(chunk, query.doc_boost)
            if score > 0:
                scored.append((score, chunk.chunk_id, chunk))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                citation=chunk.title,
                content=chunk.text,
                score=score,
                retrieved_at_iteration=iteration,
            )
            for score, _, chunk in scored[:k]
        ]
