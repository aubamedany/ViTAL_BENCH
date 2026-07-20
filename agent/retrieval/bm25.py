"""BM25Retriever — R-BM25 (AGENT_PLAN.md §3.2), backed by a pyserini/Lucene
index built by `agent/build_bm25_index.py`.

Query text is segmented with the exact same `agent.retrieval.vn_tokenize.segment`
used to build the index, so indexed terms and query terms agree on Vietnamese
compound-word boundaries. Hard filters (§3.3) and the document-id score boost
are applied the same way as MockRetriever, over-fetching from Lucene first so
filtering/boosting after the fact doesn't starve the top-k.
"""

from __future__ import annotations

from pathlib import Path

from agent.corpus import Corpus, CorpusChunk
from agent.retrieval.filters import doc_boost_score, passes_effective_date, passes_vehicle_filter
from agent.retrieval.vn_tokenize import segment
from agent.schemas import RetrievalQuery, RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_DIR = ROOT / "index" / "bm25"

OVERFETCH_MULTIPLIER = 4
OVERFETCH_MIN = 50


class BM25Retriever:
    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        k1: float = 0.9,
        b: float = 0.4,
    ):
        # Some pyserini builds import their (unused-by-us) OpenAI SPLADE encoder
        # eagerly, which raises at import time if OPENAI_API_KEY isn't set.
        # BM25Retriever never touches that code path; set a dummy key only if
        # the caller hasn't set a real one.
        import os

        os.environ.setdefault("OPENAI_API_KEY", "unused-bm25-only")
        from pyserini.search.lucene import LuceneSearcher  # heavy import; keep lazy

        index_dir = Path(index_dir)
        if not index_dir.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_dir}. Build it first: "
                f"python -m agent.build_bm25_index --index {index_dir}"
            )
        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_bm25(k1=k1, b=b)

    def retrieve(
        self,
        query: RetrievalQuery,
        corpus: Corpus,
        k: int,
        iteration: int,
        as_of: str | None = None,
    ) -> list[RetrievedChunk]:
        q_text = segment(query.text).strip()
        if not q_text:
            return []

        fetch_n = max(k * OVERFETCH_MULTIPLIER, OVERFETCH_MIN)
        hits = self.searcher.search(q_text, k=fetch_n)

        scored: list[tuple[float, str, CorpusChunk]] = []
        for hit in hits:
            chunk = corpus.by_id.get(hit.docid)
            if chunk is None:
                continue  # index out of sync with corpus; skip rather than crash
            if not passes_vehicle_filter(chunk, query.vehicle_filter):
                continue
            if not passes_effective_date(chunk, as_of):
                continue
            score = float(hit.score) + doc_boost_score(chunk, query.doc_boost)
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
