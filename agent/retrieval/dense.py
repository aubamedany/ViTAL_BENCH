"""DenseRetriever — BGE-M3 dense embeddings behind the same `Retriever`
protocol `BM25Retriever`/`MockRetriever` implement (QUERY_GEN_SPEC.md v3
Step 5). Verified worth building before writing this: on the LUAT_36_2024
partition (raw question as query, gold rank within-partition), BGE-M3 beat
this corpus's BM25 by +8.4 points top-5 (43.1% vs 34.7%) and cut median gold
rank from 25 to 8 — the spec's own gate ("if dense does not beat BM25 on
LUAT, it will not help anywhere, stop") passes clearly.

Field selection matters and is NOT optional (Step 5): embeds ONLY the
"Tên điều hoặc mục" (heading) + "Nội dung" (body) — never the full chunk
text. The `Văn bản:`/`Trích dẫn:`/`Ngữ cảnh:` preamble is boilerplate
repeated near-verbatim across most chunks; BM25 is immune (IDF flattens
terms common to every document) but a dense encoder pools everything into
one vector and that boilerplate collapses distinct chunks toward the same
point.

Embeddings are cached to disk keyed by `corpus.version` (the sha1 the
existing `Corpus.load()` already computes) — encoding is fast (seconds for
~1000 chunks) but there's no reason to pay it more than once per corpus
version, and this keeps `DenseRetriever()` cheap to construct across
repeated runs/notebook cells.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from agent.corpus import Corpus, CorpusChunk
from agent.retrieval.filters import apply_hard_filters, doc_boost_score
from agent.schemas import RetrievalQuery, RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = ROOT / "index" / "dense"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"

HEAD_RE = re.compile(r"Tên điều hoặc mục:\s*([^.]+?)\.\s*(?:Phương tiện|Nội dung):")
# One ND chunk (ND_168_2024_D7_K3) lacks the "Phương tiện:"/"Nội dung:" label
# entirely — same corpus quirk noted in scripts/build_banks.py. Recovered by
# matching up to the first digit-enumerated clause instead.
HEAD_FALLBACK_RE = re.compile(r"Tên điều hoặc mục:\s*([^.]+?)\.\s+\d+\.")


def extract_dense_text(text: str) -> str:
    """Heading + body only — see module docstring."""
    m = HEAD_RE.search(text)
    if m:
        heading = m.group(1).strip()
    else:
        m2 = HEAD_FALLBACK_RE.search(text)
        heading = m2.group(1).strip() if m2 else ""
    body = text.split("Nội dung:", 1)[-1].strip()
    return f"{heading}. {body}".strip()


class DenseRetriever:
    """BGE-M3 dense retrieval, hard-filtered/doc_boost-scored the same way
    `BM25Retriever` is (`agent/retrieval/filters.py`) so the two are
    interchangeable behind the `Retriever` protocol."""

    def __init__(
        self,
        corpus: Corpus,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        use_fp16: bool = True,
    ):
        from FlagEmbedding import BGEM3FlagModel  # heavy import, deferred to call time

        self.corpus = corpus
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        self._chunk_ids, self._embeddings = self._load_or_encode_corpus(corpus, Path(cache_dir))
        self._id_to_row = {cid: i for i, cid in enumerate(self._chunk_ids)}

    def _cache_paths(self, cache_dir: Path, corpus_version: str) -> tuple[Path, Path]:
        key = hashlib.sha1(f"{corpus_version}:{DEFAULT_MODEL_NAME}".encode()).hexdigest()[:16]
        return cache_dir / f"{key}.npy", cache_dir / f"{key}.ids.json"

    def _load_or_encode_corpus(self, corpus: Corpus, cache_dir: Path) -> tuple[list[str], np.ndarray]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path, ids_path = self._cache_paths(cache_dir, corpus.version)
        if emb_path.exists() and ids_path.exists():
            chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            embeddings = np.load(emb_path)
            if len(chunk_ids) == len(corpus) and embeddings.shape[0] == len(chunk_ids):
                return chunk_ids, embeddings
        # Cache miss (new corpus version, or a stale/partial cache) — encode fresh.
        chunk_ids = [c.chunk_id for c in corpus.chunks]
        texts = [extract_dense_text(c.text) for c in corpus.chunks]
        embeddings = np.asarray(self._model.encode(texts, batch_size=32, max_length=512)["dense_vecs"])
        np.save(emb_path, embeddings)
        ids_path.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")
        return chunk_ids, embeddings

    def retrieve(
        self,
        query: RetrievalQuery,
        corpus: Corpus,
        k: int,
        as_of: str | None = None,
    ) -> list[RetrievedChunk]:
        if corpus is not self.corpus and corpus.version != self.corpus.version:
            raise ValueError(
                "DenseRetriever was built for a different corpus version — construct a new "
                "DenseRetriever(corpus) for a different/updated corpus rather than reusing this one."
            )
        candidates = apply_hard_filters(corpus.chunks, query, as_of)
        if not candidates:
            return []

        q_emb = np.asarray(self._model.encode([query.text], max_length=512)["dense_vecs"])[0]
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        scored: list[tuple[float, str, CorpusChunk]] = []
        for chunk in candidates:
            row = self._id_to_row.get(chunk.chunk_id)
            if row is None:
                continue  # chunk in corpus but not in the embedding cache (shouldn't happen; fail soft)
            vec = self._embeddings[row]
            cos_sim = float(np.dot(vec, q_norm) / (np.linalg.norm(vec) + 1e-9))
            score = cos_sim + doc_boost_score(chunk, query.doc_boost)
            scored.append((score, chunk.chunk_id, chunk))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            RetrievedChunk(chunk_id=chunk.chunk_id, citation=chunk.title, content=chunk.text, score=score)
            for score, _, chunk in scored[:k]
        ]
