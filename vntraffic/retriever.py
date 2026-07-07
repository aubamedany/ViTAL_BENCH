"""FAISS retriever with a SHA-keyed retrieval cache (EXPERIMENT_PLAN.md §6).

* Index: FAISS ``IndexFlatIP`` over L2-normalized embeddings (exact cosine
  search — the corpus is small, no ANN needed).
* Retriever is **model-independent**: retrieval depends only on the query
  string, so the cache is shared across the whole model ladder.
* Cache key: ``sha256(query ‖ corpus_version ‖ k ‖ embedder_id)``. A hit
  returns chunk_ids + scores without touching the embedder, so E4 re-runs never
  re-embed. The cache is a plain JSON file, committable alongside results.

faiss / numpy are imported lazily inside the methods that need them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import constants as C
from .corpus import Corpus
from .embedder import Embedder


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "score": round(float(self.score), 6),
                "rank": self.rank}


def _cache_key(query: str, corpus_version: str, k: int, embedder_id: str) -> str:
    payload = "\x1f".join([query.strip(), corpus_version, str(k), embedder_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RetrievalCache:
    """In-memory dict backed by a single JSON file."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._data: Dict[str, List[dict]] = {}
        self._dirty = False
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, key: str) -> Optional[List[RetrievalHit]]:
        raw = self._data.get(key)
        if raw is None:
            return None
        return [RetrievalHit(h["chunk_id"], h["score"], h["rank"]) for h in raw]

    def put(self, key: str, hits: Sequence[RetrievalHit]) -> None:
        self._data[key] = [h.to_dict() for h in hits]
        self._dirty = True

    def flush(self) -> None:
        if self.path and self._dirty:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = False


class Retriever:
    """FAISS retriever over the statute corpus."""

    def __init__(self, corpus: Corpus, embedder: Embedder,
                 index=None, cache_path: Optional[str] = None):
        self.corpus = corpus
        self.embedder = embedder
        self.index = index
        self.chunk_ids: List[str] = corpus.chunk_ids
        self.cache = RetrievalCache(cache_path)

    # ---- build / persist ------------------------------------------------- #
    @classmethod
    def build(cls, corpus: Corpus, embedder: Embedder,
              cache_path: Optional[str] = None) -> "Retriever":
        import faiss  # lazy
        texts = [c.text_for_embedding for c in corpus.chunks]
        emb = embedder.encode_passages(texts).astype("float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(corpus, embedder, index=index, cache_path=cache_path)

    def save(self, directory: str) -> None:
        import faiss  # lazy
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        meta = {
            "chunk_ids": self.chunk_ids,
            "corpus_version": self.corpus.version,
            "embedder_id": self.embedder.id,
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str, corpus: Corpus, embedder: Embedder,
             cache_path: Optional[str] = None) -> "Retriever":
        import faiss  # lazy
        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("corpus_version") != corpus.version:
            raise ValueError(
                f"index corpus_version {meta.get('corpus_version')} != "
                f"corpus {corpus.version}; rebuild the index")
        r = cls(corpus, embedder, index=index, cache_path=cache_path)
        r.chunk_ids = meta["chunk_ids"]
        return r

    # ---- search ---------------------------------------------------------- #
    def search(self, query: str, k: int = C.DEFAULT_TOP_K) -> List[RetrievalHit]:
        key = _cache_key(query, self.corpus.version, k, self.embedder.id)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vec = self.embedder.encode_query(query).astype("float32").reshape(1, -1)
        scores, idxs = self.index.search(vec, k)
        hits = [RetrievalHit(self.chunk_ids[i], float(s), rank)
                for rank, (i, s) in enumerate(zip(idxs[0], scores[0])) if i != -1]
        self.cache.put(key, hits)
        return hits

    def flush_cache(self) -> None:
        self.cache.flush()


# --------------------------------------------------------------------------- #
# Lexical retriever — dependency-free TF-IDF cosine (no numpy/faiss/torch)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalRetriever:
    """Pure-Python TF-IDF cosine retriever over ``text_for_embedding``.

    Shares the ``search`` / ``flush_cache`` / ``corpus`` / ``chunk_ids`` surface
    with :class:`Retriever`, so conditions and the agent are agnostic to which is
    used. It needs no heavy dependencies, which makes it a zero-setup smoke-test
    backend and a lexical sanity baseline (select via ``retriever: lexical`` in a
    RunConfig). ``embedder_id`` in the cache key is the fixed string ``lexical``.
    """

    embedder_id = "lexical"

    def __init__(self, corpus: Corpus, cache_path: Optional[str] = None):
        self.corpus = corpus
        self.chunk_ids: List[str] = corpus.chunk_ids
        self.cache = RetrievalCache(cache_path)
        # build tf-idf
        docs = [_tokenize(c.text_for_embedding) for c in corpus.chunks]
        df: Counter = Counter()
        for toks in docs:
            df.update(set(toks))
        n = max(1, len(docs))
        self._idf: Dict[str, float] = {
            t: math.log((n + 1) / (dfi + 1)) + 1.0 for t, dfi in df.items()
        }
        self._vecs: List[Dict[str, float]] = [self._vectorize(toks) for toks in docs]

    @property
    def embedder(self):  # for parity with Retriever.info logging
        class _E:  # noqa: D401 - tiny shim
            id = "lexical"
        return _E()

    def _vectorize(self, toks: Sequence[str]) -> Dict[str, float]:
        tf = Counter(toks)
        vec = {t: (c / len(toks)) * self._idf.get(t, 0.0) for t, c in tf.items() if toks}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _score(self, qv: Dict[str, float], dv: Dict[str, float]) -> float:
        # iterate over the smaller vector
        if len(qv) > len(dv):
            qv, dv = dv, qv
        return sum(v * dv.get(t, 0.0) for t, v in qv.items())

    def search(self, query: str, k: int = C.DEFAULT_TOP_K) -> List[RetrievalHit]:
        key = _cache_key(query, self.corpus.version, k, self.embedder_id)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        qv = self._vectorize(_tokenize(query))
        scored = sorted(
            ((self._score(qv, dv), i) for i, dv in enumerate(self._vecs)),
            key=lambda x: x[0], reverse=True)[:k]
        hits = [RetrievalHit(self.chunk_ids[i], float(s), rank)
                for rank, (s, i) in enumerate(scored)]
        self.cache.put(key, hits)
        return hits

    def flush_cache(self) -> None:
        self.cache.flush()
