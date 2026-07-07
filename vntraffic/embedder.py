"""Text embedding wrapper (EXPERIMENT_PLAN.md §6, ablation A4).

Default embedder is a Vietnamese-capable multilingual encoder
(``intfloat/multilingual-e5-base``). The e5 family expects ``query:`` /
``passage:`` prefixes, which this wrapper adds automatically; other models get
no prefix. The embedder id is part of the retrieval cache key so swapping it
(A4) invalidates the cache.

sentence-transformers / torch are imported lazily so the rest of the package
runs without them.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from . import constants as C

# Model ids that use the e5 query:/passage: prefixing convention.
_E5_PREFIX_MODELS = ("e5", "multilingual-e5")


def _uses_e5_prefix(model_id: str) -> bool:
    low = model_id.lower()
    return any(tok in low for tok in _E5_PREFIX_MODELS)


class Embedder:
    """Lazy sentence-transformers embedder with L2-normalized output."""

    def __init__(self, model_id: str = C.DEFAULT_EMBEDDER,
                 device: Optional[str] = None, batch_size: int = 64,
                 use_prefix: Optional[bool] = None):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.use_prefix = _uses_e5_prefix(model_id) if use_prefix is None else use_prefix
        self._model = None  # loaded on first use

    # ---- identity used in cache keys ------------------------------------ #
    @property
    def id(self) -> str:
        return self.model_id

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._ensure().get_sentence_embedding_dimension())

    def _prep(self, texts: Sequence[str], kind: str) -> List[str]:
        if not self.use_prefix:
            return list(texts)
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + t for t in texts]

    def encode_passages(self, texts: Sequence[str]):
        model = self._ensure()
        return model.encode(self._prep(texts, "passage"),
                            batch_size=self.batch_size,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False)

    def encode_queries(self, texts: Sequence[str]):
        model = self._ensure()
        return model.encode(self._prep(texts, "query"),
                            batch_size=self.batch_size,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False)

    def encode_query(self, text: str):
        return self.encode_queries([text])[0]
