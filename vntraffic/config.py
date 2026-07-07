"""Run configuration (EXPERIMENT_PLAN.md §11: one config file per model × condition).

A RunConfig fully determines a run: which items, which condition, which model +
backend, retrieval settings, decoding, and output location. Configs are YAML
(JSON also accepted); no manual prompt edits happen between runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from . import constants as C

try:
    import yaml
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    _HAVE_YAML = False


@dataclass
class RunConfig:
    # identity
    run_id: str
    condition: str                       # E1|E2|E3|E4
    model_key: str                       # registry key
    backend: str = "hf"                  # hf|openai|mock

    # data selection
    data: str = os.path.join(C.DATASET_V1_DIR, "data.json")
    corpus: str = C.CORPUS_JSON
    corpus_aliases: Optional[str] = None
    tier: str = C.TIER_LAW               # detection|law
    categories: Optional[List[str]] = None
    item_ids: Optional[List[int]] = None  # explicit subset (e.g. pilot sample)

    # retrieval
    retriever: str = "faiss"             # faiss (default) | lexical (dep-free baseline)
    embedder: str = C.DEFAULT_EMBEDDER
    index_dir: Optional[str] = None      # default derived from embedder
    top_k: int = C.DEFAULT_TOP_K
    budget: int = C.DEFAULT_AGENT_BUDGET  # E4 only
    cache_path: Optional[str] = None     # retrieval cache (default derived)

    # decoding / hardware
    max_new_tokens: int = 1024
    seed: int = C.DEFAULT_DECODING_SEED
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_tiles: int = 12                  # InternVL tiling
    # openai backend
    base_url: Optional[str] = None
    served_model_id: Optional[str] = None

    # evaluation options
    circular: bool = False               # CircularEval: 4 cyclic option permutations
    limit: Optional[int] = None          # cap #items (debug/pilot)

    # output
    out_dir: str = C.RUNS_DIR

    # ---- derived ---------------------------------------------------------- #
    def resolved_index_dir(self) -> str:
        return self.index_dir or os.path.join(C.INDEX_DIR, self.embedder.replace("/", "__"))

    def resolved_cache_path(self) -> str:
        if self.cache_path:
            return self.cache_path
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        return os.path.join(C.CACHE_DIR, f"retrieval_{self.embedder.replace('/', '__')}.json")

    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.run_id)

    # ---- io --------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if _HAVE_YAML and path.endswith((".yaml", ".yml")):
                yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
            else:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            if _HAVE_YAML and path.endswith((".yaml", ".yml")):
                d = yaml.safe_load(f)
            else:
                d = json.load(f)
        return cls.from_dict(d)
