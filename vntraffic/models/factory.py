"""Backend factory: build a VLMClient from a registry key + backend name."""

from __future__ import annotations

from typing import Optional

from .base import VLMClient
from .mock import MockClient
from .registry import get_spec


def build_client(model_key: str, backend: str = "hf",
                 device: str = "cuda", dtype: str = "bfloat16",
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 served_model_id: Optional[str] = None,
                 max_tiles: int = 12) -> VLMClient:
    """Construct a client.

    backend:
      "hf"     local Transformers (default) — uses the registry HF checkpoint;
      "openai" OpenAI-compatible server (vLLM) — ``served_model_id`` overrides
               the model name sent to the server (defaults to the HF id);
      "mock"   deterministic dependency-free stub.
    """
    if backend == "mock":
        return MockClient(model_key=model_key, model_id=f"mock/{model_key}")

    if backend == "hf":
        from .hf import build_hf_client
        return build_hf_client(model_key, device=device, dtype=dtype, max_tiles=max_tiles)

    if backend == "openai":
        from .openai_api import OpenAIClient
        spec = get_spec(model_key)
        return OpenAIClient(model_id=served_model_id or spec.hf_id, model_key=model_key,
                            base_url=base_url, api_key=api_key)

    raise ValueError(f"unknown backend '{backend}' (expected hf|openai|mock)")
