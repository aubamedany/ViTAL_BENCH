"""OpenAI-compatible HTTP backend (EXPERIMENT_PLAN.md §4).

Point this at any OpenAI-compatible server — in particular ``vllm serve`` — for
~5-10x faster sweeps than local HF (E4 is tens of thousands of model calls).
Images are inlined as base64 ``data:`` URIs, so no external host is contacted
for image content. Decoding is greedy (temperature 0) with a fixed seed.

The ``openai`` package is imported lazily.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import List, Optional, Sequence

from .base import GenerationConfig, VLMClient


def _data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class OpenAIClient(VLMClient):
    backend = "openai"

    def __init__(self, model_id: str, model_key: Optional[str] = None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 revision: Optional[str] = None, timeout: float = 120.0):
        super().__init__(model_key or model_id, model_id, revision)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout = timeout
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout)
        return self._client

    def generate(self, prompt: str, images: Optional[Sequence[str]] = None,
                 system: Optional[str] = None,
                 gen_cfg: Optional[GenerationConfig] = None) -> str:
        gen_cfg = gen_cfg or GenerationConfig()
        client = self._ensure()
        content: List[dict] = []
        for p in (images or []):
            content.append({"type": "image_url", "image_url": {"url": _data_uri(p)}})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        resp = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            max_tokens=gen_cfg.max_new_tokens,
            seed=gen_cfg.seed,
        )
        return (resp.choices[0].message.content or "").strip()
