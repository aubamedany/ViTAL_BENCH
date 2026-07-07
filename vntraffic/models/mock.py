"""Deterministic mock backend (no torch, no downloads, no GPU).

Used for dry runs that exercise the full runner → JSONL → metrics → tables
pipeline without any model. It is deterministic (seeded from the prompt + model
key) and deliberately prompt-agnostic:

* it returns a single JSON object carrying a *superset* of the fields every
  parser looks for (a bare answer letter, a perception string, a query, a
  sufficiency flag, a determination), so the same mock serves E1-E4 steps;
* for citations it echoes chunk-id-like tokens found **in the prompt** (i.e. the
  retrieved context), so mock determinations stay within the retrieved set and
  the citation-whitelist / grounding paths get exercised.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import List, Optional, Sequence

from .. import constants as C
from .base import GenerationConfig, VLMClient

# chunk-id-like tokens, e.g. LUAT_36_2024_D33_K3, ND_168_2024_D6_K5, QCVN_41_2024_M1
_CHUNK_RE = re.compile(r"\b[A-Z]{2,}_[A-Z0-9]+(?:_[A-Z0-9]+)+\b")


class MockClient(VLMClient):
    backend = "mock"

    def __init__(self, model_key: str = "mock", model_id: str = "mock/deterministic",
                 revision: Optional[str] = None):
        super().__init__(model_key, model_id, revision or "mock-v1")

    def _rng_letter(self, prompt: str) -> str:
        h = hashlib.sha256((self.model_key + "\x1f" + prompt).encode("utf-8")).digest()
        return C.OPTION_LETTERS[h[0] % 4]

    def generate(self, prompt: str, images: Optional[Sequence[str]] = None,
                 system: Optional[str] = None,
                 gen_cfg: Optional[GenerationConfig] = None) -> str:
        letter = self._rng_letter(prompt)
        cited = list(dict.fromkeys(_CHUNK_RE.findall(prompt)))[:2]
        obj = {
            "answer": letter,
            "perception": "[mock] scene description",
            "query": "[mock] legal query",
            "sufficient": True,
            "violated_behavior": "[mock] behavior",
            "applicable_provisions": cited,
            "penalty": "[mock] penalty range",
            "supplementary_consequences": "[mock] none",
            "confidence": 0.5,
        }
        return json.dumps(obj, ensure_ascii=False)
