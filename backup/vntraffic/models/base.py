"""VLMClient interface and generation config (EXPERIMENT_PLAN.md §4).

Every condition talks to models through the same narrow interface: a single-turn
``generate(prompt, images, system, gen_cfg) -> str``. That is all the E1-E4
prompts need (each step is one image-optional instruction), and keeping it
minimal lets HF, an OpenAI-compatible server, and the mock be swapped freely.

Decoding is greedy (temperature 0) with a fixed seed by default, per the plan's
reproducibility requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .. import constants as C


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    do_sample: bool = False        # greedy
    max_new_tokens: int = 1024
    top_p: float = 1.0
    seed: int = C.DEFAULT_DECODING_SEED

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
        }


class VLMClient:
    """Abstract vision-language client.

    Subclasses implement :meth:`generate`. ``model_key`` is the registry key
    (e.g. ``internvl3-8b``); ``model_id`` is the concrete checkpoint/backend id;
    ``revision`` (optional) is logged for reproducibility.
    """

    backend: str = "abstract"

    def __init__(self, model_key: str, model_id: str, revision: Optional[str] = None):
        self.model_key = model_key
        self.model_id = model_id
        self.revision = revision

    def generate(self, prompt: str, images: Optional[Sequence[str]] = None,
                 system: Optional[str] = None,
                 gen_cfg: Optional[GenerationConfig] = None) -> str:
        """Return the model's text completion for a single-turn prompt.

        images: list of image file paths (may be empty/None for text-only turns,
        e.g. the Phase-2 SELECT step).
        """
        raise NotImplementedError

    # informative metadata for the run manifest
    def info(self) -> dict:
        return {
            "backend": self.backend,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "revision": self.revision,
        }
