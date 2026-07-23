"""Vision-language model backends (EXPERIMENT_PLAN.md §4).

A single ``VLMClient`` interface with interchangeable backends:
  * ``hf``      — HuggingFace Transformers (local, default; per-family adapters)
  * ``openai``  — OpenAI-compatible HTTP API (point at a vLLM server for speed)
  * ``mock``    — deterministic, dependency-free (dry runs, plumbing tests)

Use ``build_client(model_key, backend=...)`` to construct one from the registry.
"""

from .base import GenerationConfig, VLMClient  # noqa: F401
from .factory import build_client  # noqa: F401
