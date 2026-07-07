"""Model ladder registry (EXPERIMENT_PLAN.md §4, hard cap ≤14B).

Maps short registry keys to concrete HuggingFace checkpoints, the adapter
family that knows how to prompt them, and default backend. All entries are
image-capable and ≤14B parameters (no model above 14B; text-only checkpoints
excluded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

FAMILY_INTERNVL = "internvl"
FAMILY_QWEN_VL = "qwen_vl"
FAMILY_MLLAMA = "mllama"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    params_b: float          # approximate parameter count in billions (≤14)
    is_primary_ladder: bool  # InternVL scaling family (E1, E4 at all sizes)
    leakage_subset: bool     # runs full E1-E4 (leakage curve, §4)


# EXPERIMENT_PLAN.md §4 model ladder.
_SPECS: List[ModelSpec] = [
    # Primary scaling family — InternVL 1/2/8/14B (E1, E4 at all sizes)
    ModelSpec("internvl3-1b",  "OpenGVLab/InternVL3-1B",  FAMILY_INTERNVL, 1.0,  True,  False),
    ModelSpec("internvl3-2b",  "OpenGVLab/InternVL3-2B",  FAMILY_INTERNVL, 2.0,  True,  False),
    ModelSpec("internvl3-8b",  "OpenGVLab/InternVL3-8B",  FAMILY_INTERNVL, 8.0,  True,  True),
    ModelSpec("internvl3-14b", "OpenGVLab/InternVL3-14B", FAMILY_INTERNVL, 14.0, True,  False),
    # Cross-family checks (E1, E4); Qwen2.5-VL-7B is also on the leakage curve
    ModelSpec("qwen2.5-vl-3b", "Qwen/Qwen2.5-VL-3B-Instruct", FAMILY_QWEN_VL, 3.0, False, False),
    ModelSpec("qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct", FAMILY_QWEN_VL, 7.0, False, True),
    ModelSpec("llama-3.2-11b-vision", "meta-llama/Llama-3.2-11B-Vision-Instruct",
              FAMILY_MLLAMA, 11.0, False, False),
]

REGISTRY: Dict[str, ModelSpec] = {s.key: s for s in _SPECS}

# Default "best model" for ablations (§8), overridable if the pilot disagrees.
DEFAULT_BEST_MODEL = "internvl3-14b"


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"unknown model key '{key}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def ladder_keys() -> List[str]:
    """All 7 checkpoints, smallest-first (run cheap models first, §11)."""
    return [s.key for s in sorted(_SPECS, key=lambda x: x.params_b)]


def leakage_subset_keys() -> List[str]:
    """Models that run the full E1-E4 leakage curve (§4)."""
    return [s.key for s in _SPECS if s.leakage_subset]
