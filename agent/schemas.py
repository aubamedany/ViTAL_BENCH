"""Data contracts for ViTAL-Agent (AGENT_PLAN_v2.md).

Enforced with pydantic: a schema violation is a bug, not a warning.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Perception(BaseModel):
    """PERCEIVE output — the only step that sees pixels, and it's
    question-directed: look only at what the question names, then predict
    the violation. OPEN VOCABULARY (free text, never an enum) — the schema
    constrains structure, never vocabulary. Never contains answer options."""

    focus: str  # what the question is about, e.g. "người điều khiển xe mô tô, mũ bảo hiểm"
    vehicle: str | None = None  # free text, normalized to a corpus filter term at retrieval time
    observed: str  # what's actually there, scene language
    violation: str | None = None  # statute-register phrasing; None = no violation
    confidence: float


class RetrievalQuery(BaseModel):
    text: str
    vehicle_filter: list[str] = Field(default_factory=list)  # normalize_vehicle() output; empty = fail-open
    doc_boost: list[str] = Field(default_factory=list)  # e.g. ["ND_168_2024"]


class RetrievedChunk(BaseModel):
    chunk_id: str
    citation: str
    content: str
    score: float


class Assessment(BaseModel):
    """ASSESS output (AGENT_PLAN_v2.md §3.4) — merges the sufficiency
    judgment and query refinement into one call per refine-loop iteration.
    `reason` is listed first so a model's autoregressive JSON generation is
    conditioned by a short justification before it commits to `sufficient`.
    Deliberately NOT v1's SufficiencyReport: no slots, no per-chunk
    attribution — a holistic judgment, one call."""

    reason: str = ""
    sufficient: bool
    refined_query: str | None = None  # statute-register text; ignored once sufficient=True


class Selection(BaseModel):
    """Phase 2 output — reasons over the retrieved statute text directly
    (not a lossy pre-synthesized summary)."""

    reasoning: str  # Vietnamese, must reference the statute text
    answer: Literal["A", "B", "C", "D"]
    citation: str | None  # chunk_id — MUST be one of the retrieved chunks, or None if none retrieved
    match_quality: Literal["exact", "partial", "poor"]
