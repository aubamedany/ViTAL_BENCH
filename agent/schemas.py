"""Data contracts for VNTraffic-Agent (AGENT_PLAN.md §4).

Enforced with pydantic: a schema violation is a bug, not a warning.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Slot = Literal["violated_behavior", "penalty", "supplementary", "required_behavior"]
Intent = Literal[
    "behavior_identification",
    "required_behavior",
    "penalty_amount",
    "supplementary_consequence",
]


class LegalEntity(BaseModel):
    """PERCEIVE-v2 output unit — OPEN VOCABULARY. `kind`/`state` are the model's
    own words; the schema constrains structure, never vocabulary."""

    kind: str
    description: str
    state: str | None = None
    confidence: float
    legally_relevant_because: str


class SceneDescription(BaseModel):
    """PERCEIVE output. Never contains answer options (invariant I1)."""

    entities: list[LegalEntity] = Field(default_factory=list)
    observed_behavior: str
    scene_notes: str | None = None


class RetrievalQuery(BaseModel):
    text: str
    intent: Intent
    vehicle_filter: list[str] = Field(default_factory=list)  # normalize_vehicle() output; empty = fail-open
    doc_boost: list[str] = Field(default_factory=list)  # e.g. ["ND_168_2024"]


class RetrievedChunk(BaseModel):
    chunk_id: str
    citation: str
    content: str
    score: float
    retrieved_at_iteration: int


class SlotEvidence(BaseModel):
    slot: Slot
    filled: bool
    supporting_chunk_id: str | None = None  # checkable post-hoc against chunk text
    justification: str


class SufficiencyReport(BaseModel):
    """ASSESS output — replaces a bare boolean so sufficiency is checkable."""

    required_slots: list[Slot]
    slot_evidence: list[SlotEvidence]
    overall_sufficient: bool  # DERIVED: all required slots filled
    unfilled_slots: list[str]


class LegalDetermination(BaseModel):
    """THE ARTIFACT THAT CROSSES THE OPTION WALL. `applicable_provisions` MUST
    be a subset of the retrieved chunk_ids (invariant I2)."""

    violated_behavior: str | None
    applicable_provisions: list[str] = Field(default_factory=list)
    penalty_min_vnd: int | None = None
    penalty_max_vnd: int | None = None
    supplementary_consequences: list[str] = Field(default_factory=list)
    reasoning: str
    confidence: float
    insufficient_evidence: bool = False
    iterations_used: int = 0


class Selection(BaseModel):
    """Phase 2 output."""

    answer: Literal["A", "B", "C", "D"]
    match_quality: Literal["exact", "partial", "poor"]
    rationale: str
