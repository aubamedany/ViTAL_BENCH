"""VLMLLMClient — real model backend for agent.llm.LLMClient (AGENT_PLAN.md
§2), adapting answer_vqa.VLM (the same model wrapper run.sh/answer_vqa.py use
for qwen3vl/llama/internvl) to the seven-method Protocol the agent pipeline
talks to. Nothing in agent/pipeline.py changes to use this instead of
MockLLM — that swappability is the point (§1 "training-free" + the module
docstring in agent/llm.py).

Only ENUMERATE ever receives the image (I1: FORMULATE/REFINE/ASSESS/EMIT are
text-only; SELECT is the only place options ever appear, and it never sees
the image or a retriever either, per I3).

Real model JSON is unreliable, especially at small scale (§10: "small models
will fight the schema"). Every call goes through `_chat_json`, which retries
once with an explicit schema reminder (§8 failure-mode table: "Retry x2 with
schema reminder") before falling back to a safe empty/degenerate structure —
the pipeline's own guards (entity confidence gate, citation closure, REFINE's
early-stop) are designed to absorb a degenerate response rather than crash.
`parse_failure_count` is exposed so a run can report the parse-failure rate
per model, itself a finding per §10.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_vqa import PRESETS, VLM  # noqa: E402  (heavy import, deferred to call time by caller)

from agent.prompts.assess import ASSESS_SYSTEM_PROMPT, build_assess_user_prompt
from agent.prompts.emit import EMIT_SYSTEM_PROMPT, build_emit_user_prompt
from agent.prompts.formulate import (
    FORMULATE_FEWSHOT,
    FORMULATE_SYSTEM_PROMPT,
    build_formulate_user_prompt,
    build_refine_suffix,
)
from agent.prompts.perceive import (
    ENUMERATE_SYSTEM_PROMPT,
    INSPECT_SYSTEM_PROMPT,
    build_enumerate_user_prompt,
    build_inspect_user_prompt,
)
from agent.prompts.select import SELECT_SYSTEM_PROMPT, build_select_user_prompt
from agent.schemas import LegalDetermination, LegalEntity, RetrievedChunk, SceneDescription

_VALID_INTENTS = {
    "behavior_identification",
    "required_behavior",
    "penalty_amount",
    "supplementary_consequence",
}
_VALID_SLOTS = {"violated_behavior", "penalty", "supplementary", "required_behavior"}
_VALID_MATCH_QUALITY = {"exact", "partial", "poor"}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_JSON_REMINDER = "\n\nCHỈ TRẢ LỜI ĐÚNG ĐỊNH DẠNG JSON NHƯ TRÊN, không thêm chữ nào khác, không dùng markdown."

DEFAULT_JSON_MAX_NEW_TOKENS = 640
DEFAULT_MAX_CHUNK_CHARS = 800


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    stripped = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Output sanitizers — real model JSON is untrusted; never let a malformed or
# out-of-schema field reach pydantic validation as a crash.
# --------------------------------------------------------------------------- #


def _sanitize_enumerate(raw: dict) -> dict:
    entities = []
    for e in raw.get("entities", []) if isinstance(raw, dict) else []:
        if not isinstance(e, dict) or not e.get("kind"):
            continue
        try:
            confidence = float(e.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        entities.append(
            {
                "kind": str(e["kind"]),
                "description": str(e.get("description", "")),
                "state": e.get("state") if isinstance(e.get("state"), str) else None,
                "confidence": max(0.0, min(1.0, confidence)),
                "legally_relevant_because": str(e.get("legally_relevant_because", "")),
            }
        )
    return {
        "entities": entities,
        "observed_behavior": str(raw.get("observed_behavior", "")) if isinstance(raw, dict) else "",
        "scene_notes": raw.get("scene_notes") if isinstance(raw, dict) and isinstance(raw.get("scene_notes"), str) else None,
    }


def _sanitize_queries(raw: dict) -> dict:
    queries = []
    for q in raw.get("queries", []) if isinstance(raw, dict) else []:
        if not isinstance(q, dict) or not q.get("text"):
            continue
        intent = q.get("intent") if q.get("intent") in _VALID_INTENTS else "behavior_identification"
        queries.append(
            {
                "text": str(q["text"]),
                "type": q.get("type", ""),
                "intent": intent,
                "vehicle_mentioned": q.get("vehicle_mentioned") if isinstance(q.get("vehicle_mentioned"), str) else None,
            }
        )
    return {"queries": queries}


def _sanitize_slot_evidence(raw: dict, required_slots: list[str]) -> dict:
    evidence = []
    for item in raw.get("slot_evidence", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict) or item.get("slot") not in _VALID_SLOTS:
            continue
        evidence.append(
            {
                "slot": item["slot"],
                "filled": bool(item.get("filled")),
                "supporting_chunk_id": item.get("supporting_chunk_id") if isinstance(item.get("supporting_chunk_id"), str) else None,
                "justification": str(item.get("justification", "")),
            }
        )
    return {"slot_evidence": evidence}


def _int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _sanitize_determination(raw: dict, observed_behavior: str) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    provisions = [p for p in raw.get("applicable_provisions", []) if isinstance(p, str)]
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    violated_behavior = raw.get("violated_behavior")
    if not isinstance(violated_behavior, str) or not violated_behavior.strip():
        violated_behavior = observed_behavior or None
    return {
        "violated_behavior": violated_behavior,
        "applicable_provisions": provisions,
        "penalty_min_vnd": _int_or_none(raw.get("penalty_min_vnd")),
        "penalty_max_vnd": _int_or_none(raw.get("penalty_max_vnd")),
        "supplementary_consequences": [s for s in raw.get("supplementary_consequences", []) if isinstance(s, str)],
        "reasoning": str(raw.get("reasoning", "")),
        "confidence": confidence,
        "insufficient_evidence": bool(raw.get("insufficient_evidence", not provisions)),
    }


def _sanitize_selection(raw: dict, options: dict[str, str]) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    answer = raw.get("answer")
    if answer not in options:
        return {"answer": next(iter(options)), "match_quality": "poor", "rationale": str(raw.get("rationale", ""))}
    quality = raw.get("match_quality") if raw.get("match_quality") in _VALID_MATCH_QUALITY else "partial"
    return {"answer": answer, "match_quality": quality, "rationale": str(raw.get("rationale", ""))}


class VLMLLMClient:
    """Real backend: wraps answer_vqa.VLM (loaded once) and drives it through
    all seven LLMClient calls as JSON-only chat turns."""

    def __init__(
        self,
        model_id: str,
        family: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        json_max_new_tokens: int = DEFAULT_JSON_MAX_NEW_TOKENS,
        seed: int = 0,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ):
        self._vlm = VLM(model_id, family, device, dtype, max_new_tokens=json_max_new_tokens)
        self._vlm.load()
        self.json_max_new_tokens = json_max_new_tokens
        self.seed = seed
        self.max_chunk_chars = max_chunk_chars
        self.parse_failure_count = 0

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "VLMLLMClient":
        preset = PRESETS[name]
        return cls(preset.hf_id, preset.family, **kwargs)

    # ----------------------------------------------------------------- #
    # Generation core
    # ----------------------------------------------------------------- #

    def _chat(self, system: str, user: str, image_path: Path | None) -> str:
        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        content = []
        if image_path is not None:
            content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]
        inputs = self._vlm.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        inputs = inputs.to(self._vlm.model.device)
        with torch.inference_mode():
            out = self._vlm.model.generate(
                **inputs, max_new_tokens=self.json_max_new_tokens, do_sample=False,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._vlm.processor.decode(gen, skip_special_tokens=True).strip()

    def _chat_json(self, system: str, user: str, image_path: Path | None, fallback: dict) -> dict:
        prompt = user
        for _ in range(3):  # 1 initial attempt + 2 schema-reminder retries (§8)
            raw_text = self._chat(system, prompt, image_path)
            parsed = _extract_json(raw_text)
            if parsed is not None:
                return parsed
            prompt = user + _JSON_REMINDER
        self.parse_failure_count += 1
        return fallback

    # ----------------------------------------------------------------- #
    # LLMClient Protocol
    # ----------------------------------------------------------------- #

    def enumerate_entities(self, image_path: Path, question: str) -> dict:
        fallback = {"entities": [], "observed_behavior": "", "scene_notes": None}
        raw = self._chat_json(ENUMERATE_SYSTEM_PROMPT, build_enumerate_user_prompt(question), image_path, fallback)
        return _sanitize_enumerate(raw)

    def inspect_entity(self, entity_kind: str, entity_description: str) -> dict:
        # NOTE: the LLMClient Protocol doesn't thread image_path through this
        # call, and agent/pipeline.py's perceive() doesn't call it today
        # (single-pass ENUMERATE only, see AGENT_PLAN.md §2.1's two-step
        # design vs. the current one-step implementation). Implemented for
        # protocol completeness / a future two-step PERCEIVE; text-only.
        fallback = {"state": entity_description, "confidence": 0.3}
        raw = self._chat_json(INSPECT_SYSTEM_PROMPT, build_inspect_user_prompt(entity_kind, entity_description), None, fallback)
        if not isinstance(raw, dict) or "state" not in raw:
            return fallback
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.3))))
        except (TypeError, ValueError):
            confidence = 0.3
        return {"state": str(raw.get("state", entity_description)), "confidence": confidence}

    def formulate_queries(self, entities: list[LegalEntity], question: str) -> dict:
        system = FORMULATE_SYSTEM_PROMPT + "\n" + FORMULATE_FEWSHOT
        fallback = {"queries": []}
        raw = self._chat_json(system, build_formulate_user_prompt(entities, question), None, fallback)
        return _sanitize_queries(raw)

    def refine_queries(
        self,
        entities: list[LegalEntity],
        question: str,
        unfilled_slots: list[str],
        found_citations: list[str],
        tried_queries: list[str],
    ) -> dict:
        system = FORMULATE_SYSTEM_PROMPT + "\n" + FORMULATE_FEWSHOT
        user = build_formulate_user_prompt(entities, question) + "\n\n" + build_refine_suffix(
            unfilled_slots, found_citations, tried_queries
        )
        fallback = {"queries": []}
        raw = self._chat_json(system, user, None, fallback)
        return _sanitize_queries(raw)

    def assess(
        self,
        question: str,
        observed_behavior: str,
        required_slots: list[str],
        retrieved: list[RetrievedChunk],
    ) -> dict:
        truncated = [c.model_copy(update={"content": c.content[: self.max_chunk_chars]}) for c in retrieved]
        user = build_assess_user_prompt(question, observed_behavior, required_slots, truncated)
        fallback = {
            "slot_evidence": [
                {"slot": s, "filled": False, "supporting_chunk_id": None, "justification": ""} for s in required_slots
            ]
        }
        raw = self._chat_json(ASSESS_SYSTEM_PROMPT, user, None, fallback)
        return _sanitize_slot_evidence(raw, required_slots)

    def emit_determination(
        self, question: str, scene: SceneDescription, retrieved: list[RetrievedChunk], iterations_used: int
    ) -> dict:
        truncated = [c.model_copy(update={"content": c.content[: self.max_chunk_chars]}) for c in retrieved]
        user = build_emit_user_prompt(question, scene, truncated)
        fallback = {
            "violated_behavior": scene.observed_behavior or None,
            "applicable_provisions": [],
            "penalty_min_vnd": None,
            "penalty_max_vnd": None,
            "supplementary_consequences": [],
            "reasoning": "(khong the tao ket luan do loi phan tich JSON)",
            "confidence": 0.2,
            "insufficient_evidence": True,
        }
        raw = self._chat_json(EMIT_SYSTEM_PROMPT, user, None, fallback)
        result = _sanitize_determination(raw, scene.observed_behavior)
        result["iterations_used"] = iterations_used
        return result

    def select(self, determination: LegalDetermination, options: dict[str, str]) -> dict:
        fallback = {"answer": next(iter(options)), "match_quality": "poor", "rationale": "(loi phan tich JSON)"}
        raw = self._chat_json(SELECT_SYSTEM_PROMPT, build_select_user_prompt(determination, options), None, fallback)
        return _sanitize_selection(raw, options)
