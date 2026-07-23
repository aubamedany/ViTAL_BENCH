"""VLMLLMClient — real model backend for agent.llm.LLMClient (AGENT_PLAN_v2.md
§2-3.4), adapting answer_vqa.VLM (the same model wrapper run_100.sh/answer_vqa.py
use for qwen3vl/llama/internvl) to the three-method Protocol the agent
pipeline talks to. Nothing in agent/pipeline.py changes to use this instead of
MockLLM — that swappability is the point.

PERCEIVE and SELECT both receive the image (PERCEIVE always, SELECT because
it now re-examines the image alongside the retrieved statute text); ASSESS is
text-only (it only ever sees perception's output, never re-looks at pixels).
SELECT is the only call that ever sees the options.

Real model JSON is unreliable, especially at small scale. Every call goes
through `_chat_json`, which retries once with an explicit schema reminder
before falling back to a safe empty/degenerate structure — the pipeline's own
guards (citation closure) are designed to absorb a degenerate response rather
than crash. `parse_failure_count` is exposed so a run can report the
parse-failure rate per model.
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
from agent.prompts.perceive import PERCEIVE_SYSTEM_PROMPT, build_perceive_user_prompt
from agent.prompts.select import SELECT_SYSTEM_PROMPT, build_select_user_prompt
from agent.schemas import Perception, RetrievedChunk

_VALID_MATCH_QUALITY = {"exact", "partial", "poor"}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_JSON_REMINDER = "\n\nCHỈ TRẢ LỜI ĐÚNG ĐỊNH DẠNG JSON NHƯ TRÊN, không thêm chữ nào khác, không dùng markdown."


# Observed on qwen3vl-2b (logs/agent/e4_agent/qwen3vl-2b.json): the model's
# free-text "reasoning"/"reason" field (first in the schema, generated before
# answer/citation/match_quality) alone runs up to 470 tokens for legitimate
# non-degenerate cases -- the full JSON for that record came to 507/512,
# right at the old ceiling. Any reasoning slightly longer got the response
# truncated before the closing fields, producing invalid JSON that fell back
# to an arbitrary answer pick on 42% of that run's samples. 1024 gives
# roughly 2x headroom over the longest observed legitimate response.
DEFAULT_JSON_MAX_NEW_TOKENS = 1024
# A khoản chunk is self-contained: it carries the penalty range AND every
# điểm sub-item (AGENT_PLAN_v2.md §3.3) — often 8-12 bullets enumerating
# unrelated behaviors under one fine bracket. Truncating too aggressively
# silently breaks that guarantee: at the old default of 800, 38% of the
# corpus's chunks got cut off mid-bullet, discarding whichever điểm happened
# to sit later in the list — including, in one observed case, the exact
# "mang vác vật cồng kềnh" (bulky-cargo) bullet a "cong_kenh" item needed.
# 3500 covers ~95% of chunks (measured on law_db.json) without truncation.
DEFAULT_MAX_CHUNK_CHARS = 3500


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


def _sanitize_perception(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    violation = raw.get("violation")
    if not isinstance(violation, str) or not violation.strip():
        violation = None
    vehicle = raw.get("vehicle")
    if not isinstance(vehicle, str) or not vehicle.strip():
        vehicle = None
    return {
        "focus": str(raw.get("focus", "")),
        "vehicle": vehicle,
        "observed": str(raw.get("observed", "")),
        "violation": violation,
        "confidence": confidence,
    }


def _sanitize_assessment(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    sufficient = bool(raw.get("sufficient", False))
    refined_query = raw.get("refined_query")
    if not isinstance(refined_query, str) or not refined_query.strip():
        refined_query = None
    if sufficient:
        refined_query = None  # ignore a redundant refined_query when already sufficient
    return {"reason": str(raw.get("reason", "")), "sufficient": sufficient, "refined_query": refined_query}


def _sanitize_selection(raw: dict, options: dict[str, str], retrieved_ids: set[str]) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    answer = raw.get("answer")
    if answer not in options:
        return {
            "reasoning": str(raw.get("reasoning", "")),
            "answer": next(iter(options)),
            "citation": None,
            "match_quality": "poor",
        }
    quality = raw.get("match_quality") if raw.get("match_quality") in _VALID_MATCH_QUALITY else "partial"
    citation = raw.get("citation")
    if not isinstance(citation, str) or citation not in retrieved_ids:
        citation = None
    return {
        "reasoning": str(raw.get("reasoning", "")),
        "answer": answer,
        "citation": citation,
        "match_quality": quality,
    }


class VLMLLMClient:
    """Real backend: wraps answer_vqa.VLM (loaded once) and drives it through
    all three LLMClient calls as JSON-only chat turns."""

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
        for _ in range(3):  # 1 initial attempt + 2 schema-reminder retries
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

    def perceive(self, image_path: Path, question: str) -> dict:
        fallback = {"focus": "", "vehicle": None, "observed": "", "violation": None, "confidence": 0.2}
        raw = self._chat_json(PERCEIVE_SYSTEM_PROMPT, build_perceive_user_prompt(question), image_path, fallback)
        return _sanitize_perception(raw)

    def assess(
        self,
        question: str,
        perception: Perception,
        retrieved: list[RetrievedChunk],
        queries_tried: list[str],
    ) -> dict:
        truncated = [c.model_copy(update={"content": c.content[: self.max_chunk_chars]}) for c in retrieved]
        user = build_assess_user_prompt(question, perception, truncated, queries_tried)
        # Fallback favors STOPPING the loop (sufficient=True), not retrying: a
        # model that just failed to produce valid JSON here is unlikely to
        # succeed against a harder prompt (a longer chunk list) on retry —
        # let SELECT's own poor-match-quality path absorb the degraded case.
        fallback = {"reason": "(loi phan tich JSON - dung vong lap)", "sufficient": True, "refined_query": None}
        raw = self._chat_json(ASSESS_SYSTEM_PROMPT, user, None, fallback)  # text-only: Phase 1, no image
        return _sanitize_assessment(raw)

    def select(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        options: dict[str, str],
        image_path: Path,
    ) -> dict:
        truncated = [c.model_copy(update={"content": c.content[: self.max_chunk_chars]}) for c in retrieved]
        user = build_select_user_prompt(question, truncated, options)
        fallback = {
            "reasoning": "(khong the tao ket luan do loi phan tich JSON)",
            "answer": next(iter(options)),
            "citation": None,
            "match_quality": "poor",
        }
        raw = self._chat_json(SELECT_SYSTEM_PROMPT, user, image_path, fallback)
        retrieved_ids = {c.chunk_id for c in retrieved}
        return _sanitize_selection(raw, options, retrieved_ids)
