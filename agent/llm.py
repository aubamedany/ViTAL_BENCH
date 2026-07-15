"""LLM client interface, the I1 option-blindness guard, and MockLLM.

`LLMClient` is the swappable seam for the real backend (e.g. adapting
answer_vqa.py's VLM class to this Protocol). `MockLLM` is a deterministic,
rule-based stand-in used to exercise Algorithm 1 & 2 (AGENT_PLAN.md §5)
end-to-end against real law.json/law_db.json rows without any model download.
It does NOT implement real perception or legal reasoning — see its docstring.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Protocol

from agent.schemas import LegalDetermination, LegalEntity

# --------------------------------------------------------------------------- #
# I1 — option-blindness guard (AGENT_PLAN.md §4, "do this literally")
# --------------------------------------------------------------------------- #

_assertion_count = 0


class OptionLeakageError(RuntimeError):
    pass


def get_assertion_count() -> int:
    return _assertion_count


def reset_assertion_count() -> None:
    global _assertion_count
    _assertion_count = 0


def _fold(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def assert_no_option_leakage(prompt: str, options: dict[str, str] | None) -> None:
    """Raise if any option's text (normalized) appears in a Phase-1 prompt.

    Runs before every Phase-1 LLM call; the count is logged for the paper's
    reproducibility appendix (§4). Phase-1 functions never receive `options`
    in production — this is a runtime trip-wire, not the only line of
    defense. Skips options shorter than 3 folded characters to avoid
    false positives on trivial substrings.
    """
    global _assertion_count
    _assertion_count += 1
    if not options:
        return
    folded_prompt = _fold(prompt)
    for letter, text in options.items():
        folded_option = _fold(text.strip())
        if len(folded_option) >= 3 and folded_option in folded_prompt:
            raise OptionLeakageError(
                f"option {letter} text leaked into a Phase-1 prompt: {text!r}"
            )


# --------------------------------------------------------------------------- #
# LLMClient protocol
# --------------------------------------------------------------------------- #


class LLMClient(Protocol):
    def enumerate_entities(self, image_path: Path, question: str) -> dict: ...

    def inspect_entity(self, entity_kind: str, entity_description: str) -> dict: ...

    def formulate_queries(self, entities: list[LegalEntity], question: str) -> dict: ...

    def refine_queries(
        self,
        entities: list[LegalEntity],
        question: str,
        unfilled_slots: list[str],
        found_citations: list[str],
        tried_queries: list[str],
    ) -> dict: ...

    def assess(
        self,
        question: str,
        observed_behavior: str,
        required_slots: list[str],
        retrieved: list,
    ) -> dict: ...

    def emit_determination(
        self, question: str, scene, retrieved: list, iterations_used: int
    ) -> dict: ...

    def select(self, determination: LegalDetermination, options: dict[str, str]) -> dict: ...


# --------------------------------------------------------------------------- #
# MockLLM — deterministic wiring stub, NOT a real reasoning backend
# --------------------------------------------------------------------------- #

_PENALTY_RE = re.compile(r"(\d[\d\.]*)\s*(?:đồng|000 đồng)?\s*đến\s*(\d[\d\.]*)\s*đồng")
_SUPPLEMENTARY_RE = re.compile(
    r"(trừ\s+\d+\s*điểm[^.;\n]*|tước quyền sử dụng giấy phép lái xe[^.;\n]*)",
    re.IGNORECASE,
)

# image path's parent-folder name (e.g. "/data/khong_mu_bao_hiem/1.jpg" ->
# "khong_mu_bao_hiem") -> an ORACLE entity hint. This is a labeled shortcut:
# there is no real VLM wired in yet, so MockLLM reads the category folder name
# instead of the pixels. Never report this as PERCEIVE accuracy — that
# requires a real VLM scored against the detection tier (AGENT_PLAN.md M2.5).
_TYPE_IMAGE_HINTS: dict[str, dict] = {
    "cong_kenh": {
        "kind": "xe chở hàng cồng kềnh",
        "state": "hàng hóa xếp trên xe vượt quá giới hạn cho phép",
        "legally_relevant_because": "xếp hàng hóa vượt quá kích thước quy định là hành vi bị xử phạt",
    },
    "khong_mu_bao_hiem": {
        "kind": "người điều khiển xe mô tô",
        "state": "không đội mũ bảo hiểm",
        "legally_relevant_because": "không đội mũ bảo hiểm khi điều khiển xe mô tô là hành vi bị xử phạt",
    },
    "khong_tuan_thu_den_tin_hieu": {
        "kind": "đèn tín hiệu giao thông",
        "state": "đèn đỏ, phương tiện vẫn di chuyển",
        "legally_relevant_because": "không chấp hành hiệu lệnh của đèn tín hiệu là hành vi bị xử phạt",
    },
    "nguoc_chieu": {
        "kind": "phương tiện",
        "state": "đi ngược chiều của đường một chiều",
        "legally_relevant_because": "đi ngược chiều đường là hành vi bị xử phạt",
    },
    "sai_lan": {
        "kind": "phương tiện",
        "state": "đi không đúng phần đường hoặc làn đường quy định",
        "legally_relevant_because": "đi sai làn đường là hành vi bị xử phạt",
    },
    "dung_qua_vach": {
        "kind": "phương tiện",
        "state": "dừng đè lên vạch dừng",
        "legally_relevant_because": "không chấp hành vạch kẻ đường là hành vi bị xử phạt",
    },
    "sai_bien_bao": {
        "kind": "phương tiện",
        "state": "không chấp hành hiệu lệnh của biển báo",
        "legally_relevant_because": "không chấp hành hiệu lệnh biển báo là hành vi bị xử phạt",
    },
    "di_vao_duong_cam": {
        "kind": "phương tiện",
        "state": "đi vào đường có biển cấm",
        "legally_relevant_because": "đi vào đường cấm là hành vi bị xử phạt",
    },
}


def _tokenize(text: str) -> set[str]:
    folded = _fold(text)
    return set(re.findall(r"[a-z0-9]+", folded))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class MockLLM:
    """Deterministic, rule-based stand-in for the real VLM/LLM backends
    (AGENT_PLAN.md §2.1/§2.2). Exists purely so agent/pipeline.py can be
    built and smoke-tested end-to-end on real data before any model is
    wired in. Swap for a real backend behind this same Protocol; nothing
    in agent/pipeline.py changes.
    """

    def enumerate_entities(self, image_path: Path, question: str) -> dict:
        type_image = Path(image_path).parent.name
        hint = _TYPE_IMAGE_HINTS.get(type_image)
        if hint is None:
            return {"entities": [], "observed_behavior": "", "scene_notes": None}
        entity = {
            "kind": hint["kind"],
            "description": hint["state"],
            "state": hint["state"],
            "confidence": 0.6,
            "legally_relevant_because": hint["legally_relevant_because"],
        }
        return {
            "entities": [entity],
            "observed_behavior": hint["state"],
            "scene_notes": None,
        }

    def inspect_entity(self, entity_kind: str, entity_description: str) -> dict:
        return {"state": entity_description, "confidence": 0.6}

    def formulate_queries(self, entities: list[LegalEntity], question: str) -> dict:
        if not entities:
            return {"queries": []}
        e = entities[0]
        vehicle = e.kind
        return {
            "queries": [
                {
                    "text": f"người điều khiển {vehicle} {e.state or e.description}",
                    "type": "conduct",
                    "intent": "behavior_identification",
                    "vehicle_mentioned": vehicle,
                },
                {
                    "text": f"phạt tiền đối với người điều khiển {vehicle} {e.state or e.description}",
                    "type": "penalty",
                    "intent": "penalty_amount",
                    "vehicle_mentioned": vehicle,
                },
            ]
        }

    def refine_queries(
        self,
        entities: list[LegalEntity],
        question: str,
        unfilled_slots: list[str],
        found_citations: list[str],
        tried_queries: list[str],
    ) -> dict:
        if not entities or not unfilled_slots:
            return {"queries": []}
        e = entities[0]
        vehicle = e.kind
        queries = []
        if "supplementary" in unfilled_slots:
            queries.append(
                {
                    "text": f"trừ điểm giấy phép lái xe người điều khiển {vehicle} {e.state or e.description}",
                    "type": "penalty",
                    "intent": "supplementary_consequence",
                    "vehicle_mentioned": vehicle,
                }
            )
        if "required_behavior" in unfilled_slots or "violated_behavior" in unfilled_slots:
            queries.append(
                {
                    "text": f"{vehicle} không được {e.state or e.description}",
                    "type": "rule",
                    "intent": "required_behavior",
                    "vehicle_mentioned": vehicle,
                }
            )
        return {"queries": [q for q in queries if q["text"] not in tried_queries]}

    def assess(
        self,
        question: str,
        observed_behavior: str,
        required_slots: list[str],
        retrieved: list,
    ) -> dict:
        slot_evidence = []
        for slot in required_slots:
            best_chunk, best_score = None, 0.0
            for chunk in retrieved:
                if slot == "penalty" and "đồng" not in chunk.content:
                    continue
                if slot == "supplementary" and not _SUPPLEMENTARY_RE.search(chunk.content):
                    continue
                score = _overlap(observed_behavior, chunk.content)
                if score > best_score:
                    best_chunk, best_score = chunk, score
            if best_chunk is None and retrieved and slot in ("violated_behavior", "required_behavior"):
                # any retrieved chunk with nonzero scene overlap can support these slots
                for chunk in retrieved:
                    score = _overlap(observed_behavior, chunk.content)
                    if score > best_score:
                        best_chunk, best_score = chunk, score
            slot_evidence.append(
                {
                    "slot": slot,
                    "filled": best_chunk is not None,
                    "supporting_chunk_id": best_chunk.chunk_id if best_chunk else None,
                    "justification": (best_chunk.content[:120] if best_chunk else ""),
                }
            )
        return {"slot_evidence": slot_evidence}

    def emit_determination(
        self, question: str, scene, retrieved: list, iterations_used: int
    ) -> dict:
        cited_ids = [c.chunk_id for c in retrieved]
        penalty_min = penalty_max = None
        supplementary: list[str] = []
        for chunk in retrieved:
            m = _PENALTY_RE.search(chunk.content)
            if m and penalty_min is None:
                penalty_min = int(m.group(1).replace(".", ""))
                penalty_max = int(m.group(2).replace(".", ""))
            for sm in _SUPPLEMENTARY_RE.finditer(chunk.content):
                supplementary.append(sm.group(1).strip())

        violated_behavior = scene.observed_behavior or None
        insufficient = not cited_ids
        return {
            "violated_behavior": violated_behavior,
            "applicable_provisions": cited_ids,
            "penalty_min_vnd": penalty_min,
            "penalty_max_vnd": penalty_max,
            "supplementary_consequences": supplementary[:3],
            "reasoning": (
                f"Hành vi quan sát: {violated_behavior or '(không xác định)'}. "
                f"Điều khoản liên quan: {', '.join(cited_ids) or '(không có)'}."
            ),
            "confidence": 0.75 if cited_ids and not insufficient else 0.3,
            "insufficient_evidence": insufficient,
            "iterations_used": iterations_used,
        }

    def select(self, determination: LegalDetermination, options: dict[str, str]) -> dict:
        basis = determination.reasoning + " " + (determination.violated_behavior or "")
        scores = {L: _overlap(basis, text) for L, text in options.items()}
        # tie-break deterministically on option letter (A before D)
        best_letter = max(scores, key=lambda L: (scores[L], -ord(L)))
        best_score = scores[best_letter]
        ranked = sorted(scores.values(), reverse=True)
        if best_score == 0.0:
            # no lexical signal at all -> fall back to the first option, poor quality
            best_letter = next(iter(options))
            quality = "poor"
        elif len(ranked) > 1 and ranked[0] == ranked[1]:
            quality = "partial"
        else:
            quality = "exact" if best_score > 0.2 else "partial"
        return {
            "answer": best_letter,
            "match_quality": quality,
            "rationale": f"Lựa chọn khớp nhất với kết luận (overlap={best_score:.2f}).",
        }
