"""LLM client interface, the I1 option-blindness guard, and MockLLM.

`LLMClient` is the swappable seam for the real backend (agent/llm_vlm.py
adapts answer_vqa.py's VLM class to it). `MockLLM` is a deterministic,
rule-based stand-in used to exercise the pipeline (AGENT_PLAN_v2.md §2)
end-to-end against real law.json/law_db.json rows without any model download.
It does NOT implement real perception or legal reasoning — see its docstring.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Protocol

from agent.schemas import Perception, RetrievedChunk

# --------------------------------------------------------------------------- #
# I1 — option-blindness guard: the retrieval query is built from the image
# and the question only, never from the options (AGENT_PLAN_v2.md §1).
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

    Runs before every Phase-1 LLM call; the count is logged for reproducibility.
    Phase-1 functions never receive `options` in production — this is a
    runtime trip-wire, not the only line of defense. Skips options shorter
    than 3 folded characters to avoid false positives on trivial substrings.
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
# LLMClient protocol — PERCEIVE (image, once), ASSESS (text-only, up to
# max_iterations-1 times, the refine loop), SELECT (image + statute text,
# once). FORMULATE is string assembly and RETRIEVE has no LLM
# (agent/pipeline.py).
# --------------------------------------------------------------------------- #


class LLMClient(Protocol):
    def perceive(self, image_path: Path, question: str) -> dict: ...

    def assess(
        self,
        question: str,
        perception: Perception,
        retrieved: list[RetrievedChunk],
        queries_tried: list[str],
    ) -> dict: ...

    def select(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        options: dict[str, str],
        image_path: Path,
    ) -> dict: ...


# --------------------------------------------------------------------------- #
# MockLLM — deterministic wiring stub, NOT a real reasoning backend
# --------------------------------------------------------------------------- #

# image path's parent-folder name (e.g. "/data/khong_mu_bao_hiem/1.jpg" ->
# "khong_mu_bao_hiem") -> an ORACLE hint. This is a labeled shortcut: there is
# no real VLM wired in yet, so MockLLM reads the category folder name instead
# of the pixels. Never report this as PERCEIVE accuracy.
_TYPE_IMAGE_HINTS: dict[str, dict] = {
    "cong_kenh": {
        "focus": "xe chở hàng cồng kềnh",
        "vehicle": "xe mô tô",
        "observed": "hàng hóa xếp trên xe vượt quá giới hạn cho phép",
        "violation": "xếp hàng hóa trên xe vượt quá chiều rộng giá đèo hàng, vượt quá chiều cao quy định",
    },
    "khong_mu_bao_hiem": {
        "focus": "người điều khiển xe mô tô",
        "vehicle": "xe mô tô",
        "observed": "không đội mũ bảo hiểm",
        "violation": "không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng quy cách khi điều khiển xe tham gia giao thông",
    },
    "khong_tuan_thu_den_tin_hieu": {
        "focus": "đèn tín hiệu giao thông",
        "vehicle": "xe mô tô",
        "observed": "đèn đỏ, phương tiện vẫn di chuyển",
        "violation": "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    },
    "nguoc_chieu": {
        "focus": "phương tiện",
        "vehicle": "xe mô tô",
        "observed": "đi ngược chiều của đường một chiều",
        "violation": "đi ngược chiều của đường một chiều, đi ngược chiều trên đường có biển \"Cấm đi ngược chiều\"",
    },
    "sai_lan": {
        "focus": "phương tiện",
        "vehicle": "xe mô tô",
        "observed": "đi không đúng phần đường hoặc làn đường quy định",
        "violation": "đi không đúng phần đường hoặc làn đường quy định",
    },
    "dung_qua_vach": {
        "focus": "phương tiện",
        "vehicle": "xe mô tô",
        "observed": "dừng đè lên vạch dừng",
        "violation": "không chấp hành hiệu lệnh, chỉ dẫn của vạch kẻ đường, dừng xe vượt quá vạch dừng",
    },
    "sai_bien_bao": {
        "focus": "phương tiện",
        "vehicle": "xe mô tô",
        "observed": "không chấp hành hiệu lệnh của biển báo",
        "violation": "không chấp hành hiệu lệnh, chỉ dẫn của biển báo hiệu đường bộ",
    },
    "di_vao_duong_cam": {
        "focus": "phương tiện",
        "vehicle": "xe mô tô",
        "observed": "đi vào đường có biển cấm",
        "violation": "đi vào đường có biển báo hiệu có nội dung cấm đi vào đối với loại phương tiện đang điều khiển",
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


_MOCK_ASSESS_MIN_OVERLAP = 0.08  # local to MockLLM; not agent.pipeline's retrieval-score threshold


class MockLLM:
    """Deterministic, rule-based stand-in for the real VLM/LLM backends.
    Exists purely so agent/pipeline.py can be built and smoke-tested
    end-to-end on real data before any model is wired in. Swap for
    agent/llm_vlm.py's VLMLLMClient behind this same Protocol; nothing in
    agent/pipeline.py changes.
    """

    def perceive(self, image_path: Path, question: str) -> dict:
        type_image = Path(image_path).parent.name
        hint = _TYPE_IMAGE_HINTS.get(type_image)
        if hint is None:
            return {"focus": "", "vehicle": None, "observed": "", "violation": None, "confidence": 0.3}
        return {**hint, "confidence": 0.6}

    def assess(
        self,
        question: str,
        perception: Perception,
        retrieved: list[RetrievedChunk],
        queries_tried: list[str],
    ) -> dict:
        basis = perception.violation or perception.observed or perception.focus
        best_overlap = max((_overlap(basis, c.content) for c in retrieved), default=0.0)
        if best_overlap >= _MOCK_ASSESS_MIN_OVERLAP:
            return {
                "reason": f"mock: overlap={best_overlap:.2f} >= threshold",
                "sufficient": True,
                "refined_query": None,
            }
        refined = f"{basis} {perception.vehicle or ''}".strip()
        if not refined or refined in queries_tried:
            return {"reason": "mock: no new query to try", "sufficient": False, "refined_query": None}
        return {
            "reason": f"mock: overlap={best_overlap:.2f} < threshold",
            "sufficient": False,
            "refined_query": refined,
        }

    def select(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        options: dict[str, str],
        image_path: Path,
    ) -> dict:
        basis = " ".join(c.content for c in retrieved)
        scores = {L: _overlap(basis, text) for L, text in options.items()}
        best_letter = max(scores, key=lambda L: (scores[L], -ord(L)))
        best_score = scores[best_letter]
        ranked = sorted(scores.values(), reverse=True)

        citation = None
        if retrieved:
            citation = max(retrieved, key=lambda c: _overlap(c.content, options[best_letter])).chunk_id

        if best_score == 0.0:
            best_letter = next(iter(options))
            quality = "poor"
        elif len(ranked) > 1 and ranked[0] == ranked[1]:
            quality = "partial"
        else:
            quality = "exact" if best_score > 0.2 else "partial"

        return {
            "reasoning": f"Lựa chọn khớp nhất với nội dung điều khoản đã truy xuất (overlap={best_score:.2f}).",
            "answer": best_letter,
            "citation": citation,
            "match_quality": quality,
        }
