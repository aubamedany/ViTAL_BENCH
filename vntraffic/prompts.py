"""Prompt templates and robust output parsers (EXPERIMENT_PLAN.md §3).

All prompts are frozen here (one config file per run means *no* manual prompt
edits between runs, §11). Content is Vietnamese; every step asks for a small
JSON object so parsing is deterministic, with regex fallbacks for models that
wrap or malform the JSON.

Conditions and the steps they use:
  E1               ANSWER (image + question + options)
  E2 / E3          ANSWER_WITH_CITATIONS (adds retrieved chunks; E3's retrieval
                   query also saw the options — the prompt itself is identical)
  E4 Phase 1       PERCEIVE → FORMULATE → ASSESS/REFINE → DETERMINATION
                   (options never shown)
  E4 Phase 2       SELECT (determination + question + options; no image)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

from . import constants as C

# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_options(options: Dict[str, str],
                   letters: Sequence[str] = C.OPTION_LETTERS) -> str:
    return "\n".join(f"{L}. {options[L]}" for L in letters)


def render_chunks(chunks: Sequence) -> str:
    """Render retrieved chunks as an id-labelled context block."""
    lines = []
    for ch in chunks:
        cid = ch["chunk_id"] if isinstance(ch, dict) else ch.chunk_id
        text = ch["text"] if isinstance(ch, dict) else getattr(ch, "content", "")
        lines.append(f"[{cid}] {text}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #
SYSTEM_VI = ("Bạn là trợ lý phân tích vi phạm giao thông đường bộ theo pháp luật "
             "Việt Nam (Luật 36/2024/QH15 và Nghị định 168/2024/NĐ-CP). "
             "Hãy trả lời chính xác, ngắn gọn và đúng định dạng được yêu cầu.")

# --------------------------------------------------------------------------- #
# E1 — parametric answer
# --------------------------------------------------------------------------- #
def build_e1_answer(question: str, options: Dict[str, str],
                    letters: Sequence[str] = C.OPTION_LETTERS) -> str:
    return (
        f"Quan sát hình ảnh và trả lời câu hỏi trắc nghiệm sau.\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Các lựa chọn:\n{render_options(options, letters)}\n\n"
        f"Chỉ chọn MỘT đáp án đúng. Trả về JSON: "
        f'{{"answer": "<A|B|C|D>"}}'
    )


# --------------------------------------------------------------------------- #
# E2 / E3 — answer grounded in retrieved chunks (identical prompt; the two
# conditions differ only in what query produced `chunks`, see conditions.py)
# --------------------------------------------------------------------------- #
def build_rag_answer(question: str, options: Dict[str, str], chunks: Sequence,
                     letters: Sequence[str] = C.OPTION_LETTERS) -> str:
    return (
        f"Sử dụng hình ảnh và các trích dẫn pháp luật dưới đây để trả lời.\n\n"
        f"Các trích dẫn pháp luật (mỗi trích dẫn có mã trong ngoặc vuông):\n"
        f"{render_chunks(chunks)}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Các lựa chọn:\n{render_options(options, letters)}\n\n"
        f"Chọn MỘT đáp án đúng và liệt kê mã các trích dẫn đã dùng để suy luận "
        f"(chỉ dùng mã có trong danh sách trên). Trả về JSON: "
        f'{{"answer": "<A|B|C|D>", "citations": ["<chunk_id>", ...]}}'
    )


# --------------------------------------------------------------------------- #
# E4 Phase 1 — option-blind loop
# --------------------------------------------------------------------------- #
def build_perceive(question: str) -> str:
    return (
        f"Mô tả khách quan những gì quan sát được trong ảnh có liên quan đến "
        f"trật tự, an toàn giao thông (phương tiện, người, biển báo, vạch kẻ "
        f"đường, tín hiệu, hành vi đang diễn ra). KHÔNG suy đoán đáp án.\n\n"
        f"Bối cảnh câu hỏi (không kèm lựa chọn): {question}\n\n"
        f'Trả về JSON: {{"perception": "<mô tả>"}}'
    )


def build_formulate(question: str, perception: str) -> str:
    return (
        f"Dựa trên mô tả cảnh và câu hỏi, hãy tạo MỘT truy vấn tìm kiếm ngắn gọn "
        f"bằng tiếng Việt để tra cứu điều khoản pháp luật liên quan (nêu hành vi "
        f"và loại phương tiện nếu có). KHÔNG nhắc tới bất kỳ lựa chọn đáp án nào.\n\n"
        f"Mô tả cảnh: {perception}\n\n"
        f"Câu hỏi: {question}\n\n"
        f'Trả về JSON: {{"query": "<truy vấn>"}}'
    )


def build_assess(question: str, perception: str, chunks: Sequence) -> str:
    return (
        f"Đánh giá xem các trích dẫn pháp luật sau có ĐỦ để xác định điều khoản "
        f"áp dụng cho tình huống hay chưa.\n\n"
        f"Mô tả cảnh: {perception}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Trích dẫn hiện có:\n{render_chunks(chunks)}\n\n"
        f"Nếu chưa đủ, đưa ra MỘT truy vấn tìm kiếm mới, khác trước. "
        f'Trả về JSON: {{"sufficient": <true|false>, "query": "<truy vấn mới hoặc rỗng>"}}'
    )


def build_determination(question: str, perception: str, chunks: Sequence) -> str:
    return (
        f"Đưa ra kết luận pháp lý cho tình huống, CHỈ dựa trên các trích dẫn đã "
        f"truy xuất dưới đây. Mọi điều khoản viện dẫn phải nằm trong danh sách mã "
        f"này.\n\n"
        f"Mô tả cảnh: {perception}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Trích dẫn đã truy xuất:\n{render_chunks(chunks)}\n\n"
        f"Trả về JSON:\n"
        f'{{"violated_behavior": "<hành vi vi phạm hoặc \'không vi phạm\'>", '
        f'"applicable_provisions": ["<chunk_id>", ...], '
        f'"penalty": "<mức phạt nếu có>", '
        f'"supplementary_consequences": "<hình thức bổ sung nếu có>", '
        f'"confidence": <0..1>}}'
    )


# --------------------------------------------------------------------------- #
# E4 Phase 2 — SELECT (no image, no retrieval)
# --------------------------------------------------------------------------- #
def build_select(question: str, options: Dict[str, str], determination: dict,
                 letters: Sequence[str] = C.OPTION_LETTERS) -> str:
    det_str = json.dumps(determination, ensure_ascii=False, indent=2)
    return (
        f"Dưới đây là kết luận pháp lý đã được xác định TRƯỚC khi nhìn thấy các "
        f"lựa chọn. Hãy ánh xạ kết luận này sang đúng MỘT lựa chọn trắc nghiệm. "
        f"KHÔNG suy luận lại về hình ảnh, chỉ chọn lựa chọn khớp nhất với kết luận.\n\n"
        f"Kết luận pháp lý:\n{det_str}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Các lựa chọn:\n{render_options(options, letters)}\n\n"
        f'Trả về JSON: {{"answer": "<A|B|C|D>"}}'
    )


# ========================================================================== #
# Parsers (robust: JSON-first, regex fallback)
# ========================================================================== #
_LETTER_RE = re.compile(r'"answer"\s*:\s*"?\s*([ABCD])', re.IGNORECASE)
_BARE_LETTER_RE = re.compile(r'\b([ABCD])\b')
_CHUNK_ID_RE = re.compile(r"[A-Za-zĐđ0-9]+_[A-Za-zĐđ0-9_]+")


def extract_json(text: str) -> Optional[dict]:
    """Return the first balanced JSON object in *text*, or None.

    Handles ```json fences and leading/trailing prose by scanning for the first
    '{' and matching braces (string-aware).
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break  # try next '{'
        start = text.find("{", start + 1)
    return None


def parse_answer_letter(text: str) -> Optional[str]:
    """Extract the chosen A/B/C/D letter, JSON-first then regex."""
    obj = extract_json(text)
    if obj is not None:
        val = obj.get("answer")
        if isinstance(val, str) and val.strip().upper()[:1] in C.OPTION_LETTERS:
            return val.strip().upper()[0]
    m = _LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _BARE_LETTER_RE.search(text.strip())
    if m:
        return m.group(1).upper()
    return None


def parse_citations(text: str,
                    keys: Sequence[str] = ("citations", "applicable_provisions")) -> List[str]:
    """Extract cited chunk_ids from a response (JSON list preferred)."""
    obj = extract_json(text)
    if obj is not None:
        for key in keys:
            val = obj.get(key)
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
    # fallback: bracketed [CHUNK_ID] tokens or bare chunk-id-like tokens
    bracketed = re.findall(r"\[([A-Za-zĐđ0-9_]+)\]", text)
    if bracketed:
        return bracketed
    return []


def parse_query(text: str) -> str:
    obj = extract_json(text)
    if obj is not None and isinstance(obj.get("query"), str):
        return obj["query"].strip()
    # fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip().strip('"')
        if line:
            return line
    return text.strip()


def parse_assessment(text: str) -> dict:
    obj = extract_json(text) or {}
    sufficient = obj.get("sufficient")
    if isinstance(sufficient, str):
        sufficient = sufficient.strip().lower() in ("true", "yes", "đủ", "1")
    return {"sufficient": bool(sufficient), "query": (obj.get("query") or "").strip()}


def parse_determination(text: str) -> dict:
    obj = extract_json(text) or {}
    return {
        "violated_behavior": obj.get("violated_behavior", ""),
        "applicable_provisions": [
            str(x).strip() for x in obj.get("applicable_provisions", []) if str(x).strip()
        ] if isinstance(obj.get("applicable_provisions"), list) else [],
        "penalty": obj.get("penalty", ""),
        "supplementary_consequences": obj.get("supplementary_consequences", ""),
        "confidence": obj.get("confidence", None),
        "_raw": text,
    }
