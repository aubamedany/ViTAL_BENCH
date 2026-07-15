"""ASSESS prompt — structured SufficiencyReport (AGENT_PLAN.md §2.1).

Deliberately not a bare "sufficient: true/false": ASSESS names, for each
required slot, the specific chunk_id that supplies it, so its own accuracy is
checkable post-hoc against the chunk text (ASSESS precision, §7). The final
`overall_sufficient` is *derived* in code from the slot evidence, never taken
verbatim from the LLM — that is the point of the upgrade over a self-reported
boolean, so this module's parser recomputes it rather than trusting the field.
Option-blind: only the question, scene, and retrieved chunks are shown.
"""

from __future__ import annotations

from agent.schemas import RetrievedChunk, SlotEvidence, SufficiencyReport

# Query intent -> the SufficiencyReport slot it is meant to fill (§2.1 ASSESS).
INTENT_TO_SLOT: dict[str, str] = {
    "behavior_identification": "violated_behavior",
    "required_behavior": "required_behavior",
    "penalty_amount": "penalty",
    "supplementary_consequence": "supplementary",
}

ASSESS_SYSTEM_PROMPT = """\
Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. Nhiệm vụ: đánh giá xem các
điều khoản đã truy xuất có đủ để trả lời câu hỏi pháp lý hay không — CHỈ dựa trên nội
dung các điều khoản được cung cấp, KHÔNG được suy đoán nội dung không có trong đó.

Với MỖI mục trong danh sách "các nội dung cần có" bên dưới, hãy xác định:
- mục đó đã được điều khoản nào đáp ứng (ghi rõ chunk_id), nếu có
- trích dẫn/diễn giải ngắn gọn phần nội dung trong điều khoản đó chứng minh điều này

Nếu KHÔNG điều khoản nào trong danh sách đáp ứng được một mục, đánh dấu mục đó là
chưa đáp ứng (filled: false, supporting_chunk_id: null).

KHÔNG được tự nhận định "đủ hay không đủ" một cách chung chung — chỉ báo cáo bằng
chứng cho từng mục cụ thể. TUYỆT ĐỐI KHÔNG đề cập phương án trả lời trắc nghiệm nào.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{
  "slot_evidence": [
    {"slot": "...", "filled": true, "supporting_chunk_id": "...", "justification": "..."}
  ]
}
"""


def build_assess_user_prompt(
    question: str,
    observed_behavior: str,
    required_slots: list[str],
    retrieved: list[RetrievedChunk],
) -> str:
    chunk_lines = [f"[{c.chunk_id}] {c.content}" for c in retrieved]
    return (
        f"Câu hỏi: {question.strip()}\n"
        f"Hành vi quan sát được: {observed_behavior}\n"
        f"Các nội dung cần có: {', '.join(required_slots)}\n\n"
        "Các điều khoản đã truy xuất:\n" + "\n".join(chunk_lines)
    )


def parse_assess_output(
    raw_json: dict, required_slots: list[str], retrieved_ids: set[str]
) -> SufficiencyReport:
    """Recompute overall_sufficient/unfilled_slots from slot evidence rather
    than trusting any self-reported flag (§2.1). A slot claimed `filled` whose
    `supporting_chunk_id` isn't actually in the retrieved set is downgraded to
    unfilled — the LLM cannot claim sufficiency via a chunk it wasn't shown."""
    evidence: list[SlotEvidence] = []
    for item in raw_json.get("slot_evidence", []):
        chunk_id = item.get("supporting_chunk_id")
        filled = bool(item.get("filled")) and chunk_id in retrieved_ids
        evidence.append(
            SlotEvidence(
                slot=item["slot"],
                filled=filled,
                supporting_chunk_id=chunk_id if filled else None,
                justification=item.get("justification", ""),
            )
        )

    filled_slots = {e.slot for e in evidence if e.filled}
    unfilled = [s for s in required_slots if s not in filled_slots]
    return SufficiencyReport(
        required_slots=required_slots,
        slot_evidence=evidence,
        overall_sufficient=len(unfilled) == 0,
        unfilled_slots=unfilled,
    )
