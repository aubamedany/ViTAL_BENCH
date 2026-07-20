"""EMIT prompt — Phase-1's final synthesis step (AGENT_PLAN.md §2.1 "Emit
LegalDetermination"). Option-blind: only the question, observed behavior, and
retrieved chunks are shown, never the MCQ options (I1). Mirrors the house
style of formulate.py/assess.py: Vietnamese, JSON-only output.

"applicable_provisions" is instructed to draw only from the listed chunk_ids —
a soft guard. The hard guarantee (invariant I2, citation closure) is enforced
in code by agent/pipeline.py's emit(), which rejects/re-emits or drops
ungrounded citations; this prompt just reduces how often that has to fire.
"""

from __future__ import annotations

from agent.schemas import RetrievedChunk, SceneDescription

EMIT_SYSTEM_PROMPT = """\
Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. Dựa CHỈ VÀO các điều khoản
được liệt kê bên dưới (không dùng kiến thức khác, không suy đoán), hãy đưa ra KẾT LUẬN
PHÁP LÝ cho hành vi quan sát được.

QUY TẮC:
1. "applicable_provisions" CHỈ được chứa các chunk_id có trong danh sách điều khoản
   được cung cấp — TUYỆT ĐỐI KHÔNG bịa ra chunk_id không có trong danh sách.
2. Mức phạt (penalty_min_vnd/penalty_max_vnd) phải lấy đúng từ nội dung điều khoản
   được cung cấp, không suy đoán. Để null nếu không có điều khoản nào nêu mức phạt.
3. "insufficient_evidence": true nếu các điều khoản được cung cấp KHÔNG đủ để xác
   định hành vi vi phạm, mức phạt, hoặc hình thức xử phạt bổ sung liên quan đến câu hỏi.
4. TUYỆT ĐỐI KHÔNG đề cập đến bất kỳ phương án trả lời trắc nghiệm nào — bạn sẽ
   không được cung cấp phương án nào, chỉ có câu hỏi và các điều khoản.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{
  "violated_behavior": "..." (hoặc null nếu không xác định vi phạm),
  "applicable_provisions": ["chunk_id", ...],
  "penalty_min_vnd": số nguyên hoặc null,
  "penalty_max_vnd": số nguyên hoặc null,
  "supplementary_consequences": ["...", ...],
  "reasoning": "giải thích ngắn gọn bằng tiếng Việt, có trích dẫn điều khoản",
  "confidence": 0.0-1.0,
  "insufficient_evidence": true|false
}
"""


def build_emit_user_prompt(
    question: str, scene: SceneDescription, retrieved: list[RetrievedChunk]
) -> str:
    chunk_lines = [f"[{c.chunk_id}] {c.content}" for c in retrieved]
    return (
        f"Câu hỏi: {question.strip()}\n"
        f"Hành vi quan sát được: {scene.observed_behavior}\n\n"
        "Các điều khoản đã truy xuất (CHỈ được trích dẫn chunk_id từ danh sách này):\n"
        + ("\n".join(chunk_lines) if chunk_lines else "(không có điều khoản nào được truy xuất)")
    )
