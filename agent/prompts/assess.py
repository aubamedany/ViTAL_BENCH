"""ASSESS prompt (AGENT_PLAN_v2.md §3.4) — the sufficiency judgment + query
refinement for agent/pipeline.py's run_phase1 refine loop. Merges what v1
split into ASSESS+REFINE into a single call, and drops v1's SufficiencyReport
/ per-slot-chunk-attribution machinery in favor of a holistic judgment.

Text-only: does NOT re-look at the image. Perception happens exactly once,
before the loop starts; this call only ever sees perception's *output*
(violation/vehicle), never pixels.

Option-blind: question, perception fields, retrieved chunk text, and prior
query texts reach this prompt — never `options`.
"""

from __future__ import annotations

from agent.schemas import Perception, RetrievedChunk

ASSESS_SYSTEM_PROMPT = """\
Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. Nhiệm vụ: đánh giá xem các
điều khoản đã truy xuất có đủ để xác định hành vi vi phạm và (nếu câu hỏi hỏi mức
phạt) mức phạt hay chưa.

QUY TẮC:
1. CHỈ dựa vào nội dung điều khoản được cung cấp, không dùng kiến thức khác.
2. Nếu CHƯA đủ, đề xuất MỘT truy vấn MỚI để tìm điều khoản còn thiếu. Truy vấn PHẢI
   dùng đúng văn phong pháp luật (như: "vượt đèn đỏ" → "không chấp hành hiệu lệnh của
   đèn tín hiệu giao thông"; "không đội mũ" → "không đội mũ bảo hiểm hoặc đội mũ bảo
   hiểm không cài quai đúng quy cách") — truy vấn sai văn phong sẽ không tìm được luật.
3. Truy vấn mới PHẢI khác các truy vấn đã thử, nhắm đúng vào nội dung còn thiếu
   (hành vi vi phạm, hoặc mức phạt, hoặc hình thức xử phạt bổ sung).
4. Nếu câu hỏi hỏi về hình thức xử phạt bổ sung (trừ điểm, tước quyền sử dụng giấy
   phép lái xe): các điều khoản về hậu quả bổ sung THAM CHIẾU CHÉO đến điểm/khoản CỤ
   THỂ của điều khoản định nghĩa hành vi (ví dụ "áp dụng đối với điểm b khoản 3"). CHỈ
   coi là "đủ" khi đã có CẢ (a) điều khoản xác định hành vi thuộc điểm/khoản nào, VÀ
   (b) điều khoản về hậu quả bổ sung nêu rõ danh sách điểm/khoản mà nó áp dụng — thiếu
   một trong hai thì CHƯA đủ, kể cả khi đã có điều khoản nêu số điểm cho MỘT hành vi
   khác.
5. Nếu đã đủ, "refined_query" PHẢI là null.
6. TUYỆT ĐỐI KHÔNG đề cập đến bất kỳ phương án trả lời trắc nghiệm nào — bạn sẽ
   không được cung cấp phương án nào.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{"reason": "lý do ngắn gọn", "sufficient": true|false,
 "refined_query": "truy vấn mới, hoặc null nếu đã đủ"}
"""


def build_assess_user_prompt(
    question: str,
    perception: Perception,
    retrieved: list[RetrievedChunk],
    queries_tried: list[str],
) -> str:
    chunk_lines = [f"[{c.chunk_id}] {c.content}" for c in retrieved]
    return (
        f"Câu hỏi: {question.strip()}\n"
        f"Hành vi vi phạm dự đoán: {perception.violation or '(không có)'}\n"
        f"Phương tiện: {perception.vehicle or '(không xác định)'}\n\n"
        "Các điều khoản đã truy xuất:\n"
        + ("\n".join(chunk_lines) if chunk_lines else "(chưa truy xuất được điều khoản nào)")
        + "\n\nCác truy vấn đã thử (không đề xuất lại nguyên văn):\n"
        + ("\n".join(f"- {q}" for q in queries_tried) if queries_tried else "(chưa có)")
    )
