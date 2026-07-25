"""ASSESS prompt (AGENT_PLAN_v2.md §3.4) — the sufficiency judgment + query
refinement for agent/pipeline.py's run_phase1 refine loop. Merges what v1
split into ASSESS+REFINE into a single call, and drops v1's SufficiencyReport
/ per-slot-chunk-attribution machinery in favor of a holistic judgment.

Text-only: does NOT re-look at the image. Perception happens exactly once,
before the loop starts; this call only ever sees perception's *output*
(violation/vehicle), never pixels.

v2.4: options are attached here too (AGENT_PLAN_v2.md §1) — used only to
judge whether the retrieved chunks already distinguish between them, never
to pick one. The option wall now starts at SELECT, not Phase 1.

v2.7: `refined_query` guidance (rules 2a-2c) now asks explicitly for the
same three things agent/pipeline.py's `_build_query_text()` enforces
deterministically for the rebuilt query — state the vehicle, match the
question's actual focus (mức phạt / trừ điểm / tước quyền / plain behavior),
no scene-descriptive filler that wouldn't appear in a legal text. The
deterministic rebuild guarantees the vehicle/consequence framing regardless
of what the model does; this prompt change targets the part that can't be
enforced that way — the core phrase itself.
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
2a. Truy vấn PHẢI bắt đầu bằng LOẠI PHƯƠNG TIỆN (dùng đúng giá trị "Phương tiện" ở
   trên, ví dụ "xe mô tô") — nhiều điều khoản được đặt tiêu đề theo loại phương tiện, bỏ
   loại phương tiện sẽ bỏ lỡ manh mối khớp từ ngữ quan trọng.
2b. Truy vấn PHẢI bám sát ĐÚNG TRỌNG TÂM câu hỏi đang hỏi — nếu câu hỏi hỏi về mức
   phạt tiền, dùng "mức phạt tiền"; nếu hỏi về trừ điểm giấy phép lái xe, dùng "trừ
   điểm giấy phép lái xe"; nếu hỏi về tước quyền sử dụng giấy phép lái xe, dùng "tước
   quyền sử dụng giấy phép lái xe"; nếu hỏi về hành vi/lỗi vi phạm nói chung, chỉ nêu
   hành vi, KHÔNG thêm cụm nào về mức phạt/trừ điểm nếu câu hỏi không hỏi về chúng.
2c. TUYỆT ĐỐI KHÔNG đưa vào truy vấn các chi tiết mô tả cảnh KHÔNG xuất hiện trong văn
   bản pháp luật (màu sắc, kích thước, hình dáng cụ thể của vật thể, thời tiết, bối
   cảnh xung quanh...) — chỉ dùng từ ngữ có thể xuất hiện trong điều khoản.
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
6. Nếu có các PHƯƠNG ÁN TRẢ LỜI được cung cấp: dùng chúng CHỈ để đánh giá xem các điều
   khoản đã truy xuất đã đủ để PHÂN BIỆT giữa các phương án hay chưa (ví dụ hai phương
   án nêu hai mức phạt khác nhau — cần điều khoản nêu đúng mức áp dụng) — KHÔNG chọn
   phương án nào ở bước này, việc chọn phương án thuộc bước khác.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{"reason": "lý do ngắn gọn", "sufficient": true|false,
 "refined_query": "truy vấn mới, hoặc null nếu đã đủ"}
"""


def build_assess_user_prompt(
    question: str,
    perception: Perception,
    retrieved: list[RetrievedChunk],
    queries_tried: list[str],
    options: dict[str, str] | None = None,
) -> str:
    chunk_lines = [f"[{c.chunk_id}] {c.content}" for c in retrieved]
    parts = [
        f"Câu hỏi: {question.strip()}",
        f"Hành vi vi phạm dự đoán: {perception.violation or '(không có)'}",
        f"Phương tiện: {perception.vehicle or '(không xác định)'}",
    ]
    if options:
        parts.append("")
        parts.append("Các phương án trả lời (chỉ để đánh giá tính đủ, KHÔNG chọn ở bước này):")
        parts.extend(f"{L}. {text}" for L, text in options.items())
    parts.append("")
    parts.append("Các điều khoản đã truy xuất:")
    parts.append("\n".join(chunk_lines) if chunk_lines else "(chưa truy xuất được điều khoản nào)")
    parts.append("")
    parts.append("Các truy vấn đã thử (không đề xuất lại nguyên văn):")
    parts.append("\n".join(f"- {q}" for q in queries_tried) if queries_tried else "(chưa có)")
    return "\n".join(parts)
