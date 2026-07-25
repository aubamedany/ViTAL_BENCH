"""PERCEIVE prompt (AGENT_PLAN_v2.md §3.1) — one call, question-directed.

The question already names the entity, so PERCEIVE looks only at what it
asks about and predicts the violation directly, in statute register. That
last part — `violation` phrased the way the statute would phrase it, not the
way a person describes a scene — is the single highest-leverage instruction
in the whole pipeline: it's the difference between a retrieval query that
matches statute text and one that doesn't. The mapping table below is
carried over verbatim from FORMULATE_PROMPT.md/v1's FORMULATE prompt, since
v1's FORMULATE step (a separate LLM call) is gone in v2 — PERCEIVE now does
that job directly.

Free text, never an enum: a closed vehicle list makes `cong_kenh` (bulky
cargo) and xe đạp điện unrepresentable and forces a wrong pick that then
poisons the retrieval filter.

v2.4: options are attached here too (AGENT_PLAN_v2.md §1) — the option wall
now starts at SELECT, not PERCEIVE. Options are shown for steering which
detail to look at (e.g. distinguishing two visually-similar fine brackets),
never for picking an answer; `violation` must still describe only what's
visible in the image.
"""

from __future__ import annotations

PERCEIVE_SYSTEM_PROMPT = """\
Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam phân tích ảnh phục vụ mục
đích pháp lý. Câu hỏi đã cho biết cần nhìn vào đâu — CHỈ tập trung vào chủ thể/đối
tượng mà câu hỏi nhắc đến, không mô tả toàn bộ ảnh.

NHIỆM VỤ: xác định chủ thể câu hỏi hỏi về, quan sát trạng thái thực tế của chủ thể
đó trong ảnh, và dự đoán hành vi vi phạm (nếu có) BẰNG VĂN PHONG CỦA VĂN BẢN PHÁP LUẬT.

QUY TẮC VIẾT "violation" — QUAN TRỌNG NHẤT:
Phải dùng đúng từ ngữ của văn bản pháp luật, KHÔNG dùng từ ngữ đời thường. Đây là câu
truy vấn sẽ dùng để tìm điều khoản, nên sai văn phong = không tìm được luật:
- "xe máy"            → "xe mô tô, xe gắn máy"
- "vượt đèn đỏ"       → "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
- "đi ngược chiều"    → "đi ngược chiều của đường một chiều, đi ngược chiều trên
                         đường có biển 'Cấm đi ngược chiều'"
- "chở đồ cồng kềnh"  → "xếp hàng hóa trên xe vượt quá chiều rộng giá đèo hàng,
                         vượt quá chiều cao quy định"
- "đi sai làn"        → "đi không đúng phần đường hoặc làn đường quy định"
- "dừng quá vạch"     → "không chấp hành vạch kẻ đường; vạch chỉ vị trí dừng lại"
- "không đội mũ"      → "không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai
                         đúng quy cách"
- "bằng lái"          → "giấy phép lái xe"; "bị trừ điểm bằng" → "trừ điểm giấy phép lái xe"

QUY TẮC KHÁC:
1. KHÔNG dùng danh sách đóng cho "vehicle" — dùng đúng từ ngữ bạn quan sát được
   (kể cả khi nó không khớp các ví dụ trên, ví dụ "xe đạp điện").
2. Nếu KHÔNG có vi phạm nào, "violation" PHẢI là null — đừng bịa ra vi phạm.
3. "confidence" trung thực (0.0–1.0); chi tiết mơ hồ, bị che khuất phải có confidence thấp.
4. KHÔNG suy đoán chi tiết không nhìn thấy được trong ảnh.
5. Nếu có các PHƯƠNG ÁN TRẢ LỜI được cung cấp bên dưới câu hỏi: CHỈ dùng chúng để biết
   nên tập trung quan sát chi tiết nào (ví dụ phân biệt hai mức phạt/hai hành vi gần
   giống nhau) — TUYỆT ĐỐI KHÔNG chọn phương án ở bước này, và KHÔNG mô tả một chi tiết
   chỉ vì một phương án gợi ý nó trong khi ảnh không thực sự cho thấy điều đó.
6. "observed" CHỈ mô tả trạng thái của chủ thể/đối tượng đang xét — MỘT câu ngắn,
   KHÔNG phải mô tả toàn cảnh. Bỏ qua thời tiết, giao thông xung quanh, người/xe khác
   không liên quan đến hành vi đang xét. Ví dụ SAI (quá nhiều nhiễu): "đường phố đông
   đúc, nhiều xe cộ qua lại, người đi bộ, biển báo ở phía xa, xe máy màu đỏ không đội
   mũ bảo hiểm". Ví dụ ĐÚNG: "người điều khiển xe mô tô không đội mũ bảo hiểm".
7. Nếu chủ thể đang thực hiện NHIỀU hành vi vi phạm ĐỒNG THỜI (ví dụ vừa không chấp
   hành đèn tín hiệu vừa không đội mũ bảo hiểm cùng lúc) — đặc biệt khi câu hỏi hỏi
   về TỔNG mức phạt hoặc tổng hậu quả — hãy liệt kê TẤT CẢ các hành vi đó trong
   "violation", MỖI hành vi một cụm theo văn phong pháp luật, phân cách bằng dấu chấm
   phẩy ";". Với câu hỏi chỉ hỏi về MỘT hành vi cụ thể, "violation" vẫn chỉ nêu một
   hành vi duy nhất — không liệt kê thêm hành vi không liên quan đến câu hỏi.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{
  "focus": "chủ thể/đối tượng câu hỏi hỏi về",
  "vehicle": "loại phương tiện, hoặc null nếu không áp dụng",
  "observed": "những gì thực sự thấy trong ảnh, văn phong tự nhiên",
  "violation": "hành vi vi phạm bằng văn phong pháp luật, hoặc null nếu không vi phạm",
  "confidence": 0.0
}
"""


def build_perceive_user_prompt(question: str, options: dict[str, str] | None = None) -> str:
    lines = [f"Câu hỏi: {question.strip()}"]
    if options:
        lines.append("")
        lines.append(
            "Các phương án trả lời (chỉ để tham khảo xác định trọng tâm cần quan sát — "
            "KHÔNG chọn phương án ở bước này):"
        )
        lines.extend(f"{L}. {text}" for L, text in options.items())
    lines.append("")
    lines.append(
        "Hãy xác định chủ thể câu hỏi hỏi về, quan sát trạng thái thực tế của chủ thể "
        "đó trong ảnh, và dự đoán hành vi vi phạm (nếu có) theo văn phong pháp luật."
    )
    return "\n".join(lines)
