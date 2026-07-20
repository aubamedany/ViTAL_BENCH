"""PERCEIVE-v2 prompts — ENUMERATE + INSPECT (AGENT_PLAN.md §2.1).

Not given verbatim anywhere (FORMULATE_PROMPT.md only specifies FORMULATE/
REFINE); drafted to match its house style: Vietnamese, JSON-only output,
open-vocabulary (no closed enum for `kind`/`state` — see §2.1 "Design rules").
Both calls are option-blind: only the image and the question ever reach them.
"""

from __future__ import annotations

ENUMERATE_SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích ảnh giao thông đường bộ Việt Nam phục vụ mục đích pháp lý.
Nhiệm vụ: liệt kê MỌI thực thể trong ảnh có thể liên quan đến việc xác định vi phạm
pháp luật giao thông — đây là bước trích xuất YẾU TỐ PHÁP LÝ (chủ thể, hành vi, hoàn
cảnh), không phải mô tả ảnh chung chung.

Các loại thực thể sau chỉ là VÍ DỤ MINH HỌA, KHÔNG PHẢI danh sách đầy đủ — hãy dùng
đúng từ ngữ của riêng bạn cho bất cứ thực thể nào bạn quan sát được, kể cả khi nó
không khớp với các ví dụ dưới đây:
- phương tiện (loại xe, số lượng người trên xe, hàng hóa chở theo)
- biển báo, vạch kẻ đường, đèn tín hiệu
- người đi bộ, người điều khiển phương tiện (trang bị bảo hộ, tư thế)
- vật cản, chướng ngại vật, phần đường/làn đường

QUY TẮC:
1. KHÔNG dùng danh sách đóng — gọi tên thực thể bằng từ ngữ tự nhiên của bạn.
2. Với mỗi thực thể, giải thích NGẮN GỌN vì sao nó có thể liên quan đến pháp luật
   giao thông (trường "legally_relevant_because"). Một thực thể không giải thích
   được lý do liên quan sẽ bị loại khỏi bước truy vấn sau, nhưng vẫn được ghi log.
3. Gán "confidence" trung thực (0.0–1.0) cho từng thực thể — thực thể không chắc
   chắn (mơ hồ, bị che khuất, khó nhìn rõ) phải có confidence thấp.
4. KHÔNG suy đoán các chi tiết bạn không thể nhìn thấy trong ảnh.
5. TUYỆT ĐỐI KHÔNG đề cập đến bất kỳ phương án trả lời trắc nghiệm nào — bạn sẽ
   không được cung cấp phương án nào, chỉ có ảnh và câu hỏi.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{
  "entities": [
    {"kind": "...", "description": "...", "confidence": 0.0,
     "legally_relevant_because": "..."}
  ],
  "observed_behavior": "...",
  "scene_notes": "..."
}
"""

INSPECT_SYSTEM_PROMPT = """\
Bạn đang kiểm tra một thực thể cụ thể trong ảnh giao thông để phục vụ mục đích pháp lý.
Chỉ trả lời về thực thể được nêu tên bên dưới, dựa trên những gì thực sự nhìn thấy
trong ảnh. KHÔNG đề cập đến phương án trả lời trắc nghiệm nào.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{"state": "...", "confidence": 0.0}
"""


def build_enumerate_user_prompt(question: str) -> str:
    return (
        f"Câu hỏi pháp lý liên quan đến ảnh: {question.strip()}\n"
        "Hãy liệt kê mọi thực thể trong ảnh có thể liên quan đến việc xác định vi phạm "
        "pháp luật giao thông."
    )


def build_inspect_user_prompt(entity_kind: str, entity_description: str) -> str:
    return (
        f"Thực thể cần kiểm tra: {entity_kind} ({entity_description}).\n"
        "Thuộc tính pháp lý quan trọng nhất của thực thể này là gì "
        "(ví dụ: trạng thái đèn tín hiệu, có đội mũ bảo hiểm hay không, "
        "mã biển báo, vị trí so với vạch kẻ đường)?"
    )
