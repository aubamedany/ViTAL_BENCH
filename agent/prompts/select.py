"""SELECT prompt — Phase 2, the only module that ever sees the options
(AGENT_PLAN_v2.md §4). Multimodal: the image is re-attached here (as a
separate content block in the LLM call, not textually embedded — see
agent/llm_vlm.py's select()), alongside the *actual retrieved statute text*
rather than a frozen, lossy determination summary written before the model
knew what it would need to distinguish. Options often turn on a detail a
summary would drop (a specific fine range, a particular condition), and a
static PERCEIVE field can drop a visual detail SELECT needs; letting SELECT
look at the image again, now with the actual candidate law in hand, catches
what a frozen summary would miss. This does not reopen the option wall:
retrieval is already finished by the time SELECT runs, and SELECT has no
retriever/corpus handle — it cannot issue a new retrieval no matter what it
notices once options are visible.
"""

from __future__ import annotations

from agent.schemas import RetrievedChunk

SELECT_SYSTEM_PROMPT = """\
Bạn nhận được một HÌNH ẢNH, các ĐIỀU KHOẢN PHÁP LUẬT đã truy xuất, một câu hỏi, và
bốn phương án trắc nghiệm A/B/C/D. Nhiệm vụ: quan sát hình ảnh, suy luận TỪ nội dung
điều khoản để trả lời câu hỏi, rồi chọn phương án khớp nhất.

QUY TẮC:
1. CHỈ dựa vào (a) những gì quan sát được trực tiếp trong ảnh và (b) nội dung điều
   khoản được cung cấp — KHÔNG dùng kiến thức pháp luật nào khác ngoài các điều
   khoản này.
2. THAM CHIẾU CHÉO GIỮA CÁC ĐIỂM/KHOẢN — các điều luật LIÊN KẾT với nhau, phải suy
   luận qua NHIỀU điều khoản, không dừng lại ở điều khoản đầu tiên có vẻ liên quan.
   Nhiều điều khoản (đặc biệt khoản về hình thức xử phạt bổ sung — trừ điểm, tước
   quyền sử dụng giấy phép lái xe) chỉ áp dụng cho ĐÚNG các điểm/khoản được liệt kê
   TƯỜNG MINH. Ví dụ: "Thực hiện hành vi quy định tại điểm b khoản 3... Điều này bị
   trừ điểm giấy phép lái xe 02 điểm" CHỈ áp dụng cho ĐÚNG điểm b khoản 3 — KHÔNG áp
   dụng cho điểm a, c, d, đ, e... của CÙNG khoản 3, dù cùng nằm trong một khoản với
   điểm b. Khi xét hậu quả cho một hành vi cụ thể, PHẢI làm đủ 3 bước:
   (1) xác định hành vi đó khớp với ĐIỂM NÀO của điều khoản định nghĩa hành vi (ví dụ
       hành vi "mang vác vật cồng kềnh" là điểm e khoản 3, KHÔNG phải điểm b khoản 3);
   (2) đọc kỹ điều khoản về hậu quả (trừ điểm/tước quyền) để biết nó liệt kê CHÍNH XÁC
       điểm/khoản nào;
   (3) kiểm tra xem điểm xác định ở bước (1) CÓ nằm trong danh sách ở bước (2) hay
       KHÔNG — nếu KHÔNG có trong danh sách, hậu quả đó KHÔNG áp dụng cho hành vi này,
       dù điều khoản có nêu hậu quả cho các điểm KHÁC cùng khoản.
3. "citation" PHẢI là chunk_id của MỘT điều khoản trong danh sách được cung cấp —
   TUYỆT ĐỐI KHÔNG bịa ra chunk_id không có trong danh sách. Nếu không có điều khoản
   nào được cung cấp (danh sách rỗng), để "citation": null.
4. Nếu câu hỏi hỏi TỔNG mức phạt (hoặc tổng hậu quả) cho NHIỀU hành vi vi phạm xảy ra
   ĐỒNG THỜI (ví dụ vừa không chấp hành đèn tín hiệu vừa không đội mũ bảo hiểm): xác
   định mức phạt của TỪNG hành vi riêng biệt trong các điều khoản đã truy xuất, rồi
   CỘNG DỒN lại để ra tổng (cộng min với min, cộng max với max) — KHÔNG chỉ chọn mức
   phạt của MỘT hành vi rồi coi đó là tổng. "reasoning" phải nêu rõ phép cộng (mức
   phạt hành vi 1 + mức phạt hành vi 2 = tổng). "citation" vẫn chỉ nhận MỘT chunk_id
   — chọn chunk_id của điều khoản có mức phạt cao nhất trong số đã cộng.
5. Nếu không có phương án nào khớp hoàn toàn với nội dung điều khoản, chọn phương án
   GẦN NHẤT và đặt "match_quality": "poor".
6. RIÊNG với câu hỏi về SỐ ĐIỂM bị trừ trên giấy phép lái xe ("trừ bao nhiêu điểm",
   "bị trừ mấy điểm", ...): KHÔNG áp dụng quy tắc 5 — KHÔNG được đoán một số điểm cụ
   thể chỉ vì nó "gần đúng nhất". Áp dụng quy tắc 2: chỉ chọn một số điểm nếu điểm
   khớp với hành vi đang xét THỰC SỰ được liệt kê tường minh trong điều khoản về trừ
   điểm. Nếu điểm đó KHÔNG được liệt kê (dù khoản trừ điểm CÓ nêu số điểm cho các
   điểm khác), hãy chọn phương án "không bị trừ điểm" (nếu có trong các phương án) —
   đây không phải thiếu thông tin cần đoán, mà là kết luận đúng theo quy tắc 2.
7. "reasoning" phải nối kết những gì quan sát được trong ảnh VỚI nội dung điều khoản
   cụ thể, bằng tiếng Việt — không chỉ diễn giải lại điều khoản một cách tách biệt.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{"reasoning": "...", "answer": "A|B|C|D", "citation": "chunk_id hoặc null",
 "match_quality": "exact|partial|poor"}
"""


def build_select_user_prompt(
    question: str, retrieved: list[RetrievedChunk], options: dict[str, str]
) -> str:
    chunk_lines = [f"[{c.chunk_id}] {c.content}" for c in retrieved]
    option_lines = [f"{L}. {text}" for L, text in options.items()]
    return (
        "Quan sát kỹ hình ảnh đính kèm và đối chiếu với nội dung điều khoản dưới đây.\n\n"
        f"Câu hỏi: {question.strip()}\n\n"
        "Các điều khoản đã truy xuất:\n"
        + ("\n".join(chunk_lines) if chunk_lines else "(không có điều khoản nào được truy xuất)")
        + "\n\nCác phương án:\n"
        + "\n".join(option_lines)
    )
