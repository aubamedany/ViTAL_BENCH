"""SELECT prompt — Phase 2, the only module that ever sees the options
(AGENT_PLAN.md §2.2). No retrieval, no image, no re-reasoning about the scene:
it maps a frozen LegalDetermination onto the four options.
"""

from __future__ import annotations

from agent.schemas import LegalDetermination

SELECT_SYSTEM_PROMPT = """\
Bạn nhận được một KẾT LUẬN PHÁP LÝ đã được xác định sẵn (hành vi vi phạm, điều khoản
áp dụng, mức phạt, hình thức xử phạt bổ sung) và bốn phương án trắc nghiệm A/B/C/D.

Nhiệm vụ DUY NHẤT: chọn phương án khớp với kết luận pháp lý này. KHÔNG suy luận lại
từ đầu, KHÔNG tự tra cứu thêm điều khoản nào khác — chỉ dùng nội dung kết luận đã
cho. Nếu không có phương án nào khớp hoàn toàn, chọn phương án GẦN NHẤT và đặt
"match_quality": "poor".

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{"answer": "A|B|C|D", "match_quality": "exact|partial|poor", "rationale": "..."}
"""


def build_select_user_prompt(
    determination: LegalDetermination, options: dict[str, str]
) -> str:
    det_lines = [
        f"Hành vi vi phạm: {determination.violated_behavior or '(không xác định vi phạm)'}",
        f"Điều khoản áp dụng: {', '.join(determination.applicable_provisions) or '(không có)'}",
    ]
    if determination.penalty_min_vnd is not None or determination.penalty_max_vnd is not None:
        det_lines.append(
            f"Mức phạt: {determination.penalty_min_vnd} - {determination.penalty_max_vnd} đồng"
        )
    if determination.supplementary_consequences:
        det_lines.append(
            "Hình thức bổ sung: " + ", ".join(determination.supplementary_consequences)
        )
    det_lines.append(f"Lý giải: {determination.reasoning}")
    if determination.insufficient_evidence:
        det_lines.append("(Lưu ý: kết luận này dựa trên bằng chứng CHƯA ĐẦY ĐỦ.)")

    option_lines = [f"{L}. {text}" for L, text in options.items()]
    return (
        "Kết luận pháp lý:\n" + "\n".join(det_lines) +
        "\n\nCác phương án:\n" + "\n".join(option_lines)
    )
