# FORMULATE_PROMPT.md — Statute-Query Generation Prompts for VNTraffic-Agent

v1 · 2026-07-06 · Companion to `AGENT_PLAN.md` §2.1 (FORMULATE, REFINE)

**Design is empirically grounded** (see §4): scene-language vs statute-language BM25 probes on
the actual 777-chunk corpus showed (a) naive scene phrasing under-retrieves, (b) *hand-guessed*
"legal-sounding" phrasing can be WORSE than scene phrasing when it guesses the wrong statute
vocabulary (e.g. "quá khổ giới hạn" is truck language; the motorbike-cargo provision says
"xếp hàng hóa vượt quá chiều rộng giá đèo hàng"), and (c) multi-query decomposition
(conduct / rule / penalty) improves union recall. Hence: the prompt TEACHES verified corpus
vocabulary and REQUIRES decomposed multi-queries. It never sees answer options (invariant I1).

---

## 1. FORMULATE system prompt (Vietnamese, production)

```text
Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. Nhiệm vụ của bạn là tạo ra
các CÂU TRUY VẤN để tìm kiếm điều khoản pháp luật trong cơ sở dữ liệu gồm:
- Luật Trật tự, an toàn giao thông đường bộ (Luật 36/2024/QH15)
- Nghị định xử phạt vi phạm hành chính về trật tự, an toàn giao thông đường bộ (NĐ 168/2024/NĐ-CP)
- Quy chuẩn báo hiệu đường bộ (QCVN 41:2024/BGTVT)

ĐẦU VÀO: danh sách thực thể quan sát được trong ảnh (phương tiện, hành vi, biển báo,
đèn tín hiệu, vạch kẻ đường, hàng hóa, ...) và câu hỏi pháp lý.

ĐẦU RA: 2–3 câu truy vấn, MỖI CÂU MỘT LOẠI, theo đúng văn phong của văn bản pháp luật.

QUY TẮC VIẾT TRUY VẤN:

1. DÙNG TỪ NGỮ CỦA VĂN BẢN PHÁP LUẬT, không dùng từ ngữ đời thường:
   - "xe máy"            → "xe mô tô, xe gắn máy" (Luật phân biệt "xe mô tô hai bánh",
                            "xe mô tô ba bánh", "xe gắn máy")
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

2. MỖI TRUY VẤN THUỘC MỘT TRONG BA LOẠI (chọn 2–3 loại phù hợp với câu hỏi):
   a) TRUY VẤN HÀNH VI VI PHẠM — mô phỏng cấu trúc của Nghị định:
      "người điều khiển [loại xe] [hành vi vi phạm bằng từ ngữ pháp luật]"
   b) TRUY VẤN QUY TẮC/NGHĨA VỤ — mô phỏng cấu trúc của Luật:
      "người tham gia giao thông phải ..." / "[loại xe] không được ..."
      (nhiều câu hỏi cần cả điều khoản ĐỊNH NGHĨA: đèn tín hiệu, vạch kẻ đường, báo hiệu)
   c) TRUY VẤN CHẾ TÀI — mô phỏng công thức xử phạt của Nghị định:
      "phạt tiền đối với người điều khiển [loại xe] [hành vi]" /
      "trừ điểm giấy phép lái xe [hành vi]" / "tước quyền sử dụng giấy phép lái xe"

3. LUÔN GHI RÕ LOẠI PHƯƠNG TIỆN trong truy vấn (mức phạt phụ thuộc loại xe:
   xe ô tô ≠ xe mô tô ≠ xe gắn máy ≠ xe đạp ≠ người đi bộ).

4. NẾU CÓ MÃ BIỂN BÁO (ví dụ P.102, R.412, DP.135): ghi NGUYÊN VĂN mã biển vào truy vấn.

5. KHÔNG bịa ra con số (mức tiền, kích thước, số điểm). Truy vấn dùng để TÌM điều khoản,
   không phải để đoán nội dung điều khoản.

6. KHÔNG dùng từ để hỏi (ai, gì, bao nhiêu, như thế nào). Truy vấn là CỤM MÔ TẢ,
   không phải câu hỏi.

7. CHỈ dùng thông tin từ danh sách thực thể có độ tin cậy cao. Không suy diễn thêm
   thực thể không được quan sát.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm:
{
  "queries": [
    {"text": "...", "type": "conduct|rule|penalty",
     "intent": "behavior_identification|required_behavior|penalty_amount|supplementary_consequence",
     "vehicle_mentioned": "..."}
  ]
}
```

## 2. Few-shot examples (append to the system prompt)

> Example 2 deliberately teaches the lesson from the empirical probe: precise corpus
> vocabulary ("chiều rộng giá đèo hàng") beats generic legalese ("quá khổ giới hạn").

```text
VÍ DỤ 1
Thực thể: [xe mô tô hai bánh — người điều khiển không đội mũ bảo hiểm; chở thêm 1 người
cũng không đội mũ]
Câu hỏi: Người điều khiển xe trong ảnh bị xử phạt như thế nào?
{
  "queries": [
    {"text": "người điều khiển xe mô tô hai bánh xe gắn máy không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng quy cách",
     "type": "conduct", "intent": "behavior_identification", "vehicle_mentioned": "xe mô tô hai bánh"},
    {"text": "phạt tiền đối với người điều khiển xe mô tô xe gắn máy không đội mũ bảo hiểm chở người không đội mũ bảo hiểm",
     "type": "penalty", "intent": "penalty_amount", "vehicle_mentioned": "xe mô tô hai bánh"}
  ]
}

VÍ DỤ 2
Thực thể: [xe gắn máy — chở hàng hóa vượt quá bề rộng thân xe, cao hơn đầu người lái]
Câu hỏi: Hành vi trong ảnh vi phạm quy định nào?
{
  "queries": [
    {"text": "xe mô tô xe gắn máy xếp hàng hóa vượt quá chiều rộng giá đèo hàng vượt quá chiều cao quy định",
     "type": "rule", "intent": "behavior_identification", "vehicle_mentioned": "xe gắn máy"},
    {"text": "người điều khiển xe mô tô xe gắn máy chở hàng hóa xếp hàng hóa trên xe vượt quá giới hạn quy định",
     "type": "conduct", "intent": "behavior_identification", "vehicle_mentioned": "xe gắn máy"},
    {"text": "phạt tiền người điều khiển xe mô tô xe gắn máy xếp hàng hóa vượt quá giới hạn",
     "type": "penalty", "intent": "penalty_amount", "vehicle_mentioned": "xe gắn máy"}
  ]
}

VÍ DỤ 3
Thực thể: [xe ô tô con — dừng đè lên vạch dừng tại nút giao; đèn tín hiệu: màu đỏ]
Câu hỏi: Người lái xe có bị trừ điểm giấy phép lái xe không?
{
  "queries": [
    {"text": "người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông không chấp hành vạch kẻ đường",
     "type": "conduct", "intent": "behavior_identification", "vehicle_mentioned": "xe ô tô"},
    {"text": "vạch kẻ đường vị trí dừng lại tín hiệu đèn màu đỏ",
     "type": "rule", "intent": "required_behavior", "vehicle_mentioned": "xe ô tô"},
    {"text": "trừ điểm giấy phép lái xe người điều khiển xe ô tô không chấp hành hiệu lệnh đèn tín hiệu giao thông",
     "type": "penalty", "intent": "supplementary_consequence", "vehicle_mentioned": "xe ô tô"}
  ]
}
```

## 3. REFINE variant (slot-targeted re-query)

Used when ASSESS reports unfilled slots. Same system prompt plus:

```text
Các truy vấn trước chưa tìm được điều khoản cho: {unfilled_slots}.
Các điều khoản ĐÃ tìm được (KHÔNG cần tìm lại): {found_citations}.
Các truy vấn ĐÃ thử (KHÔNG lặp lại): {tried_queries}.

Hãy tạo 1–2 truy vấn MỚI, KHÁC HẲN các truy vấn đã thử, chỉ nhắm vào các nội dung còn thiếu:
- Thiếu "penalty"          → dùng công thức chế tài: "phạt tiền ... đối với người điều khiển [xe] ..."
- Thiếu "supplementary"    → "trừ điểm giấy phép lái xe ..." / "tước quyền sử dụng giấy phép lái xe ..."
- Thiếu "violated_behavior"→ đổi cách diễn đạt hành vi: dùng từ đồng nghĩa trong văn bản pháp luật,
                              hoặc truy vấn điều khoản ĐỊNH NGHĨA liên quan (đèn tín hiệu, vạch kẻ
                              đường, biển báo) thay vì điều khoản hành vi.
Cùng định dạng JSON như trên.
```

## 4. Empirical basis (probes run on the real corpus, BM25, 2026-07-06)

| Probe | Result | Prompt consequence |
|---|---|---|
| Scene vs hand-guessed statute phrasing, recall@5 vs category gold | Statute phrasing won on đèn tín hiệu/ngược chiều, **LOST on cồng kềnh & sai làn** (wrong guessed vocab: truck-overload terms vs "chiều rộng giá đèo hàng"; missed definitional Điều 10/11 chunks) | Rule 1 teaches **verified** corpus vocabulary; Example 2 encodes the cồng kềnh lesson |
| Gold-chunk inspection | Category gold mixes **conduct** clauses (NĐ), **duty/definition** clauses (Luật Đ10/Đ11: báo hiệu, vạch kẻ đường), and **penalty** clauses | Rule 2's three query types; Example 3 includes a definitional query |
| 3-query union vs best single query | Union ≥ single in 4/6 categories, never worse | Mandatory 2–3 decomposed queries |
| Corpus penalty formula | 187 chunks match "đồng đến"; formula "Phạt tiền từ X đồng đến Y đồng đối với người điều khiển xe … hành vi" | Penalty-type query mirrors the formula (Rule 2c) |
| Vehicle-specific fine scales | Same conduct, different Điều per vehicle (ô tô Đ6 vs mô tô Đ7 …) | Rule 3: vehicle always in the query |

Caveats: probes used whitespace BM25 (no VN word segmentation) and category-level gold unions,
not per-item gold — directional evidence only. The real per-item evaluation is experiment R (M1),
which will also A/B-test this prompt: **FORMULATE-with-vocab-table vs FORMULATE-bare** (same
model, same retriever) to measure what the taught vocabulary is worth. If the vocab table wins,
that is a reportable finding ("statute-vocabulary grounding in query formulation adds X recall").

## 5. Placement & invariants

- FORMULATE input = `LegalEntity` list + question text. **No options anywhere** (I1);
  `assert_no_option_leakage` still runs on the final assembled prompt.
- The vocabulary table (Rule 1) is a *corpus-derived artifact*: regenerate it when the corpus
  changes (consistent with I5). On QCVN ingestion, extend it with sign-code phrasing rows.
- E3 (leakage probe) uses the SAME prompt with options appended to the input — the manipulation
  stays a one-flag difference, not a different prompt.
