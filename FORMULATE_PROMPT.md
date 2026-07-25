# FORMULATE_PROMPT.md — Statute-Query Generation Prompts for VNTraffic-Agent

v2 · 2026-07-24 · Companion to `QUERY_GEN_SPEC.md` §6.2 — supersedes v1's three-way
conduct/rule/penalty split with the two-channel (P=ND_168_2024, R=LUAT_36_2024) design.
Three changes from v1, per QUERY_GEN_SPEC.md §6.2: (1) sign-code/channel-S instructions
deleted — QCVN 41 is no longer in the retrieval corpus; (2) the vocabulary table below was
re-verified against the pruned 775-chunk corpus (LUAT_36_2024 441 + ND_168_2024 334) — a
different set from v1's 777-chunk corpus, not a subset — and three entries were corrected
where their old target phrasing no longer matches actual corpus text (see inline notes);
(3) output schema gains `channel` (§6.2.3), replacing the old `type`/`intent` pair.

**Design is empirically grounded** (see §4): scene-language vs statute-language BM25 probes on
the corpus showed (a) naive scene phrasing under-retrieves, (b) *hand-guessed*
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

ĐẦU VÀO: danh sách thực thể quan sát được trong ảnh (phương tiện, hành vi, biển báo,
đèn tín hiệu, vạch kẻ đường, hàng hóa, ...) và câu hỏi pháp lý.

ĐẦU RA: 2–3 câu truy vấn, MỖI CÂU MỘT KÊNH, theo đúng văn phong của văn bản pháp luật.

QUY TẮC VIẾT TRUY VẤN:

1. DÙNG TỪ NGỮ CỦA VĂN BẢN PHÁP LUẬT, không dùng từ ngữ đời thường (bảng dưới đã được đối
   chiếu lại với kho ngữ liệu 775 chunk hiện tại — v2, 2026-07-24):
   - "xe máy"            → "xe mô tô, xe gắn máy" (Luật phân biệt "xe mô tô hai bánh",
                            "xe mô tô ba bánh", "xe gắn máy")
   - "vượt đèn đỏ"       → "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
   - "đi ngược chiều"    → "đi ngược chiều của đường một chiều, đi ngược chiều trên
                            đường có biển 'Cấm đi ngược chiều'"
   - "chở đồ cồng kềnh"  → "xếp hàng hóa vượt quá chiều rộng giá đèo hàng, vượt quá
                            chiều cao quy định" (SỬA v2: bỏ "trên xe" — cụm này không
                            khớp corpus hiện tại; "xếp hàng hóa vượt quá chiều rộng giá
                            đèo hàng" mới khớp, LUAT_36_2024_D33_K5)
   - "đi sai làn"        → "đi không đúng phần đường hoặc làn đường quy định"
   - "dừng quá vạch"     → "phải dừng lại trước vạch dừng" (SỬA v2: mục tiêu cũ "không
                            chấp hành vạch kẻ đường; vạch chỉ vị trí dừng lại" không có
                            trong corpus hiện tại — cụm nghĩa vụ đã xác nhận thuộc
                            LUAT_36_2024_D11_K4/D24_K1; chưa xác nhận được vế xử phạt
                            (NĐ) tương ứng, cần kiểm tra thêm nếu câu hỏi cần khoản phạt)
   - "không đội mũ"      → "không đội "mũ bảo hiểm cho người đi mô tô, xe máy" hoặc đội
                            "mũ bảo hiểm cho người đi mô tô, xe máy" không cài quai đúng
                            quy cách" (SỬA v2: NĐ 168 dùng đúng cụm có ngoặc kép này,
                            không phải "mũ bảo hiểm" trơn — xem ND_168_2024_D7_K2)
   - "bằng lái"          → "giấy phép lái xe"; "bị trừ điểm bằng" → "trừ điểm giấy phép lái xe"

2. MỖI TRUY VẤN THUỘC MỘT TRONG HAI KÊNH (chọn 2–3 truy vấn phù hợp với câu hỏi; cả hai
   kênh LUÔN được tạo — QUERY_GEN_SPEC.md §3: 50.1% số câu hỏi cần điều khoản từ cả hai
   văn bản, và một lệnh gọi FORMULATE đã tạo cả hai kênh nên kênh thứ hai không tốn thêm
   lệnh gọi nào):
   a) KÊNH P (penalty) — nhắm vào NĐ 168/2024/NĐ-CP, gồm cả hành vi vi phạm LẪN chế tài
      (hai việc này nằm chung một văn bản/cấu trúc khoản):
      "người điều khiển [loại xe] [hành vi vi phạm bằng từ ngữ pháp luật]" /
      "phạt tiền đối với người điều khiển [loại xe] [hành vi]" /
      "trừ điểm giấy phép lái xe [hành vi]" / "tước quyền sử dụng giấy phép lái xe"
   b) KÊNH R (rule) — nhắm vào Luật 36/2024/QH15: nghĩa vụ, quy tắc, định nghĩa:
      "người tham gia giao thông phải ..." / "[loại xe] không được ..."
      (nhiều câu hỏi cần cả điều khoản ĐỊNH NGHĨA: đèn tín hiệu, vạch kẻ đường, báo hiệu)

3. LUÔN GHI RÕ LOẠI PHƯƠNG TIỆN trong truy vấn (mức phạt phụ thuộc loại xe:
   xe ô tô ≠ xe mô tô ≠ xe gắn máy ≠ xe đạp ≠ người đi bộ).

4. KHÔNG bịa ra con số (mức tiền, kích thước, số điểm). Truy vấn dùng để TÌM điều khoản,
   không phải để đoán nội dung điều khoản.

5. KHÔNG dùng từ để hỏi (ai, gì, bao nhiêu, như thế nào). Truy vấn là CỤM MÔ TẢ,
   không phải câu hỏi.

6. CHỈ dùng thông tin từ danh sách thực thể có độ tin cậy cao. Không suy diễn thêm
   thực thể không được quan sát.

ĐỊNH DẠNG ĐẦU RA — JSON, không giải thích thêm (§6.2.3 — "channel" thay cho "type"/"intent" cũ):
{
  "queries": [
    {"text": "...", "channel": "P|R", "vehicle_mentioned": "..."}
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
    {"text": "người điều khiển xe mô tô hai bánh xe gắn máy không đội \"mũ bảo hiểm cho người đi mô tô, xe máy\" hoặc đội \"mũ bảo hiểm cho người đi mô tô, xe máy\" không cài quai đúng quy cách",
     "channel": "P", "vehicle_mentioned": "xe mô tô hai bánh"},
    {"text": "phạt tiền đối với người điều khiển xe mô tô xe gắn máy không đội mũ bảo hiểm chở người không đội mũ bảo hiểm",
     "channel": "P", "vehicle_mentioned": "xe mô tô hai bánh"}
  ]
}

VÍ DỤ 2
Thực thể: [xe gắn máy — chở hàng hóa vượt quá bề rộng thân xe, cao hơn đầu người lái]
Câu hỏi: Hành vi trong ảnh vi phạm quy định nào?
{
  "queries": [
    {"text": "xe mô tô, xe gắn máy không được xếp hàng hóa vượt quá chiều rộng giá đèo hàng vượt quá chiều cao quy định",
     "channel": "R", "vehicle_mentioned": "xe gắn máy"},
    {"text": "người điều khiển xe mô tô xe gắn máy xếp hàng hóa vượt quá chiều rộng giá đèo hàng",
     "channel": "P", "vehicle_mentioned": "xe gắn máy"},
    {"text": "phạt tiền người điều khiển xe mô tô xe gắn máy xếp hàng hóa vượt quá giới hạn",
     "channel": "P", "vehicle_mentioned": "xe gắn máy"}
  ]
}

VÍ DỤ 3
Thực thể: [xe ô tô con — dừng đè lên vạch dừng tại nút giao; đèn tín hiệu: màu đỏ]
Câu hỏi: Người lái xe có bị trừ điểm giấy phép lái xe không?
{
  "queries": [
    {"text": "người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
     "channel": "P", "vehicle_mentioned": "xe ô tô"},
    {"text": "tín hiệu đèn màu đỏ là cấm đi, người tham gia giao thông phải dừng lại trước vạch dừng",
     "channel": "R", "vehicle_mentioned": "xe ô tô"},
    {"text": "trừ điểm giấy phép lái xe người điều khiển xe ô tô không chấp hành hiệu lệnh đèn tín hiệu giao thông",
     "channel": "P", "vehicle_mentioned": "xe ô tô"}
  ]
}
```

## 3. REFINE variant (channel-targeted re-query)

**v2 note:** the trigger logic that decides *when* to call this (structural gap → off-vocabulary
→ ASSESS, in that order) is `QUERY_GEN_SPEC.md` §6.4's `Assessment{missing_channel: "P"|"R"|None}`
— an orchestration/schema change in the refine loop itself, not this prompt file, and out of scope
for the query-GENERATION work this file covers. What changes here is only the re-query instruction
once a channel is known to be missing:

```text
Kênh còn thiếu điều khoản: {missing_channel}  (P = NĐ 168/2024/NĐ-CP, R = Luật 36/2024/QH15)
Các điều khoản ĐÃ tìm được (KHÔNG cần tìm lại): {found_citations}.
Các truy vấn ĐÃ thử (KHÔNG lặp lại): {tried_queries}.

Hãy tạo 1–2 truy vấn MỚI cho ĐÚNG kênh {missing_channel}, KHÁC HẲN các truy vấn đã thử:
- Thiếu kênh P (chế tài/hành vi)  → dùng công thức chế tài: "phạt tiền ... đối với người điều
                                     khiển [xe] ..." / "trừ điểm giấy phép lái xe ..." /
                                     "tước quyền sử dụng giấy phép lái xe ..."
- Thiếu kênh R (quy tắc/nghĩa vụ) → đổi cách diễn đạt hành vi sang thể nghĩa vụ: dùng từ đồng
                                     nghĩa trong văn bản pháp luật, hoặc truy vấn điều khoản
                                     ĐỊNH NGHĨA liên quan (đèn tín hiệu, vạch kẻ đường, báo hiệu)
                                     thay vì điều khoản hành vi.
Cùng định dạng JSON như trên, "channel" PHẢI bằng {missing_channel}.
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
- The vocabulary table (Rule 1) is a *corpus-derived artifact*: regenerate/re-verify it whenever
  the corpus changes (consistent with I5) — v2's re-verification against the 775-chunk pruned
  corpus (this file's header) is itself an instance of that rule, not a one-off.
- QCVN 41 is permanently out of the retrieval corpus (`QUERY_GEN_SPEC.md` §2, §11) — this is not
  "not yet ingested," it's excluded by design. Do not extend the vocabulary table with sign-code
  rows; channel-S is deleted, not paused.
- E3 (leakage probe) uses the SAME prompt with options appended to the input — the manipulation
  stays a one-flag difference, not a different prompt.
- **Not covered by this file**: `channel`'s two values map 1:1 onto `agent/query_gen.py`'s
  `CHANNEL_DOC` (P→ND_168_2024, R→LUAT_36_2024) — keep them in sync if either changes. The actual
  LLM call that runs this prompt (an `LLMClient.formulate_channels()`-shaped method) is not yet
  wired into `agent/llm.py`/`agent/llm_vlm.py`; `agent/query_gen.py::parse_channel_queries()` is
  the seam that call's JSON output should be passed through.
