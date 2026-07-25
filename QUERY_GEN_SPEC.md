# QUERY_GEN_PLAN.md — Query Generation

v3 · simplified · MCQ accuracy only
Detailed version with the discarded alternatives: `QUERY_GEN_PLAN_v2_detailed.md`

Two mechanisms: **route the question to a document**, then **close the citation graph**. Nothing else.

---

## Corpus: three partitions

Already separable by `chunk_id` prefix, no new data needed.

```
LUAT_36_2024   441 chunks   rules and obligations   "phải / không được"
ND_168_2024    334 chunks   sanctions               "Phạt tiền từ" (56% of chunks; 0% of LUAT)
QCVN_41_2024   296 chunks   sign and marking semantics
```

QCVN stays in the corpus. It was a distractor under global top-k — it held 35.3% of top-5 slots for items
that never needed it — but routing removes that by construction, and keeping it preserves gold coverage for
the 133 items whose only gold is QCVN.

---

## Step 1 — Perceive, then route

**PERCEIVE runs first and its output feeds the router.** The pipeline already runs it; the router just has to
read it.

Routing on question text alone fails on generic questions. Item 129 asks only *"Chiếc ô tô màu trắng ở phía
bên trái ảnh đang thực hiện hành vi nào sau đây?"*, yet its gold includes `QCVN_41_2024_PLB_3` — the
prohibitory-sign semantics. The sign is in the image, not in the sentence.

| question type | n | falls to the default route |
|---|---|---|
| generic (`hành vi nào` / `đang thực hiện`) | 444 | **89.9%** |
| all others | 1,945 | 24.1% |

For those 444 items the question carries no routing signal at all, and 132 of them need QCVN, which the
default `{LUAT, ND}` never reaches. The same holds for the query itself: a generic question contains no
description of the violation, so perception is the only source of query content — routing is reusing
something the pipeline already computed.

```python
ROUTE_ND   = r"(mức phạt|phạt bao nhiêu|bị phạt|số tiền|xử phạt|chế tài|trừ.{0,12}điểm|tước quyền|giấy phép lái xe)"
ROUTE_QCVN = r"(biển|vạch|đèn tín hiệu|làn đường|lề đường|dải phân cách)"
ROUTE_LUAT = r"(phải thực hiện|phải làm gì|xử lý như thế nào|phải chấp hành|có được phép|theo quy định|nguyên tắc|quy tắc)"

def route(question: str, perception: Perception) -> set[str]:
    text = question + " " + perception.observed      # sign/marking mentions come from the image
    r = {d for d, p in [("ND", ROUTE_ND), ("QCVN", ROUTE_QCVN), ("LUAT", ROUTE_LUAT)]
         if re.search(p, text, re.I)}
    return r or {"LUAT", "ND"}
```

Note `ROUTE_QCVN` matches bare `biển`, not `biển báo`. Narrowing it to `biển báo|biển hiệu|biển số` drops
QCVN routing recall from 62.7% to 44.2%, because the corpus and the questions both say *biển cấm*.

Router recall on question text alone, measured against gold:

| document | routed when gold needs it |
|---|---|
| ND 168 | 94.1% |
| Luật 36 | 84.7% |
| QCVN | 62.7% |

76.3% of items get every document their gold requires. Perception input should lift the QCVN arm; it will not
move MCQ much, because QCVN-governed items are answered at 84.3% with no statute at all and injecting the
gold provision moves them by −0.7%. Do it because it is nearly free and structurally right, not for the
points.

## Step 2 — Retrieve within the routed documents

Split the budget evenly across the routed documents and retrieve inside each independently. **Not** a global
top-k over the merged corpus.

| configuration (budget 12) | full_recall |
|---|---|
| global top-12 | 14.3% |
| **hard routing** | **20.8%** |
| soft routing, 1 slot reserved per unrouted document | 19.7% |

Reserving a slot for unrouted documents is worse than not doing it. Route hard.

---

## Step 3 — Close the citation graph

Build once, offline, from the corpus. Parse every internal citation into a directed graph:

```
"điểm b khoản 3 Điều này"  -> containing chunk's own article, clause 3, point b
"khoản 5 Điều 7"           -> same document, article 7, clause 5
```

**Traverse both directions.** A conduct provision and the sanction chunk that cites it form a pair, and the
graph makes the pair reachable from either end:

| retrieval lands on | traversal | recovers the other end |
|---|---|---|
| conduct chunk | reverse | 75.3% |
| sanction chunk | forward | 75.3% |

Which end retrieval lands on follows from step 1: a question about a violation reaches the conduct provision,
a question about a fine or point deduction reaches the sanction chunk.

This matters because 62 chunks in the corpus are pure cross-references — the most-cited gold chunk in the
benchmark, `ND_168_2024_D7_K13`, reads in full *"Thực hiện hành vi quy định tại điểm b khoản 3; khoản 5; ...
bị trừ điểm giấy phép lái xe 06 điểm"*. It shares no vocabulary and no meaning with any description of the
conduct it sanctions. **29.7% of items need one**, and no query reaches them. Only the graph does.

```python
def expand(retrieved, graph, query, retriever, max_rev=8, max_fwd=8):
    """1 hop. Additive — expanded chunks rank below retrieved ones, never displace them.
    Reverse: take all (fan-in <= 11).
    Forward: rerank the fan-out by `query` and keep the top max_fwd."""
```

Forward needs reranking because a sanction chunk fans out to the conduct provisions it covers, and the query
describes conduct, so it discriminates among them: top-5 of the reranked pool captures the gold conduct chunk
**84.6%** of the time. Reverse does not need it.

Tag chunks `source: "retrieval" | "graph_fwd" | "graph_rev"` so the ablation can attribute gains.

---

## Step 4 — Measure

`scripts/eval_retrieval.py`, BM25, no GPU.

- Primary: **MCQ accuracy**, all 2,414 items.
- Diagnostic: **`full_recall@k`** = `gold ⊆ retrieved_topk`, over the 2,389 items with gold.
  Plain recall is uninformative — most items need multiple chunks and a partial set cannot be answered.
- Break out by `needs_xref` (716 items) and `source`.

| run | configuration |
|---|---|
| A | current pipeline, global top-k — baseline |
| B | A + routing |
| C | B + graph expansion |
| E0 | gold injected — **labeled ceiling, never a system result** |

C − B is the paper's number. Reference band for qwen3vl-4b on the legal tier: zero-shot 64.2%,
E0-oracle 82.2%.

---

## Gates

| condition | action |
|---|---|
| graph recovers < 42% of the 716 (reverse, from conduct gold) | parser is broken — debug, do not ship |
| B − A < 2 pts MCQ | drop routing, keep global top-k |
| perception-fed routing changes MCQ by < 1 pt | keep it anyway — it costs nothing and fixes the generic-question case |
| C − B < 3 pts MCQ on the 716-item subset | expansion is underperforming — investigate before continuing |

All thresholds and budgets are set on a development split partitioned **by image**, not by item — items from
the same image share gold.

---

## Step 5 — Dense retrieval as a second engine

Run only after steps 1–4 are measured. It changes how well the pipeline retrieves, not what the paper claims.

### Where it is aimed

Gold rank **within its own routed partition**, raw question as query:

| partition | gold refs | top-5 | median rank | beyond rank 20 |
|---|---|---|---|---|
| ND 168 (334 chunks) | 1,077 | 32.1% | **13** | 39.6% |
| Luật 36 (441 chunks) | 1,654 | 27.9% | **64** | 64.6% |
| QCVN (296 chunks) | 752 | 12.8% | **68** | 77.5% |
| cross-reference (inside ND) | 938 | 7.2% | 81 | 80.1% |

BM25 already works on the Decree: its provisions describe conduct in concrete terms that overlap the
question's vocabulary. **Luật 36 is where it fails** — the largest gold population, and a median rank of 64
out of 441. The Law states obligations abstractly (*"phải chấp hành hiệu lệnh"*) while questions describe a
concrete scene, which is exactly the semantic gap a dense encoder is built to cross.

Dense will **not** help the cross-reference population. Those chunks are semantically about point deductions,
not about the violation, so a dense encoder places them no closer than BM25 does. Step 3 is the only
mechanism that reaches them, and the two are independent.

All figures use the raw question as query, so they are a floor. With perception-fed queries the ranks improve
and the headroom shrinks — treat the table as targeting information, not as an expected gain.

### Model and indexing

BGE-M3. **Field selection differs from BM25 and is not optional:**

```python
# BM25:  index the full chunk text — it benefits from "Khoản 3, Điều 7, 168/2024/NĐ-CP"
# dense: index ONLY "Tên điều hoặc mục" + "Nội dung"
```

A median **43.6%** of each chunk's text is metadata boilerplate (`Văn bản:` / `Trích dẫn:` / `Ngữ cảnh:`),
and 284 of 774 non-QCVN chunks are more than half boilerplate. BM25 is immune — IDF flattens terms that
appear in every chunk. A dense encoder is not: it pools the boilerplate into the vector and collapses every
Decree chunk toward the same point.

BGE-M3 emits dense, sparse, and ColBERT representations from one pass. The sparse head is a learned lexical
scorer, so `hybrid` can come from a single model rather than two indexes — worth trying before wiring BM25
and dense together by hand.

### Integration

One flag behind `agent/retrieval/base.py`:

| engine | index |
|---|---|
| `lexical` | BM25 over full text |
| `dense` | BGE-M3 over heading + body |
| `hybrid` | both, fused by RRF |

Routing (step 1) is unchanged — it selects partitions, not scorers. Graph expansion (step 3) reranks the
forward fan-out with **whichever engine is active**, not always BM25.

1,071 chunks encode in seconds. The real costs are the model download and the integration.

### What to run

Do not run the full grid. Four cells answer the question:

| | lexical | dense | hybrid |
|---|---|---|---|
| routing + graph (run C) | baseline | ✓ | ✓ |
| per-partition recall, LUAT only | ✓ | ✓ | |

The LUAT-only row is the diagnostic: if dense does not beat BM25 there, it will not help anywhere, since ND
is already well served and QCVN barely affects MCQ.

### Gates

| condition | action |
|---|---|
| dense − lexical < 2 pts MCQ at run C | keep `lexical`, report the comparison, move on |
| dense does not beat lexical on LUAT partition recall | stop — the mechanism is not doing what it was chosen for |
| hybrid ≈ dense | ship whichever is simpler to run |

---

## Also optional

**Anchor snapping** against a bank of 910 act phrases parsed from the Decree, forcing queries into statutory
phrasing. Note the interaction: snapping exists because models asked for statutory phrasing often cannot
produce it. If dense retrieval bridges that gap on its own, snapping is redundant — check before building it.