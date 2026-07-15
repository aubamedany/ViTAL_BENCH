# AGENT_PLAN.md — VNTraffic-Agent

**Design & implementation plan** · v1 · 2026-07-06
Companion to `EXPERIMENT_PLAN.md` (what we measure) and `research-proposal.md` (what we claim).
This document specifies **what we build**.

---

## 0. What the agent is, in one paragraph

VNTraffic-Agent is a **training-free, model-agnostic, two-phase** pipeline that turns a
traffic-scene image + a legal question into a **`LegalDetermination`** — a structured verdict
citing only statute chunks it actually retrieved — and only then maps that verdict onto the
multiple-choice options. The defining property is the **option wall**: Phase 1 never sees the
answer options, and Phase 2 never retrieves. Everything in this document exists to make that
wall real rather than a promise in a prompt.

---

## 1. Design goals and invariants

The agent must satisfy five invariants. Each is testable, and each maps to a paper claim.

| # | Invariant | Why | Enforced by |
|---|-----------|-----|-------------|
| **I1** | **Option-blindness.** No answer option text reaches any Phase-1 module. | Kills answer leakage into retrieval (Claim K2) | Type system: Phase-1 functions never receive the options; assert at runtime |
| **I2** | **Citation closure.** Every `chunk_id` in the final determination appears in the union of retrieved sets. | No hallucinated law | Post-hoc validator; violations → determination rejected, re-emitted |
| **I3** | **No retrieval in Phase 2.** SELECT maps a *frozen* determination onto options. | Prevents back-door leakage after the wall | Phase 2 has no retriever handle |
| **I4** | **Training-free.** No gradient touches any weight. | Amendment-robustness argument (Claim K5) | Only inference APIs used |
| **I5** | **Corpus-swappability.** Changing the statute DB changes behavior with zero code change. | The deployment story | Corpus version string is a config field; index rebuilt from JSON |

> **Bonus property worth stating in the paper:** because Phase 1 never sees options, it is
> *also* structurally immune to the option-length artifact (correct option is uniquely longest
> in 57% of law items). Phase 2 is not immune — which is another reason the data-level
> length rebalancing gate must still land. Option-blindness protects the *reasoning*, not the
> *selection*.

---

## 2. Architecture

```
                        ┌───────────────────── PHASE 1: OPTION-BLIND ─────────────────────┐
   image ──────────────►│                                                                  │
   question ───────────►│  PERCEIVE ──► FORMULATE ──► RETRIEVE ──► ASSESS ──┐              │
                        │      ▲                          ▲                 │              │
                        │      │                          │            sufficient?         │
                        │      │                     REFINE ◄── no ────────┘              │
                        │      │                                          │ yes            │
                        │   SceneDescription                              ▼               │
                        │                                        LegalDetermination        │
                        └──────────────────────────────────────────────┬───────────────────┘
                                                                       │  ═══ OPTION WALL ═══
   options A/B/C/D ────────────────────────────────────────────────────┼──────────┐
                                                                       ▼          ▼
                        ┌───────────────── PHASE 2: SELECT (no retrieval) ─────────────────┐
                        │            SELECT(determination, options) ──► answer ∈ {A,B,C,D} │
                        └──────────────────────────────────────────────────────────────────┘
```

### 2.1 Phase-1 modules

**PERCEIVE (v2 — entity-directed, open-vocabulary)** — the only module that sees pixels.
Two steps, both option-blind:

1. **ENUMERATE** — one VLM call: *"list every entity in this scene that could matter legally"*
   (vehicles, signs, lights, markings, persons, cargo, obstructions, … — examples in the prompt
   are explicitly marked *illustrative, not exhaustive*). The rationale: statutes are written in
   *elements* — a subject, a conduct, circumstances — so perception's job is legal-element
   extraction, not scene captioning.
2. **INSPECT** — for each enumerated entity, one short *targeted* VLM question against the image
   ("What state is the traffic light in?", "Is the rider wearing a helmet?", "What is the sign's
   shape/color/code?"). Targeted questions reliably beat one describe-everything pass, especially
   at 1–8B scale where undirected captioning goes vague. Each answer comes back as a typed
   `LegalEntity` with its own confidence.

Design rules:
- **Open vocabulary everywhere.** `kind` and all attributes are free text in the model's own
  words. The schema constrains *structure*, never *vocabulary* — a closed enum makes the
  benchmark's own `cong_kenh` (bulky cargo) and legally-distinct cases like xe đạp điện
  unrepresentable, and forces wrong picks that then poison retrieval filters (fail-closed, the
  unacceptable failure mode).
- **The `legally_relevant_because` field does the filtering an enum would have done**: an entity
  the model cannot legally justify is dropped from query formulation (but kept in the log).
- **Hallucination guard:** entities below a confidence threshold are excluded from FORMULATE —
  a phantom "red light" retrieves red-light provisions and yields a confidently wrong verdict;
  a missed entity merely under-retrieves. Asymmetric costs, asymmetric policy.
- **Corpus-derived hints (optional, consistent with I5):** the illustrative examples in the
  ENUMERATE prompt may be *derived from the corpus itself* (the entities NĐ-168/Luật 36 actually
  condition on), so guidance is statute-grounded and updates when the corpus does. Hints, never
  a closed list.
- **No external object detector.** Same VLM, better-directed prompts — an external detector
  breaks the training-free story and the model-agnostic claim for marginal gain on this corpus.

**FORMULATE** — LLM call, extracted `LegalEntity` list + question → 1–3 retrieval queries in
*statute language*, composed from **entity–element pairs**: (xe mô tô, không đội mũ bảo hiểm) →
the NĐ-168 phrasing; (đèn tín hiệu: đỏ, vượt) → red-light provisions. When QCVN 41 is ingested,
a recognized sign *code* (e.g. "P.102") becomes a near-exact lexical key into it — the biggest
single win available for the worst-gradeable sign categories. Also emits a `query_intent`
∈ {behavior_identification, required_behavior, penalty_amount, supplementary_consequence},
which drives retrieval filters (below).

**RETRIEVE** — no LLM. Hybrid dense+sparse over the statute corpus with hard filters (§3).

**ASSESS** — LLM call, question + retrieved chunks (no options) → a **structured
`SufficiencyReport`**, not a bare boolean. This is a deliberate upgrade: a single self-reported
`sufficient=true` is the LLM grading its own homework, and a reviewer will (rightly) distrust it.
Instead ASSESS decomposes sufficiency into the *slots the query intent requires* and, for each
slot, names the specific `chunk_id` that supplies it. Because each claimed slot points at a
chunk, ASSESS's own accuracy is **checkable post-hoc** against the chunk text — so we can report
*ASSESS precision* (how often "sufficient" was actually sufficient) as a validated number rather
than trusting the flag. `overall_sufficient` is then a *derived* property (all required slots
filled), not a free-floating opinion. A determination is emitted when `overall_sufficient` or
budget is exhausted (then `confidence` low, `insufficient_evidence: true`).

**REFINE** — LLM call, given the *unfilled slots* from the `SufficiencyReport` (not a vague
`missing` string), rewrites/expands queries targeting exactly those slots. Loop back to RETRIEVE.
Bounded by budget `B` (default 3). **Early-stop guard:** if two consecutive iterations retrieve
substantially the same chunk set (Jaccard ≥ 0.8), stop — more iterations won't help; record
`stop_reason ∈ {sufficient, budget, no_progress}`. Every iteration's queries, retrieved sets, and
stop reason are logged (feeds the budget ablation A1 and defends the choice of B=3).

**Emit `LegalDetermination`** — the frozen artifact that crosses the wall.

### 2.2 Phase-2 module

**SELECT** — LLM call, `LegalDetermination` + options → letter. Explicit instruction: *decide
using only the determination; if no option matches, choose the closest and set
`match_quality: poor`*. `match_quality` is logged and becomes a diagnostic: a high
poor-match rate means Phase 1 is under-determining, not that SELECT is broken.

---

## 3. Retrieval layer (the part that decides whether any of this works)

Corpus: 777 clause-level chunks (LUAT_36_2024: 441, ND_168_2024: 334, QCVN_41_2024: 2 stubs).
Small enough that **exact search is correct and free** — no ANN, no approximation.

### 3.1 Key structural fact to exploit

A khoản chunk is **self-contained**: `ND_168_2024_D7_K3` holds both the penalty range
("Phạt tiền từ 600.000 đồng đến 800.000 đồng") *and* every điểm sub-item (a, b, c…) describing
the behaviors that incur it. Consequences:

- Retrieving one clause chunk answers **behavior + penalty together**. Penalty questions do not
  need a second hop.
- The điểm→khoản citation fallback is **semantically sound**, not a lossy approximation: the
  clause text contains the point text. Say this explicitly in the paper; it converts an apparent
  data defect into a justified design choice.
- Corpus signal inventory: 162 chunks contain monetary ranges, 37 contain `trừ điểm`
  (license-point deduction), 22 contain `tước quyền sử dụng` (license suspension).

### 3.2 Retrieval and the retriever comparison (experiment R)

The agent's retriever is a **swappable component behind one interface** (`retrieve(query, k,
filters) -> list[RetrievedChunk]`). This is deliberate: retriever choice is an empirical
question we answer with a dedicated experiment, not a design assumption. The agent loop, prompts,
filters, and fusion are held identical; only the scoring function underneath changes.

**Retrievers under comparison (experiment R, run standalone at M1 before any agent iteration):**

| ID | Retriever | Type | Why it's in the comparison |
|----|-----------|------|----------------------------|
| R-BM25 | BM25 (Okapi) over `content` | sparse/lexical | Statute language is legalistic and *lexical* — exact terms ("mũ bảo hiểm", "biển báo P.102", "trừ điểm") carry the signal. Strong, tuning-free, and the honest baseline every dense method must beat. |
| R-BGE | **BGE-M3** over `text_for_embedding` | dense multilingual | SOTA-class multilingual embedder with strong Vietnamese; supports dense + sparse + multi-vector from one model. Primary dense candidate. |
| R-E5 | multilingual-e5-base | dense multilingual | Widely-used baseline dense encoder; sanity anchor for BGE. |
| R-Vi | Vietnamese-specific encoder (e.g. `vietnamese-bi-encoder` / `sup-SimCSE-VietNamese`) | dense monolingual | Tests whether a VN-tuned encoder beats general multilingual on VN legal text. |
| R-Hybrid | best-dense ⊕ BM25 via RRF (k=60) | hybrid | Fusion of lexical + semantic; expected top performer. |
| R-Rerank | R-Hybrid → cross-encoder rerank (BGE-reranker-v2-m3) | hybrid+rerank | Only promoted if R-Hybrid shows recall@10 ≫ recall@3 (headroom a reranker can capture). |

Notes that matter for a fair comparison:
- **BGE-M3 is one model with three modes** (dense / sparse / multi-vector). Report dense-only for
  the clean dense-vs-sparse contrast, and optionally its native hybrid as a second hybrid point.
- **Same chunks, same field policy:** dense retrievers embed `text_for_embedding` (already
  formatted with citation + context + content); BM25 tokenizes `content`. Document this — it is a
  confound if one method silently gets richer text.
- **Vietnamese tokenization for BM25** is not optional: use a VN word segmenter (e.g. `pyvi` /
  `underthesea`) before BM25, or BM25 is crippled on compound words and the comparison is unfair
  to the lexical baseline. NFC-normalize first (the `Đ`/`đ` issue bites the tokenizer too).
- Corpus is 777 chunks → exact search for all dense methods; no ANN, no approximation, so the
  comparison reflects the *encoder*, not the index.

**What experiment R measures (per retriever):** recall@{1,3,5,10} and MRR of the **gold** chunks,
on the gradeable subset, reported overall **and per legal subtype** (penalty vs behavior vs
required vs supplementary) and per image category. Subtype breakdown is the interesting part:
the prediction is that BM25 wins on `penalty_amount` (monetary strings are lexical) while dense
wins on `behavior_identification` (paraphrase gap between scene language and statute language).
If that split appears, it is a finding on its own and a direct argument *for* the hybrid.

**How R feeds the rest of the paper:** the winner of R (call it R\*) becomes the agent's default
retriever for the main sweep. R is also the retrieval-recall ceiling from M1 — if R\* recall@5 is
low, grounding F1 cannot exceed it and the bottleneck is retrieval, not reasoning (which the
oracle-retrieval condition, §6.1, then confirms). Retriever choice is therefore reported twice:
as a standalone table (experiment R) and as an ablation on final grounded accuracy (does a better
retriever actually move the end-to-end number, or does the agent loop absorb the difference?).

### 3.3 Hard filters (free precision, from metadata)

- **`vehicle_type`**: PERCEIVE's open-vocabulary vehicle observation is mapped to corpus terms
  by `normalize_vehicle()` at retrieval time (§4). If mapped: keep chunks whose `vehicle_type`
  contains it **or is empty** (empty = generic, 55% of chunks; never filter those out) — prunes
  ~30–40% of the space at zero cost. If unmappable: **fail open, no filter** (the filter is an
  optimization, never a gate).
- **`document_id` by intent**: `penalty_amount` / `supplementary_consequence` → prefer
  `ND_168_2024` (the sanctions decree); `required_behavior` → prefer `LUAT_36_2024` (the
  rules-of-the-road law). Implement as a **score boost, not a hard exclusion** — 32% of items
  need ≥2 documents, so hard exclusion would break cross-document reasoning.
- **`effective_date`**: filter to provisions in force at a configurable `as_of` date
  (default 2025-01-01). Costs nothing now; it is the *mechanism* behind the amendment-robustness
  claim, so build it in and demo it.

### 3.4 `resolve_citation()` — shared by agent and evaluator

Resolution order: exact → case-folded → điểm→khoản fallback (regex must include `Đ`/`đ`, not
just `[A-Z]`) → unresolved. Emits a coverage report every run. Used in two places: the
evaluator (grounding metrics) and the agent's I2 validator (citation closure).

### 3.5 The QCVN 41 hole — agent-side consequence

`QCVN_41_2024_M1`/`M2` are 69- and 657-character stubs, not the sign regulation. 63 unique gold
citations point into it. **Until ingestion, the agent cannot retrieve sign provisions and will
correctly report `insufficient_evidence` on sign-dependent items** (`sai_bien_bao` 48%
gradeable, `di_vao_duong_cam` 36%). This is the *right* behavior — an honest agent abstains —
but it must not be silently scored as a reasoning failure. Two obligations:
1. Report `insufficient_evidence` rate per category alongside accuracy.
2. Either ingest QCVN (Path A) or pre-register the 974-item gradeable subset for grounding
   metrics (Path B). Never mix.

---

## 4. Data contracts

These are the interfaces. Enforce with `pydantic`; a schema violation is a bug, not a warning.

```python
class LegalEntity(BaseModel):               # PERCEIVE-v2 output unit — OPEN VOCABULARY
    kind: str                                # model's own label: "xe máy", "hàng hóa chở theo",
                                             # "vạch dừng", "đèn tín hiệu", ... never an enum
    description: str                         # what it is, scene language
    state: str | None                        # the legally-loaded attribute:
                                             # "đèn đỏ", "vượt quá thùng xe", "không đội mũ"
    confidence: float                        # gates use in FORMULATE (hallucination guard)
    legally_relevant_because: str            # forced justification — does the enum's old job

class SceneDescription(BaseModel):          # PERCEIVE output — never contains options
    entities: list[LegalEntity]              # open-vocabulary, entity-directed extraction
    observed_behavior: str                   # the conduct, in Vietnamese scene language
    scene_notes: str | None
```

**Vehicle-filter normalization (retrieval-time, NOT in the schema):** the corpus's
`vehicle_type` metadata uses six fixed values (`xe_mo_to`, `xe_o_to`, `xe_tai`, `xe_dau_keo`,
`xe_dap`, `nguoi_di_bo`). The constraint "match corpus vocabulary" belongs to the *filter*, not
to *perception*. A `normalize_vehicle(described_as) -> corpus_term | None` mapper — built from
the corpus's own observed values, so it updates when the corpus does (invariant I5) — translates
the model's free text ("xe máy", "mô tô", "motorbike") at retrieval time. **Fail-open rule:**
unmappable → `None` → no filter applied, retrieve over everything. Losing the filter on an odd
phrasing costs a little pruning; a wrong forced mapping silently excludes the correct statutes.
The filter is an optimization, never a gate.

```python
class RetrievalQuery(BaseModel):
    text: str                                # statute-language query
    intent: Literal["behavior_identification","required_behavior",
                    "penalty_amount","supplementary_consequence"]
    vehicle_filter: list[str]                # NORMALIZED corpus terms from normalize_vehicle();
                                             # empty list = fail-open, no filter applied
    doc_boost: list[str]                     # e.g. ["ND_168_2024"]

class RetrievedChunk(BaseModel):
    chunk_id: str; citation: str; content: str
    score: float; retrieved_at_iteration: int

class SlotEvidence(BaseModel):              # one required piece of the answer
    slot: Literal["violated_behavior","penalty","supplementary","required_behavior"]
    filled: bool
    supporting_chunk_id: str | None         # the chunk that supplies it (checkable post-hoc!)
    justification: str                       # short quote/paraphrase from that chunk

class SufficiencyReport(BaseModel):         # ASSESS output — replaces the bare bool
    required_slots: list[Literal["violated_behavior","penalty",
                                 "supplementary","required_behavior"]]  # set by query intent
    slot_evidence: list[SlotEvidence]
    overall_sufficient: bool                 # DERIVED: all required slots filled
    unfilled_slots: list[str]                # drives REFINE precisely

class LegalDetermination(BaseModel):        # THE ARTIFACT THAT CROSSES THE WALL
    violated_behavior: str | None            # None ⇒ no violation
    applicable_provisions: list[str]         # chunk_ids — MUST ⊆ retrieved (I2)
    penalty_min_vnd: int | None
    penalty_max_vnd: int | None
    supplementary_consequences: list[str]    # e.g. "trừ 4 điểm GPLX"
    reasoning: str                           # Vietnamese, cites provisions inline
    confidence: float
    insufficient_evidence: bool
    iterations_used: int

class Selection(BaseModel):                  # Phase 2 output
    answer: Literal["A","B","C","D"]
    match_quality: Literal["exact","partial","poor"]
    rationale: str
```

**I1 enforcement (do this literally):** Phase-1 function signatures accept `question: str` and
never an `options` argument. A single `assert_no_option_leakage(prompt, options)` runs before
every Phase-1 LLM call and raises if any option string (normalized) appears in the prompt. Log
the assertion count; report it in the paper's reproducibility appendix. *An invariant you can't
test is a slogan.*

---

## 5. Algorithms (paper-ready)

**Algorithm 1 — Phase 1: Option-Blind Legal Determination**

```
Input: image I, question q, corpus C, budget B, as_of date d
Output: LegalDetermination D

s ← PERCEIVE(I, q)          # ENUMERATE entities → INSPECT each (targeted); open vocab; options never passed
Q ← FORMULATE(s, q)         # entity–element pairs → 1..3 statute-language queries; sets intent→slots
                            # vehicle_filter ← normalize_vehicle(s.entities) — fail open if unmappable
R ← ∅ ; t ← 0 ; R_prev ← ∅
repeat
    t ← t + 1
    for each query in Q:
        R ← R ∪ RETRIEVE(query, C, filters=(query.vehicle_filter, d), k)   # RETRIEVE = swappable R*
    sr ← ASSESS(q, s, R)                            # → SufficiencyReport (options never passed)
    if sr.overall_sufficient: stop_reason ← "sufficient"; break
    if jaccard(chunk_ids(R), chunk_ids(R_prev)) ≥ 0.8: stop_reason ← "no_progress"; break
    if t = B: stop_reason ← "budget"; break
    Q ← REFINE(q, s, sr.unfilled_slots)             # target exactly the missing slots
    R_prev ← R
D ← EMIT(q, s, R)                                   # applicable_provisions ⊆ chunk_ids(R)
assert citation_closure(D, R)                        # invariant I2
if stop_reason ≠ "sufficient": D.insufficient_evidence ← true; D.confidence ← low
return D, stop_reason
```

**Algorithm 2 — Phase 2: Selection**

```
Input: determination D, options O = {A,B,C,D}
Output: Selection
# No retriever, no image, no re-reasoning about the scene.
return SELECT(D, O)          # match_quality logged for diagnostics
```

---

## 6. Baselines are the same code path

This is a **methodological requirement**, not an implementation convenience. If E1–E3 are
written as separate scripts, any E4 advantage is confounded with prompt-tuning asymmetry, and a
reviewer will say so. Implement all four conditions as **flags on one pipeline**:

| Condition | `retrieval` | `query_source` | `iterations` | `options_visible_phase1` |
|---|---|---|---|---|
| E1 parametric | off | — | 0 | n/a |
| E2 RAG-Q | on | question | 1 | **false** |
| E3 RAG-QO | on | question + options | 1 | **true** ← the leakage probe |
| E4 Agent | on | agent-formulated | B (3) | **false** |
| **E4-single** | on | agent-formulated | **1** | false | ← single-pass; "does the loop earn its compute?" |
| **E0-oracle** | **gold chunks** | — (skip retrieval) | 0 | false | ← reasoning ceiling; isolates retriever vs reasoner |

E3 is the *only* configuration where `assert_no_option_leakage` is disabled by design. That
single flag is the experimental manipulation the whole paper turns on. Make it loud in the code.

**Two conditions that turn ambiguous numbers into diagnosed ones — treat both as blocking, not optional:**

- **E0-oracle (oracle retrieval).** Feed the agent the *gold* chunks directly and let it produce
  the determination. This decomposes every failure into *retrieval couldn't find the law* vs
  *the model had the law and still reasoned wrong*. Without it, a low grounded-accuracy number is
  uninterpretable — you cannot tell a reviewer whether to fix the retriever or the reasoner, and
  those imply completely different papers. E0-oracle − E4 = the retrieval gap; 100% − E0-oracle =
  the reasoning gap. It is nearly free (a flag swapping retrieved-set for gold-set) and it is the
  single most diagnostic experiment in the plan.
- **E4-single (single-pass agent).** Budget-1 version of the full agent. If E4 ≈ E4-single, the
  iterative loop is *not* the contribution and the paper must say so honestly (and lean harder on
  option-blindness + the benchmark). Reported as its own row so "does agentic iteration help?" is
  answered with a number, not a hope. Note this differs from E2 only in query source (agent-
  formulated vs raw question), which cleanly attributes any gain to FORMULATE.

---

## 6b. Grounding-correctness metric — FROZEN before the sweep

> This definition is committed **now**, in writing, before any headline number exists.
> Post-hoc metric choice on the paper's central table (T3, "citations correct") is the fastest
> way to lose a reviewer's trust. It is frozen here so it cannot be tuned to the result.

Let `G` = resolved gold citation set for an item, `P` = resolved predicted citation set
(both passed through `resolve_citation()`, §3.4). Report **two** numbers, always together:

1. **Strict grounding (primary, used in T3):** correct iff `G ⊆ P` — the agent cited
   *everything the item requires* (extra retrieved chunks don't penalize; over-retrieval is not
   the sin, missing law is). This is the binary that populates the leakage 2×2.
2. **Citation F1 (secondary, partial credit):** per-item precision/recall/F1 of `P` against `G`,
   macro-averaged. This exists because mean citations/item ≈ 2 and 24% of items need ≥3, so
   strict all-or-nothing would score a 2-of-3 agent as fully ungrounded and understate real
   competence. F1 shows the near-misses that strict hides.

Rules that are part of the freeze:
- **Granularity:** matching is at khoản level (điểm folded to its khoản via the fallback), because
  the DB is khoản-level and the clause chunk contains the point text (§3.1). A điểm-gold matched
  by its khoản-chunk counts as correct.
- **No-violation / insufficient items:** if `G` is empty (no-violation items) grounding is scored
  on whether `P` is *also* empty (a hallucinated citation on a no-violation image is a
  false-violation error, tracked in §RQ4). Items where the agent emits `insufficient_evidence` are
  reported in a **separate abstention column**, never silently counted as grounding failures.
- **Reporting pair is mandatory:** every grounded-accuracy figure in the paper shows strict AND F1.
  If they diverge sharply, that divergence is itself reported and discussed (it means the agent is
  systematically getting *most* of the citations but rarely *all*).

The same `G`/`P` machinery, with `resolve_citation()` shared between agent (I2) and evaluator,
guarantees the agent's self-check and the paper's grading use identical resolution logic — no
gap where the agent thinks it grounded but the grader disagrees for a normalization reason.

---

## 7. Logging (the leakage table is built from logs, not from re-runs)

Per item, per condition, one JSONL record containing: `item_id`, condition, model, corpus
version, seed, `SceneDescription`, every iteration's `RetrievalQuery` list and retrieved
`chunk_id`s with scores, final `LegalDetermination`, `Selection`, gold answer, gold citations,
resolved gold citations, wall-clock, token counts.

This directly yields:
- **T3 leakage 2×2**: answer-correct × citation-correct, per condition.
- **Retrieval-overlap diagnostic**: Jaccard(retrieved_E2, retrieved_E3) on the same items, and
  the rate at which E3's retrieved chunks lexically match *distractor* options. This is the
  *mechanistic* evidence for K2 — not just the outcome-level inference. Predicted signature:
  E3's sets skew toward chunks containing distractor monetary strings on `penalty_amount` items.
- **Abstention analysis**: `insufficient_evidence` rate by category (needed for the QCVN story).
- **Budget curve** (A1): accuracy/grounding vs `iterations_used`, split by `stop_reason`.
- **Retrieval-vs-reasoning decomposition**: E0-oracle − E4 (retrieval gap) and 100% − E0-oracle
  (reasoning gap), per subtype — tells the reader which half to fix.
- **Confidence calibration (free figure):** `confidence` is already logged per determination, so
  plot a reliability curve (confidence vs empirical correctness) + ECE. For a *legal* system,
  "the agent knows when it doesn't know" is a genuinely valuable property, and it pairs with the
  honest abstention on empty-QCVN sign items — turning the QCVN gap from an embarrassment into
  evidence of calibration.
- **ASSESS precision:** for each slot ASSESS marked `filled`, check its `supporting_chunk_id`
  actually contains the slot content → precision of the sufficiency judgment (validates that the
  loop stops for real reasons, not LLM optimism).

---

## 8. Failure modes and guards

| Failure | Symptom | Guard |
|---|---|---|
| Hallucinated citation | `chunk_id` not in retrieved set | I2 validator → reject, re-emit once, else `insufficient_evidence` |
| **Hallucinated entity** | phantom "đèn đỏ" → confidently wrong red-light verdict | confidence gate: low-confidence entities excluded from FORMULATE (kept in log); asymmetric policy — a missed entity only under-retrieves, a phantom one poisons the verdict |
| **Unmappable vehicle term** | `normalize_vehicle()` returns None | **fail open**: no vehicle filter, retrieve over everything; never force a mapping |
| Malformed JSON from LLM | parse error | Retry ×2 with schema reminder; then structured-output mode; log parse-failure rate per model (small models will fail more — report it). ENUMERATE+INSPECT helps here: short targeted answers parse better than one big JSON |
| Phase-1 prompt contains an option | leakage | `assert_no_option_leakage` raises |
| Agent loops without progress | same queries each iteration | Dedup queries; if REFINE returns a query already tried, break early |
| Sign question, empty QCVN | agent abstains | Expected; report `insufficient_evidence` separately from wrong answers |
| SELECT rescues a bad determination via option length | inflated E4 accuracy | Report `match_quality`; and fix the data (length rebalance gate) |
| Retrieval returns generic Điều 1 chunks | low-signal top-k | BM25 half of hybrid + `vehicle_type` filter; monitor recall@k against gold |

---

## 9. Implementation roadmap

**M0 — Foundations (blocking everything).**
`resolve_citation()` + coverage report. Corpus loader with version hash. `pydantic` schemas.
Deliverable: `python -m vnt.resolve --report` prints exact/case/point-fallback/unresolved counts.

**M1 — Retriever comparison, evaluated standalone (experiment R).**
Build the swappable retriever interface + RRF + filters, then run **all retrievers** (R-BM25,
R-BGE, R-E5, R-Vi, R-Hybrid, optionally R-Rerank) against gold chunks: recall@{1,3,5,10} + MRR,
on the gradeable subset, overall / per-subtype / per-category. **Do this before any agent
exists** — it is the ceiling on grounding F1 and it selects R\* for the main sweep. If R\*
recall@5 is 0.5, no agent can exceed it and retrieval is the bottleneck, not reasoning.
Deliverables: the retriever-comparison table + the chosen R\*. *Do not skip this step.*

**M2 — Ceilings: oracle-SELECT and oracle-retrieval.**
(a) Feed **gold** determinations into SELECT → oracle-SELECT accuracy (expect >95%; lower = broken
MCQ mapping or bad items). (b) Stand up **E0-oracle** (gold chunks → determination) → the
reasoning ceiling. Together M1 (retrieval ceiling) and M2 (reasoning + selection ceilings) bound
the whole system before a single agent iteration runs.

**M2.5 — Validate PERCEIVE-v2 against the detection tier (free gold labels).**
The 1,156 detection-tier questions ask precisely about the entities PERCEIVE extracts — what
vehicle, what sign, what light. So the entity extractor is scored **directly against detection-
tier gold on the same images**, before it ever feeds the legal pipeline. Deliverables:
(a) per-entity-type accuracy (vehicles vs signs vs lights vs cargo…), per model size;
(b) the **discovered entity taxonomy**: cluster the open-vocabulary `kind` strings post-hoc
(embedding clustering or one LLM pass over the vocabulary) — the taxonomy is *derived from what
the model extracted*, never imposed pre-extraction; it becomes an appendix table
("the entities the model found legally relevant") and the grouping used in error analysis.
This milestone enables the paper's three-way failure decomposition: **perception wrong**
(entity ≠ detection gold) vs **retrieval wrong** (entity right, gold chunk missed — from M1/E4
logs) vs **reasoning wrong** (chunks right, determination wrong — from E0-oracle). Almost no
benchmark can do this decomposition because they lack paired perception/legal tiers over the
same images; it is the dataset's strongest structural asset — use it.

**M3 — Phase 1 single-shot** = E4-single (budget 1). End-to-end first numbers, and the baseline
that answers "does the loop help?"

**M4 — Full agent loop** (structured ASSESS + slot-targeted REFINE + early-stop, budget B).
A1 budget curve and `stop_reason` distribution fall out.

**M5 — E1, E2, E3 as flags.** The leakage probe (E3) + naive-RAG (E2). Run the pilot (200
stratified items, InternVL-8B, R\*) and check the go/no-go criteria from `EXPERIMENT_PLAN` §9.2.

**M6 — Main sweep + ablations**, including:
- the retriever ablation (does R\* vs R-BM25 move *end-to-end* grounded accuracy, or does the
  agent loop absorb the retriever difference?);
- **E4-entity vs E4-prose**: PERCEIVE-v2 (entity-directed extraction) vs a single undirected
  scene-description pass, all else fixed. If entity extraction wins, that is a result
  ("element-aligned perception improves grounded legal accuracy by X points"); expectation:
  gains concentrate on sign/light-dependent categories and at smaller model sizes, where
  undirected captioning is weakest. If open-vocabulary extraction flails at 1–2B, report it as a
  capability-threshold finding, not a design failure.

Note the ordering logic: M1 and M2 bound the system from below (retrieval) and above (reasoning,
selection) before any agent iteration. Most agent projects discover in week 5 that retrieval
recall was 0.4 the whole time.

---

## 10. Engineering notes

- **Caching is not optional.** Key on `sha1(query ‖ corpus_version ‖ k ‖ filters)`. E4 × 1,319
  items × B=3 × 7 checkpoints re-embeds the same queries constantly otherwise.
- **Determinism:** temperature 0, fixed seed, pinned model revisions. Log all three.
- **Small models will fight the schema.** InternVL-1B/2B may fail structured output. Budget for
  a parse-failure fallback and *report the rate* — "the agent needs ≥7B to emit valid
  determinations" is itself a finding (capability threshold), not an embarrassment.
- **Vietnamese normalization:** NFC-normalize everything. `Đ`/`đ` and diacritic variants bite in
  regex, BM25 tokenization, and chunk_id matching alike. One `normalize()` function, used everywhere.
- **Cost control:** run the ladder smallest-first — pipeline bugs surface on the 1B model for free.

---

## 11. What I'd build first, concretely

1. `resolve_citation()` + coverage report (M0) — gates every metric in the paper.
2. Retriever comparison, recall@k across R-BM25 / R-BGE / R-E5 / R-Vi / R-Hybrid (M1, experiment R)
   — tells us whether the agent is even possible and picks R\*. If gold recall@5 is low, the
   bottleneck is retrieval, not reasoning, and everything downstream changes.
3. The two ceilings (M2): oracle-SELECT (validates the MCQ mapping) and E0-oracle (the reasoning
   ceiling) — both cheap, both bound the system before Phase 1 is built.

Only then does building the agent loop make sense.
