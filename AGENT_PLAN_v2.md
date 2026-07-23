# AGENT_PLAN.md — ViTAL-Agent (simplified)

**v2.3 · 2026-07-22** · Replaces v2.2. v2.1 replaced the retry-once heuristic with a
bounded, LLM-judged refine loop, and made SELECT multimodal. v2.2 was a correctness
pass on FORMULATE/SELECT for total-fine and point-deduction questions. v2.3 adds an
explicit cross-referencing discipline to both ASSESS and SELECT: NĐ 168's
supplementary-sanction khoản (trừ điểm/tước quyền) reference *specific* điểm/khoản of
the behavior-defining article, not the whole khoản — SELECT must check whether the
observed behavior's specific điểm is actually named before applying that consequence,
and ASSESS must not declare "sufficient" without both the behavior-defining chunk and
the cross-referencing consequence chunk in hand (§3.4, §4).

---

## 0. What changed from v1, and why

Four simplifications, all driven by a property of the benchmark that v1 didn't exploit: **the question already tells you what to look at, and each question is usually about exactly one violation** (named exception: "total fine" questions spanning several simultaneous violations, §3.2).

| Module | v1 | v2.1 |
|---|---|---|
| PERCEIVE | 2 calls: ENUMERATE all legally-relevant entities (open vocab), then INSPECT each one | **1 call**, question-directed: look only at the entities the question names, and predict the violation. Runs exactly once per item — never re-run inside the refine loop below. |
| FORMULATE | 1–3 queries across intents, entity–element pairs | **1 query** (2 if the question asks for a penalty), single violation type. On a refine-loop iteration, exactly 1 new query, built from ASSESS's `refined_query`. |
| ASSESS/REFINE | structured `SufficiencyReport` with per-slot chunk attribution, budget 3 | **one merged ASSESS call** (sufficiency judgment + query refinement in a single call), budget/`max_iterations` 2 — the loop re-queries only, it never re-runs PERCEIVE. See §3.4. |
| SELECT | maps a frozen determination onto options | **reasons over the image + statute text + question + options** |

Net effect vs. v1: still **4–6 LLM calls per item → 2–3ish**, easier to debug, easier to describe in a 7-page paper. Precisely: at the default `max_iterations=2`, ASSESS always runs exactly once, so the common e4_agent path is a **fixed 3 calls** (PERCEIVE, ASSESS, SELECT); `max_iterations=1` collapses back to 2 calls (no ASSESS). The option wall — the actual contribution — is untouched.

---

## 1. The one invariant that matters

> **The retrieval query is built from the image and the question only. Never from the options.**

Everything else in this document is negotiable. This is not.

Why: on the legal tier the correct option is uniquely the longest 49% of the time, and picking the longest scores 53.7%. A retriever that sees the options will embed the correct option's text, fetch the statute that matches it, and then "confirm" it. That looks like retrieval working. It is the answer key leaking into the pipeline.

Enforced structurally: **Phase-1 functions never receive the options as an argument.** Not a prompt instruction — a function signature.

**Note on SELECT (§4):** passing retrieved statute text into SELECT does *not* break this. The chunks were chosen before any option was visible. The wall protects *retrieval*, not *reading*.

---

## 2. Pipeline

```
image + question
      │
      ▼
 ① PERCEIVE   → what's in the image, focused on what the question asks about
      │          + predicted violation. Runs ONCE — never re-run below.
      ▼
 ② FORMULATE  → 1 query in statute language (2 if penalty asked)
      │
      ▼
 ③ RETRIEVE   → top-k chunks from the law corpus (BM25 + filters)
      │
      ▼
 ④ ASSESS     → enough to answer? if not, refine the query and loop back to ③
      │            (query only — perception is not re-run; capped at max_iterations)
      │
      ═════ OPTION WALL ═════
      ▼
 ⑤ SELECT     → reason over (image + statute chunks + question + options) → letter + citation
```

Three LLM calls in the common path at the default `max_iterations=2` (PERCEIVE, ASSESS, SELECT — ASSESS always runs exactly once at N=2). Two calls at `max_iterations=1` (no ASSESS). FORMULATE can be folded into the PERCEIVE call — see §3.2.

---

## 3. Phase 1 (option-blind)

### 3.1 PERCEIVE — question-directed, one call

The question already names the entity. *"Người điều khiển xe mô tô trong hình có vi phạm quy định về mũ bảo hiểm không?"* tells you: look at the motorbike rider, look at the helmet. Enumerating every legally-relevant object in the scene is wasted effort and, at 1–8B scale, wasted accuracy — undirected captioning goes vague.

So: **read the question, extract what it asks about, look at only that, then predict the violation.**

```python
class Perception(BaseModel):
    focus: str            # what the question is about: "người điều khiển xe mô tô, mũ bảo hiểm"
    vehicle: str | None    # for the retrieval filter: "xe máy", "ô tô", ...
    observed: str          # what's actually there: "người lái không đội mũ bảo hiểm"
    violation: str | None  # the predicted fault, in statute language. None = no violation.
    confidence: float
```

Two rules worth keeping from v1:

- **`violation` must be phrased as the statute would phrase it.** ❌ "xe máy đi qua ngã tư khi đèn đỏ" → ✅ "không chấp hành hiệu lệnh của đèn tín hiệu giao thông". This is one line in the prompt and it is the single highest-leverage line in the codebase — it's the difference between a query that matches statute text and one that doesn't.
- **Free text, not enums.** Constrain structure, never vocabulary. A closed vehicle enum makes `cong_kenh` (bulky cargo) and xe đạp điện unrepresentable and forces a wrong pick that then poisons the retrieval filter.

If `violation` is None, the agent is claiming no violation. Carry that through — the benchmark has a `du_lieu_khong_loi` category and this is the correct behaviour there.

### 3.2 FORMULATE — one query per violation

Each question usually targets exactly one violation type, so generate **one** query, not a spread across intents.

**Exception: "total" questions.** A question can ask for the *total* fine/consequence across several violations happening at once (e.g. running a red light *and* no helmet, same frame). For those, PERCEIVE names every violation in `violation`, semicolon-separated (§3.1), and FORMULATE issues one query pair *per named violation* — otherwise RETRIEVE only ever finds the one violation the base query's lexical match happens to favor, and SELECT is left guessing at a "total" it can't actually compute.

```
for v in split(perception.violation, ";"):
    query = v                                       # already in statute language
    if question asks for a fine/penalty/consequence:
        query2 = v + " mức phạt " + vehicle          # NĐ 168 is where penalties live
```

SELECT then sums the per-violation fine ranges it finds among the retrieved chunks (min+min, max+max) rather than reporting one violation's range as if it were the total — see §4.

**Implementation note:** since `violation` is already statute-phrased, FORMULATE is mostly string assembly — fold it into the PERCEIVE call's output and skip the separate LLM call. That's the third call saved.

### 3.3 RETRIEVE

Hybrid BM25 + dense with RRF over the ~1,073-chunk corpus. Corpus is small, so exact search — no ANN.

Filters:
- `vehicle_type` — keep chunks matching `perception.vehicle` **or with an empty list** (empty = applies to everyone, ~55% of chunks). **Unmappable vehicle → no filter at all.** Fail open. Losing the filter costs a little precision; a wrong forced filter silently deletes the correct statute.
- `effective_date ≤ as_of` (default 2025-01-01) — the mechanism behind the "update the corpus, not the weights" claim. Cheap to build, so build it.
- Document boost by question type: penalty → NĐ 168, rule → Luật 36, sign → QCVN 41. **Boost, not hard filter** — many items need two documents.

One useful structural fact: **a khoản chunk is self-contained** — it carries the penalty range *and* all its điểm sub-items. So one retrieval answers behaviour and penalty together, and the điểm→khoản citation fallback is semantically sound rather than a lossy hack. Worth one sentence in the paper; it turns an apparent data defect into a justified design choice.

**Refine loop (replaces v1's ASSESS/REFINE and v2's retry-once heuristic) — see §3.4.**

### 3.4 ASSESS — one call, sufficiency + refinement

```python
class Assessment(BaseModel):
    reason: str = ""                   # short justification, emitted FIRST
    sufficient: bool
    refined_query: str | None = None   # statute-register text; ignored once sufficient=True
```

One call does both jobs v1 split across two: judge whether the retrieved chunks are
enough to answer the question, and — if not — propose the next query to try. `reason`
is listed first in the schema/prompt on purpose: a model's autoregressive JSON
generation is then conditioned by a short justification *before* it commits to the
`sufficient` boolean, which is free chain-of-thought inside one call rather than a
second one.

**Loop mechanics:** PERCEIVE runs once, before the loop starts. Each iteration:
RETRIEVE (using the current query set) → accumulate the results into the running pool
as a **union, max-score-wins — never replace** (a chunk found in iteration 1 must
survive into iteration 2, or a legitimate citation could be lost) → if this is the
last allowed iteration, stop (no ASSESS call — there's no budget left to act on the
judgment); otherwise call ASSESS. If `sufficient`, stop. If not, and `refined_query`
is non-empty and hasn't been tried yet, build **one** new query from it (reusing the
same `vehicle_filter`/`doc_boost` computed once from `perception` — never re-asked of
the LLM) and loop back to RETRIEVE. Otherwise stop (`no_new_query`). Bounded by
`max_iterations` (default 2).

Right before SELECT, truncate the accumulated pool to `MAX_CONTEXT_CHUNKS` (6),
sorted by score — "don't retrieve a lot of chunks" has to hold for what SELECT
actually sees, not just for `k` on a single call, since chunks accumulate across
iterations.

This replaces v1's `SufficiencyReport`: no `required_slots`, no per-chunk attribution,
no slot-level tracking — a holistic judgment, still one call, not two.

---

## 4. Phase 2 — SELECT

Give the model the **image**, the retrieved statute text, the question, and the
options. Ask it to reason from the image and the law to the answer, and to name the
provision it used.

```python
class Selection(BaseModel):
    reasoning: str          # Vietnamese, must reference the statute text
    answer: Literal["A","B","C","D"]
    citation: str           # chunk_id — MUST be one of the retrieved chunks
    match_quality: Literal["exact","partial","poor"]
```

Two guards:

- **Citation closure.** The cited `chunk_id` must be in the retrieved set. Anything else is a hallucination — reject and re-emit once. This is what makes the citation auditable rather than decorative.
- **`match_quality`.** If no option matches what the law says, the model picks the closest and flags `poor`. A high poor-match rate means Phase 1 under-determined; it's a diagnostic, not a failure.

**Two named exceptions to "pick the closest option" (v2.2):**
- **Total-fine questions.** If the question asks for the combined fine across several simultaneous violations, SELECT must find each violation's range among the retrieved chunks and *sum* them (min+min, max+max) — not report one violation's range and call it the total. Only possible because FORMULATE fanned out a query per named violation in the first place (§3.2); SELECT can't sum what was never retrieved.
- **Point-deduction ("trừ điểm") questions.** "Pick the closest option" is actively wrong here — the closest *numeric* point value is often just a different violation's point count, not evidence for this one. If no retrieved provision explicitly states a point deduction for the observed behavior, that absence is itself the answer (many violations only carry a fine, no điểm deduction): prefer a "no deduction" option over guessing a nearby number.

**Cross-referencing between điểm/khoản (v2.3).** NĐ 168's supplementary-sanction khoản name *exact* điểm/khoản, not the whole khoản: `ND_168_2024_D7_K13` điểm (a) reads "Thực hiện hành vi quy định tại điểm b khoản 3 ... Điều này bị trừ điểm giấy phép lái xe 02 điểm" — this deducts points **only** for điểm b of khoản 3, not for điểm a/c/d/đ/e of the same khoản. Observed case: a "chở hàng cồng kềnh" item's behavior is điểm (e) of `D7_K3` ("mang vác vật cồng kềnh"), not điểm (b) — so `D7_K13`'s cross-reference does not cover it, and the correct answer is "not deducted," even though `D7_K13` is in the retrieved set and does state a deduction (for a *different* điểm). SELECT must resolve this three-step check explicitly (which điểm does the behavior match → what does the consequence khoản list → is that điểm in that list) rather than pattern-matching "a trừ điểm clause exists somewhere in the retrieved set." ASSESS must not call the loop "sufficient" until it has both the behavior-defining chunk and the consequence chunk together — one without the other cannot resolve the cross-reference.

Why pass the statute text rather than v1's frozen `LegalDetermination`: the determination was a lossy summary written before the model knew what it would be asked to distinguish. Options often turn on a detail the summary dropped — a specific fine range, a particular condition. Let SELECT read the actual law. The wall already did its job upstream.

**Why the image is re-attached here too (v2.1):** PERCEIVE already looked at it once, option-blind, to drive retrieval — but `perception.violation` is itself a summary, written before SELECT knew which specific options it would need to distinguish. Letting SELECT look at the image again, now with the actual candidate law in hand, catches what a frozen text field could drop. This does not reopen the option wall: retrieval is already finished by the time SELECT runs, and SELECT still has no retriever/corpus handle — it's structurally incapable of issuing a new retrieval no matter what it notices once options are visible. (SELECT deliberately does *not* receive the `Perception` object itself — only the image, chunks, question, and options — so it isn't anchored toward repeating a possibly-wrong PERCEIVE hypothesis.)

---

## 5. Conditions to run

| ID | Setup | Question it answers |
|---|---|---|
| **E1** | closed-book, no retrieval | what the model already knows |
| **E2** | retrieve with query = question text | does retrieval help at all |
| **E3** | retrieve with query = question **+ options** | 🚨 leakage probe |
| **E4** | ours (perception-driven query) | the method |
| **E0** | oracle: gold chunk injected | ceiling — is the gap retrieval or reasoning? |

**The signature to watch in E3:** accuracy goes **up** while citation accuracy stays **flat**. That's a system finding the answer without finding the law, and it's the empirical core of the paper.

Ablations, in priority order:
1. **A2** — descriptive vs. statute-register `violation` phrasing (expect a large recall effect)
2. **A1** — question-directed vs. undirected perception
3. **A3** — with/without vehicle filter

---

## 6. Metrics

| Metric | Definition |
|---|---|
| `MCQ-Acc` | option accuracy — **always print pick-longest (53.7%) beneath it** |
| **`Grounded-Acc`** | answer ✓ **and** citation resolves to gold at khoản level ← **headline** |
| `Recall@k` | gold chunk retrieved, reported per document (Luật / NĐ / QCVN) |
| `Divergence` | P(citation wrong \| answer correct) |
| `Δ(E0 − E4)` | retrieval gap vs. reasoning gap |

`pick-longest` scores 53.7% on MCQ and **0%** on grounded. That asymmetry is why grounded accuracy is the headline.

---

## 7. Build order

1. **`resolve_citation()` + coverage report.** Gates every metric. Exact match resolves 14.3%; with case folding and điểm→khoản fallback, 91.6%. Already written — promote it out of `backup/`.
2. **Retrieval recall@k against gold, standalone.** Before any agent exists. If recall@5 is 0.5, no agent can beat 0.5 and retrieval is the bottleneck — which changes everything downstream. *Do not skip this.*
3. **E0 oracle** (gold chunks → answer). The reasoning ceiling, and it doubles as the cheapest possible test of "legal knowledge is the bottleneck" — no agent required.
4. **E4 single-shot.** First end-to-end numbers.
5. **E1/E2/E3** as flags on the same code path.
6. Pilot: 200 items, one 8B model. Check the E3 leakage signature before spending GPU on the full ladder.

---

## 8. Engineering notes

- **NFC-normalize everything.** `Đ`/`đ` and diacritics bite in regex, BM25 tokenization, and chunk_id matching alike. One `normalize()`, used everywhere.
- **Vietnamese word segmentation before BM25** (`pyvi`/`underthesea`), or the lexical baseline is crippled on compounds and the comparison is unfair.
- **Cache on `sha1(query ‖ corpus_version ‖ as_of ‖ k ‖ filters)`.** Omitting `as_of` will silently serve you the wrong law generation and you will believe it.
- **Temperature 0, fixed seed, pinned revisions.** Log all three.
- **Small models will fight the schema.** Report the parse-failure rate — "the agent needs ≥7B to emit valid output" is a capability-threshold finding, not an embarrassment.
- **Run smallest model first.** Pipeline bugs surface on the 1B for free.
- **Don't truncate chunk text below what "self-contained" (§3.3) requires.** A khoản chunk carries the penalty range *and* every điểm sub-item — often 8-12 bullets of unrelated behaviors under one fine bracket. 38% of `law_db.json`'s chunks exceed 800 characters; truncating there cuts a chunk off mid-bullet and silently discards whichever điểm happens to sit later in the list. Observed directly: an `e0_oracle` item was fed the gold chunk containing "mang vác vật cồng kềnh" (bulky cargo) at điểm (e), but at an 800-char cap the prompt cut off after điểm (a) — the model correctly read what it was shown, correctly saw no match, and answered `poor`/wrong not because it reasoned badly but because it never saw the relevant sentence. `agent/llm_vlm.py`'s `DEFAULT_MAX_CHUNK_CHARS` is 3500 (covers ~95% of chunks); tune with `--max-chunk-chars` if a corpus update shifts that distribution.

---

## 9. What I'd cut further if time runs out

In order: the refine loop (back to a retry-once heuristic) → the penalty second query → the vehicle filter. Each is an optimization.

**Never cut:** the option wall (§1), citation closure (§4), or grounded accuracy (§6). Those three are the paper.
