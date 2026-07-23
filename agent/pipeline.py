"""ViTAL-Agent pipeline (AGENT_PLAN_v2.md §2-4): PERCEIVE (1 VLM call,
question-directed, once) -> FORMULATE (string assembly, no LLM) -> RETRIEVE
(BM25) <-> ASSESS (1 LLM call per round, merged sufficiency+refine, bounded
by max_iterations) -> SELECT (1 LLM call, multimodal: image + retrieved
statute text + question + options).

Phase-1 functions never receive `options` — the retrieval query is built
from the image and the question only, never the options (AGENT_PLAN_v2.md
§1, the one invariant that matters). The refine loop only regenerates the
query; PERCEIVE runs exactly once per item, before the loop starts. SELECT
never sees a retriever handle — citation closure (the cited chunk_id must be
one actually retrieved) is enforced in `select()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.corpus import Corpus
from agent.llm import LLMClient, assert_no_option_leakage
from agent.prompts.assess import build_assess_user_prompt
from agent.prompts.perceive import build_perceive_user_prompt
from agent.resolve import CitationIndex
from agent.retrieval.base import Retriever
from agent.schemas import Assessment, Perception, RetrievalQuery, RetrievedChunk, Selection
from agent.vehicle import normalize_vehicle

DEFAULT_K = 3
DEFAULT_MAX_ITERATIONS = 2
MAX_CONTEXT_CHUNKS = 6  # cap on the accumulated pool handed to SELECT, across all iterations
# Triggers the second, ND_168-boosted FORMULATE query (§3.2) — covers not just
# fine-amount questions but any sanction NĐ 168 states alongside the fine
# (điểm/license consequences live in the same document). Without "tước
# quyền"/"hậu quả pháp lý" here, a point-deduction question never gets an
# ND_168-boosted query at all, and SELECT is left guessing over whatever
# LUAT_36-boosted chunks the base query happened to find.
PENALTY_KEYWORDS = (
    "phạt tiền", "mức phạt", "khung tiền phạt", "bị phạt", "khung nào", "bao nhiêu tiền",
    "tước quyền", "tước giấy phép", "hậu quả pháp lý",
)
# "trừ điểm" as a regex, not a literal keyword: real phrasing is often
# "trừ BAO NHIÊU điểm" / "trừ MẤY điểm" — a plain substring check for "trừ
# điểm" misses those (the words aren't contiguous).
_TRU_DIEM_RE = re.compile(r"trừ.{0,20}điểm")


@dataclass
class Phase1IterationLog:
    """One RETRIEVE (+ optional ASSESS) round inside the e4_agent refine loop."""

    iteration: int
    query_texts: list[str]
    retrieved_chunk_ids: list[str]  # chunk_ids surfaced THIS round only
    cumulative_chunk_count: int  # size of the accumulated pool after this round
    assessed: bool  # False on the final allowed iteration (ASSESS is skipped there)
    sufficient: bool | None = None
    refined_query: str | None = None
    reason: str = ""


@dataclass
class Phase1Trace:
    """Everything logging wants for one item's Phase-1 run. `iterations`/
    `iterations_used` are populated only by the e4_agent loop (run_phase1);
    the other conditions leave them at their empty/zero defaults."""

    perception: Perception
    queries: list[dict] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    iterations: list[Phase1IterationLog] = field(default_factory=list)
    iterations_used: int = 0
    retry_used: bool = False  # True iff iterations_used > 1
    stop_reason: str = ""


# --------------------------------------------------------------------------- #
# PERCEIVE — one call, question-directed, run exactly once per item (§3.1)
# --------------------------------------------------------------------------- #


def perceive(image_path: Path, question: str, llm: LLMClient) -> Perception:
    assert_no_option_leakage(build_perceive_user_prompt(question), None)  # I1 trip-wire
    raw = llm.perceive(image_path, question)
    return Perception(**raw)


# --------------------------------------------------------------------------- #
# FORMULATE — pure string assembly, no LLM call (§3.2)
# --------------------------------------------------------------------------- #


def is_penalty_question(question: str) -> bool:
    q = question.lower()
    if _TRU_DIEM_RE.search(q):
        return True
    return any(kw in q for kw in PENALTY_KEYWORDS)


def _split_violations(violation: str) -> list[str]:
    """PERCEIVE may name more than one violation, semicolon-separated, for
    questions that ask for a TOTAL across simultaneous violations (e.g.
    running a red light AND no helmet at once). One query pair per violation
    below gives RETRIEVE a chance to find each one's provisions, not just
    whichever the base query's lexical match favors."""
    parts = [v.strip() for v in violation.split(";")]
    return [v for v in parts if v]


def formulate(perception: Perception, question: str) -> list[RetrievalQuery]:
    """One query per named violation (two per violation if the question asks
    for a penalty/consequence) — `perception.violation` is already
    statute-phrased, so this is mostly string assembly (§3.2)."""
    vehicle_term = normalize_vehicle(perception.vehicle)
    vehicle_filter = [vehicle_term] if vehicle_term else []
    penalty_asked = is_penalty_question(question)

    violations = _split_violations(perception.violation) if perception.violation else [perception.observed]
    queries: list[RetrievalQuery] = []
    for v in violations:
        queries.append(RetrievalQuery(text=v, vehicle_filter=vehicle_filter, doc_boost=["LUAT_36_2024"]))
        if penalty_asked and perception.violation:
            penalty_text = f"{v} mức phạt {perception.vehicle or ''}".strip()
            queries.append(
                RetrievalQuery(text=penalty_text, vehicle_filter=vehicle_filter, doc_boost=["ND_168_2024"])
            )
    return queries


# --------------------------------------------------------------------------- #
# RETRIEVE (no LLM) — small helpers reused by both the single-pass baselines
# and the e4_agent refine loop.
# --------------------------------------------------------------------------- #


def _run_queries(
    queries: list[RetrievalQuery], corpus: Corpus, retriever: Retriever, k: int, as_of: str | None
) -> dict[str, RetrievedChunk]:
    out: dict[str, RetrievedChunk] = {}
    for q in queries:
        for chunk in retriever.retrieve(q, corpus, k, as_of):
            existing = out.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                out[chunk.chunk_id] = chunk
    return out


def _accumulate(target: dict[str, RetrievedChunk], new: dict[str, RetrievedChunk]) -> None:
    """Union, max-score-wins — never replace. A chunk found in an earlier
    iteration must survive into a later one, or a legitimate citation could
    be wrongly rejected as a hallucination downstream (see select())."""
    for cid, chunk in new.items():
        existing = target.get(cid)
        if existing is None or chunk.score > existing.score:
            target[cid] = chunk


def _truncate_to_context_cap(pool: dict[str, RetrievedChunk]) -> list[RetrievedChunk]:
    """Cap the final context handed to SELECT at MAX_CONTEXT_CHUNKS, sorted
    by score. "Don't retrieve a lot of chunks" must hold for what SELECT
    actually sees, not just per-call k — chunks accumulate across iterations."""
    ranked = sorted(pool.values(), key=lambda c: c.score, reverse=True)
    return ranked[:MAX_CONTEXT_CHUNKS]


# --------------------------------------------------------------------------- #
# SELECT — enforces citation closure (§4)
# --------------------------------------------------------------------------- #


def citation_closure(selection: Selection, retrieved_ids: set[str]) -> bool:
    return selection.citation is None or selection.citation in retrieved_ids


def select(
    question: str,
    retrieved: list[RetrievedChunk],
    options: dict[str, str],
    image_path: Path,
    llm: LLMClient,
) -> Selection:
    retrieved_ids = {c.chunk_id for c in retrieved}

    def _build() -> Selection:
        raw = llm.select(question, retrieved, options, image_path)
        return Selection(**raw)

    selection = _build()
    if not citation_closure(selection, retrieved_ids):
        selection = _build()  # reject, re-emit once
    if not citation_closure(selection, retrieved_ids):
        # Still hallucinating a citation: drop it rather than crash.
        selection.citation = None
        selection.match_quality = "poor"

    assert citation_closure(selection, retrieved_ids), "citation closure violated after filtering"
    return selection


# --------------------------------------------------------------------------- #
# Algorithm — Phase 1: option-blind PERCEIVE (once) -> [FORMULATE -> RETRIEVE
# -> ASSESS]* (refine loop, bounded by max_iterations)
# --------------------------------------------------------------------------- #


def run_phase1(
    image_path: Path,
    question: str,
    corpus: Corpus,
    retriever: Retriever,
    llm: LLMClient,
    k: int = DEFAULT_K,
    as_of: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[Perception, list[RetrievedChunk], Phase1Trace]:
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    perception = perceive(image_path, question, llm)  # exactly once — never re-run inside the loop
    vehicle_term = normalize_vehicle(perception.vehicle)
    vehicle_filter = [vehicle_term] if vehicle_term else []
    penalty_asked = is_penalty_question(question)  # constant for this item; reused every iteration

    queries = formulate(perception, question)  # iteration 0 only; formulate() is never called again
    retrieved_by_id: dict[str, RetrievedChunk] = {}
    all_query_texts: list[str] = []
    iteration_logs: list[Phase1IterationLog] = []
    stop_reason = ""

    for i in range(max_iterations):
        round_results = _run_queries(queries, corpus, retriever, k, as_of)
        _accumulate(retrieved_by_id, round_results)
        all_query_texts.extend(q.text for q in queries)
        is_last_round = i == max_iterations - 1

        if is_last_round:
            iteration_logs.append(
                Phase1IterationLog(
                    iteration=i,
                    query_texts=[q.text for q in queries],
                    retrieved_chunk_ids=sorted(round_results),
                    cumulative_chunk_count=len(retrieved_by_id),
                    assessed=False,
                )
            )
            stop_reason = "max_iterations_reached"
            break

        assess_prompt = build_assess_user_prompt(question, perception, list(retrieved_by_id.values()), all_query_texts)
        assert_no_option_leakage(assess_prompt, None)  # I1 trip-wire — ASSESS is Phase 1 too
        raw = llm.assess(question, perception, list(retrieved_by_id.values()), all_query_texts)
        assessment = Assessment(**raw)

        iteration_logs.append(
            Phase1IterationLog(
                iteration=i,
                query_texts=[q.text for q in queries],
                retrieved_chunk_ids=sorted(round_results),
                cumulative_chunk_count=len(retrieved_by_id),
                assessed=True,
                sufficient=assessment.sufficient,
                refined_query=assessment.refined_query,
                reason=assessment.reason,
            )
        )

        if assessment.sufficient:
            stop_reason = "sufficient"
            break

        refined_text = (assessment.refined_query or "").strip()
        if not refined_text or refined_text in all_query_texts:
            stop_reason = "no_new_query"
            break

        queries = [
            RetrievalQuery(
                text=refined_text,
                vehicle_filter=vehicle_filter,
                doc_boost=["ND_168_2024"] if penalty_asked else ["LUAT_36_2024"],
            )
        ]

    retrieved = _truncate_to_context_cap(retrieved_by_id)
    trace = Phase1Trace(
        perception=perception,
        queries=[{"text": t} for t in all_query_texts],
        retrieved_chunk_ids=sorted(c.chunk_id for c in retrieved),
        iterations=iteration_logs,
        iterations_used=len(iteration_logs),
        retry_used=len(iteration_logs) > 1,
        stop_reason=stop_reason,
    )
    return perception, retrieved, trace


# --------------------------------------------------------------------------- #
# Algorithm — Phase 2: SELECT (no retrieval, multimodal, no re-perception)
# --------------------------------------------------------------------------- #


def run_phase2(
    question: str,
    retrieved: list[RetrievedChunk],
    options: dict[str, str],
    image_path: Path,
    llm: LLMClient,
) -> Selection:
    return select(question, retrieved, options, image_path, llm)


# --------------------------------------------------------------------------- #
# §5 conditions — flags on one pipeline, not separate scripts, so E4 vs.
# E0-E3 is never confounded with prompt-tuning asymmetry.
# --------------------------------------------------------------------------- #

CONDITIONS = ("e0_oracle", "e1_parametric", "e2_rag_q", "e3_rag_qo", "e4_agent")


def _run_e1_parametric(
    image_path: Path, question: str, llm: LLMClient
) -> tuple[Perception, list[RetrievedChunk], Phase1Trace]:
    """No retrieval at all: SELECT reasons with nothing retrieved, so its
    citation is necessarily null — structurally ungrounded, which is the
    point of including this condition (§5, "what the model already knows")."""
    perception = perceive(image_path, question, llm)
    trace = Phase1Trace(perception=perception, queries=[], retrieved_chunk_ids=[], stop_reason="no_retrieval")
    return perception, [], trace


def _run_naive_rag(
    image_path: Path,
    question: str,
    corpus: Corpus,
    retriever: Retriever,
    llm: LLMClient,
    k: int,
    as_of: str | None,
    options: dict[str, str] | None,
) -> tuple[Perception, list[RetrievedChunk], Phase1Trace]:
    """E2 (options=None) / E3 (options=given): one retrieval pass using the
    raw question as the query — FORMULATE is skipped, which is what makes
    this "naive" relative to E4. Deliberately single-pass, no refine loop:
    E2 vs E4's comparative point is "naive query vs. agent-formulated query,"
    which a loop would confound with an iteration-budget manipulation. When
    `options` is given (E3, the leakage probe, §5), option text is appended
    to the retrieval query and `assert_no_option_leakage` is deliberately NOT
    called — that single disabled guard is the experimental manipulation the
    whole paper turns on.
    """
    perception = perceive(image_path, question, llm)

    query_text = question.strip()
    if options is not None:
        # THE LEAKAGE PROBE: options folded into the retrieval query on
        # purpose. assert_no_option_leakage is intentionally skipped below —
        # do not add it to this branch.
        query_text = query_text + " " + " ".join(options.values())
    else:
        assert_no_option_leakage(query_text, None)  # E2 stays guarded like E4

    query = RetrievalQuery(text=query_text)
    retrieved = retriever.retrieve(query, corpus, k, as_of)
    trace = Phase1Trace(
        perception=perception,
        queries=[{"text": query_text}],
        retrieved_chunk_ids=sorted(c.chunk_id for c in retrieved),
        stop_reason="single_pass_naive_rag",
    )
    return perception, retrieved, trace


def _run_e0_oracle(
    image_path: Path,
    question: str,
    corpus: Corpus,
    llm: LLMClient,
    citation_index: CitationIndex,
    gold_chunk_ids: set[str],
) -> tuple[Perception, list[RetrievedChunk], Phase1Trace]:
    """Skip RETRIEVE entirely; feed the *gold* chunks directly. Isolates the
    reasoning ceiling from the retriever (§5: E0-oracle - E4 = retrieval gap)."""
    perception = perceive(image_path, question, llm)
    resolved = citation_index.resolve_set(sorted(gold_chunk_ids))
    retrieved = [
        RetrievedChunk(chunk_id=cid, citation=corpus[cid].title, content=corpus[cid].text, score=1.0)
        for cid in sorted(resolved)
        if cid in corpus.by_id
    ]
    trace = Phase1Trace(
        perception=perception,
        queries=[],
        retrieved_chunk_ids=[c.chunk_id for c in retrieved],
        stop_reason="oracle_retrieval",
    )
    return perception, retrieved, trace


def run_condition(
    condition: str,
    image_path: Path,
    question: str,
    options: dict[str, str],
    corpus: Corpus,
    retriever: Retriever,
    llm: LLMClient,
    citation_index: CitationIndex | None = None,
    gold_chunk_ids: set[str] | None = None,
    k: int = DEFAULT_K,
    as_of: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[Perception, list[RetrievedChunk], Selection, Phase1Trace]:
    """Single dispatch point for every condition in the §5 table. Every
    branch ends the same way: run_phase2 never sees a retriever handle, and
    always only sees the retrieved chunks + question + options + image —
    which is also why threading `image_path` through this one shared
    trailing call upgrades SELECT to multimodal for all five conditions at
    once, with zero per-condition code changes needed."""
    if condition == "e4_agent":
        perception, retrieved, trace = run_phase1(
            image_path, question, corpus, retriever, llm, k=k, as_of=as_of, max_iterations=max_iterations
        )
    elif condition == "e1_parametric":
        perception, retrieved, trace = _run_e1_parametric(image_path, question, llm)
    elif condition == "e2_rag_q":
        perception, retrieved, trace = _run_naive_rag(
            image_path, question, corpus, retriever, llm, k, as_of, options=None
        )
    elif condition == "e3_rag_qo":
        perception, retrieved, trace = _run_naive_rag(
            image_path, question, corpus, retriever, llm, k, as_of, options=options
        )
    elif condition == "e0_oracle":
        if citation_index is None:
            raise ValueError("e0_oracle requires citation_index")
        perception, retrieved, trace = _run_e0_oracle(
            image_path, question, corpus, llm, citation_index, gold_chunk_ids or set()
        )
    else:
        raise ValueError(f"unknown condition: {condition!r} (expected one of {CONDITIONS})")

    selection = run_phase2(question, retrieved, options, image_path, llm)
    return perception, retrieved, selection, trace
