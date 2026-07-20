"""Algorithm 1 (option-blind Phase 1) and Algorithm 2 (Phase 2 SELECT).

Direct implementation of AGENT_PLAN.md §5. Phase-1 functions never receive
`options` (invariant I1); `run_phase2` never receives a retriever handle
(invariant I3); `emit()` enforces citation closure (invariant I2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent.corpus import Corpus
from agent.llm import LLMClient, assert_no_option_leakage
from agent.prompts.assess import (
    INTENT_TO_SLOT,
    build_assess_user_prompt,
    parse_assess_output,
)
from agent.prompts.formulate import (
    FORMULATE_SYSTEM_PROMPT,
    build_formulate_user_prompt,
    build_refine_suffix,
    parse_formulate_output,
)
from agent.resolve import CitationIndex
from agent.retrieval.base import Retriever
from agent.schemas import (
    LegalDetermination,
    LegalEntity,
    RetrievalQuery,
    RetrievedChunk,
    SceneDescription,
    Selection,
)

DEFAULT_K = 5
DEFAULT_BUDGET = 3
NO_PROGRESS_JACCARD = 0.8
LOW_CONFIDENCE = 0.3
ENTITY_CONFIDENCE_GATE = 0.2  # hallucination guard (§2.1, §8)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class Phase1Trace:
    """Everything §7 wants logged for one item's Phase-1 run."""

    scene: SceneDescription
    iterations: list[dict] = field(default_factory=list)
    stop_reason: str = ""


# --------------------------------------------------------------------------- #
# PERCEIVE
# --------------------------------------------------------------------------- #


def perceive(image_path: Path, question: str, llm: LLMClient) -> SceneDescription:
    raw = llm.enumerate_entities(image_path, question)
    entities: list[LegalEntity] = []
    for e in raw.get("entities", []):
        entity = LegalEntity(**e)
        if entity.confidence < ENTITY_CONFIDENCE_GATE:
            continue  # hallucination guard: excluded from FORMULATE, kept only in raw log
        entities.append(entity)
    return SceneDescription(
        entities=entities,
        observed_behavior=raw.get("observed_behavior", ""),
        scene_notes=raw.get("scene_notes"),
    )


# --------------------------------------------------------------------------- #
# FORMULATE / REFINE
# --------------------------------------------------------------------------- #


def formulate(
    scene: SceneDescription, question: str, llm: LLMClient
) -> list[RetrievalQuery]:
    prompt = FORMULATE_SYSTEM_PROMPT + build_formulate_user_prompt(scene.entities, question)
    assert_no_option_leakage(prompt, None)  # I1 trip-wire; Phase 1 never holds options
    raw = llm.formulate_queries(scene.entities, question)
    return parse_formulate_output(raw)


def refine(
    scene: SceneDescription,
    question: str,
    unfilled_slots: list[str],
    found_citations: list[str],
    tried_queries: list[str],
    llm: LLMClient,
) -> list[RetrievalQuery]:
    prompt = FORMULATE_SYSTEM_PROMPT + build_refine_suffix(
        unfilled_slots, found_citations, tried_queries
    )
    assert_no_option_leakage(prompt, None)
    raw = llm.refine_queries(scene.entities, question, unfilled_slots, found_citations, tried_queries)
    return parse_formulate_output(raw)


# --------------------------------------------------------------------------- #
# RETRIEVE (no LLM)
# --------------------------------------------------------------------------- #


def retrieve_all(
    queries: list[RetrievalQuery],
    corpus: Corpus,
    retriever: Retriever,
    iteration: int,
    k: int,
    as_of: str | None,
    retrieved_by_id: dict[str, RetrievedChunk],
) -> None:
    for q in queries:
        for chunk in retriever.retrieve(q, corpus, k, iteration, as_of):
            existing = retrieved_by_id.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                retrieved_by_id[chunk.chunk_id] = chunk


# --------------------------------------------------------------------------- #
# ASSESS
# --------------------------------------------------------------------------- #


def assess(
    question: str,
    scene: SceneDescription,
    required_slots: list[str],
    retrieved: list[RetrievedChunk],
    llm: LLMClient,
):
    prompt = build_assess_user_prompt(question, scene.observed_behavior, required_slots, retrieved)
    assert_no_option_leakage(prompt, None)
    raw = llm.assess(question, scene.observed_behavior, required_slots, retrieved)
    retrieved_ids = {c.chunk_id for c in retrieved}
    return parse_assess_output(raw, required_slots, retrieved_ids)


# --------------------------------------------------------------------------- #
# EMIT — enforces invariant I2 (citation closure)
# --------------------------------------------------------------------------- #


def citation_closure(
    determination: LegalDetermination, retrieved_ids: set[str], citation_index: CitationIndex
) -> bool:
    resolved = citation_index.resolve_set(determination.applicable_provisions)
    return resolved <= retrieved_ids


def emit(
    question: str,
    scene: SceneDescription,
    retrieved: list[RetrievedChunk],
    llm: LLMClient,
    iterations_used: int,
    citation_index: CitationIndex,
) -> LegalDetermination:
    retrieved_ids = {c.chunk_id for c in retrieved}

    def _build() -> LegalDetermination:
        raw = llm.emit_determination(question, scene, retrieved, iterations_used)
        return LegalDetermination(**raw)

    determination = _build()
    if not citation_closure(determination, retrieved_ids, citation_index):
        determination = _build()  # reject, re-emit once (§8 failure-mode table)
    if not citation_closure(determination, retrieved_ids, citation_index):
        # Still hallucinating a citation: drop what can't be grounded rather
        # than crash, and mark the determination as evidence-insufficient.
        determination.applicable_provisions = [
            p
            for p in determination.applicable_provisions
            if citation_index.resolve(p).resolved_id in retrieved_ids
        ]
        determination.insufficient_evidence = True
        determination.confidence = min(determination.confidence, LOW_CONFIDENCE)

    assert citation_closure(determination, retrieved_ids, citation_index), "I2 violated after filtering"
    return determination


# --------------------------------------------------------------------------- #
# Algorithm 1 — Phase 1: Option-Blind Legal Determination
# --------------------------------------------------------------------------- #


def run_phase1(
    image_path: Path,
    question: str,
    corpus: Corpus,
    retriever: Retriever,
    llm: LLMClient,
    citation_index: CitationIndex,
    budget: int = DEFAULT_BUDGET,
    k: int = DEFAULT_K,
    as_of: str | None = None,
) -> tuple[LegalDetermination, Phase1Trace]:
    scene = perceive(image_path, question, llm)
    queries = formulate(scene, question, llm)
    # required_slots fixed at first FORMULATE call; REFINE only targets the
    # unfilled subset of this same set (§2.1 ASSESS).
    required_slots = sorted({INTENT_TO_SLOT[q.intent] for q in queries if q.intent in INTENT_TO_SLOT})

    retrieved_by_id: dict[str, RetrievedChunk] = {}
    prev_ids: set[str] = set()
    tried_query_texts: list[str] = []
    trace = Phase1Trace(scene=scene)

    iteration = 0
    stop_reason = "budget"
    while True:
        iteration += 1
        tried_query_texts.extend(q.text for q in queries)
        retrieve_all(queries, corpus, retriever, iteration, k, as_of, retrieved_by_id)

        retrieved_list = list(retrieved_by_id.values())
        sr = assess(question, scene, required_slots, retrieved_list, llm)
        cur_ids = set(retrieved_by_id)

        trace.iterations.append(
            {
                "iteration": iteration,
                "queries": [q.model_dump() for q in queries],
                "retrieved_chunk_ids": sorted(cur_ids),
                "sufficiency": sr.model_dump(),
            }
        )

        if sr.overall_sufficient:
            stop_reason = "sufficient"
            break
        if prev_ids and jaccard(cur_ids, prev_ids) >= NO_PROGRESS_JACCARD:
            stop_reason = "no_progress"
            break
        if iteration >= budget:
            stop_reason = "budget"
            break

        queries = refine(
            scene, question, sr.unfilled_slots, sorted(cur_ids), tried_query_texts, llm
        )
        if not queries:
            stop_reason = "no_progress"  # REFINE has nothing new to try (§8)
            break
        prev_ids = cur_ids

    trace.stop_reason = stop_reason
    determination = emit(
        question, scene, list(retrieved_by_id.values()), llm, iteration, citation_index
    )
    if stop_reason != "sufficient":
        determination.insufficient_evidence = True
        determination.confidence = min(determination.confidence, LOW_CONFIDENCE)
    return determination, trace


# --------------------------------------------------------------------------- #
# Algorithm 2 — Phase 2: Selection (no retrieval, no image, no re-reasoning)
# --------------------------------------------------------------------------- #


def run_phase2(
    determination: LegalDetermination, options: dict[str, str], llm: LLMClient
) -> Selection:
    raw = llm.select(determination, options)
    return Selection(**raw)


# --------------------------------------------------------------------------- #
# §6 baseline conditions — "flags on one pipeline", not separate scripts, so
# the E4 vs. E1-E3 comparison is never confounded with prompt-tuning asymmetry.
# --------------------------------------------------------------------------- #

CONDITIONS = ("e0_oracle", "e1_parametric", "e2_rag_q", "e3_rag_qo", "e4_agent", "e4_single")

ALL_SLOTS = sorted(set(INTENT_TO_SLOT.values()))


def _run_e1_parametric(
    image_path: Path, question: str, llm: LLMClient, citation_index: CitationIndex
) -> tuple[LegalDetermination, Phase1Trace]:
    """No retrieval at all: the determination comes from the model's
    parametric knowledge only. `applicable_provisions` is necessarily empty
    (there is nothing retrieved to cite), so this condition is structurally
    ungrounded — that is the point of including it (§6)."""
    scene = perceive(image_path, question, llm)
    trace = Phase1Trace(scene=scene, iterations=[], stop_reason="parametric_no_retrieval")
    determination = emit(question, scene, [], llm, iterations_used=0, citation_index=citation_index)
    return determination, trace


def _run_naive_rag(
    image_path: Path,
    question: str,
    corpus: Corpus,
    retriever: Retriever,
    llm: LLMClient,
    citation_index: CitationIndex,
    k: int,
    as_of: str | None,
    options: dict[str, str] | None,
) -> tuple[LegalDetermination, Phase1Trace]:
    """RAG-Q (options=None) / RAG-QO (options=given): one retrieval pass using
    the raw question as the query — FORMULATE is skipped entirely, which is
    what makes this "naive" relative to E4. When `options` is given (RAG-QO,
    the leakage probe, §6), option text is appended to the retrieval query and
    `assert_no_option_leakage` is deliberately NOT called — that single
    disabled guard is the experimental manipulation the whole paper turns on.
    """
    scene = perceive(image_path, question, llm)

    query_text = question.strip()
    if options is not None:
        # THE LEAKAGE PROBE: options are folded into the retrieval query on
        # purpose. assert_no_option_leakage is intentionally skipped below —
        # do not add it to this branch.
        query_text = query_text + " " + " ".join(options.values())
    else:
        assert_no_option_leakage(query_text, None)  # E2 stays guarded like E4

    query = RetrievalQuery(text=query_text, intent="behavior_identification")
    retrieved_by_id: dict[str, RetrievedChunk] = {}
    retrieve_all([query], corpus, retriever, iteration=1, k=k, as_of=as_of, retrieved_by_id=retrieved_by_id)
    retrieved_list = list(retrieved_by_id.values())

    sr = assess(question, scene, ALL_SLOTS, retrieved_list, llm)
    trace = Phase1Trace(
        scene=scene,
        iterations=[
            {
                "iteration": 1,
                "queries": [query.model_dump()],
                "retrieved_chunk_ids": sorted(retrieved_by_id),
                "sufficiency": sr.model_dump(),
            }
        ],
        stop_reason="single_pass_naive_rag",
    )
    determination = emit(question, scene, retrieved_list, llm, iterations_used=1, citation_index=citation_index)
    return determination, trace


def _run_e0_oracle(
    image_path: Path,
    question: str,
    corpus: Corpus,
    llm: LLMClient,
    citation_index: CitationIndex,
    gold_chunk_ids: set[str],
) -> tuple[LegalDetermination, Phase1Trace]:
    """Skip RETRIEVE entirely; feed the *gold* chunks directly. Isolates the
    reasoning ceiling from the retriever (§6, the single most diagnostic
    experiment in the plan: E0-oracle - E4 = the retrieval gap)."""
    scene = perceive(image_path, question, llm)
    resolved = citation_index.resolve_set(sorted(gold_chunk_ids))
    retrieved_list = [
        RetrievedChunk(
            chunk_id=cid,
            citation=corpus[cid].title,
            content=corpus[cid].text,
            score=1.0,
            retrieved_at_iteration=0,
        )
        for cid in sorted(resolved)
        if cid in corpus.by_id
    ]
    sr = assess(question, scene, ALL_SLOTS, retrieved_list, llm)
    trace = Phase1Trace(
        scene=scene,
        iterations=[
            {
                "iteration": 0,
                "queries": [],
                "retrieved_chunk_ids": [c.chunk_id for c in retrieved_list],
                "sufficiency": sr.model_dump(),
            }
        ],
        stop_reason="oracle_retrieval",
    )
    determination = emit(question, scene, retrieved_list, llm, iterations_used=0, citation_index=citation_index)
    return determination, trace


def run_condition(
    condition: str,
    image_path: Path,
    question: str,
    options: dict[str, str],
    corpus: Corpus,
    retriever: Retriever,
    llm: LLMClient,
    citation_index: CitationIndex,
    gold_chunk_ids: set[str] | None = None,
    budget: int = DEFAULT_BUDGET,
    k: int = DEFAULT_K,
    as_of: str | None = None,
) -> tuple[LegalDetermination, Selection, Phase1Trace]:
    """Single dispatch point for every condition in the §6 table. Every branch
    ends the same way: run_phase2 never sees a retriever handle (I3), and
    always only sees the frozen determination + options."""
    if condition == "e4_agent":
        determination, trace = run_phase1(
            image_path, question, corpus, retriever, llm, citation_index, budget=budget, k=k, as_of=as_of
        )
    elif condition == "e4_single":
        determination, trace = run_phase1(
            image_path, question, corpus, retriever, llm, citation_index, budget=1, k=k, as_of=as_of
        )
    elif condition == "e1_parametric":
        determination, trace = _run_e1_parametric(image_path, question, llm, citation_index)
    elif condition == "e2_rag_q":
        determination, trace = _run_naive_rag(
            image_path, question, corpus, retriever, llm, citation_index, k, as_of, options=None
        )
    elif condition == "e3_rag_qo":
        determination, trace = _run_naive_rag(
            image_path, question, corpus, retriever, llm, citation_index, k, as_of, options=options
        )
    elif condition == "e0_oracle":
        determination, trace = _run_e0_oracle(
            image_path, question, corpus, llm, citation_index, gold_chunk_ids or set()
        )
    else:
        raise ValueError(f"unknown condition: {condition!r} (expected one of {CONDITIONS})")

    selection = run_phase2(determination, options, llm)
    return determination, selection, trace
