"""Retrieval per QUERY_GEN_SPEC.md v3 — "Two mechanisms: route the question
to a document, then close the citation graph. Nothing else."

This is a SEPARATE track from `agent/pipeline.py`'s shipped v2.8 FORMULATE —
different corpus handling (QCVN stays in the corpus under v3, unlike the
now-superseded v2 draft that dropped it), different retrieval structure
(route to a SET of {LUAT, ND, QCVN} document partitions and retrieve within
each independently, vs. per-violation queries with soft doc_boost). Nothing
here is wired into `agent/pipeline.py`/`agent/cli.py`'s e0-e4 condition
runner; that integration is a separate decision.

    PERCEIVE (unchanged) -> ROUTE (§1) -> RETRIEVE within routed docs (§2)
        -> EXPAND via citation graph (§3) -> [SELECT]

What's implemented here:
  - §1 `route()` — question + perception.observed -> {"LUAT", "ND", "QCVN"}
  - §2 `retrieve_within_routed()` — hard per-document retrieval, budget
    split evenly across whatever routed, never a global top-k
  - §3 `expand_with_graph()` — 1-hop citation-graph closure: reverse takes
    all neighbors (capped), forward reranks its fan-out by the query text
    and keeps the top few, chunks tagged by source for ablation

What's NOT implemented here, out of scope for this module: Step 4's
eval_retrieval.py MCQ/full_recall harness (a measurement script, not
retrieval logic — this module exposes what it needs to be measured, doesn't
run the measurement itself); Step 5's dense/BGE-M3 retrieval engine
(explicitly "run only after steps 1-4 are measured" per the spec); the LLM
call that would produce a query from PERCEIVE's output (this module assumes
callers pass `query_text` — e.g. `perception.violation`, same convention
`agent/pipeline.py` already uses — no separate FORMULATE LLM call needed
under v3's design, unlike v2's channel-tagged multi-query generation).

Anchor snapping (`Bank`/`snap`/`load_act_phrase_bank`/`load_heading_bank`,
built for v2's channel P/R design) is kept below as available but NOT part
of the v3 pipeline — the spec demotes it to "Also optional... check before
building it [further]" pending the dense-retrieval comparison in Step 5.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.corpus import Corpus
from agent.retrieval.base import Retriever
from agent.retrieval.vn_tokenize import segment
from agent.schemas import Perception, RetrievalQuery, RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BANKS_DIR = ROOT / "banks"

# --------------------------------------------------------------------------- #
# §1 — route the question (+ perception) to a document partition
# --------------------------------------------------------------------------- #

ROUTE_LABEL_TO_DOC = {"ND": "ND_168_2024", "LUAT": "LUAT_36_2024", "QCVN": "QCVN_41_2024"}

# Patterns exactly as given in QUERY_GEN_SPEC.md §1.
ROUTE_ND_RE = re.compile(
    r"(mức phạt|phạt bao nhiêu|bị phạt|số tiền|xử phạt|chế tài|trừ.{0,12}điểm|tước quyền|giấy phép lái xe)", re.I
)
# Deliberately bare "biển", not "biển báo" (§1: narrowing it drops measured
# QCVN routing recall from 62.7% to ~40% on this corpus — verified directly;
# the corpus and the questions both say "biển cấm", not "biển báo cấm").
ROUTE_QCVN_RE = re.compile(r"(biển|vạch|đèn tín hiệu|làn đường|lề đường|dải phân cách)", re.I)
ROUTE_LUAT_RE = re.compile(
    r"(phải thực hiện|phải làm gì|xử lý như thế nào|phải chấp hành|có được phép|theo quy định|nguyên tắc|quy tắc)",
    re.I,
)


def route(question: str, perception: Perception | None = None) -> set[str]:
    """§1. Routes on question text + `perception.observed` together — sign/
    marking mentions come from the IMAGE, not always the question (spec's
    example: item 129 asks only "...đang thực hiện hành vi nào sau đây?",
    no routing signal in the text at all, yet its gold is QCVN_41_2024_PLB_3;
    the sign is in the image). `perception` is optional so callers without a
    PERCEIVE result yet (or doing a pure text-routing measurement, e.g. §1's
    own reported recall table) can still call this.

    Defaults to {"LUAT", "ND"} when nothing matches — never QCVN, since a
    generic/unmatched question skews LUAT+ND in the gold distribution (§1),
    and defaulting QCVN in would be a wrong guess more often than a hit."""
    text = question if perception is None else f"{question} {perception.observed}"
    routed: set[str] = set()
    if ROUTE_ND_RE.search(text):
        routed.add("ND")
    if ROUTE_QCVN_RE.search(text):
        routed.add("QCVN")
    if ROUTE_LUAT_RE.search(text):
        routed.add("LUAT")
    return routed or {"LUAT", "ND"}


# --------------------------------------------------------------------------- #
# §2 — retrieve within the routed documents (hard routing, not global top-k)
# --------------------------------------------------------------------------- #


def retrieve_from_document(
    query_text: str,
    doc_id: str,
    k: int,
    corpus: Corpus,
    retriever: Retriever,
    as_of: str | None = None,
) -> list[RetrievedChunk]:
    """Top-`k` chunks BELONGING TO `doc_id` — a hard per-document budget,
    not a soft `doc_boost` scoring bonus. Retrieves against the WHOLE corpus
    with a generous k first — the corpus is ~1000 chunks, so this is cheap —
    then filters to `doc_id` and truncates, so document-filtering doesn't
    starve `k` of candidates that a smaller internal overfetch would have
    already cut."""
    if k <= 0:
        return []
    query = RetrievalQuery(text=query_text)
    overfetch_k = max(k, len(corpus))
    hits = retriever.retrieve(query, corpus, overfetch_k, as_of)
    doc_hits = [c for c in hits if corpus.by_id.get(c.chunk_id) and corpus.by_id[c.chunk_id].document_id == doc_id]
    return doc_hits[:k]


def retrieve_within_routed(
    query_text: str,
    routed_labels: set[str],
    k_total: int,
    corpus: Corpus,
    retriever: Retriever,
    as_of: str | None = None,
) -> list[RetrievedChunk]:
    """§2. "Split the budget evenly across the routed documents and retrieve
    inside each independently. NOT a global top-k over the merged corpus" —
    measured +6.5 full_recall points over global top-k at budget 12 (§2's
    table); a reserved-slot-for-unrouted-documents variant scored WORSE than
    hard routing, so this deliberately gives zero budget to anything not
    routed.

    Order: grouped by document (stable order LUAT, ND, QCVN), each group
    internally score-sorted. Not merged into one globally-sorted list — that
    would reintroduce the exact cross-document score-comparison problem
    hard routing exists to avoid (a document whose BM25 scores run
    numerically higher for unrelated reasons would still crowd out the
    other route's results if resorted together)."""
    labels = [l for l in ("LUAT", "ND", "QCVN") if l in routed_labels]
    if not labels:
        return []
    base = k_total // len(labels)
    remainder = k_total - base * len(labels)
    results: list[RetrievedChunk] = []
    for i, label in enumerate(labels):
        k = base + (1 if i < remainder else 0)  # remainder goes to the first documents in stable order
        results.extend(retrieve_from_document(query_text, ROUTE_LABEL_TO_DOC[label], k, corpus, retriever, as_of))
    return results


# --------------------------------------------------------------------------- #
# §3 — close the citation graph (EXPAND)
# --------------------------------------------------------------------------- #


def load_citation_graph(path: Path | str = DEFAULT_BANKS_DIR / "citation_graph.json") -> dict[str, dict[str, list[str]]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TaggedChunk(BaseModel):
    """A chunk plus where it came from, so the ablation can attribute gains
    (§3: 'Tag chunks source: "retrieval" | "graph_fwd" | "graph_rev"')."""

    chunk: RetrievedChunk
    source: Literal["retrieval", "graph_fwd", "graph_rev"]


def expand_with_graph(
    retrieved: list[RetrievedChunk],
    graph: dict[str, dict[str, list[str]]],
    query_text: str,
    corpus: Corpus,
    max_rev: int = 8,
    max_fwd: int = 8,
) -> list[TaggedChunk]:
    """§3. 1 hop, additive — expanded chunks are appended AFTER `retrieved`,
    never displacing or reordering it (a `TaggedChunk` list, in exactly that
    priority order: retrieval, then graph_rev, then graph_fwd).

    Reverse: take ALL neighbors of anything retrieved. No reranking needed —
    §3 measured fan-in already bounded (<=11) in this graph, so `max_rev` is
    a safety cap, not the operative constraint; reverse recovers the sanction
    chunk from a conduct hit (or vice versa) 75.3% of the time regardless of
    which specific neighbor gets picked, because a conduct provision's
    sanction is usually singular.

    Forward: the opposite shape — a SANCTION chunk's fan-out covers MULTIPLE
    conduct provisions it applies to (§3's D7_K13 example alone points at 4+
    khoản), so forward expansion needs to DISCRIMINATE among them. Reranked
    by BM25 against `query_text` (the same text RETRIEVE used — it describes
    the conduct, which is exactly what should separate the right provision
    from its siblings in the fan-out) and truncated to `max_fwd` — §3: "top-5
    of the reranked pool captures the gold conduct chunk 84.6% of the time."
    """
    forward, reverse = graph.get("forward", {}), graph.get("reverse", {})
    seen = {c.chunk_id for c in retrieved}
    tagged = [TaggedChunk(chunk=c, source="retrieval") for c in retrieved]

    rev_ids: list[str] = []
    fwd_ids: list[str] = []
    for c in retrieved:
        for t in reverse.get(c.chunk_id, []):
            if t not in seen:
                rev_ids.append(t)
                seen.add(t)
        for t in forward.get(c.chunk_id, []):
            if t not in seen:
                fwd_ids.append(t)
                seen.add(t)

    for cid in rev_ids[:max_rev]:
        chunk = corpus.by_id.get(cid)
        if chunk:
            tagged.append(
                TaggedChunk(
                    chunk=RetrievedChunk(chunk_id=cid, citation=chunk.title, content=chunk.text, score=0.0),
                    source="graph_rev",
                )
            )

    fwd_ids = [cid for cid in fwd_ids if cid in corpus.by_id]
    if fwd_ids:
        bank = Bank([(cid, corpus.by_id[cid].text, [cid]) for cid in fwd_ids])
        scores = bank.bm25_scores(query_text)
        ranked = sorted(range(len(fwd_ids)), key=lambda i: scores[i], reverse=True)
        for i in ranked[:max_fwd]:
            cid = fwd_ids[i]
            chunk = corpus.by_id[cid]
            tagged.append(
                TaggedChunk(
                    chunk=RetrievedChunk(chunk_id=cid, citation=chunk.title, content=chunk.text, score=scores[i]),
                    source="graph_fwd",
                )
            )

    return tagged


# --------------------------------------------------------------------------- #
# Orchestration — route (§1) -> retrieve within routed docs (§2) -> expand (§3)
# --------------------------------------------------------------------------- #


def retrieve_with_expansion(
    question: str,
    query_text: str,
    perception: Perception | None,
    corpus: Corpus,
    retriever: Retriever,
    graph: dict[str, dict[str, list[str]]],
    k_total: int = 12,
    max_rev: int = 8,
    max_fwd: int = 8,
    as_of: str | None = None,
) -> list[TaggedChunk]:
    """The whole v3 pipeline in one call: route -> retrieve within routed
    documents -> close the citation graph. `query_text` is separate from
    `question` on purpose — routing (§1) uses the raw question + perception,
    but the actual search string is whatever FORMULATE/PERCEIVE produced
    (e.g. `perception.violation`, statute-register)."""
    routed = route(question, perception)
    retrieved = retrieve_within_routed(query_text, routed, k_total, corpus, retriever, as_of)
    return expand_with_graph(retrieved, graph, query_text, corpus, max_rev, max_fwd)


# --------------------------------------------------------------------------- #
# Anchor snapping — v2-era, NOT part of the v3 pipeline (see module
# docstring). Kept available: still-correct code, just not on the critical
# path until/unless the dense-retrieval comparison (Step 5) shows it's still
# worth it.
# --------------------------------------------------------------------------- #


class Anchor(BaseModel):
    anchor_id: str  # phrase_id (act_phrases) or heading text (headings)
    text: str
    chunk_ids: list[str] = Field(default_factory=list)  # what this anchor expands to on a snap
    score: float


class Bank:
    """One snapping target set (either act_phrases.json's phrase list or
    headings.json's heading->chunk_ids map), pre-tokenized for repeated
    BM25 scoring against many free-text queries. Also reused internally by
    `expand_with_graph()` to rerank a forward fan-out — same math, different
    corpus of "documents" (candidate chunks instead of bank phrases)."""

    def __init__(self, entries: list[tuple[str, str, list[str]]]):
        """`entries`: list of (anchor_id, text, chunk_ids)."""
        self.ids = [e[0] for e in entries]
        self.texts = [e[1] for e in entries]
        self.chunk_ids = [e[2] for e in entries]
        self._doc_tokens = [segment(t).lower().split() for t in self.texts]
        self._df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = (sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)) if self._doc_tokens else 0.0
        self._n = len(self._doc_tokens)

    def __len__(self) -> int:
        return self._n

    def bm25_scores(self, query_text: str, k1: float = 0.9, b: float = 0.4) -> list[float]:
        """Standard Okapi BM25 over this bank's entries, tokenized the same
        way the real corpus is (agent.retrieval.vn_tokenize.segment) so
        snapping and the corpus BM25 index agree on term boundaries."""
        if self._n == 0:
            return []
        q_terms = segment(query_text).lower().split()
        scores = [0.0] * self._n
        for term in set(q_terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for i, doc_toks in enumerate(self._doc_tokens):
                tf = doc_toks.count(term)
                if tf == 0:
                    continue
                dl = len(doc_toks) or 1
                denom = tf + k1 * (1 - b + b * dl / (self._avgdl or 1))
                scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores


def load_act_phrase_bank(path: Path | str = DEFAULT_BANKS_DIR / "act_phrases.json") -> Bank:
    """`is_xref` entries are excluded — pointer phrases match everything and
    describe nothing; they belong to the citation graph, not a snapping
    bank."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [(e["phrase_id"], e["text"], [e["chunk_id"]]) for e in raw if not e.get("is_xref", False)]
    return Bank(entries)


def load_heading_bank(path: Path | str = DEFAULT_BANKS_DIR / "headings.json") -> Bank:
    """Snapping to a heading expands to every chunk filed under it."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [(heading, heading, chunk_ids) for heading, chunk_ids in raw.items()]
    return Bank(entries)


def snap(free_text: str, bank: Bank, top_n: int = 2, min_score: float = 0.0) -> list[Anchor]:
    """Returns [] if the best score is below `min_score` (TAU)."""
    if len(bank) == 0:
        return []
    scores = bank.bm25_scores(free_text)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    anchors = []
    for i in ranked[:top_n]:
        if scores[i] < min_score:
            break
        anchors.append(
            Anchor(anchor_id=bank.ids[i], text=bank.texts[i], chunk_ids=bank.chunk_ids[i], score=scores[i])
        )
    return anchors


def rrf_fuse(ranked_lists: list[list[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion (Cormack et al. 2009's standard rrf_k=60).
    Not used by the v3 pipeline (§2/§3 use hard routing + append-after-not-
    fuse, not RRF) — kept as a utility for anyone still using the snapping
    machinery above, which used RRF to combine a generated-text query with
    its snapped anchor under the old v2 design."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if chunk.chunk_id not in best_chunk or chunk.score > best_chunk[chunk.chunk_id].score:
                best_chunk[chunk.chunk_id] = chunk
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [best_chunk[cid].model_copy(update={"score": scores[cid]}) for cid in ordered_ids]
