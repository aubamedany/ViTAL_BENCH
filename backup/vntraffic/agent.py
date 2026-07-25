"""VNTraffic-Agent: the option-blind two-phase agent (EXPERIMENT_PLAN.md §3, Claim 4).

Structural option-blindness (not a prompt convention):
  * Phase 1 (PERCEIVE → FORMULATE → RETRIEVE → ASSESS → REFINE ≤ B → DETERMINE)
    never receives the answer options — they are not passed into any Phase-1
    call. It emits a ``LegalDetermination`` that may cite **only** retrieved
    chunk_ids; provisions outside the retrieved set are dropped and logged
    (the citation whitelist).
  * Phase 2 (SELECT) maps the frozen determination to A/B/C/D with **no image**
    and **no retriever handle** — it cannot re-perceive or retrieve.

Every step is recorded in an audit trail (queries, retrieved sets + scores, raw
model outputs, whitelist drops) — required for the leakage decomposition and the
budget ablation A1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import constants as C
from . import prompts as P
from .models.base import GenerationConfig, VLMClient
from .resolver import resolve_citation
from .retriever import Retriever


@dataclass
class AgentResult:
    answer: Optional[str]                     # final A/B/C/D (may be None if unparsable)
    perception: str
    determination: dict                       # post-whitelist LegalDetermination
    retrieved_chunk_ids: List[str]            # union across all retrieval rounds
    kept_citations: List[str]                 # provisions surviving the whitelist
    dropped_citations: List[str]              # provisions cited but not retrieved
    budget_used: int                          # number of retrieval rounds actually run
    audit: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "perception": self.perception,
            "determination": self.determination,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "kept_citations": self.kept_citations,
            "dropped_citations": self.dropped_citations,
            "budget_used": self.budget_used,
            "audit": self.audit,
        }


class VNTrafficAgent:
    def __init__(self, client: VLMClient, retriever: Retriever,
                 budget: int = C.DEFAULT_AGENT_BUDGET, top_k: int = C.DEFAULT_TOP_K,
                 gen_cfg: Optional[GenerationConfig] = None):
        self.client = client
        self.retriever = retriever
        self.budget = budget
        self.top_k = top_k
        self.gen_cfg = gen_cfg or GenerationConfig()

    # ---- helpers --------------------------------------------------------- #
    def _chunks_payload(self, chunk_ids: List[str]) -> List[dict]:
        """Build {chunk_id, text} payloads from the corpus for prompt rendering."""
        out = []
        for cid in chunk_ids:
            ch = self.retriever.corpus.get(cid)
            out.append({"chunk_id": cid, "text": ch.content if ch else ""})
        return out

    def _accumulate(self, ordered: "Dict[str, float]", hits) -> None:
        for h in hits:
            if h.chunk_id not in ordered or h.score > ordered[h.chunk_id]:
                ordered[h.chunk_id] = h.score

    # ---- Phase 1 --------------------------------------------------------- #
    def run_phase1(self, image_path: Optional[str], question: str) -> AgentResult:
        audit: List[dict] = []

        # PERCEIVE (image + question, NO options)
        perceive_prompt = P.build_perceive(question)
        perceive_raw = self.client.generate(
            perceive_prompt, images=[image_path] if image_path else None,
            system=P.SYSTEM_VI, gen_cfg=self.gen_cfg)
        perception = P.extract_json(perceive_raw) or {}
        perception = perception.get("perception") if isinstance(perception, dict) else None
        perception = perception or perceive_raw.strip()
        audit.append({"step": "perceive", "raw": perceive_raw, "perception": perception})

        # FORMULATE first query
        formulate_raw = self.client.generate(
            P.build_formulate(question, perception), system=P.SYSTEM_VI, gen_cfg=self.gen_cfg)
        query = P.parse_query(formulate_raw)
        audit.append({"step": "formulate", "raw": formulate_raw, "query": query})

        # RETRIEVE / ASSESS / REFINE loop
        retrieved: Dict[str, float] = {}
        rounds = 0
        for i in range(self.budget):
            hits = self.retriever.search(query, k=self.top_k)
            self._accumulate(retrieved, hits)
            rounds += 1
            audit.append({"step": "retrieve", "round": i, "query": query,
                          "hits": [h.to_dict() for h in hits]})
            if i == self.budget - 1:
                break  # budget exhausted
            chunks = self._chunks_payload(list(retrieved.keys()))
            assess_raw = self.client.generate(
                P.build_assess(question, perception, chunks),
                system=P.SYSTEM_VI, gen_cfg=self.gen_cfg)
            assessment = P.parse_assessment(assess_raw)
            audit.append({"step": "assess", "round": i, "raw": assess_raw,
                          "assessment": assessment})
            if assessment["sufficient"]:
                break
            if assessment["query"]:
                query = assessment["query"]  # REFINE

        # DETERMINATION (still option-blind)
        retrieved_ids = list(retrieved.keys())
        chunks = self._chunks_payload(retrieved_ids)
        det_raw = self.client.generate(
            P.build_determination(question, perception, chunks),
            system=P.SYSTEM_VI, gen_cfg=self.gen_cfg)
        determination = P.parse_determination(det_raw)
        audit.append({"step": "determination", "raw": det_raw})

        # citation whitelist: keep only provisions that map into the retrieved set
        retrieved_resolved = {
            resolve_citation(cid, self.retriever.corpus).chunk_id or cid
            for cid in retrieved_ids
        }
        kept, dropped = [], []
        for prov in determination["applicable_provisions"]:
            r = resolve_citation(prov, self.retriever.corpus)
            target = r.chunk_id or prov
            if prov in retrieved_ids or target in retrieved_resolved:
                kept.append(prov)
            else:
                dropped.append(prov)
        determination["applicable_provisions"] = kept
        audit.append({"step": "whitelist", "kept": kept, "dropped": dropped})

        return AgentResult(
            answer=None, perception=perception, determination=determination,
            retrieved_chunk_ids=retrieved_ids, kept_citations=kept,
            dropped_citations=dropped, budget_used=rounds, audit=audit,
        )

    # ---- Phase 2 --------------------------------------------------------- #
    def run_phase2(self, question: str, options: Dict[str, str],
                   result: AgentResult) -> AgentResult:
        select_prompt = P.build_select(question, options, result.determination)
        # NB: no image, no retriever — pure mapping of the frozen determination
        select_raw = self.client.generate(select_prompt, images=None,
                                           system=P.SYSTEM_VI, gen_cfg=self.gen_cfg)
        result.answer = P.parse_answer_letter(select_raw)
        result.audit.append({"step": "select", "raw": select_raw, "answer": result.answer})
        return result

    # ---- full item ------------------------------------------------------- #
    def run(self, image_path: Optional[str], question: str,
            options: Dict[str, str]) -> AgentResult:
        result = self.run_phase1(image_path, question)
        return self.run_phase2(question, options, result)
