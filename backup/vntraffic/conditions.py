"""The E1-E4 conditions behind one interface (EXPERIMENT_PLAN.md §3).

Each condition maps a QAItem to an ItemResult. The rungs differ by exactly one
variable:

  E1  parametric      no retrieval
  E2  RAG-Q           retrieval query = question only          (options hidden)
  E3  RAG-QO          retrieval query = question + options      (options exposed)
  E4  VNTraffic-Agent option-blind two-phase agent

E2 and E3 share every line except ``build_query`` — that single difference is
what isolates option leakage (Claim 3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import constants as C
from . import prompts as P
from .agent import VNTrafficAgent
from .dataset import QAItem, Permutation, apply_permutation
from .models.base import GenerationConfig, VLMClient
from .retriever import Retriever


@dataclass
class ItemResult:
    id: int
    condition: str
    tier: str
    category: str
    gt: str
    answer: Optional[str]
    correct: bool
    pred_citations: List[str] = field(default_factory=list)
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieval_query: Optional[str] = None
    determination: Optional[dict] = None
    budget_used: Optional[int] = None            # E4 retrieval rounds actually run
    permutation: Optional[Permutation] = None   # non-identity only for CircularEval
    raw: Optional[str] = None                    # raw model output (E1-E3)
    audit: Optional[List[dict]] = None           # agent trail (E4)
    timing_s: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "condition": self.condition, "tier": self.tier,
            "category": self.category, "gt": self.gt, "answer": self.answer,
            "correct": self.correct, "pred_citations": self.pred_citations,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieval_query": self.retrieval_query,
            "determination": self.determination,
            "budget_used": self.budget_used,
            "timing_s": round(self.timing_s, 4),
        }
        if self.permutation and self.permutation != {L: L for L in C.OPTION_LETTERS}:
            d["permutation"] = self.permutation
        if self.raw is not None:
            d["raw"] = self.raw
        if self.audit is not None:
            d["audit"] = self.audit
        return d


class Condition:
    """Base condition. Subclasses implement :meth:`_run`."""
    name = "abstract"
    uses_retrieval = False

    def __init__(self, client: VLMClient, image_root: str = C.IMAGE_ROOT,
                 retriever: Optional[Retriever] = None,
                 gen_cfg: Optional[GenerationConfig] = None,
                 top_k: int = C.DEFAULT_TOP_K, budget: int = C.DEFAULT_AGENT_BUDGET):
        self.client = client
        self.image_root = image_root
        self.retriever = retriever
        self.gen_cfg = gen_cfg or GenerationConfig()
        self.top_k = top_k
        self.budget = budget

    def run_item(self, item: QAItem,
                 permutation: Optional[Permutation] = None) -> ItemResult:
        eff = apply_permutation(item, permutation) if permutation else item
        t0 = time.perf_counter()
        res = self._run(eff)
        res.timing_s = time.perf_counter() - t0
        res.permutation = permutation
        res.correct = (res.answer == eff.gt)
        # store GT of the *effective* (possibly permuted) item so `correct` is consistent
        res.gt = eff.gt
        return res

    def _run(self, item: QAItem) -> ItemResult:
        raise NotImplementedError

    def _base_result(self, item: QAItem) -> ItemResult:
        return ItemResult(id=item.id, condition=self.name, tier=item.tier,
                          category=item.category, gt=item.gt, answer=None, correct=False)


class E1Parametric(Condition):
    name = C.COND_E1

    def _run(self, item: QAItem) -> ItemResult:
        res = self._base_result(item)
        prompt = P.build_e1_answer(item.question, item.options)
        raw = self.client.generate(prompt, images=[item.image_path(self.image_root)],
                                    system=P.SYSTEM_VI, gen_cfg=self.gen_cfg)
        res.answer = P.parse_answer_letter(raw)
        res.raw = raw
        return res


class _RagCondition(Condition):
    uses_retrieval = True

    def build_query(self, item: QAItem) -> str:
        raise NotImplementedError

    def _run(self, item: QAItem) -> ItemResult:
        res = self._base_result(item)
        query = self.build_query(item)
        hits = self.retriever.search(query, k=self.top_k)
        chunks = []
        for h in hits:
            ch = self.retriever.corpus.get(h.chunk_id)
            chunks.append({"chunk_id": h.chunk_id, "text": ch.content if ch else ""})
        prompt = P.build_rag_answer(item.question, item.options, chunks)
        raw = self.client.generate(prompt, images=[item.image_path(self.image_root)],
                                    system=P.SYSTEM_VI, gen_cfg=self.gen_cfg)
        res.answer = P.parse_answer_letter(raw)
        res.pred_citations = P.parse_citations(raw)
        res.retrieved_chunk_ids = [h.chunk_id for h in hits]
        res.retrieval_query = query
        res.raw = raw
        return res


class E2RagQ(_RagCondition):
    name = C.COND_E2

    def build_query(self, item: QAItem) -> str:
        return item.question  # options hidden


class E3RagQO(_RagCondition):
    name = C.COND_E3

    def build_query(self, item: QAItem) -> str:
        # options exposed to retrieval — the leakage probe (the ONLY diff vs E2)
        return item.question + "\n" + P.render_options(item.options)


class E4Agent(Condition):
    name = C.COND_E4
    uses_retrieval = True

    def _run(self, item: QAItem) -> ItemResult:
        res = self._base_result(item)
        agent = VNTrafficAgent(self.client, self.retriever, budget=self.budget,
                               top_k=self.top_k, gen_cfg=self.gen_cfg)
        ar = agent.run(item.image_path(self.image_root), item.question, item.options)
        res.answer = ar.answer
        res.pred_citations = ar.kept_citations
        res.retrieved_chunk_ids = ar.retrieved_chunk_ids
        res.determination = ar.determination
        res.budget_used = ar.budget_used
        res.audit = ar.audit
        return res


_CONDITIONS = {c.name: c for c in (E1Parametric, E2RagQ, E3RagQO, E4Agent)}


def build_condition(name: str, **kwargs) -> Condition:
    if name not in _CONDITIONS:
        raise KeyError(f"unknown condition '{name}'. Known: {sorted(_CONDITIONS)}")
    return _CONDITIONS[name](**kwargs)
