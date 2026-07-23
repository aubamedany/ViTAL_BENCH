"""Per-item JSONL logging — the leakage table and retrieval-vs-reasoning
decomposition are built from these logs, not from re-runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.pipeline import Phase1Trace
from agent.schemas import RetrievedChunk, Selection


def build_log_record(
    item_id: Any,
    condition: str,
    model: str,
    corpus_version: str,
    seed: int,
    trace: Phase1Trace | None,
    retrieved: list[RetrievedChunk],
    selection: Selection | None,
    gold_answer: str | None,
    gold_citations_raw: list[str],
    gold_citations_resolved: list[str],
    wall_clock_sec: float,
    assertion_count: int,
    extra: dict | None = None,
) -> dict:
    record = {
        "item_id": item_id,
        "condition": condition,
        "model": model,
        "corpus_version": corpus_version,
        "seed": seed,
        "perception": trace.perception.model_dump() if trace else None,
        "queries": trace.queries if trace else [],
        "iterations": [asdict(it) for it in trace.iterations] if trace else [],
        "iterations_used": trace.iterations_used if trace else 0,
        "retry_used": trace.retry_used if trace else False,
        "stop_reason": trace.stop_reason if trace else None,
        "retrieved_chunk_ids": [c.chunk_id for c in retrieved],
        "selection": selection.model_dump() if selection else None,
        "gold_answer": gold_answer,
        "gold_citations_raw": gold_citations_raw,
        "gold_citations_resolved": gold_citations_resolved,
        "wall_clock_sec": wall_clock_sec,
        "option_leakage_assertions": assertion_count,
    }
    if extra:
        record.update(extra)
    return record


def append_jsonl(path: Path | str, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
