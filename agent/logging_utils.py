"""Per-item JSONL logging (AGENT_PLAN.md §7) — the leakage table, budget curve,
and retrieval-vs-reasoning decomposition are all built from these logs, not
from re-runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.pipeline import Phase1Trace
from agent.schemas import LegalDetermination, Selection


def build_log_record(
    item_id: Any,
    condition: str,
    model: str,
    corpus_version: str,
    seed: int,
    trace: Phase1Trace | None,
    determination: LegalDetermination | None,
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
        "scene": trace.scene.model_dump() if trace else None,
        "iterations": trace.iterations if trace else [],
        "stop_reason": trace.stop_reason if trace else None,
        "determination": determination.model_dump() if determination else None,
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
