"""Statute corpus loader (AGENT_PLAN.md §1 invariant I5, §3.3 document_id).

Loading goes through one function so the corpus version is always known and
logged, and so swapping law_db.json for a re-ingested corpus (e.g. with QCVN
41 properly chunked, §3.5) requires zero code change elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = ROOT / "law_db.json"


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    text: str
    title: str
    document_id: str  # derived from chunk_id, e.g. "LUAT_36_2024"
    # Not present in today's law_db.json records — read as "" / None until the
    # corpus is enriched with this metadata (§3.3). Hard filters over these
    # fields fail open (no-op) until then; see agent/retrieval/filters.py.
    vehicle_type: str = ""
    effective_date: str | None = None


def _document_id(chunk_id: str) -> str:
    """chunk_id is always PREFIX_NUM_YEAR_... (LUAT_36_2024_..., ND_168_2024_...,
    QCVN_41_2024_...) — the document id is the first three underscore tokens."""
    parts = chunk_id.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else chunk_id


class Corpus:
    """In-memory statute corpus with a version hash for reproducibility (§10)."""

    def __init__(self, chunks: list[CorpusChunk], version: str, path: Path):
        self.chunks = chunks
        self.version = version
        self.path = path
        self.by_id: dict[str, CorpusChunk] = {c.chunk_id: c for c in chunks}

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CORPUS_PATH) -> "Corpus":
        path = Path(path)
        raw_bytes = path.read_bytes()
        version = hashlib.sha1(raw_bytes).hexdigest()[:12]
        records = json.loads(raw_bytes.decode("utf-8"))
        chunks = [
            CorpusChunk(
                chunk_id=rec["chunk_id"],
                text=rec["text"],
                title=rec.get("title", ""),
                document_id=_document_id(rec["chunk_id"]),
                vehicle_type=rec.get("vehicle_type", ""),
                effective_date=rec.get("effective_date"),
            )
            for rec in records
        ]
        return cls(chunks, version, path)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, chunk_id: str) -> CorpusChunk:
        return self.by_id[chunk_id]

    def document_ids(self) -> set[str]:
        return {c.document_id for c in self.chunks}
