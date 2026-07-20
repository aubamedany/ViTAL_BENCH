"""Statute chunk corpus loading and versioning (EXPERIMENT_PLAN.md §6).

The corpus (legal_traffic_chunks.json) is a flat JSON list of chunks:

    chunk_id        str        UPPERCASE, e.g. "LUAT_36_2024_D84_K2", "QCVN_41_2024_M1"
    document_id     str        LUAT_36_2024 | ND_168_2024 | QCVN_41_2024 | ...
    article         str
    article_title   str
    section_context str
    clause          str|null
    document_name   str
    citation        str        human-readable VN citation
    content         str        statute text
    effective_date  str
    vehicle_type    list[str]
    text_for_embedding str     pre-formatted embedding string

The corpus version string is a content hash so that any change to the corpus
(e.g. QCVN ingestion) yields a new version and invalidates retrieval caches.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import constants as C


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text_for_embedding: str
    content: str
    citation: str = ""
    article: str = ""
    article_title: str = ""
    section_context: str = ""
    clause: Optional[str] = None
    document_name: str = ""
    effective_date: str = ""
    vehicle_type: List[str] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        known = {"chunk_id", "document_id", "text_for_embedding", "content",
                 "citation", "article", "article_title", "section_context",
                 "clause", "document_name", "effective_date", "vehicle_type"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            chunk_id=d["chunk_id"],
            document_id=d.get("document_id", ""),
            text_for_embedding=d.get("text_for_embedding") or d.get("content", ""),
            content=d.get("content", ""),
            citation=d.get("citation", ""),
            article=str(d.get("article", "")),
            article_title=d.get("article_title", ""),
            section_context=d.get("section_context", ""),
            clause=d.get("clause"),
            document_name=d.get("document_name", ""),
            effective_date=d.get("effective_date", ""),
            vehicle_type=list(d.get("vehicle_type", []) or []),
            extra=extra,
        )

    def to_dict(self) -> dict:
        d = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "article": self.article,
            "article_title": self.article_title,
            "section_context": self.section_context,
            "clause": self.clause,
            "document_name": self.document_name,
            "citation": self.citation,
            "content": self.content,
            "effective_date": self.effective_date,
            "vehicle_type": list(self.vehicle_type),
            "text_for_embedding": self.text_for_embedding,
        }
        d.update(self.extra)
        return d


class Corpus:
    """Holds chunks and provides id lookup + a content-hash version string."""

    def __init__(self, chunks: Sequence[Chunk], version: str,
                 aliases: Optional[Dict[str, str]] = None):
        self.chunks: List[Chunk] = list(chunks)
        self.version = version
        self._by_id: Dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        # case-insensitive lookup index (chunk_ids are uppercase, but be safe)
        self._by_id_upper: Dict[str, Chunk] = {c.chunk_id.upper(): c for c in self.chunks}
        # citation-alias map (UPPERCASE citation -> chunk_id), populated by QCVN
        # ingestion for irregular citation formats that no case/point rule reaches.
        # Empty for the shipped corpus, so resolver coverage is unchanged.
        self.aliases: Dict[str, str] = {
            k.upper(): v for k, v in (aliases or {}).items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    def get(self, chunk_id: str) -> Optional[Chunk]:
        c = self._by_id.get(chunk_id)
        if c is not None:
            return c
        return self._by_id_upper.get(chunk_id.upper())

    def has(self, chunk_id: str) -> bool:
        return self.get(chunk_id) is not None

    def alias_target(self, citation: str) -> Optional[str]:
        """Return the chunk_id an alias maps *citation* to, if any."""
        cid = self.aliases.get(citation.upper())
        if cid is not None and self.has(cid):
            return self.get(cid).chunk_id
        return None

    def document_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.chunks:
            out[c.document_id] = out.get(c.document_id, 0) + 1
        return out


def corpus_version(path: str = C.CORPUS_JSON) -> str:
    """12-char content hash of the corpus file, stamped on every run."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:12]


def default_alias_path(corpus_path: str) -> str:
    """Sibling alias file for a corpus, e.g. ``x.json`` → ``x.aliases.json``."""
    base, ext = os.path.splitext(corpus_path)
    return base + ".aliases.json"


def load_corpus(path: str = C.CORPUS_JSON,
                aliases_path: Optional[str] = None) -> Corpus:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [Chunk.from_dict(d) for d in raw]
    aliases: Dict[str, str] = {}
    ap = aliases_path or default_alias_path(path)
    if os.path.exists(ap):
        with open(ap, "r", encoding="utf-8") as f:
            aliases = json.load(f)
    # The corpus version folds in the alias file too, so ingesting aliases
    # invalidates retrieval/grounding caches just like changing chunks.
    ver = corpus_version(path)
    if aliases:
        h = hashlib.sha256((ver + json.dumps(aliases, sort_keys=True)).encode())
        ver = h.hexdigest()[:12]
    return Corpus(chunks, version=ver, aliases=aliases)


def save_corpus(corpus: Corpus, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in corpus.chunks], f, ensure_ascii=False, indent=2)
