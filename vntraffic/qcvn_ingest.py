"""QCVN 41:2024 ingestion tool (EXPERIMENT_PLAN.md §7.2, Path A enablement).

The shipped corpus has only 2 QCVN chunks, so 72 distinct gold citations (the
QCVN sign-code articles/appendices + a few TT circular refs) are unresolvable
today. This tool is the bridge from Path B (run now on the 974-item gradeable
subset) to Path A (ingest QCVN → all law items gradeable), to be run later once
the QCVN 41:2024 source text is available.

Workflow:

  1. ``build_skeleton(items, corpus)`` scans the unresolvable gold citations and
     writes a YAML skeleton with one entry per unresolvable citation. A human
     fills ``content`` (and optionally ``article_title`` etc.) from the official
     QCVN text. Each entry proposes a ``chunk_id`` and lists the citation(s) it
     ``covers``.
  2. ``merge_skeleton(...)`` reads the filled YAML, builds schema-conforming
     ``Chunk`` objects (constructing ``text_for_embedding`` with the same
     template the corpus uses), writes an augmented corpus + a citation-alias
     map, and prints the coverage delta.

Nothing here fabricates legal text: skeleton entries with an empty ``content``
are skipped on merge and reported, so a partial fill simply resolves fewer
citations — never wrong ones.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import constants as C
from .corpus import Chunk, Corpus, load_corpus, save_corpus
from .resolver import resolve_citation, gold_citations_from_items

try:  # PyYAML is in requirements.txt; degrade to JSON if unavailable.
    import yaml
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    _HAVE_YAML = False


# text_for_embedding template, matching the shipped corpus format.
_EMBED_TEMPLATE = ("Văn bản: {document_name}. Trích dẫn: {citation}. "
                   "Ngữ cảnh: {section_context}. Tên điều hoặc mục: {article_title}. "
                   "Nội dung: {content}")


def build_text_for_embedding(document_name: str, citation: str, section_context: str,
                             article_title: str, content: str) -> str:
    return _EMBED_TEMPLATE.format(
        document_name=document_name, citation=citation,
        section_context=section_context, article_title=article_title, content=content,
    )


def _proposed_chunk_id(citation: str) -> str:
    """A safe default chunk_id for a QCVN/TT citation: uppercase the string."""
    return citation.upper()


def unresolvable_gold_citations(items: Sequence, corpus: Corpus) -> List[str]:
    """Distinct gold citations that do not resolve against *corpus*."""
    distinct = sorted(set(gold_citations_from_items(items)))
    return [c for c in distinct if not resolve_citation(c, corpus).resolved]


def build_skeleton(items: Sequence, corpus: Corpus, path: str) -> int:
    """Write a fill-in skeleton for every unresolvable gold citation.

    Returns the number of skeleton entries written.
    """
    unresolved = unresolvable_gold_citations(items, corpus)
    # Count how many items reference each citation, to help prioritize filling.
    ref_counts: Dict[str, int] = {}
    for it in items:
        for c in getattr(it, "law", []) or []:
            if c in unresolved:
                ref_counts[c] = ref_counts.get(c, 0) + 1

    entries = []
    for cit in unresolved:
        entries.append({
            "citation": cit,                       # the gold citation to resolve
            "covers": [cit],                       # citation strings this chunk grades
            "n_items_referencing": ref_counts.get(cit, 0),
            "chunk_id": _proposed_chunk_id(cit),   # editable
            "document_id": "QCVN_41_2024",         # editable
            "document_name": "Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ",
            "article": "",
            "article_title": "",
            "section_context": "",
            "clause": None,
            "citation_human": "",                  # human-readable VN citation
            "effective_date": "2025-01-01",
            "content": "",                         # <-- fill from official QCVN text
        })

    doc = {
        "_instructions": (
            "Fill 'content' (and ideally article_title/section_context) from the "
            "official QCVN 41:2024 text. Keep or edit 'chunk_id'. Multiple "
            "citations that map to the same statutory unit may share a chunk_id "
            "and list all of them under 'covers'. Entries with empty 'content' "
            "are skipped on merge."
        ),
        "corpus_version_at_skeleton": corpus.version,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _dump(doc, path)
    return len(entries)


def _dump(doc: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if _HAVE_YAML and path.endswith((".yaml", ".yml")):
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
        else:
            json.dump(doc, f, ensure_ascii=False, indent=2)


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if _HAVE_YAML and path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


@dataclass
class MergeResult:
    n_entries: int
    n_filled: int                 # entries with non-empty content, merged as chunks
    n_skipped_empty: int
    n_new_chunks: int
    n_aliases: int
    coverage_before: dict
    coverage_after: dict
    out_corpus_path: str
    out_alias_path: str


def merge_skeleton(items: Sequence, corpus: Corpus, skeleton_path: str,
                   out_corpus_path: str,
                   out_alias_path: Optional[str] = None) -> MergeResult:
    """Merge a filled skeleton into the corpus + alias map."""
    from .resolver import build_coverage_report  # local import avoids cycle at import time

    doc = _load(skeleton_path)
    entries = doc.get("entries", [])
    golds = gold_citations_from_items(items)
    cov_before = build_coverage_report(golds, corpus).counts

    new_chunks: List[Chunk] = []
    aliases: Dict[str, str] = dict(corpus.aliases)
    seen_chunk_ids = set(corpus._by_id.keys())
    n_filled = n_skipped = 0

    for e in entries:
        content = (e.get("content") or "").strip()
        if not content:
            n_skipped += 1
            continue
        n_filled += 1
        chunk_id = (e.get("chunk_id") or _proposed_chunk_id(e["citation"])).strip()
        citation_human = e.get("citation_human") or e.get("citation")
        if chunk_id not in seen_chunk_ids:
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=e.get("document_id", "QCVN_41_2024"),
                content=content,
                citation=citation_human,
                article=str(e.get("article", "")),
                article_title=e.get("article_title", ""),
                section_context=e.get("section_context", ""),
                clause=e.get("clause"),
                document_name=e.get("document_name", ""),
                effective_date=e.get("effective_date", "2025-01-01"),
                vehicle_type=list(e.get("vehicle_type", []) or []),
                text_for_embedding=build_text_for_embedding(
                    e.get("document_name", ""), citation_human,
                    e.get("section_context", ""), e.get("article_title", ""), content),
            )
            new_chunks.append(chunk)
            seen_chunk_ids.add(chunk_id)
        # register aliases for every covered citation
        for cov in (e.get("covers") or [e["citation"]]):
            aliases[cov.upper()] = chunk_id

    merged = Corpus(list(corpus.chunks) + new_chunks, version="pending", aliases=aliases)
    save_corpus(merged, out_corpus_path)
    out_alias_path = out_alias_path or (os.path.splitext(out_corpus_path)[0] + ".aliases.json")
    with open(out_alias_path, "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False, indent=2)

    # reload so version reflects the augmented corpus + aliases
    reloaded = load_corpus(out_corpus_path, out_alias_path)
    cov_after = build_coverage_report(golds, reloaded).counts

    return MergeResult(
        n_entries=len(entries), n_filled=n_filled, n_skipped_empty=n_skipped,
        n_new_chunks=len(new_chunks), n_aliases=len(aliases),
        coverage_before=cov_before, coverage_after=cov_after,
        out_corpus_path=out_corpus_path, out_alias_path=out_alias_path,
    )
