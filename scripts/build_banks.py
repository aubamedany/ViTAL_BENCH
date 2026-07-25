#!/usr/bin/env python
"""Offline bank builder for QUERY_GEN_SPEC.md §4 — the artifacts FORMULATE's
snapping (§6.1) and EXPAND's citation-graph closure (§5) need:
act_phrases.json (§4.3, channel P), headings.json (§4.4, channel R+P), and
citation_graph.json (§4.1, EXPAND's reverse/forward pointer graph).

Deliberately NOT built here (a different concern — retrieval-time filtering,
not query generation or graph expansion): vehicle_meta.json (§4.2). See
QUERY_GEN_SPEC.md §9 for the full offline-artifact order.

Usage:
    python -m scripts.build_banks [--corpus law_db.json] [--out-dir banks]
    python -m scripts.build_banks --verify [--questions law.json]
        also runs §4.1's acceptance test: reverse expansion must recover
        >=1 xref gold chunk for >=42% of items that need one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHUNK_ID_RE = re.compile(r"^(LUAT_36_2024|ND_168_2024)_D(\d+)_K(\d+)$")

# §4.3/§4.1 — one lettered sub-item per match. Multiline: ^ anchors to the
# start of each line, so this only fires on genuine "a) ..." list items, not
# incidental "a)" substrings mid-sentence.
ITEM_RE = re.compile(r"(?m)^([a-zđ]{1,2})\)\s*(.+?)(?=;\s*$|\.\s*$|$)")

# A phrase body that itself just points elsewhere ("quy định tại điểm...")
# describes nothing on its own — it belongs to the citation graph (§4.1),
# not the snapping bank (§4.3). ".{0,25}" (not bare whitespace) because real
# phrasing varies: "quy định tại điểm b khoản 3" but also "quy định tại một
# trong các điểm, khoản sau của Điều này" — same idiom as agent/pipeline.py's
# _TRU_DIEM_RE/_TUOC_QUYEN_RE (a literal-adjacency check misses real phrasing
# with words in between).
XREF_RE = re.compile(r"quy định tại.{0,25}(điểm|khoản)")

# §4.2's extractor, reused here only to populate act_phrases.json's
# vehicle_type field — NOT a full implementation of §4.2 (that also changes
# CorpusChunk/passes_vehicle_filter, a retrieval-filtering concern, not
# query generation).
VEHICLE_RE = re.compile(r"Phương tiện:\s*([^.]+?)\.\s*(?:Nội dung|Tên điều)")

# §4.4. Primary pattern per spec. One ND chunk (ND_168_2024_D7_K3) lacks the
# "Phương tiện:"/"Nội dung:" label entirely (a pre-existing corpus quirk,
# also the chunk behind the DEFAULT_MAX_CHUNK_CHARS truncation bug fixed
# earlier — see agent/llm_vlm.py) — the fallback recovers it by matching up
# to the first digit-enumerated clause instead.
HEAD_RE = re.compile(r"Tên điều hoặc mục:\s*([^.]+?)\.\s*(?:Phương tiện|Nội dung):")
HEAD_FALLBACK_RE = re.compile(r"Tên điều hoặc mục:\s*([^.]+?)\.\s+\d+\.")

# --------------------------------------------------------------------------- #
# §4.1 — citation graph (điểm-aware pointer parser)
# --------------------------------------------------------------------------- #

# "Điều này" / "Điều 7" markers. Applied two ways below: (a) to find the
# GOVERNING article for a list of khoản mentions that share one trailing
# marker (real phrasing: "điểm b khoản 3; khoản 5; ... điểm a khoản 7 Điều
# này" — only the LAST segment states the article, for the whole list); (b)
# to find a marker attached directly after one specific khoản mention, which
# overrides the governing one for just that mention.
DIEU_RE = re.compile(r"Điều\s+(này|\d+)")
KHOAN_RE = re.compile(r"khoản\s+(\d+)")
# A Điều marker immediately followed by this is pointing at an EXTERNAL
# document (e.g. "Điều 6 của Luật Xử lý vi phạm hành chính") — not a chunk
# in this corpus. Excluding these is what keeps forward edges from landing
# on articles that don't exist in the pruned 775-chunk corpus.
EXTERNAL_DOC_RE = re.compile(r"^\s*(?:của\s+Luật|của\s+Nghị định\s+\d)")


def parse_chunk_id(chunk_id: str) -> tuple[str, int, int] | None:
    m = CHUNK_ID_RE.match(chunk_id)
    if not m:
        return None
    doc, dieu, khoan = m.groups()
    return doc, int(dieu), int(khoan)


def extract_vehicle_types(text: str) -> list[str]:
    m = VEHICLE_RE.search(text)
    if not m:
        return []
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


def extract_heading(text: str) -> str | None:
    m = HEAD_RE.search(text)
    if m:
        return m.group(1).strip()
    m = HEAD_FALLBACK_RE.search(text)
    return m.group(1).strip() if m else None


def build_act_phrases(chunks: list[dict]) -> list[dict]:
    """§4.3 — channel P (ND_168) anchor bank."""
    entries = []
    for chunk in chunks:
        if not chunk["chunk_id"].startswith("ND_168_2024"):
            continue
        parsed = parse_chunk_id(chunk["chunk_id"])
        if parsed is None:
            continue
        _, dieu, khoan = parsed
        vehicle_types = extract_vehicle_types(chunk["text"])
        body = chunk["text"].split("Nội dung:", 1)[-1]
        for letter, phrase_text in ITEM_RE.findall(body):
            phrase_text = phrase_text.strip()
            if not phrase_text:
                continue
            entries.append(
                {
                    "phrase_id": f"{chunk['chunk_id']}_{letter}",
                    "chunk_id": chunk["chunk_id"],
                    "dieu": dieu,
                    "khoan": khoan,
                    "diem": letter,
                    "text": phrase_text,
                    "vehicle_type": vehicle_types,
                    "is_xref": bool(XREF_RE.search(phrase_text)),
                }
            )
    return entries


def build_headings(chunks: list[dict]) -> dict[str, list[str]]:
    """v2-era anchor-snapping bank (act_phrases.json's counterpart) — kept
    available but NOT part of the v3 design (QUERY_GEN_SPEC.md: "Two
    mechanisms: route the question to a document, then close the citation
    graph. Nothing else." — anchor snapping is explicitly demoted to
    "Also optional... check before building it"). Restricted to LUAT_36/
    ND_168 only (142 unique headings, verified) — QCVN is back in the corpus
    under v3, and QCVN headings were never part of this bank's scope."""
    headings: dict[str, list[str]] = {}
    for chunk in chunks:
        if not (chunk["chunk_id"].startswith("LUAT_36_2024") or chunk["chunk_id"].startswith("ND_168_2024")):
            continue
        heading = extract_heading(chunk["text"])
        if heading is None:
            continue
        headings.setdefault(heading, []).append(chunk["chunk_id"])
    return headings


def _governing_dieu(item_text: str, own_dieu: int) -> int:
    """Last valid (non-external) Điều marker anywhere in the lettered item's
    text, else `own_dieu` (an unstated article means "this one" in context)."""
    best = None
    for m in DIEU_RE.finditer(item_text):
        tail = item_text[m.end() : m.end() + 40]
        if EXTERNAL_DOC_RE.match(tail):
            continue
        best = own_dieu if m.group(1) == "này" else int(m.group(1))
    return best if best is not None else own_dieu


def _own_dieu_for_khoan(item_text: str, khoan_match: re.Match, own_dieu: int) -> int | None:
    """A Điều marker attached DIRECTLY after this specific khoản mention
    (e.g. "điểm g khoản 5 Điều này;") overrides the governing one — real
    phrasing sometimes mixes several khoản from different articles in one
    semicolon list, each carrying its own marker."""
    tail = item_text[khoan_match.end() : khoan_match.end() + 20].lstrip()
    m = DIEU_RE.match(tail)
    if not m:
        return None
    after = tail[m.end() : m.end() + 40]
    if EXTERNAL_DOC_RE.match(after):
        return None
    return own_dieu if m.group(1) == "này" else int(m.group(1))


def parse_pointers(chunk: dict) -> list[str]:
    """§4.1 — điểm-aware: resolves to khoản-level chunk_ids (the corpus's
    chunking granularity), but only considers a mention a pointer if it sits
    inside a "quy định tại" reference clause — a bare "khoản 3" elsewhere in
    a chunk (there aren't many, but corpus text is not adversarial-proofed)
    would otherwise produce a spurious edge."""
    parsed = parse_chunk_id(chunk["chunk_id"])
    if parsed is None:
        return []
    doc, own_dieu, _ = parsed
    body = chunk["text"].split("Nội dung:", 1)[-1]
    pointers: set[str] = set()
    for m in ITEM_RE.finditer(body):
        item_text = m.group(0)
        if "quy định tại" not in item_text:
            continue
        governing_dieu = _governing_dieu(item_text, own_dieu)
        for km in KHOAN_RE.finditer(item_text):
            khoan_n = int(km.group(1))
            dieu_n = _own_dieu_for_khoan(item_text, km, own_dieu)
            if dieu_n is None:
                dieu_n = governing_dieu
            target = f"{doc}_D{dieu_n}_K{khoan_n}"
            if target != chunk["chunk_id"]:
                pointers.add(target)
    return sorted(pointers)


def build_citation_graph(chunks: list[dict]) -> dict[str, dict[str, list[str]]]:
    """§4.1. Emits both directions; reverse is the one EXPAND actually uses
    for recovery (§4.1: forward 1.4% recovery, reverse 41.8%) — forward is
    kept too (§4.1: "adds 0.4 points and is nearly free") but capped
    separately at expansion time (§5), not here (this stores the full graph,
    uncapped — MAX_ADD lives in the runtime expand() function)."""
    by_id = {c["chunk_id"]: c for c in chunks}
    forward: dict[str, list[str]] = {}
    for chunk in chunks:
        pointers = [t for t in parse_pointers(chunk) if t in by_id]  # drop dead links (see build --verify output)
        if pointers:
            forward[chunk["chunk_id"]] = pointers

    reverse: dict[str, list[str]] = {}
    for src, targets in forward.items():
        for t in targets:
            reverse.setdefault(t, []).append(src)
    for t in reverse:
        reverse[t] = sorted(reverse[t])

    return {"forward": forward, "reverse": reverse}


def _xref_chunks(chunks: list[dict]) -> set[str]:
    """A chunk counts as 'cross-reference-only' (§1: "carries no violation
    vocabulary") if EVERY lettered item in it is itself a bare pointer
    (XREF_RE). Used only by --verify's acceptance test, not written to any
    bank file."""
    xref: set[str] = set()
    for chunk in chunks:
        if not chunk["chunk_id"].startswith("ND_168_2024"):
            continue
        body = chunk["text"].split("Nội dung:", 1)[-1]
        items = ITEM_RE.findall(body)
        if items and all(XREF_RE.search(text) for _, text in items):
            xref.add(chunk["chunk_id"])
    return xref


def verify_citation_graph(chunks: list[dict], graph: dict, law_path: Path) -> None:
    """§4.1's acceptance test: reverse expansion must recover >=1 xref gold
    chunk for >=42% of items whose gold citations need one. Requires
    agent.resolve.CitationIndex, so this is opt-in (--verify), not run by
    default — build_banks.py itself has no other dependency on `agent`."""
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from agent.resolve import CitationIndex

    forward, reverse = graph["forward"], graph["reverse"]
    out_degrees = [len(v) for v in forward.values()]
    in_degrees = [len(v) for v in reverse.values()]
    print(f"citation_graph.json: {len(forward)} chunks with >=1 forward pointer, "
          f"mean out-degree {sum(out_degrees)/len(out_degrees):.2f}, max {max(out_degrees)}")
    print(f"                     {len(reverse)} chunks with >=1 incoming pointer, "
          f"mean fan-in {sum(in_degrees)/len(in_degrees):.2f}, max {max(in_degrees)}")

    xref_chunks = _xref_chunks(chunks)
    citation_index = CitationIndex.from_file(str(ROOT / "law_db.json"))
    law = json.loads(law_path.read_text(encoding="utf-8"))

    affected = []
    for rec in law:
        gold_resolved = citation_index.resolve_set(rec.get("law", []))
        xref_gold = gold_resolved & xref_chunks
        non_xref_gold = gold_resolved - xref_chunks
        if xref_gold:
            affected.append((xref_gold, non_xref_gold))

    recovered = 0
    for xref_gold, non_xref_gold in affected:
        expanded = set()
        for cid in non_xref_gold:
            expanded.update(reverse.get(cid, []))
        if expanded & xref_gold:
            recovered += 1

    rate = recovered / len(affected) if affected else 0.0
    print(f"xref chunks (all-pointer): {len(xref_chunks)}")
    print(f"items needing >=1 xref gold chunk: {len(affected)}")
    print(f"reverse-expansion recovery: {recovered}/{len(affected)} = {rate:.1%}  "
          f"(gate: >=42.0%, {'PASS' if rate >= 0.42 else 'FAIL — debug parser, do not ship'})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(ROOT / "law_db.json"))
    p.add_argument("--out-dir", default=str(ROOT / "banks"))
    p.add_argument("--verify", action="store_true", help="run §4.1's acceptance test after building")
    p.add_argument("--questions", default=str(ROOT / "law.json"), help="only used with --verify")
    args = p.parse_args()

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(corpus_path.read_text(encoding="utf-8"))

    act_phrases = build_act_phrases(chunks)
    nd_chunk_ids = {e["chunk_id"] for e in act_phrases}
    (out_dir / "act_phrases.json").write_text(
        json.dumps(act_phrases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"act_phrases.json: {len(act_phrases)} entries from {len(nd_chunk_ids)} ND chunks")

    headings = build_headings(chunks)
    (out_dir / "headings.json").write_text(
        json.dumps(headings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"headings.json: {len(headings)} unique headings")

    graph = build_citation_graph(chunks)
    (out_dir / "citation_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"citation_graph.json: {len(graph['forward'])} forward, {len(graph['reverse'])} reverse entries")

    if args.verify:
        print()
        verify_citation_graph(chunks, graph, Path(args.questions))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
