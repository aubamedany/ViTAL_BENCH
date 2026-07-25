#!/usr/bin/env python
"""Strip the HTML/Word-export markup embedded in QCVN_41_2024 chunks (the
only document in law_db.json affected — verified LUAT_36_2024/ND_168_2024
contain none).

Source pattern: QCVN's ingestion pipeline pasted raw Word-HTML table markup
wrapped in a custom `<<TABLE: ... /TABLE>>` sentinel (`<table>`/`<tr>`/`<td>`/
`<span lang="...">`/`<o:p>`/VML shapes/conditional comments and all), plus
`<<IMAGE: filename.jpg /IMAGE>>` image references. 217/296 QCVN chunks
contain this; the worst (QCVN_41_2024_PLM_4) is 238KB of markup for what
becomes ~12KB of actual readable content.

Design:
- `<<IMAGE: ... /IMAGE>>` markers contain no HTML and are left untouched —
  not in scope ("HTML tags"), and the filename itself is retrievable-corpus
  noise either way, not something cleaning should manufacture an opinion on.
- `<<TABLE: <html> /TABLE>>` blocks: parse the inner HTML, extract each row
  as `cell | cell | cell`, rows joined by `; ` — readable, compact, keeps
  the actual table data instead of just deleting it.
- Everywhere else in the text: verified zero genuine HTML tags exist (a few
  `<`/`>` characters are math inequalities like "50 < R <= 100", not
  markup) — left completely untouched to avoid corrupting them.

Cell-text extraction correctness note (found by testing, not obvious):
Word splits a single word across multiple `<span lang="EN">`/`<span
lang="VI">` runs (e.g. "Lo" + "ại đường" -> "Loại đường"), with the space
between logical words living INSIDE one span's own text, at the boundary.
`BeautifulSoup.get_text(separator, strip=True)` strips leading/trailing
whitespace from EACH fragment before joining -- which deletes exactly that
boundary space ("ờng " -> "ờng"), silently merging two words ("Đường ô tô"
-> "Đườngô tô"). Fixed by extracting each `<p>` with `strip=False` (preserve
internal boundary spaces) and only collapsing/stripping whitespace on the
fully-joined paragraph text; separate `<p>` children within one `<td>` (76
cells in this corpus) are then joined with an explicit space, since a
paragraph break is a real boundary a bare concatenation would erase.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent

TABLE_BLOCK_RE = re.compile(r"<<TABLE:\s*(.*?)\s*/TABLE>>", re.DOTALL)
WS_RE = re.compile(r"\s+")


def _cell_text(td) -> str:
    paragraphs = td.find_all("p") or [td]
    parts = []
    for p in paragraphs:
        # strip=False: preserve boundary whitespace Word left INSIDE a span's
        # own text (see module docstring) -- stripping per-fragment merges words.
        t = WS_RE.sub(" ", p.get_text("", strip=False)).strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def clean_table_block(match: re.Match) -> str:
    soup = BeautifulSoup(match.group(1), "html.parser")
    table = soup.find("table")
    rows_out: list[str] = []
    if table:
        for tr in table.find_all("tr"):
            cells = [_cell_text(td) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows_out.append(" | ".join(cells))
    if not rows_out:
        # No <table> found, or it had no non-empty rows -- fall back to
        # whatever text is in the block rather than dropping it silently.
        text = WS_RE.sub(" ", soup.get_text("", strip=False)).strip()
        rows_out = [text] if text else []
    return "<<TABLE: " + "; ".join(rows_out) + " /TABLE>>" if rows_out else ""


def clean_chunk_text(text: str) -> str:
    return TABLE_BLOCK_RE.sub(clean_table_block, text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(ROOT / "law_db.json"))
    args = p.parse_args()

    corpus_path = Path(args.corpus)
    chunks = json.loads(corpus_path.read_text(encoding="utf-8"))

    n_changed = 0
    chars_before = chars_after = 0
    for chunk in chunks:
        if not chunk["chunk_id"].startswith("QCVN_41_2024"):
            continue
        original = chunk["text"]
        cleaned = clean_chunk_text(original)
        if cleaned != original:
            n_changed += 1
            chars_before += len(original)
            chars_after += len(cleaned)
            chunk["text"] = cleaned

    corpus_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cleaned {n_changed} QCVN chunks")
    print(f"chars: {chars_before} -> {chars_after} ({chars_after - chars_before:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
