"""Vietnamese word segmentation, shared by the BM25 index builder and
BM25Retriever (AGENT_PLAN.md §3.2: "Vietnamese tokenization for BM25 is not
optional ... or the comparison is unfair to the lexical baseline").

One function, used on both sides of the index so a chunk indexed as
"mô_tô" is queried as "mô_tô", never as two separate unsegmented tokens.
"""

from __future__ import annotations

import unicodedata

from pyvi import ViTokenizer


def segment(text: str) -> str:
    """NFC-normalize then word-segment. Multi-syllable Vietnamese words come
    back joined with underscores (e.g. "mô_tô"), which Lucene's whitespace
    analyzer then treats as a single term."""
    normalized = unicodedata.normalize("NFC", text)
    return ViTokenizer.tokenize(normalized)
