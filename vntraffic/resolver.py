"""Citation resolution and coverage reporting (EXPERIMENT_PLAN.md §7.1).

Gold citations in data.json are authored at *điểm* (point) granularity and in
mixed case, e.g. ``Luat_36_2024_D33_K3_DD``. The chunk DB stops at *khoản*
(clause) granularity and is UPPERCASE, e.g. ``LUAT_36_2024_D33_K3``. Grounding
is therefore graded at clause granularity: both gold and predicted citations are
passed through ``resolve_citation`` first, which normalizes case and drops the
trailing point suffix when needed.

Match order (must not be reordered — it defines the coverage buckets):
  1. exact           citation == chunk_id
  2. case-folded     citation.upper() == chunk_id.upper()
  3. point-fallback  strip a trailing point suffix, then retry 1-2
  4. unresolved

The point suffix is ``_<POINT>`` where POINT is a single ``[A-Za-z]`` letter,
the Vietnamese ``Đ``/``đ`` (điểm đ), or the ASCII digraph ``DD`` that the data
also uses for điểm đ. A naive ``[A-Z]`` regex silently drops the two ``_Đ``
citations — hence the explicit alternation and the coverage report that makes
any silent under-counting visible.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .corpus import Corpus

# One trailing point suffix. Order in the group matters: try the two-char "DD"
# digraph before the single-letter class so "_DD" strips as a unit.
_POINT_SUFFIX_RE = re.compile(r"_(?:DD|dd|[A-Za-zĐđ])$")

METHOD_EXACT = "exact"
METHOD_CASE = "case_only"
METHOD_ALIAS = "alias"          # QCVN-ingestion alias map (empty for shipped corpus)
METHOD_POINT = "point_fallback"
METHOD_UNRESOLVED = "unresolved"


@dataclass
class Resolution:
    citation: str                 # the input citation string
    chunk_id: Optional[str]       # resolved corpus chunk_id (clause granularity) or None
    method: str                   # one of the METHOD_* constants

    @property
    def resolved(self) -> bool:
        return self.chunk_id is not None


def _strip_point_suffix(citation: str) -> Optional[str]:
    """Return the citation with one trailing point suffix removed, or None."""
    m = _POINT_SUFFIX_RE.search(citation)
    if not m:
        return None
    return citation[: m.start()]


def resolve_citation(citation: str, corpus: Corpus) -> Resolution:
    """Resolve a single citation string to a corpus chunk_id (clause level)."""
    if not citation:
        return Resolution(citation, None, METHOD_UNRESOLVED)

    # 1. exact
    if citation in corpus._by_id:  # exact, case-sensitive
        return Resolution(citation, citation, METHOD_EXACT)

    # 2. case-folded
    up = citation.upper()
    ch = corpus._by_id_upper.get(up)
    if ch is not None:
        return Resolution(citation, ch.chunk_id, METHOD_CASE)

    # 2.5 alias map (populated only by QCVN ingestion; empty otherwise)
    alias_cid = corpus.alias_target(citation)
    if alias_cid is not None:
        return Resolution(citation, alias_cid, METHOD_ALIAS)

    # 3. point-fallback: strip a trailing point suffix, retry exact then case
    base = _strip_point_suffix(citation)
    if base is not None:
        if base in corpus._by_id:
            return Resolution(citation, base, METHOD_POINT)
        ch = corpus._by_id_upper.get(base.upper())
        if ch is not None:
            return Resolution(citation, ch.chunk_id, METHOD_POINT)

    # 4. unresolved
    return Resolution(citation, None, METHOD_UNRESOLVED)


def resolve_many(citations: Iterable[str], corpus: Corpus) -> List[Resolution]:
    return [resolve_citation(c, corpus) for c in citations]


def resolved_chunk_ids(citations: Iterable[str], corpus: Corpus) -> List[str]:
    """Resolve and return the deduplicated set of chunk_ids (order-preserving)."""
    seen: Dict[str, None] = {}
    for r in resolve_many(citations, corpus):
        if r.chunk_id is not None and r.chunk_id not in seen:
            seen[r.chunk_id] = None
    return list(seen.keys())


def _document_family(citation: str) -> str:
    """Coarse document family for the coverage breakdown."""
    up = citation.upper()
    for fam in ("LUAT_36_2024", "ND_168_2024", "QCVN_41_2024", "TT_51_2024",
                "TT_12_2025", "QCVN41"):
        if up.startswith(fam):
            return fam
    return up.split("_")[0] if "_" in up else up


# --------------------------------------------------------------------------- #
# Coverage report (emitted per run, EXPERIMENT_PLAN.md §7.1)
# --------------------------------------------------------------------------- #
@dataclass
class CoverageReport:
    n_distinct: int
    counts: Dict[str, int]                       # method -> count over distinct citations
    by_family: Dict[str, Dict[str, int]]         # family -> {method -> count}
    unresolved: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_distinct": self.n_distinct,
            "counts": self.counts,
            "by_family": self.by_family,
            "unresolved": sorted(self.unresolved),
        }

    def summary_line(self) -> str:
        c = self.counts
        base = (f"citations: {self.n_distinct} distinct | "
                f"exact={c.get(METHOD_EXACT,0)} "
                f"case_only={c.get(METHOD_CASE,0)} ")
        if c.get(METHOD_ALIAS, 0):
            base += f"alias={c.get(METHOD_ALIAS,0)} "
        return base + (f"point_fallback={c.get(METHOD_POINT,0)} "
                       f"unresolved={c.get(METHOD_UNRESOLVED,0)}")


def build_coverage_report(citations: Iterable[str], corpus: Corpus) -> CoverageReport:
    """Classify each *distinct* citation into the four coverage buckets."""
    distinct = sorted(set(citations))
    counts: Counter = Counter()
    by_family: Dict[str, Counter] = {}
    unresolved: List[str] = []
    for cit in distinct:
        r = resolve_citation(cit, corpus)
        counts[r.method] += 1
        fam = _document_family(cit)
        by_family.setdefault(fam, Counter())[r.method] += 1
        if not r.resolved:
            unresolved.append(cit)
    # ensure all buckets present
    for m in (METHOD_EXACT, METHOD_CASE, METHOD_ALIAS, METHOD_POINT, METHOD_UNRESOLVED):
        counts.setdefault(m, 0)
    return CoverageReport(
        n_distinct=len(distinct),
        counts=dict(counts),
        by_family={f: dict(c) for f, c in by_family.items()},
        unresolved=unresolved,
    )


def gold_citations_from_items(items: Sequence) -> List[str]:
    """Flatten all gold citations across law items (with repeats)."""
    out: List[str] = []
    for it in items:
        if getattr(it, "law", None):
            out.extend(it.law)
    return out
