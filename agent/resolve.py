#!/usr/bin/env python
"""resolve_citation() + coverage report — AGENT_PLAN.md §3.4 / M0 deliverable.

Resolution order: exact -> case-folded -> điểm→khoản fallback -> unresolved.
Shared by the agent's I2 citation-closure validator and the evaluator's
grounding metrics (§6b), so both use identical resolution logic.

    python -m agent.resolve --report
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A "structural" điều/khoản token, e.g. D6, K5. resolve_citation() never strips
# past one of these — only trailing điểm-level letter tokens (A, B, DD, I, ...)
# are candidates for the point->khoan fallback.
_STRUCT_TOKEN = re.compile(r"^[DK]\d+$")


def normalize(text: str) -> str:
    """NFC-normalize + uppercase. Vietnamese Đ/đ and diacritic variants bite
    chunk_id matching and regexes alike if this isn't applied everywhere (§10)."""
    return unicodedata.normalize("NFC", text).strip().upper()


@dataclass(frozen=True)
class Resolution:
    query: str
    resolved_id: str | None
    status: str  # "exact" | "case_folded" | "point_fallback" | "unresolved"
    strip_count: int = 0


class CitationIndex:
    """Wraps law_db.json chunk_ids for resolution and lookup."""

    def __init__(self, db_records: list[dict]):
        self._by_norm: dict[str, str] = {
            normalize(rec["chunk_id"]): rec["chunk_id"] for rec in db_records
        }
        self.records_by_id = {rec["chunk_id"]: rec for rec in db_records}

    @classmethod
    def from_file(cls, path: Path | str = ROOT / "law_db.json") -> "CitationIndex":
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(records)

    def __contains__(self, chunk_id: str) -> bool:
        return normalize(chunk_id) in self._by_norm

    def resolve(self, raw_id: str) -> Resolution:
        norm = normalize(raw_id)
        if norm in self._by_norm:
            canonical = self._by_norm[norm]
            status = "exact" if canonical == raw_id else "case_folded"
            return Resolution(raw_id, canonical, status, 0)

        parts = norm.split("_")
        stripped = 0
        while len(parts) > 1:
            last = parts[-1]
            if _STRUCT_TOKEN.match(last):
                break  # never strip a Dieu/Khoan token itself
            parts = parts[:-1]
            stripped += 1
            candidate = "_".join(parts)
            if candidate in self._by_norm:
                return Resolution(
                    raw_id, self._by_norm[candidate], "point_fallback", stripped
                )

        return Resolution(raw_id, None, "unresolved", stripped)

    def resolve_set(self, raw_ids: list[str]) -> set[str]:
        """Resolve a list of citations to the set of canonical chunk_ids that
        matched; unresolved ids are dropped. Used for grounding metrics (§6b)
        and the I2 citation-closure check."""
        out: set[str] = set()
        for rid in raw_ids:
            r = self.resolve(rid)
            if r.resolved_id is not None:
                out.add(r.resolved_id)
        return out


def coverage_report(
    index: CitationIndex, law_path: Path | str = ROOT / "law.json"
) -> dict:
    law = json.loads(Path(law_path).read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    unresolved_ids: set[str] = set()
    seen: set[str] = set()
    for rec in law:
        for raw_id in rec.get("law", []):
            if raw_id in seen:
                continue
            seen.add(raw_id)
            r = index.resolve(raw_id)
            counts[r.status] += 1
            if r.status == "unresolved":
                unresolved_ids.add(raw_id)
    return {
        "unique_citations": len(seen),
        "exact": counts["exact"],
        "case_folded": counts["case_folded"],
        "point_fallback": counts["point_fallback"],
        "unresolved": counts["unresolved"],
        "unresolved_ids": sorted(unresolved_ids),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--law", default=str(ROOT / "law.json"))
    p.add_argument("--db", default=str(ROOT / "law_db.json"))
    p.add_argument(
        "--report", action="store_true", help="also list every unresolved citation"
    )
    args = p.parse_args(argv)

    index = CitationIndex.from_file(args.db)
    report = coverage_report(index, args.law)

    print(f"unique citations       : {report['unique_citations']}")
    print(f"  exact match          : {report['exact']}")
    print(f"  case-folded match    : {report['case_folded']}")
    print(f"  point->khoan fallback: {report['point_fallback']}")
    print(f"  unresolved           : {report['unresolved']}")
    if args.report and report["unresolved_ids"]:
        print("unresolved ids:")
        for uid in report["unresolved_ids"]:
            print(f"  - {uid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
