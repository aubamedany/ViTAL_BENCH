"""Rule-based ``law_subtype`` tagger with human review (EXPERIMENT_PLAN.md §7.4).

The plan is explicit: *do not guess tags*. So this module never writes a tag
straight into the frozen dataset. Instead it runs a transparent keyword/regex
tagger over the Vietnamese question text and emits a **review CSV** carrying,
for every law item, the proposed subtype and the exact rule evidence that fired.
A human edits the ``reviewed_subtype`` column; only then does ``apply_reviewed``
merge confirmed tags back in. If tagging slips, the subtype table is simply
dropped (metrics treat ``law_subtype`` as optional).

Subtypes (§7.4):
  behavior_identification    which violation is shown
  required_behavior          what the correct/lawful behavior is
  penalty_amount             the monetary fine (prediction: hardest — pure retrieval)
  supplementary_consequence  license points, seizure, revocation, suspension, ...
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

SUBTYPES = (
    "behavior_identification",
    "required_behavior",
    "penalty_amount",
    "supplementary_consequence",
)

# Rules are ordered most-specific → most-generic. Each pattern is matched
# case-insensitively against the question text. The evidence string records
# which patterns fired so a human can audit the proposal.
#
# Patterns are deliberately conservative: they target vocabulary that is
# characteristic of the subtype, not merely compatible with it.
_RULES: List[Tuple[str, List[str]]] = [
    # penalty_amount — asks about money / fine level
    ("penalty_amount", [
        r"phạt\s+tiền",
        r"mức\s+phạt",
        r"mức\s+tiền",
        r"số\s+tiền",
        r"bao\s+nhiêu\s+tiền",
        r"bị\s+phạt\s+bao\s+nhiêu",
        r"\bphạt\b.*\bđồng\b",
        r"\btriệu\s+đồng\b",
    ]),
    # supplementary_consequence — license points, seizure, suspension, ...
    ("supplementary_consequence", [
        r"trừ\s+điểm",
        r"điểm\s+giấy\s+phép",
        r"tước\s+(quyền\s+sử\s+dụng\s+)?giấy\s+phép",
        r"tước\s+bằng",
        r"giấy\s+phép\s+lái\s+xe",
        r"bằng\s+lái",
        r"tịch\s+thu",
        r"tạm\s+giữ",
        r"xử\s+phạt\s+bổ\s+sung",
        r"hình\s+thức.*bổ\s+sung",
    ]),
    # required_behavior — what the lawful / correct action is
    ("required_behavior", [
        r"phải\s+làm\s+gì",
        r"cần\s+(phải\s+)?làm",
        r"đúng\s+quy\s+định",
        r"theo\s+quy\s+định.*phải",
        r"hành\s+vi\s+đúng",
        r"nên\s+làm",
        r"được\s+phép",
        r"quy\s+định\s+yêu\s+cầu",
        r"bắt\s+buộc\s+phải",
    ]),
    # behavior_identification — which violation / behavior is shown
    ("behavior_identification", [
        r"hành\s+vi\s+(nào|gì|vi\s+phạm)",
        r"đang\s+(thực\s+hiện|vi\s+phạm)",
        r"lỗi\s+(nào|gì)",
        r"vi\s+phạm\s+(nào|gì|luật)",
        r"thực\s+hiện\s+hành\s+vi",
    ]),
]

_COMPILED: List[Tuple[str, List[re.Pattern]]] = [
    (sub, [re.compile(p, re.IGNORECASE) for p in pats]) for sub, pats in _RULES
]


@dataclass
class TagProposal:
    id: int
    category: str
    question: str
    proposed_subtype: Optional[str]         # highest-priority matched rule, or None
    candidate_subtypes: List[str]           # every subtype whose rule fired
    evidence: str                           # "subtype:matched_pattern; ..."


def propose_subtype(question: str) -> Tuple[Optional[str], List[str], str]:
    """Return (proposed_subtype, candidates, evidence) for a question."""
    candidates: List[str] = []
    evidence_bits: List[str] = []
    for subtype, patterns in _COMPILED:
        hits = [p.pattern for p in patterns if p.search(question)]
        if hits:
            candidates.append(subtype)
            evidence_bits.append(f"{subtype}:{'|'.join(hits)}")
    proposed = candidates[0] if candidates else None  # rules are priority-ordered
    return proposed, candidates, "; ".join(evidence_bits)


def build_proposals(law_items: Sequence) -> List[TagProposal]:
    out: List[TagProposal] = []
    for it in law_items:
        proposed, candidates, evidence = propose_subtype(it.question)
        out.append(TagProposal(
            id=it.id, category=it.category, question=it.question,
            proposed_subtype=proposed, candidate_subtypes=candidates, evidence=evidence,
        ))
    return out


_CSV_FIELDS = ["id", "category", "proposed_subtype", "reviewed_subtype",
               "candidate_subtypes", "evidence", "question"]


def write_review_csv(proposals: Sequence[TagProposal], path: str) -> None:
    """Emit the review CSV. ``reviewed_subtype`` is pre-filled with the proposal
    so a reviewer only has to correct disagreements, then re-save."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for p in proposals:
            w.writerow({
                "id": p.id,
                "category": p.category,
                "proposed_subtype": p.proposed_subtype or "",
                "reviewed_subtype": p.proposed_subtype or "",  # editable default
                "candidate_subtypes": "|".join(p.candidate_subtypes),
                "evidence": p.evidence,
                "question": p.question,
            })


def read_reviewed_csv(path: str) -> Dict[int, str]:
    """Read a reviewed CSV → {id: subtype}, keeping only valid, non-empty tags."""
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sub = (row.get("reviewed_subtype") or "").strip()
            if sub in SUBTYPES:
                out[int(row["id"])] = sub
    return out


def apply_reviewed(items: Sequence, reviewed: Dict[int, str]) -> int:
    """Merge reviewed tags onto items in place. Returns count applied."""
    n = 0
    for it in items:
        if it.id in reviewed:
            it.law_subtype = reviewed[it.id]
            n += 1
    return n


def coverage_summary(proposals: Sequence[TagProposal]) -> Dict[str, int]:
    """How many items each proposed subtype covers (for a quick sanity look)."""
    counts: Dict[str, int] = {s: 0 for s in SUBTYPES}
    counts["_untagged"] = 0
    for p in proposals:
        if p.proposed_subtype:
            counts[p.proposed_subtype] += 1
        else:
            counts["_untagged"] += 1
    return counts
