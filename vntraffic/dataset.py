"""Dataset loading, the QAItem model, and option-permutation utilities.

The QA file (data.json) is a flat JSON list of items with this schema
(verified against the shipped data):

    id            int            unique, contiguous 1..N
    image         str            "/data/<category>/<n>.jpg" (leading slash → repo-root)
    question      str            Vietnamese question text
    A,B,C,D       str            the four options (options are NOT in the question)
    GT            str            answer key, one of "A"/"B"/"C"/"D"
    type_question str            "detection" | "law"  (the tier)
    type_image    str            category (== image folder)
    law           list[str]      gold citation ids; law-tier only, may be empty
    law_subtype   str  OPTIONAL  added post-hoc by the reviewed subtype tagger (§7.4)

Option permutations are represented as a mapping from a *new* option letter to
the *source* letter in the item, e.g. {"A": "C", "B": "A", ...} means "the text
shown as A comes from the item's original C". This single representation is
reused by both the detection rebalance (§7.3) and CircularEval (§5.5).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import constants as C


# --------------------------------------------------------------------------- #
# QAItem
# --------------------------------------------------------------------------- #
@dataclass
class QAItem:
    id: int
    image: str
    question: str
    options: Dict[str, str]           # {"A": text, "B": text, "C": text, "D": text}
    gt: str                           # "A"/"B"/"C"/"D"
    tier: str                         # "detection"/"law"
    category: str                     # type_image
    law: List[str] = field(default_factory=list)   # gold citations (law tier)
    law_subtype: Optional[str] = None
    extra: Dict = field(default_factory=dict)      # any unmodeled fields, round-tripped

    # ---- serialization ---------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: dict) -> "QAItem":
        known = {"id", "image", "question", "A", "B", "C", "D", "GT",
                 "type_question", "type_image", "law", "law_subtype"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=int(d["id"]),
            image=d["image"],
            question=d["question"],
            options={L: d[L] for L in C.OPTION_LETTERS},
            gt=d["GT"],
            tier=d["type_question"],
            category=d["type_image"],
            law=list(d.get("law", []) or []),
            law_subtype=d.get("law_subtype"),
            extra=extra,
        )

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "image": self.image,
            "question": self.question,
            **{L: self.options[L] for L in C.OPTION_LETTERS},
            "GT": self.gt,
            "type_question": self.tier,
            "type_image": self.category,
        }
        if self.tier == C.TIER_LAW:
            d["law"] = list(self.law)
        if self.law_subtype is not None:
            d["law_subtype"] = self.law_subtype
        d.update(self.extra)
        return d

    # ---- convenience ------------------------------------------------------ #
    @property
    def is_law(self) -> bool:
        return self.tier == C.TIER_LAW

    @property
    def is_detection(self) -> bool:
        return self.tier == C.TIER_DETECTION

    def image_path(self, image_root: str = C.IMAGE_ROOT) -> str:
        """Absolute path on disk. data.json paths look like '/data/x/1.jpg'."""
        rel = self.image.lstrip("/")
        return os.path.join(image_root, rel)

    def gt_text(self) -> str:
        return self.options[self.gt]

    def option_block(self, letters: Sequence[str] = C.OPTION_LETTERS) -> str:
        """Rendered 'A. ...\\nB. ...' block for prompts."""
        return "\n".join(f"{L}. {self.options[L]}" for L in letters)


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #
def load_items(path: str = C.DATA_JSON) -> List[QAItem]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [QAItem.from_dict(d) for d in raw]


def save_items(items: Iterable[QAItem], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([it.to_dict() for it in items], f, ensure_ascii=False, indent=2)


def filter_tier(items: Sequence[QAItem], tier: str) -> List[QAItem]:
    return [it for it in items if it.tier == tier]


def index_by_id(items: Sequence[QAItem]) -> Dict[int, QAItem]:
    return {it.id: it for it in items}


# --------------------------------------------------------------------------- #
# Option permutations (shared by rebalance §7.3 and CircularEval §5.5)
# --------------------------------------------------------------------------- #
Permutation = Dict[str, str]  # new_letter -> source_letter

IDENTITY_PERM: Permutation = {L: L for L in C.OPTION_LETTERS}


def apply_permutation(item: QAItem, perm: Permutation) -> QAItem:
    """Return a copy of *item* with options rearranged per *perm*.

    perm maps each new option letter to the source letter it draws its text
    from. GT is remapped so the correct answer still points at the same text.
    """
    # source_letter -> new_letter (inverse), to find where GT text now lives
    inverse = {src: new for new, src in perm.items()}
    new_options = {new: item.options[src] for new, src in perm.items()}
    new_gt = inverse[item.gt]
    return QAItem(
        id=item.id,
        image=item.image,
        question=item.question,
        options=new_options,
        gt=new_gt,
        tier=item.tier,
        category=item.category,
        law=list(item.law),
        law_subtype=item.law_subtype,
        extra=dict(item.extra),
    )


def cyclic_permutations() -> List[Tuple[int, Permutation]]:
    """The 4 cyclic option shifts used by CircularEval (§5.5).

    Shift s maps new letter i to source letter (i - s) mod 4, i.e. every option
    is rotated by s positions. s=0 is the identity.
    Returns list of (shift, permutation).
    """
    letters = C.OPTION_LETTERS
    n = len(letters)
    out: List[Tuple[int, Permutation]] = []
    for s in range(n):
        perm = {letters[i]: letters[(i - s) % n] for i in range(n)}
        out.append((s, perm))
    return out


def permutation_to_target_gt(item: QAItem, target_gt: str) -> Permutation:
    """Build a permutation that moves the correct answer to *target_gt*.

    The correct text goes to the target letter; the other three texts fill the
    remaining slots in their original relative order. Deterministic given the
    item and target (no randomness) so it is fully reproducible.
    """
    if target_gt == item.gt:
        return dict(IDENTITY_PERM)
    others = [L for L in C.OPTION_LETTERS if L != item.gt]          # sources to place
    slots = [L for L in C.OPTION_LETTERS if L != target_gt]         # destinations
    perm: Permutation = {target_gt: item.gt}
    for slot, src in zip(slots, others):
        perm[slot] = src
    return perm
