"""normalize_vehicle() — retrieval-time vehicle-term normalization (AGENT_PLAN.md §3.3/§4).

The constraint "match corpus vocabulary" belongs to the *filter*, not to
perception: PERCEIVE stays open-vocabulary, and this module maps the model's
free text ("xe máy", "mô tô", "motorbike") onto the corpus's fixed
`vehicle_type` terms at retrieval time.

NOTE on corpus coverage: law_db.json's chunks (as currently ingested) carry no
`vehicle_type` metadata field, so the hard filter described in §3.3 has nothing
to filter against yet — every chunk is effectively "generic" until that
metadata is added to the corpus. The mapping below is kept so the fail-open
contract and the filter's call signature are correct today, and so the filter
starts doing real pruning the moment `vehicle_type` is ingested (invariant I5:
corpus-swappable, zero code change).
"""

from __future__ import annotations

import unicodedata

# Corpus's fixed vehicle_type vocabulary (AGENT_PLAN.md §3.3/§4).
CORPUS_VEHICLE_TERMS = (
    "xe_mo_to",
    "xe_o_to",
    "xe_tai",
    "xe_dau_keo",
    "xe_dap",
    "nguoi_di_bo",
)

# Free-text phrase -> corpus term. Keys are matched as normalized substrings,
# longest-key-first, so more specific phrases win over generic ones.
_ALIASES: dict[str, str] = {
    "xe mo to": "xe_mo_to",
    "xe gan may": "xe_mo_to",
    "xe may": "xe_mo_to",
    "moto": "xe_mo_to",
    "motorbike": "xe_mo_to",
    "motorcycle": "xe_mo_to",
    "xe o to con": "xe_o_to",
    "xe o to": "xe_o_to",
    "o to": "xe_o_to",
    "car": "xe_o_to",
    "xe tai": "xe_tai",
    "truck": "xe_tai",
    "xe dau keo": "xe_dau_keo",
    "xe container": "xe_dau_keo",
    "so mi ro mooc": "xe_dau_keo",
    "xe dap dien": "xe_dap",
    "xe dap": "xe_dap",
    "bicycle": "xe_dap",
    "nguoi di bo": "nguoi_di_bo",
    "pedestrian": "nguoi_di_bo",
}


def _fold(text: str) -> str:
    """Lowercase + strip diacritics so 'xe máy' and 'xe may' match the same key."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


_FOLDED_ALIASES = sorted(
    ((_fold(k), v) for k, v in _ALIASES.items()), key=lambda kv: -len(kv[0])
)


def normalize_vehicle(described_as: str | None) -> str | None:
    """Map open-vocabulary vehicle text to a corpus vehicle_type term.

    Fail-open by design (§3.3/§8): an unmappable description returns None,
    meaning "no filter" — retrieve over everything. A wrong forced mapping
    would silently exclude the correct statutes, which is the unacceptable
    failure mode; losing a bit of pruning on odd phrasing is cheap by
    comparison.
    """
    if not described_as:
        return None
    folded = _fold(described_as)
    for key, term in _FOLDED_ALIASES:
        if key in folded:
            return term
    return None
