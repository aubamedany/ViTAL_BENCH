#!/usr/bin/env python
"""Streamlit app to categorize datasets/questions/law.json questions into reasoning-type buckets.

Shows one question (image + question text + A-D options) at a time and lets
the annotator pick one of three categories. Choices are persisted incrementally
to ``datasets/legal/law_categories.json`` as ``[{"id": ..., "category": ...}, ...]``.

Run with:
    streamlit run annotate_law.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "datasets/questions/law.json"
OUTPUT_PATH = ROOT / "datasets/legal/law_categories.json"

CATEGORIES = [
    "Violation identification",
    "Regulation Application",
    "Multi-step legal Reasoning",
]


@st.cache_data
def load_records() -> list[dict]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def load_annotations() -> dict:
    if OUTPUT_PATH.exists():
        rows = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        return {row["id"]: row["category"] for row in rows}
    return {}


def save_annotations(annotations: dict) -> None:
    rows = [{"id": rid, "category": cat} for rid, cat in sorted(annotations.items())]
    OUTPUT_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_image_path(rel: str) -> Path:
    return (ROOT / rel.lstrip("/\\")).resolve()


def main() -> None:
    st.set_page_config(page_title="Law VQA Categorization", layout="wide")

    records = load_records()
    n = len(records)

    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    annotations = st.session_state.annotations
    idx = st.session_state.idx
    rec = records[idx]

    st.title("Traffic Law VQA — Question Categorization")
    st.progress(len(annotations) / n)
    st.caption(f"{len(annotations)} / {n} labeled")

    with st.sidebar:
        st.header("Navigate")
        jump = st.number_input(
            "Go to question # (1-based)", min_value=1, max_value=n,
            value=idx + 1, step=1,
        )
        if st.button("Go"):
            st.session_state.idx = jump - 1
            st.rerun()
        if st.button("Jump to first unlabeled"):
            for i, r in enumerate(records):
                if r["id"] not in annotations:
                    st.session_state.idx = i
                    st.rerun()
            else:
                st.info("All questions labeled.")

        st.divider()
        st.subheader("Label counts")
        counts = {c: 0 for c in CATEGORIES}
        for cat in annotations.values():
            counts[cat] = counts.get(cat, 0) + 1
        for c in CATEGORIES:
            st.write(f"{c}: {counts[c]}")

    col_img, col_q = st.columns([1, 1])

    with col_img:
        img_path = resolve_image_path(rec["image"])
        if img_path.exists():
            st.image(str(img_path), width="stretch")
        else:
            st.warning(f"Image not found: {img_path}")

    with col_q:
        st.subheader(f"Question {idx + 1} / {n}  (id={rec['id']})")
        st.write(rec["question"])
        for L in ("A", "B", "C", "D"):
            marker = " ✅" if rec.get("GT") == L else ""
            st.write(f"**{L}.** {rec[L]}{marker}")

        current = annotations.get(rec["id"])
        default_index = CATEGORIES.index(current) if current in CATEGORIES else None
        choice = st.radio(
            "Category", CATEGORIES, index=default_index, key=f"cat_{rec['id']}"
        )

        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("Previous", disabled=idx == 0, width="stretch"):
                st.session_state.idx = max(0, idx - 1)
                st.rerun()
        with nav2:
            if st.button("Save", width="stretch", disabled=choice is None):
                annotations[rec["id"]] = choice
                save_annotations(annotations)
                st.toast("Saved")
        with nav3:
            if st.button(
                "Save & Next", type="primary", width="stretch",
                disabled=choice is None or idx == n - 1,
            ):
                annotations[rec["id"]] = choice
                save_annotations(annotations)
                st.session_state.idx = min(n - 1, idx + 1)
                st.rerun()


if __name__ == "__main__":
    main()
