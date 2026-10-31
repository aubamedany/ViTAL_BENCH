#!/usr/bin/env python3
"""Check the canonical ViTALBench release statistics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED = {
    "questions": 4536,
    "detection_questions": 2122,
    "law_questions": 2414,
    "images": 551,
    "image_categories": 11,
    "law_corpus_chunks": 990,
    "law_categories": {
        "Violation identification": 482,
        "Regulation Application": 1601,
        "Multi-step legal Reasoning": 331,
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_stats(root: Path) -> dict[str, object]:
    detection = json.loads((root / "detection.json").read_text())
    law = json.loads((root / "law.json").read_text())
    law_categories = json.loads((root / "law_categories.json").read_text())
    images = [
        path
        for path in (root / "data").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return {
        "questions": len(detection) + len(law),
        "detection_questions": len(detection),
        "law_questions": len(law),
        "images": len(images),
        "image_categories": len({path.parent.name for path in images}),
        "law_corpus_chunks": len(json.loads((root / "law_db.json").read_text())),
        "law_categories": {
            category: sum(1 for item in law_categories if item["category"] == category)
            for category in EXPECTED["law_categories"]
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    stats = load_stats(root)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    if stats != EXPECTED:
        print("Release statistics do not match EXPECTED.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
