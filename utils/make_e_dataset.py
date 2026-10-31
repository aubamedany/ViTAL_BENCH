#!/usr/bin/env python
"""Create an "E-option" variant of the VQA dataset.

Appends a fifth option ``E`` ("unknown" / không xác định) to every record in
``data.json`` and writes the result to ``E_Datas.json``. Key order is preserved
with ``E`` inserted right after ``D`` (before ``GT``); every other field,
including the ground-truth letter, is left untouched.

Usage
-----
    python make_e_dataset.py
    python make_e_dataset.py --in data.json --out E_Datas.json
    python make_e_dataset.py --text "Không xác định."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Default fifth-option text. Vietnamese to match the rest of the options,
# meaning "Cannot be determined / unknown".
DEFAULT_E_TEXT = "Không xác định."


def add_e_option(rec: dict, text: str) -> dict:
    """Return a copy of ``rec`` with an ``E`` option inserted after ``D``."""
    out: dict = {}
    for key, val in rec.items():
        if key == "GT":  # E goes just before GT to keep A-B-C-D-E contiguous
            out["E"] = text
        out[key] = val
    if "E" not in out:  # no GT field -> just append at the end
        out["E"] = text
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="datasets/questions/law.json", help="input dataset")
    p.add_argument("--out", default="datasets/questions/law_e.json", help="output dataset")
    p.add_argument("--text", default=DEFAULT_E_TEXT, help="text for option E")
    p.add_argument("--letter", default="E", help="option letter to add (default: E)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    in_path = Path(args.inp)
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")
    records = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        sys.exit("expected data.json to be a list of records")

    new_records = []
    skipped = 0
    for rec in records:
        if args.letter in rec:  # already has this option; leave as-is
            skipped += 1
            new_records.append(rec)
            continue
        new_records.append(add_e_option(rec, args.text))

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(new_records, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"read  {len(records)} records from {in_path}")
    print(f"added option '{args.letter}: {args.text}'"
          + (f" ({skipped} already had it, left unchanged)" if skipped else ""))
    print(f"wrote {len(new_records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
