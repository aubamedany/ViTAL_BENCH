#!/usr/bin/env python
"""Answer the multiple-choice VQA questions in ``data.json`` with a local VLM.

Priority ladder (all ≤14B, image-capable):

  * ``qwen3vl``  -> Qwen/Qwen3-VL-8B-Instruct        (default, priority)
  * ``llama``    -> meta-llama/Llama-3.2-11B-Vision-Instruct
  * ``internvl`` -> OpenGVLab/InternVL3-8B

Each ``data.json`` record is an image + a Vietnamese question with options A-D
and a ground-truth letter ``GT``. The model is shown the image and prompted to
reply with a single letter; the reply is parsed back to A/B/C/D, scored against
``GT``, and written out with per-record predictions and an accuracy summary.

The three families are loaded through the modern unified transformers API
(``AutoModelForImageTextToText`` + ``AutoProcessor`` with a chat template that
embeds images directly), so one code path serves all of them. Heavy imports are
deferred so ``--help`` and a dry run work on a CPU-only box.

Examples
--------
    # default: Qwen3-VL-8B on all questions
    python answer_vqa.py

    # try another family, limit to first 50 for a smoke test
    python answer_vqa.py --model llama --limit 50

    # point at an explicit checkpoint / different data root
    python answer_vqa.py --model-id OpenGVLab/InternVL3-14B --family internvl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Model presets (short name -> checkpoint + adapter family). All ≤14B.
# --------------------------------------------------------------------------- #
FAMILY_QWEN = "qwen"
FAMILY_LLAMA = "llama"
FAMILY_INTERNVL = "internvl"


@dataclass(frozen=True)
class Preset:
    hf_id: str
    family: str


PRESETS: Dict[str, Preset] = {
    # --- priority ---
    "qwen3vl-8b":   Preset("Qwen/Qwen3-VL-8B-Instruct", FAMILY_QWEN),
    "qwen3vl-4b":   Preset("Qwen/Qwen3-VL-4B-Instruct", FAMILY_QWEN),
    "qwen3vl-2b":   Preset("Qwen/Qwen3-VL-2B-Instruct", FAMILY_QWEN),
    "qwen2_5vl-7b":  Preset("Qwen/Qwen2.5-VL-7B-Instruct", FAMILY_QWEN),
    "qwen2_5vl-3b":  Preset("Qwen/Qwen2.5-VL-3B-Instruct", FAMILY_QWEN),
    # --- llama ---
    "llama":        Preset("meta-llama/Llama-3.2-11B-Vision-Instruct", FAMILY_LLAMA),
    # --- internvl ---
    "internvl-8b":     Preset("OpenGVLab/InternVL3-8B-hf", FAMILY_INTERNVL),
    "internvl-14b": Preset("OpenGVLab/InternVL3-14B-hf", FAMILY_INTERNVL),
    "internvl-2b":  Preset("OpenGVLab/InternVL3-2B-hf", FAMILY_INTERNVL),
    "internvl-1b":  Preset("OpenGVLab/InternVL3-1B-hf", FAMILY_INTERNVL),
    "internvl-9b":  Preset("OpenGVLab/InternVL3-9B-hf", FAMILY_INTERNVL),
}

OPTION_LETTERS = ("A", "B", "C", "D")
OPTION_LETTERS_E = ("A", "B", "C", "D", "E")

# Default text for option E when a record lacks its own "E" field.
DEFAULT_E_TEXT = "Không xác định."


def option_letters(include_e: bool):
    return OPTION_LETTERS_E if include_e else OPTION_LETTERS


def system_prompt(include_e: bool) -> str:
    letters = ", ".join(option_letters(include_e))
    base = ("Bạn là trợ lý trả lời câu hỏi trắc nghiệm về giao thông đường bộ "
            "Việt Nam dựa trên hình ảnh. "
            f"Chỉ trả lời bằng đúng một chữ cái: {letters}.")
    if include_e:
        base += (" Nếu bạn không có đủ thông tin để trả lời, "
                 "hãy chọn đáp án E.")
    return base


def build_prompt(rec: dict, include_e: bool = False) -> str:
    """Vietnamese multiple-choice prompt for one record.

    When ``include_e`` is set, a fifth option E is appended (taken from the
    record's ``E`` field, or ``DEFAULT_E_TEXT``) together with an instruction to
    pick E when there is not enough information to answer.
    """
    letters = option_letters(include_e)
    lines = [rec["question"].strip(), ""]
    for L in letters:
        if L == "E" and "E" not in rec:
            text = DEFAULT_E_TEXT
        else:
            text = rec[L].strip()
        lines.append(f"{L}. {text}")
    lines.append("")
    joined = ", ".join(letters)
    instruction = (f"Hãy chọn đáp án đúng nhất. "
                   f"Chỉ trả lời bằng một chữ cái {joined}.")
    if include_e:
        instruction += (" Nếu không có đủ thông tin để trả lời, hãy chọn E.")
    lines.append(instruction)
    return "\n".join(lines)


def parse_letter(text: str, include_e: bool = False) -> Optional[str]:
    """Extract the predicted option letter from a model reply.

    Preference order: an explicit "Đáp án: X" / "answer X" style tag, then the
    first standalone option letter, then any option-letter character. Returns
    None if the reply contains no option letter at all.
    """
    if not text:
        return None
    cls = "ABCDE" if include_e else "ABCD"
    upper = text.upper()
    # "Đáp án: B", "Answer: C", "Choice D", "B." etc.
    m = re.search(rf"(?:ĐÁP\s*ÁN|ANSWER|CHOICE|LỰA\s*CHỌN)\s*[:\-]?\s*([{cls}])", upper)
    if m:
        return m.group(1)
    # standalone letter (word boundary), e.g. "B" or "B."
    m = re.search(rf"\b([{cls}])\b", upper)
    if m:
        return m.group(1)
    m = re.search(rf"[{cls}]", upper)
    return m.group(0) if m else None


def resolve_image_path(rel: str, data_root: Path) -> Path:
    """Map a data.json image field ("/data/foo/1.jpg") to a real path."""
    rel = rel.lstrip("/\\")
    return (data_root / rel).resolve()


# --------------------------------------------------------------------------- #
# Model wrapper — unified transformers image-text-to-text path
# --------------------------------------------------------------------------- #
class VLM:
    def __init__(self, model_id: str, family: str, device: str, dtype: str,
                 max_new_tokens: int = 16):
        self.model_id = model_id
        self.family = family
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def _torch_dtype(self):
        import torch
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32, "auto": "auto"}[self.dtype]

    def load(self):
        import torch  # noqa: F401  (ensures a clear error if torch is missing)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[load] {self.model_id} (family={self.family}, dtype={self.dtype}, "
              f"device={self.device}) ...", flush=True)
        t0 = time.time()
        # trust_remote_code helps InternVL / newer checkpoints; harmless for others.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)
        device_map = self.device if self.device != "cpu" else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype(),
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        if device_map is None:
            self.model.to(self.device)
        print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    def _messages(self, image_path: Path, prompt: str, system: str) -> List[dict]:
        return [
            {"role": "system",
             "content": [{"type": "text", "text": system}]},
            {"role": "user",
             "content": [
                 {"type": "image", "image": str(image_path)},
                 {"type": "text", "text": prompt},
             ]},
        ]

    def answer(self, image_path: Path, prompt: str, system: str,
               seed: int = 0) -> str:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = self._messages(image_path, prompt, system)
        # Modern processors resolve the "image" fields in the template and return
        # pixel tensors directly. This is the single unified path across families.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.processor.decode(gen, skip_special_tokens=True)
        return text.strip()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


## python answer_vqa.py --model qwen3vl-2b --type   --options 5


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3vl-2b", choices=sorted(PRESETS),
                   help="preset model to run (default: qwen3vl)")
    p.add_argument("--model-id", default=None,
                   help="explicit HF checkpoint id (overrides --model's id)")
    p.add_argument("--family", default=None,
                   choices=[FAMILY_QWEN, FAMILY_LLAMA, FAMILY_INTERNVL],
                   help="adapter family when using --model-id")
    p.add_argument("--data", default=None,
                   help="path to data file (auto: {type}.json or {type}_e.json "
                        "based on --type and --options)")
    p.add_argument("--data-root", default=".",
                   help="root that image paths are relative to (default: repo root)")
    p.add_argument("--type", default="law", choices=["all", "detection", "law"],
                   help="only run questions of this type_question (default: all)")
    p.add_argument("--options", default="4", choices=["4", "5"],
                   help="4 = A-D only; 5 = add option E 'unknown' with an "
                        "instruction to pick E when unsure (default: 4). "
                        "Auto-selects {type}.json or {type}_e.json")
    p.add_argument("--out", default=None,
                   help="output json path; overrides automatic name "
                        "logs/<model>/<type>_(w|wo)_E.json, e.g. logs/qwen3vl-4b/law_w_E.json")
    p.add_argument("--limit", type=int, default=None,
                   help="only run the first N questions (smoke test)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the per-sample log line (keep progress + summary)")
    p.add_argument("--device", default="cuda",
                   help="cuda | cpu | cuda:0 | auto (default: cuda)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "auto"])
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    # Windows consoles default to cp1252; Vietnamese/model text would crash prints.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args(argv)

    # Auto-select data file based on --type and --options if not specified
    if args.data is None:
        if args.type == "all":
            sys.exit("--data is required when --type=all; provide explicit data file")
        e_suffix = "_e" if args.options == "5" else ""
        args.data = f"{args.type}{e_suffix}.json"

    if args.model_id:
        model_id = args.model_id
        family = args.family
        if family is None:
            sys.exit("--family is required when --model-id is given")
        model_tag = re.sub(r"[^A-Za-z0-9.-]+", "_", model_id.split("/")[-1])
    else:
        preset = PRESETS[args.model]
        model_id, family = preset.hf_id, preset.family
        model_tag = args.model

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"data file not found: {data_path}")
    records = json.loads(data_path.read_text(encoding="utf-8"))

    if args.type != "all":
        records = [r for r in records if r.get("type_question") == args.type]
        if not records:
            sys.exit(f"no questions with type_question == '{args.type}'")
    if args.limit is not None:
        records = records[:args.limit]

    include_e = args.options == "5"
    system = system_prompt(include_e)

    data_root = Path(args.data_root).resolve()
    if args.out:
        out_path = Path(args.out)
    else:
        # logs/<model_tag>/<type>_(w|wo)_E.json  e.g. logs/qwen3vl-4b/law_w_E.json
        e_tag = "w_E" if include_e else "wo_E"
        out_path = Path("logs") / model_tag / f"{args.type}_{e_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data] {len(records)} questions from {data_path} "
          f"(type={args.type}, options={'ABCDE' if include_e else 'ABCD'})",
          flush=True)

    vlm = VLM(model_id, family, args.device, args.dtype, args.max_new_tokens)
    vlm.load()

    results: List[dict] = []
    n_correct = 0
    n_graded = 0
    t0 = time.time()

    for i, rec in enumerate(records, 1):
        img = resolve_image_path(rec["image"], data_root)
        pred: Optional[str] = None
        raw = ""
        error: Optional[str] = None
        if not img.exists():
            error = f"image not found: {img}"
        else:
            try:
                raw = vlm.answer(img, build_prompt(rec, include_e), system,
                                 seed=args.seed)
                pred = parse_letter(raw, include_e)
            except Exception as exc:  # keep going; record the failure
                error = f"{type(exc).__name__}: {exc}"

        gt = rec.get("GT")
        correct = pred is not None and gt is not None and pred == gt
        if gt is not None and error is None:
            n_graded += 1
            n_correct += int(correct)

        results.append({
            "id": rec.get("id"),
            "image": rec["image"],
            "type_question": rec.get("type_question"),
            "type_image": rec.get("type_image"),
            "gt": gt,
            "pred": pred,
            "correct": correct,
            "raw": raw,
            "error": error,
        })

        # per-sample log: image path, id, question, options, GT, prediction
        if not args.quiet:
            print(f"[{i}/{len(records)}] id={rec.get('id')}", flush=True)
            print(f"  image   : {rec['image']}")
            print(f"  question: {rec['question'].strip()}")
            for L in option_letters(include_e):
                if L == "E" and "E" not in rec:
                    text = DEFAULT_E_TEXT
                else:
                    text = rec[L].strip()
                print(f"  {L}: {text}")
            print(f"  GT: {gt}   pred: {pred}")

        if i % 25 == 0 or i == len(records):
            acc = (n_correct / n_graded) if n_graded else 0.0
            rate = i / (time.time() - t0)
            print(f"[{i}/{len(records)}] acc={acc:.3f} "
                  f"({n_correct}/{n_graded}) {rate:.2f} q/s", flush=True)

    # ------------------------------------------------------------------ #
    # Breakdowns
    # ------------------------------------------------------------------ #
    def breakdown(key: str) -> Dict[str, dict]:
        agg: Dict[str, List[int]] = {}
        for r in results:
            if r["error"] is not None or r["gt"] is None:
                continue
            k = r.get(key) or "unknown"
            c, t = agg.setdefault(k, [0, 0])
            agg[k] = [c + int(r["correct"]), t + 1]
        return {k: {"correct": c, "total": t, "accuracy": (c / t if t else 0.0)}
                for k, (c, t) in sorted(agg.items())}

    overall = (n_correct / n_graded) if n_graded else 0.0
    summary = {
        "model": model_tag,
        "model_id": model_id,
        "family": family,
        "type": args.type,
        "options": "ABCDE" if include_e else "ABCD",
        "n_questions": len(records),
        "n_graded": n_graded,
        "n_correct": n_correct,
        "accuracy": overall,
        "elapsed_sec": round(time.time() - t0, 1),
        "by_type_question": breakdown("type_question"),
        "by_type_image": breakdown("type_image"),
    }

    out_path.write_text(
        json.dumps({"summary": summary, "predictions": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"model    : {model_tag} ({model_id})")
    print(f"accuracy : {overall:.4f}  ({n_correct}/{n_graded})")
    print("by question type:")
    for k, v in summary["by_type_question"].items():
        print(f"  {k:12s} {v['accuracy']:.3f}  ({v['correct']}/{v['total']})")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
