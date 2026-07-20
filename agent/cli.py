#!/usr/bin/env python
"""VNTraffic-Agent driver — runs the §6 baseline/agent conditions over
law.json and reports accuracy + the frozen §6b grounding metrics.

Retrieval defaults to BM25Retriever (agent/retrieval/bm25.py), a pyserini/Lucene
index over law_db.json built by `python -m agent.build_bm25_index` (build it
once before the first run). `--retriever mock` swaps in the deterministic,
zero-download MockRetriever for pipeline dry-runs instead.

LLM defaults to MockLLM (deterministic, zero-download stand-in — see
agent/llm.py). `--llm vlm` swaps in VLMLLMClient (agent/llm_vlm.py), which
wraps the same answer_vqa.VLM model wrapper run_100.sh/answer_vqa.py use
(qwen3vl/llama/internvl) behind the LLMClient protocol; nothing in
agent/pipeline.py changes either way.

Examples
--------
    python -m agent.build_bm25_index                       # build the index once
    python -m agent.cli --condition e4_agent --limit 50
    python -m agent.cli --condition e0_oracle --limit 200
    python -m agent.cli --condition e3_rag_qo --limit 50   # the leakage probe
    python -m agent.cli --condition e4_agent --retriever mock --limit 50  # no index needed
    python -m agent.cli --condition e4_agent --llm vlm --model qwen3vl-2b --limit 20  # real model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent.corpus import Corpus
from agent.grounding import citation_prf1, strict_grounding_correct
from agent.llm import MockLLM, get_assertion_count, reset_assertion_count
from agent.logging_utils import append_jsonl, build_log_record
from agent.pipeline import CONDITIONS, run_condition
from agent.resolve import CitationIndex
from agent.retrieval.bm25 import DEFAULT_INDEX_DIR, BM25Retriever
from agent.retrieval.mock import MockRetriever

ROOT = Path(__file__).resolve().parent.parent


def resolve_image_path(rel: str, data_root: Path) -> Path:
    return (data_root / rel.lstrip("/\\")).resolve()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "law.json"))
    p.add_argument("--data-root", default=str(ROOT))
    p.add_argument("--corpus", default=str(ROOT / "law_db.json"))
    p.add_argument("--condition", default="e4_agent", choices=CONDITIONS)
    p.add_argument("--budget", type=int, default=3, help="B in the §6 table (e4_agent only)")
    p.add_argument("--k", type=int, default=5, help="chunks retrieved per query")
    p.add_argument(
        "--retriever",
        default="bm25",
        choices=("bm25", "mock"),
        help="R-BM25 (default; §3.2) or the deterministic MockRetriever used for pipeline dry-runs",
    )
    p.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="BM25 Lucene index dir (see agent/build_bm25_index.py)")
    p.add_argument("--bm25-k1", type=float, default=0.9)
    p.add_argument("--bm25-b", type=float, default=0.4)
    p.add_argument("--as-of", default=None, help="effective_date filter, e.g. 2025-01-01")
    p.add_argument(
        "--llm",
        default="mock",
        choices=("mock", "vlm"),
        help="MockLLM (default, deterministic, zero-download) or VLMLLMClient (agent/llm_vlm.py, a real local model)",
    )
    p.add_argument(
        "--model",
        default="mock",
        help="backend label recorded in the log (§7); with --llm vlm, also the answer_vqa.PRESETS name to load "
        "(e.g. qwen3vl-2b, internvl-2b) unless --model-id is given",
    )
    p.add_argument("--model-id", default=None, help="explicit HF checkpoint id for --llm vlm (overrides --model preset)")
    p.add_argument("--family", default=None, choices=("qwen", "llama", "internvl"), help="adapter family for --model-id")
    p.add_argument("--device", default="cuda", help="cuda | cpu | cuda:0 (only used by --llm vlm)")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32", "auto"))
    p.add_argument(
        "--json-max-new-tokens", type=int, default=640, help="generation budget for JSON-producing calls (--llm vlm)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="JSONL log path (default: logs/agent/<condition>.jsonl)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"data file not found: {data_path}")
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        records = records[: args.limit]

    corpus = Corpus.load(args.corpus)
    citation_index = CitationIndex.from_file(args.corpus)
    if args.retriever == "bm25":
        retriever = BM25Retriever(index_dir=args.index, k1=args.bm25_k1, b=args.bm25_b)
    else:
        retriever = MockRetriever(corpus)

    if args.llm == "vlm":
        from agent.llm_vlm import VLMLLMClient  # deferred: pulls in torch/transformers

        if args.model_id:
            if args.family is None:
                sys.exit("--family is required when --model-id is given")
            model_id, family = args.model_id, args.family
            model_label = args.model if args.model != "mock" else model_id.split("/")[-1]
        else:
            import answer_vqa

            if args.model not in answer_vqa.PRESETS:
                sys.exit(
                    f"--model {args.model!r} is not a real VLM preset; "
                    f"pass one of {sorted(answer_vqa.PRESETS)} or use --model-id/--family"
                )
            preset = answer_vqa.PRESETS[args.model]
            model_id, family = preset.hf_id, preset.family
            model_label = args.model
        llm = VLMLLMClient(
            model_id,
            family,
            device=args.device,
            dtype=args.dtype,
            json_max_new_tokens=args.json_max_new_tokens,
            seed=args.seed,
        )
    else:
        model_label = args.model
        llm = MockLLM()

    out_path = Path(args.out) if args.out else ROOT / "logs" / "agent" / f"{args.condition}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # each run starts a fresh log file

    print(
        f"[data] {len(records)} questions from {data_path} | condition={args.condition} "
        f"| corpus={len(corpus)} chunks (v{corpus.version}) | retriever={args.retriever}"
        + (f" ({args.index})" if args.retriever == "bm25" else "")
        + f" | llm={args.llm} ({model_label})",
        flush=True,
    )

    n = len(records)
    n_correct = 0
    n_graded = 0
    n_abstained = 0
    n_grounding_evaluated = 0
    n_strict_correct = 0
    sum_precision = sum_recall = sum_f1 = 0.0
    by_type_image: Dict[str, Dict[str, float]] = {}

    t0 = time.time()
    for i, rec in enumerate(records, 1):
        image_path = resolve_image_path(rec["image"], Path(args.data_root))
        options = {L: rec[L] for L in ("A", "B", "C", "D")}
        gold_citations_raw = rec.get("law", [])
        gold_resolved = citation_index.resolve_set(gold_citations_raw)

        reset_assertion_count()
        parse_failures_before = getattr(llm, "parse_failure_count", 0)
        t_item0 = time.time()
        determination, selection, trace = run_condition(
            condition=args.condition,
            image_path=image_path,
            question=rec["question"],
            options=options,
            corpus=corpus,
            retriever=retriever,
            llm=llm,
            citation_index=citation_index,
            gold_chunk_ids=gold_resolved,
            budget=args.budget,
            k=args.k,
            as_of=args.as_of,
        )
        wall_clock = time.time() - t_item0
        pred_resolved = citation_index.resolve_set(determination.applicable_provisions)

        gt = rec.get("GT")
        correct = gt is not None and selection.answer == gt
        if gt is not None:
            n_graded += 1
            n_correct += int(correct)

        if determination.insufficient_evidence:
            n_abstained += 1
        else:
            n_grounding_evaluated += 1
            n_strict_correct += int(strict_grounding_correct(gold_resolved, pred_resolved))
            precision, recall, f1 = citation_prf1(gold_resolved, pred_resolved)
            sum_precision += precision
            sum_recall += recall
            sum_f1 += f1

        type_image = rec.get("type_image") or "unknown"
        agg = by_type_image.setdefault(type_image, {"correct": 0, "total": 0})
        agg["total"] += 1
        agg["correct"] += int(correct)

        parse_failures_after = getattr(llm, "parse_failure_count", 0)
        append_jsonl(
            out_path,
            build_log_record(
                item_id=rec.get("id"),
                condition=args.condition,
                model=model_label,
                corpus_version=corpus.version,
                seed=args.seed,
                trace=trace,
                determination=determination,
                selection=selection,
                gold_answer=gt,
                gold_citations_raw=gold_citations_raw,
                gold_citations_resolved=sorted(gold_resolved),
                wall_clock_sec=wall_clock,
                assertion_count=get_assertion_count(),
                extra={
                    "type_image": rec.get("type_image"),
                    "predicted_answer": selection.answer,
                    "llm_parse_failures": parse_failures_after - parse_failures_before,
                },
            ),
        )

        if not args.quiet and (i % 100 == 0 or i == n):
            acc = n_correct / n_graded if n_graded else 0.0
            print(f"[{i}/{n}] acc={acc:.3f} ({n_correct}/{n_graded})", flush=True)

    elapsed = time.time() - t0
    accuracy = n_correct / n_graded if n_graded else 0.0
    strict_rate = n_strict_correct / n_grounding_evaluated if n_grounding_evaluated else 0.0
    macro_precision = sum_precision / n_grounding_evaluated if n_grounding_evaluated else 0.0
    macro_recall = sum_recall / n_grounding_evaluated if n_grounding_evaluated else 0.0
    macro_f1 = sum_f1 / n_grounding_evaluated if n_grounding_evaluated else 0.0
    abstention_rate = n_abstained / n if n else 0.0
    total_parse_failures = getattr(llm, "parse_failure_count", 0)

    summary = {
        "condition": args.condition,
        "model": model_label,
        "llm_backend": args.llm,
        "llm_parse_failures": total_parse_failures,
        "llm_parse_failure_rate": total_parse_failures / n if n else 0.0,
        "corpus_version": corpus.version,
        "n_questions": n,
        "n_graded": n_graded,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "abstention_rate": abstention_rate,
        "n_grounding_evaluated": n_grounding_evaluated,
        "strict_grounding_rate": strict_rate,
        "citation_precision_macro": macro_precision,
        "citation_recall_macro": macro_recall,
        "citation_f1_macro": macro_f1,
        "by_type_image": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for k, v in sorted(by_type_image.items())
        },
        "elapsed_sec": round(elapsed, 1),
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"condition        : {args.condition}")
    print(f"accuracy         : {accuracy:.4f} ({n_correct}/{n_graded})")
    print(f"abstention rate  : {abstention_rate:.4f} ({n_abstained}/{n})  [reported separately per §6b, never as a grounding failure]")
    print(f"strict grounding : {strict_rate:.4f} ({n_strict_correct}/{n_grounding_evaluated}, non-abstained items only)")
    print(f"citation F1      : {macro_f1:.4f}  (P={macro_precision:.4f} R={macro_recall:.4f})")
    if args.llm == "vlm":
        print(f"llm parse fails  : {total_parse_failures}/{n} ({summary['llm_parse_failure_rate']:.4f})")
    print(f"\nwrote {out_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
