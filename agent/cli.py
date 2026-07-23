#!/usr/bin/env python
"""ViTAL-Agent driver — runs the §5 condition ladder over law.json and
reports the §6 metric set (MCQ-Acc, pick-longest baseline, Grounded-Acc,
Divergence).

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
    python -m agent.cli --condition e4_agent --max-iterations 3 --limit 50  # widen the refine loop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from agent.corpus import Corpus
from agent.grounding import citation_wrong, grounded_correct
from agent.llm import MockLLM, get_assertion_count, reset_assertion_count
from agent.logging_utils import append_jsonl, build_log_record
from agent.pipeline import CONDITIONS, DEFAULT_K, DEFAULT_MAX_ITERATIONS, run_condition
from agent.resolve import CitationIndex
from agent.retrieval.bm25 import DEFAULT_INDEX_DIR, BM25Retriever
from agent.retrieval.mock import MockRetriever
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

ROOT = Path(__file__).resolve().parent.parent


def resolve_image_path(rel: str, data_root: Path) -> Path:
    return (data_root / rel.lstrip("/\\")).resolve()


def pick_longest(options: Dict[str, str]) -> str:
    """The baseline every retriever must beat: pick the option with the most
    characters. Always reported beneath MCQ-Acc (§6) — it scores 53.7% on
    MCQ and 0% on grounded, which is why grounded accuracy is the headline."""
    return max(options, key=lambda L: len(options[L]))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "law.json"))
    p.add_argument("--data-root", default=str(ROOT))
    p.add_argument("--corpus", default=str(ROOT / "law_db.json"))
    p.add_argument("--condition", default="e4_agent", choices=CONDITIONS)
    p.add_argument("--k", type=int, default=DEFAULT_K, help="chunks retrieved per query, per retrieval round")
    p.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="e4_agent refine-loop cap: N RETRIEVE rounds, up to N-1 ASSESS calls "
        "(ASSESS is skipped on the final round); ignored by e0/e1/e2/e3 (default: 2)",
    )
    p.add_argument(
        "--retriever",
        default="bm25",
        choices=("bm25", "mock"),
        help="R-BM25 (default; §3.3) or the deterministic MockRetriever used for pipeline dry-runs",
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
        help="backend label recorded in the log; with --llm vlm, also the answer_vqa.PRESETS name to load "
        "(e.g. qwen3vl-2b, internvl-2b) unless --model-id is given",
    )
    p.add_argument("--model-id", default=None, help="explicit HF checkpoint id for --llm vlm (overrides --model preset)")
    p.add_argument("--family", default=None, choices=("qwen", "llama", "internvl"), help="adapter family for --model-id")
    p.add_argument("--device", default="cuda", help="cuda | cpu | cuda:0 (only used by --llm vlm)")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32", "auto"))
    p.add_argument(
        "--json-max-new-tokens", type=int, default=512, help="generation budget for JSON-producing calls (--llm vlm)"
    )
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="per-chunk truncation for --llm vlm prompts (default: agent.llm_vlm.DEFAULT_MAX_CHUNK_CHARS, 3500). "
        "A khoản chunk is self-contained (penalty + all điểm sub-items) — truncating too low can silently "
        "cut off the exact điểm that matches the item.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="JSONL log path (default: logs/agent/<condition>.jsonl)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.max_iterations < 1:
        sys.exit(f"--max-iterations must be >= 1, got {args.max_iterations}")
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
        vlm_kwargs = {}
        if args.max_chunk_chars is not None:
            vlm_kwargs["max_chunk_chars"] = args.max_chunk_chars
        llm = VLMLLMClient(
            model_id,
            family,
            device=args.device,
            dtype=args.dtype,
            json_max_new_tokens=args.json_max_new_tokens,
            seed=args.seed,
            **vlm_kwargs,
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
    n_graded = 0
    n_mcq_correct = 0
    n_pick_longest_correct = 0
    n_grounded_correct = 0
    n_retry_used = 0
    n_correct_with_wrong_citation = 0  # numerator of Divergence
    n_iterations_total = 0
    stop_reason_counts: Counter = Counter()
    by_type_image: Dict[str, Dict[str, float]] = {}

    t0 = time.time()
    progress = tqdm(records, total=n, unit="q", desc=args.condition, disable=args.quiet)
    for rec in progress:
        image_path = resolve_image_path(rec["image"], Path(args.data_root))
        options = {L: rec[L] for L in ("A", "B", "C", "D")}
        gold_citations_raw = rec.get("law", [])
        gold_resolved = citation_index.resolve_set(gold_citations_raw)

        reset_assertion_count()
        parse_failures_before = getattr(llm, "parse_failure_count", 0)
        t_item0 = time.time()
        perception, retrieved, selection, trace = run_condition(
            condition=args.condition,
            image_path=image_path,
            question=rec["question"],
            options=options,
            corpus=corpus,
            retriever=retriever,
            llm=llm,
            citation_index=citation_index,
            gold_chunk_ids=gold_resolved,
            k=args.k,
            as_of=args.as_of,
            max_iterations=args.max_iterations,
        )
        wall_clock = time.time() - t_item0

        citation_resolved = None
        if selection.citation is not None:
            citation_resolved = citation_index.resolve(selection.citation).resolved_id

        gt = rec.get("GT")
        mcq_correct = gt is not None and selection.answer == gt
        pl_correct = gt is not None and pick_longest(options) == gt
        grounded = grounded_correct(mcq_correct, citation_resolved, gold_resolved)
        if gt is not None:
            n_graded += 1
            n_mcq_correct += int(mcq_correct)
            n_pick_longest_correct += int(pl_correct)
            n_grounded_correct += int(grounded)
            if mcq_correct and citation_wrong(citation_resolved, gold_resolved):
                n_correct_with_wrong_citation += 1
        if trace.retry_used:
            n_retry_used += 1
        n_iterations_total += trace.iterations_used
        stop_reason_counts[trace.stop_reason] += 1

        type_image = rec.get("type_image") or "unknown"
        agg = by_type_image.setdefault(type_image, {"mcq_correct": 0, "grounded_correct": 0, "total": 0})
        agg["total"] += 1
        agg["mcq_correct"] += int(mcq_correct)
        agg["grounded_correct"] += int(grounded)

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
                retrieved=retrieved,
                selection=selection,
                gold_answer=gt,
                gold_citations_raw=gold_citations_raw,
                gold_citations_resolved=sorted(gold_resolved),
                wall_clock_sec=wall_clock,
                assertion_count=get_assertion_count(),
                extra={
                    "type_image": rec.get("type_image"),
                    "predicted_answer": selection.answer,
                    "citation_resolved": citation_resolved,
                    "mcq_correct": mcq_correct,
                    "grounded_correct": grounded,
                    "llm_parse_failures": parse_failures_after - parse_failures_before,
                },
            ),
        )

        if n_graded:
            progress.set_postfix(
                mcq=f"{n_mcq_correct / n_graded:.3f}",
                grounded=f"{n_grounded_correct / n_graded:.3f}",
                refresh=False,
            )

    progress.close()
    elapsed = time.time() - t0
    mcq_acc = n_mcq_correct / n_graded if n_graded else 0.0
    pick_longest_acc = n_pick_longest_correct / n_graded if n_graded else 0.0
    grounded_acc = n_grounded_correct / n_graded if n_graded else 0.0
    divergence = n_correct_with_wrong_citation / n_mcq_correct if n_mcq_correct else 0.0
    retry_rate = n_retry_used / n if n else 0.0
    avg_iterations_used = n_iterations_total / n if n else 0.0
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
        "k": args.k,
        "max_iterations": args.max_iterations,
        "mcq_acc": mcq_acc,
        "pick_longest_acc": pick_longest_acc,
        "grounded_acc": grounded_acc,
        "divergence": divergence,
        "retry_rate": retry_rate,
        "avg_iterations_used": avg_iterations_used,
        "stop_reason_distribution": dict(stop_reason_counts),
        "by_type_image": {
            k: {
                "total": v["total"],
                "mcq_acc": v["mcq_correct"] / v["total"] if v["total"] else 0.0,
                "grounded_acc": v["grounded_correct"] / v["total"] if v["total"] else 0.0,
            }
            for k, v in sorted(by_type_image.items())
        },
        "elapsed_sec": round(elapsed, 1),
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"condition        : {args.condition}")
    print(f"MCQ-Acc          : {mcq_acc:.4f} ({n_mcq_correct}/{n_graded})")
    print(f"  pick-longest   : {pick_longest_acc:.4f}  [always-print baseline, §6]")
    print(f"Grounded-Acc     : {grounded_acc:.4f} ({n_grounded_correct}/{n_graded})  <- headline")
    print(f"Divergence       : {divergence:.4f}  [P(citation wrong | answer correct)]")
    print(f"retry rate       : {retry_rate:.4f} ({n_retry_used}/{n})")
    print(f"avg iterations   : {avg_iterations_used:.2f}  (max_iterations={args.max_iterations})")
    print(f"stop reasons     : {dict(stop_reason_counts)}")
    if args.llm == "vlm":
        print(f"llm parse fails  : {total_parse_failures}/{n} ({summary['llm_parse_failure_rate']:.4f})")
    print(f"\nwrote {out_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
