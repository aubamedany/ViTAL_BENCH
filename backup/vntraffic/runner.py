"""Run orchestration: config → per-item JSONL + manifest (EXPERIMENT_PLAN.md §11).

Emits one JSONL record per item (per option-permutation under CircularEval) and
a manifest with everything needed to reproduce and to grade: model info, corpus
version, resolver coverage, seed, decoding params. Runs resume by skipping
already-logged (id, perm_shift) keys, so the heavy E4 × ladder path restarts
cheaply. The retrieval cache is flushed periodically so re-runs never re-embed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, List, Optional, Set, Tuple

from . import constants as C
from .conditions import build_condition
from .config import RunConfig
from .corpus import load_corpus
from .dataset import cyclic_permutations, filter_tier, load_items
from .models import build_client
from .models.base import GenerationConfig
from .resolver import build_coverage_report, gold_citations_from_items
from .retriever import LexicalRetriever, Retriever
from .embedder import Embedder


def _select_items(cfg: RunConfig):
    items = load_items(cfg.data)
    items = filter_tier(items, cfg.tier)
    if cfg.categories:
        cats = set(cfg.categories)
        items = [it for it in items if it.category in cats]
    if cfg.item_ids:
        keep = set(cfg.item_ids)
        items = [it for it in items if it.id in keep]
    items.sort(key=lambda x: x.id)
    if cfg.limit:
        items = items[: cfg.limit]
    return items


def _load_done_keys(jsonl_path: str) -> Set[Tuple[int, int]]:
    done: Set[Tuple[int, int]] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            done.add((rec["id"], rec.get("perm_shift", 0)))
    return done


def _build_retriever(cfg: RunConfig, corpus):
    cache = cfg.resolved_cache_path()
    if cfg.retriever == "lexical":
        return LexicalRetriever(corpus, cache_path=cache)
    embedder = Embedder(cfg.embedder, device=cfg.device)
    idx = cfg.resolved_index_dir()
    if os.path.exists(os.path.join(idx, "index.faiss")):
        return Retriever.load(idx, corpus, embedder, cache_path=cache)
    # no prebuilt index → build in-memory (fine for the small corpus)
    return Retriever.build(corpus, embedder, cache_path=cache)


def run(cfg: RunConfig, progress_every: int = 25) -> str:
    """Execute a run; returns the run directory."""
    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)
    jsonl_path = os.path.join(run_dir, "results.jsonl")

    items = _select_items(cfg)
    corpus = load_corpus(cfg.corpus, cfg.corpus_aliases)

    gen_cfg = GenerationConfig(max_new_tokens=cfg.max_new_tokens, seed=cfg.seed)
    client = build_client(cfg.model_key, backend=cfg.backend, device=cfg.device,
                          dtype=cfg.dtype, base_url=cfg.base_url,
                          served_model_id=cfg.served_model_id, max_tiles=cfg.max_tiles)

    needs_retrieval = cfg.condition in C.RETRIEVAL_CONDITIONS
    retriever = _build_retriever(cfg, corpus) if needs_retrieval else None

    condition = build_condition(
        cfg.condition, client=client, image_root=C.IMAGE_ROOT, retriever=retriever,
        gen_cfg=gen_cfg, top_k=cfg.top_k, budget=cfg.budget)

    perms = cyclic_permutations() if cfg.circular else [(0, None)]
    done = _load_done_keys(jsonl_path)

    # write manifest first (so an interrupted run is still documented)
    _write_manifest(cfg, run_dir, client, corpus, items, retriever)

    n_total = len(items) * len(perms)
    n_done = 0
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for it in items:
            for shift, perm in perms:
                if (it.id, shift) in done:
                    n_done += 1
                    continue
                result = condition.run_item(it, permutation=perm)
                rec = result.to_dict()
                rec["perm_shift"] = shift
                rec["model_key"] = cfg.model_key
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_done += 1
                if retriever is not None and n_done % progress_every == 0:
                    retriever.flush_cache()
                if n_done % progress_every == 0:
                    print(f"  [{cfg.run_id}] {n_done}/{n_total}")

    if retriever is not None:
        retriever.flush_cache()
    print(f"[{cfg.run_id}] done: {n_done}/{n_total} → {jsonl_path}")
    return run_dir


def _write_manifest(cfg: RunConfig, run_dir: str, client, corpus, items, retriever):
    coverage = None
    if cfg.tier == C.TIER_LAW:
        coverage = build_coverage_report(gold_citations_from_items(items), corpus).counts
    manifest = {
        "run_id": cfg.run_id,
        "created": _dt.datetime.utcnow().isoformat() + "Z",
        "condition": cfg.condition,
        "tier": cfg.tier,
        "n_items": len(items),
        "circular": cfg.circular,
        "model": client.info(),
        "corpus_version": corpus.version,
        "retriever": (cfg.retriever if retriever is not None else None),
        "embedder": (None if retriever is None else
                     ("lexical" if cfg.retriever == "lexical" else cfg.embedder)),
        "top_k": cfg.top_k,
        "budget": cfg.budget if cfg.condition == C.COND_E4 else None,
        "decoding": {"max_new_tokens": cfg.max_new_tokens, "seed": cfg.seed,
                     "temperature": 0.0, "do_sample": False},
        "resolver_coverage": coverage,
        "config": cfg.to_dict(),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
