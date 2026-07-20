#!/usr/bin/env python
"""Build the BM25 Lucene index over law_db.json (AGENT_PLAN.md §3.2, R-BM25).

Two steps:
  1. Write a pyserini `JsonCollection` file: one line per chunk,
     {"id": chunk_id, "contents": <VN-segmented title + text>}.
  2. Shell out to `python -m pyserini.index.lucene` to build the index with
     `-pretokenized` (whitespace analyzer, no stemming/stopwording) so the
     pyvi word-segmentation done in step 1 is respected verbatim — English
     Porter stemming would otherwise mangle Vietnamese tokens.

Query time must use the exact same segmentation (agent/retrieval/vn_tokenize.py)
or indexed terms and query terms silently stop matching.

Usage
-----
    python -m agent.build_bm25_index
    python -m agent.build_bm25_index --corpus law_db.json --index index/bm25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agent.corpus import Corpus
from agent.retrieval.vn_tokenize import segment

ROOT = Path(__file__).resolve().parent.parent


def build_collection(corpus: Corpus, collection_dir: Path) -> Path:
    collection_dir.mkdir(parents=True, exist_ok=True)
    collection_file = collection_dir / "law_chunks.jsonl"
    with collection_file.open("w", encoding="utf-8") as f:
        for chunk in corpus.chunks:
            contents = segment(f"{chunk.title}\n{chunk.text}")
            f.write(json.dumps({"id": chunk.chunk_id, "contents": contents}, ensure_ascii=False) + "\n")
    return collection_file


def run_lucene_index(collection_dir: Path, index_dir: Path) -> None:
    if index_dir.exists():
        import shutil

        shutil.rmtree(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "-collection", "JsonCollection",
        "-input", str(collection_dir),
        "-index", str(index_dir),
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "1",
        "-storePositions", "-storeDocvectors", "-storeRaw",
        "-pretokenized",  # respect pyvi segmentation; no stemming/stopwords
    ]
    subprocess.run(cmd, check=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(ROOT / "law_db.json"))
    p.add_argument("--index", default=str(ROOT / "index" / "bm25"))
    p.add_argument("--collection-dir", default=None, help="default: <index>_collection (deleted-safe staging dir)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    corpus = Corpus.load(args.corpus)
    index_dir = Path(args.index)
    collection_dir = Path(args.collection_dir) if args.collection_dir else index_dir.parent / f"{index_dir.name}_collection"

    print(f"[build_bm25_index] {len(corpus)} chunks from {args.corpus} (v{corpus.version})")
    collection_file = build_collection(corpus, collection_dir)
    print(f"[build_bm25_index] wrote collection: {collection_file}")

    run_lucene_index(collection_dir, index_dir)

    meta = {"corpus_version": corpus.version, "corpus_path": str(Path(args.corpus).resolve()), "n_chunks": len(corpus)}
    (index_dir / "_corpus_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[build_bm25_index] index built at {index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
