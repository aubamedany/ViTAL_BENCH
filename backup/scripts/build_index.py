#!/usr/bin/env python3
"""Embed the statute corpus and build the FAISS index (EXPERIMENT_PLAN.md §6).

The index directory records the corpus_version and embedder_id; the retriever
refuses to load an index built against a different corpus version, so a QCVN
ingestion (which changes the corpus version) forces a rebuild.

Usage:
    python scripts/build_index.py [--corpus PATH] [--embedder MODEL_ID]
                                  [--out DIR] [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backup.vntraffic import constants as C
from backup.vntraffic.corpus import load_corpus
from backup.vntraffic.embedder import Embedder
from backup.vntraffic.retriever import Retriever


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=C.CORPUS_JSON)
    ap.add_argument("--aliases", default=None, help="alias JSON (for ingested corpora)")
    ap.add_argument("--embedder", default=C.DEFAULT_EMBEDDER)
    ap.add_argument("--out", default=None, help="index dir (default artifacts/index/<embedder>)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus, args.aliases)
    embedder = Embedder(args.embedder, device=args.device)
    out = args.out or os.path.join(C.INDEX_DIR, args.embedder.replace("/", "__"))

    print(f"Embedding {len(corpus)} chunks with {args.embedder} "
          f"(corpus v{corpus.version}) ...")
    retr = Retriever.build(corpus, embedder)
    retr.save(out)
    print(f"Index → {out}  (dim={embedder.dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
