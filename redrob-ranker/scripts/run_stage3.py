#!/usr/bin/env python3
"""Run Stage 3 — hybrid retrieval (100K -> top-K)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import (
    CANDIDATE_EMBEDDINGS_NPY,
    CANDIDATE_FEATURES_FULL_JSONL,
    JD_EMBEDDING_NPY,
    JD_REQUIREMENTS_JSON,
    RETRIEVED_TOP3K_JSONL,
)
from src.retrieval import (
    BM25_WEIGHT,
    COSINE_WEIGHT,
    DEFAULT_TOP_K,
    encode_texts,
    get_jd_text,
    hybrid_retrieve,
    load_candidate_corpus,
    load_or_build_embeddings,
    save_retrieval_results,
)
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 — hybrid retrieval")
    parser.add_argument(
        "--features",
        default=str(CANDIDATE_FEATURES_FULL_JSONL),
        help="Stage 2 features JSONL (uses canonical_text)",
    )
    parser.add_argument(
        "--jd-json",
        default=str(JD_REQUIREMENTS_JSON),
        help="Parsed JD JSON from Stage 1",
    )
    parser.add_argument(
        "--out",
        default=str(RETRIEVED_TOP3K_JSONL),
        help="Output JSONL for retrieved candidates",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cosine-weight", type=float, default=COSINE_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=BM25_WEIGHT)
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Force recompute candidate embeddings",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("Loading candidate corpus...")
    candidate_ids, candidate_texts = load_candidate_corpus(args.features)
    print(f"Loaded {len(candidate_ids):,} candidates")

    jd_text = get_jd_text(args.jd_json)
    print(f"JD text length: {len(jd_text):,} chars")

    print("Loading/building candidate embeddings (cached if available)...")
    candidate_embeddings = load_or_build_embeddings(
        candidate_texts,
        cache_path=CANDIDATE_EMBEDDINGS_NPY,
        force_rebuild=args.rebuild_embeddings,
    )

    if JD_EMBEDDING_NPY.exists() and not args.rebuild_embeddings:
        jd_embedding = np.load(JD_EMBEDDING_NPY)
    else:
        jd_embedding = encode_texts([jd_text])[0]
        np.save(JD_EMBEDDING_NPY, jd_embedding)

    print(f"Running hybrid retrieval (top {args.top_k})...")
    results = hybrid_retrieve(
        jd_text=jd_text,
        candidate_ids=candidate_ids,
        candidate_texts=candidate_texts,
        candidate_embeddings=candidate_embeddings,
        jd_embedding=jd_embedding,
        top_k=args.top_k,
        cosine_weight=args.cosine_weight,
        bm25_weight=args.bm25_weight,
    )

    out_path = save_retrieval_results(results, args.out)
    elapsed = time.time() - t0

    summary = {
        "candidates_scored": len(candidate_ids),
        "retrieved": len(results),
        "cosine_weight": args.cosine_weight,
        "bm25_weight": args.bm25_weight,
        "elapsed_seconds": round(elapsed, 2),
        "output": str(out_path),
        "top_5": results[:5],
        "bottom_5_of_retrieved": results[-5:],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
