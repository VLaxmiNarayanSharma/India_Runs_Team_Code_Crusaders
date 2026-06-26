"""Stage 3: Hybrid retrieval — dense cosine similarity + BM25."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# Avoid TensorFlow/Keras import issues on Windows; embeddings use PyTorch only.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
from rank_bm25 import BM25Okapi

COSINE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
DEFAULT_TOP_K = 3000
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 512


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", text.lower())


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def load_candidate_corpus(features_path: str | Path) -> tuple[list[str], list[str]]:
    """Load candidate IDs and canonical texts from Stage 2 JSONL."""
    ids: list[str] = []
    texts: list[str] = []
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            ids.append(record["candidate_id"])
            texts.append(record.get("canonical_text", ""))
    return ids, texts


def get_jd_text(jd_path: str | Path) -> str:
    jd = json.loads(Path(jd_path).read_text(encoding="utf-8"))
    return jd.get("canonical_jd_text") or jd.get("raw_text", "")


def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    model = _load_encoder(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def load_or_build_embeddings(
    texts: list[str],
    cache_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    force_rebuild: bool = False,
) -> np.ndarray:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            return embeddings
        print(f"Embedding cache size mismatch ({embeddings.shape[0]} vs {len(texts)}); rebuilding.")

    embeddings = encode_texts(texts, model_name=model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def cosine_similarity_scores(jd_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalized vectors is a dot product."""
    if jd_embedding.ndim == 1:
        jd_embedding = jd_embedding.reshape(1, -1)
    scores = candidate_embeddings @ jd_embedding.T
    return scores.reshape(-1).astype(np.float32)


def bm25_scores(jd_text: str, candidate_texts: list[str]) -> np.ndarray:
    corpus_tokens = [tokenize(t) for t in candidate_texts]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = tokenize(jd_text)
    scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
    return scores


def hybrid_retrieve(
    jd_text: str,
    candidate_ids: list[str],
    candidate_texts: list[str],
    candidate_embeddings: np.ndarray | None = None,
    jd_embedding: np.ndarray | None = None,
    candidate_emb_cache: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    cosine_weight: float = COSINE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    force_rebuild_embeddings: bool = False,
) -> list[dict[str, Any]]:
    if len(candidate_ids) != len(candidate_texts):
        raise ValueError("candidate_ids and candidate_texts length mismatch")

    if candidate_embeddings is None:
        if candidate_emb_cache is None:
            candidate_embeddings = encode_texts(candidate_texts, model_name=model_name)
        else:
            candidate_embeddings = load_or_build_embeddings(
                candidate_texts,
                cache_path=candidate_emb_cache,
                model_name=model_name,
                force_rebuild=force_rebuild_embeddings,
            )

    if jd_embedding is None:
        jd_embedding = encode_texts([jd_text], model_name=model_name)[0]

    cosine_raw = cosine_similarity_scores(jd_embedding, candidate_embeddings)
    bm25_raw = bm25_scores(jd_text, candidate_texts)

    cosine_norm = minmax_normalize(cosine_raw)
    bm25_norm = minmax_normalize(bm25_raw)
    final_scores = cosine_weight * cosine_norm + bm25_weight * bm25_norm

    top_k = min(top_k, len(candidate_ids))
    top_indices = np.argpartition(-final_scores, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-final_scores[top_indices])]

    results: list[dict[str, Any]] = []
    for retrieval_rank, idx in enumerate(top_indices, start=1):
        i = int(idx)
        results.append(
            {
                "retrieval_rank": retrieval_rank,
                "candidate_id": candidate_ids[i],
                "final_retrieval_score": round(float(final_scores[i]), 6),
                "cosine_score": round(float(cosine_norm[i]), 6),
                "bm25_score": round(float(bm25_norm[i]), 6),
                "cosine_raw": round(float(cosine_raw[i]), 6),
                "bm25_raw": round(float(bm25_raw[i]), 6),
            }
        )
    return results


def save_retrieval_results(results: list[dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
