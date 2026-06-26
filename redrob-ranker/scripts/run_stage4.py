#!/usr/bin/env python3
"""Stage 4 — score retrieved pool and write top-100 submission.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import (
    CANDIDATE_FEATURES_FULL_JSONL,
    CANDIDATES_JSONL,
    PRECOMPUTED_DIR,
    RETRIEVED_TOP3K_JSONL,
)
from src.reasoning import build_reasoning, load_candidate_profiles, submission_score_for_rank
from src.scoring import DEFAULT_WEIGHTS, load_features_for_ids, rank_candidates

SUBMISSION_CSV = ROOT / "submission.csv"
SCORED_TOP100_JSONL = PRECOMPUTED_DIR / "scored_top100.jsonl"


def write_submission(
    ranked_rows: list[dict],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in ranked_rows:
            writer.writerow(
                [row["candidate_id"], row["rank"], f"{row['score']:.4f}", row["reasoning"]]
            )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 — final scoring and submission")
    parser.add_argument("--retrieved", default=str(RETRIEVED_TOP3K_JSONL))
    parser.add_argument("--features", default=str(CANDIDATE_FEATURES_FULL_JSONL))
    parser.add_argument("--candidates", default=str(CANDIDATES_JSONL))
    parser.add_argument("--out", default=str(SUBMISSION_CSV))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--scores-out", default=str(SCORED_TOP100_JSONL))
    args = parser.parse_args()

    t0 = time.time()
    print("Scoring retrieved candidates...")
    ranked = rank_candidates(args.retrieved, args.features, weights=DEFAULT_WEIGHTS)
    top = ranked[: args.top_k]
    print(f"Scored {len(ranked):,} candidates; selecting top {len(top)}")

    top_ids = {x.candidate_id for x in top}
    features = load_features_for_ids(args.features, top_ids)
    profiles = load_candidate_profiles(args.candidates, top_ids)

    submission_rows: list[dict] = []
    scored_rows: list[dict] = []

    for rank, breakdown in enumerate(top, start=1):
        cid = breakdown.candidate_id
        feat = features[cid]
        profile = profiles.get(cid, {})
        reasoning = build_reasoning(profile, breakdown, feat)
        score = submission_score_for_rank(rank)
        row = {
            "candidate_id": cid,
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
            "breakdown": breakdown.to_dict(),
        }
        submission_rows.append(row)
        scored_rows.append(row)

    out_path = write_submission(submission_rows, Path(args.out))

    scores_out = Path(args.scores_out)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_out, "w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "weights": DEFAULT_WEIGHTS,
        "submission": str(out_path),
        "scored_details": str(scores_out),
        "top_10": [
            {
                "rank": r["rank"],
                "candidate_id": r["candidate_id"],
                "score": r["score"],
                "total_score": r["breakdown"]["total_score"],
                "reasoning": r["reasoning"],
            }
            for r in submission_rows[:10]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
