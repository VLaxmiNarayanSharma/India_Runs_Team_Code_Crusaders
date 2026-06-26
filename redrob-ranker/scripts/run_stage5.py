#!/usr/bin/env python3
"""Stage 5 — anti-honeypot filtering + final top-100 submission.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_loader import load_candidates_by_ids
from src.honeypot import assess_honeypot
from src.paths import (
    CANDIDATE_FEATURES_FULL_JSONL,
    CANDIDATES_JSONL,
    PRECOMPUTED_DIR,
    RETRIEVED_TOP3K_JSONL,
)
from src.reasoning import build_reasoning, load_candidate_profiles, submission_score_for_rank
from src.scoring import DEFAULT_WEIGHTS, load_features_for_ids, load_retrieved_candidates, rank_candidates

SUBMISSION_CSV = ROOT / "submission.csv"
HONEYPOT_REPORT_JSONL = PRECOMPUTED_DIR / "honeypot_assessments.jsonl"
SCORED_TOP100_JSONL = PRECOMPUTED_DIR / "scored_top100.jsonl"


def write_submission(ranked_rows: list[dict], out_path: Path) -> Path:
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
    parser = argparse.ArgumentParser(description="Stage 5 — honeypot-aware final ranking")
    parser.add_argument("--retrieved", default=str(RETRIEVED_TOP3K_JSONL))
    parser.add_argument("--features", default=str(CANDIDATE_FEATURES_FULL_JSONL))
    parser.add_argument("--candidates", default=str(CANDIDATES_JSONL))
    parser.add_argument("--out", default=str(SUBMISSION_CSV))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--honeypot-report", default=str(HONEYPOT_REPORT_JSONL))
    parser.add_argument("--scores-out", default=str(SCORED_TOP100_JSONL))
    args = parser.parse_args()

    t0 = time.time()
    retrieval_scores = load_retrieved_candidates(args.retrieved)
    candidate_ids = set(retrieval_scores)
    features = load_features_for_ids(args.features, candidate_ids)

    print(f"Assessing honeypots for {len(candidate_ids):,} retrieved candidates...")
    raw_candidates = load_candidates_by_ids(args.candidates, candidate_ids)

    honeypot_rows: list[dict] = []
    flag_counter: Counter[str] = Counter()
    honeypot_count = 0

    honeypot_report_path = Path(args.honeypot_report)
    honeypot_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(honeypot_report_path, "w", encoding="utf-8") as report_f:
        for cid in sorted(candidate_ids):
            assessment = assess_honeypot(raw_candidates[cid], features[cid])
            row = assessment.to_dict()
            honeypot_rows.append(row)
            report_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if assessment.is_honeypot:
                honeypot_count += 1
            for flag in assessment.flags:
                flag_counter[flag] += 1

    print("Re-ranking with honeypot penalties...")
    ranked = rank_candidates(
        args.retrieved,
        args.features,
        candidates_path=args.candidates,
        weights=DEFAULT_WEIGHTS,
    )
    top = ranked[: args.top_k]

    top_ids = {x.candidate_id for x in top}
    top_features = {cid: features[cid] for cid in top_ids}
    top_profiles = load_candidate_profiles(args.candidates, top_ids)

    submission_rows: list[dict] = []
    scored_rows: list[dict] = []

    for rank, breakdown in enumerate(top, start=1):
        cid = breakdown.candidate_id
        feat = top_features[cid]
        profile = top_profiles.get(cid, {})
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
    with open(scores_out, "w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "retrieved_pool": len(candidate_ids),
        "honeypots_detected": honeypot_count,
        "honeypot_flag_counts": dict(flag_counter),
        "weights": DEFAULT_WEIGHTS,
        "submission": str(out_path),
        "honeypot_report": str(honeypot_report_path),
        "top_10": [
            {
                "rank": r["rank"],
                "candidate_id": r["candidate_id"],
                "total_score": r["breakdown"]["total_score"],
                "honeypot_penalty": r["breakdown"]["honeypot_penalty"],
                "honeypot_flags": r["breakdown"]["honeypot_flags"],
                "reasoning": r["reasoning"],
            }
            for r in submission_rows[:10]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
