#!/usr/bin/env python3
"""Stage 6 — regenerate template-based reasoning and refresh submission.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import CANDIDATES_JSONL, PRECOMPUTED_DIR
from src.reasoning import build_reasoning, load_candidate_profiles, submission_score_for_rank
from src.scoring import ScoreBreakdown

SUBMISSION_CSV = ROOT / "submission.csv"
SCORED_TOP100_JSONL = PRECOMPUTED_DIR / "scored_top100.jsonl"


def write_submission(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow(
                [row["candidate_id"], row["rank"], f"{row['score']:.4f}", row["reasoning"]]
            )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 — reasoning generation")
    parser.add_argument("--scored", default=str(SCORED_TOP100_JSONL))
    parser.add_argument("--candidates", default=str(CANDIDATES_JSONL))
    parser.add_argument("--features", default=str(PRECOMPUTED_DIR / "candidate_features_full.jsonl"))
    parser.add_argument("--out", default=str(SUBMISSION_CSV))
    args = parser.parse_args()

    t0 = time.time()
    scored_rows = [json.loads(line) for line in Path(args.scored).read_text(encoding="utf-8").splitlines() if line.strip()]
    top_ids = {r["candidate_id"] for r in scored_rows}
    profiles = load_candidate_profiles(args.candidates, top_ids)

    features: dict[str, dict] = {}
    with open(args.features, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["candidate_id"] in top_ids:
                features[rec["candidate_id"]] = rec

    updated: list[dict] = []
    for row in scored_rows:
        cid = row["candidate_id"]
        bd = row["breakdown"]
        breakdown = ScoreBreakdown(
            candidate_id=cid,
            semantic_fit=bd["semantic_fit"],
            career_trajectory=bd["career_trajectory"],
            skill_depth_trust=bd["skill_depth_trust"],
            behavioral_hireability=bd["behavioral_hireability"],
            inconsistency_penalty=bd["inconsistency_penalty"],
            keyword_stuffing_penalty=bd["keyword_stuffing_penalty"],
            honeypot_penalty=bd.get("honeypot_penalty", 0.0),
            honeypot_flags=bd.get("honeypot_flags", []),
            retrieval_score=bd["retrieval_score"],
            total_score=bd["total_score"],
        )
        reasoning = build_reasoning(profiles[cid], breakdown, features[cid])
        updated.append(
            {
                "candidate_id": cid,
                "rank": row["rank"],
                "score": row["score"],
                "reasoning": reasoning,
                "breakdown": bd,
            }
        )

    out_path = write_submission(updated, Path(args.out))

    with open(args.scored, "w", encoding="utf-8") as f:
        for row in updated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "submission": str(out_path),
        "sample_reasoning": [r["reasoning"] for r in updated[:5]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
