#!/usr/bin/env python3
"""Run Stage 2: build canonical text + structured features for candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_features import process_candidates
from src.paths import (
    CANDIDATE_FEATURES_JSONL,
    JD_REQUIREMENTS_JSON,
    SAMPLE_CANDIDATES_JSON,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 — candidate representation")
    parser.add_argument(
        "--candidates",
        default=str(SAMPLE_CANDIDATES_JSON),
        help="Path to candidates.jsonl or sample_candidates.json",
    )
    parser.add_argument(
        "--jd-json",
        default=str(JD_REQUIREMENTS_JSON),
        help="Parsed JD JSON from Stage 1",
    )
    parser.add_argument(
        "--out",
        default=str(CANDIDATE_FEATURES_JSONL),
        help="Output JSONL with features per candidate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N candidates (for quick tests)",
    )
    args = parser.parse_args()

    out_path = process_candidates(
        candidates_path=args.candidates,
        jd_path=args.jd_json,
        out_path=args.out,
        limit=args.limit,
    )

    # Print a quick preview of the first record.
    with open(out_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())

    preview = {
        "candidate_id": first["candidate_id"],
        "canonical_text_chars": len(first["canonical_text"]),
        "role_fit": first["role_fit"],
        "skills": first["skills"],
        "career_coherence": first["career_coherence"],
        "authenticity": first["authenticity"],
        "hireability": {
            k: first["hireability"][k]
            for k in (
                "experience_fit",
                "location_fit",
                "open_to_work",
                "recruiter_response_rate",
                "notice_fit",
                "recency_fit",
            )
        },
        "behavioral": first["behavioral"],
        "output": str(out_path),
    }
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
