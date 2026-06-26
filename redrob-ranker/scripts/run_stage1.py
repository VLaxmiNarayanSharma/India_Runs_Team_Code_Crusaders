#!/usr/bin/env python3
"""Run Stage 1: parse job description into jd_requirements.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jd_parser import parse_job_description, save_parsed_jd
from src.paths import JOB_DESCRIPTION_DOCX, JD_REQUIREMENTS_JSON


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 — JD understanding")
    parser.add_argument(
        "--jd",
        default=str(JOB_DESCRIPTION_DOCX),
        help="Path to job description (.txt, .md, or .docx)",
    )
    parser.add_argument(
        "--out",
        default=str(JD_REQUIREMENTS_JSON),
        help="Output JSON path",
    )
    args = parser.parse_args()

    parsed = parse_job_description(args.jd)
    out_path = save_parsed_jd(parsed, args.out)

    summary = {
        "role_title": parsed.role_title,
        "role_family": parsed.role_family,
        "seniority": parsed.seniority,
        "must_have_clusters": list(parsed.must_haves.keys()),
        "nice_to_have_clusters": list(parsed.nice_to_haves.keys()),
        "experience_band": parsed.experience_band,
        "location": parsed.location,
        "work_mode": parsed.work_mode,
        "soft_signals": list(parsed.soft_signals.keys()),
        "disqualifiers": list(parsed.disqualifiers.keys()),
        "ideal_profile": list(parsed.ideal_profile.keys()),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)))


if __name__ == "__main__":
    main()
