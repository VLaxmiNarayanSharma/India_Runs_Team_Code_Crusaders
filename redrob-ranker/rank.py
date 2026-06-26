#!/usr/bin/env python3
"""End-to-end ranking entrypoint (Stages 1-4)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script)] + (extra_args or [])
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob ranker pipeline")
    parser.add_argument(
        "--candidates",
        default=str(ROOT / "data" / "precomputed" / "candidate_features_full.jsonl"),
        help="Used for Stage 2 if rebuilding features",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "submission.csv"),
        help="Final submission CSV path",
    )
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    args = parser.parse_args()

    if not args.skip_stage1:
        run("run_stage1.py")
    if not args.skip_stage2:
        run(
            "run_stage2.py",
            [
                "--candidates",
                str(ROOT.parent / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"),
                "--out",
                str(ROOT / "data" / "precomputed" / "candidate_features_full.jsonl"),
            ],
        )
    if not args.skip_stage3:
        run("run_stage3.py")
    run("run_stage5.py", ["--out", args.out])
    run("run_stage6.py", ["--out", args.out])


if __name__ == "__main__":
    main()
