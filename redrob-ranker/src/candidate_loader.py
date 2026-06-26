"""Load raw candidate profiles by ID from JSONL/JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_candidates_by_ids(
    candidates_path: str | Path,
    candidate_ids: set[str],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(candidate_ids)
    path = Path(candidates_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not remaining:
                    break
                record = json.loads(line)
                cid = record["candidate_id"]
                if cid in remaining:
                    found[cid] = record
                    remaining.remove(cid)
    elif suffix == ".json":
        for record in json.loads(path.read_text(encoding="utf-8")):
            cid = record["candidate_id"]
            if cid in candidate_ids:
                found[cid] = record
                remaining.discard(cid)
    else:
        raise ValueError(f"Unsupported candidates format: {suffix}")

    if remaining:
        raise KeyError(
            f"Missing raw profiles for {len(remaining)} candidates, e.g. {next(iter(remaining))}"
        )
    return found
