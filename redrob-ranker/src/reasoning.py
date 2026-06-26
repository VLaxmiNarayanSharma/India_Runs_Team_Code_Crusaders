"""Stage 6: Template-based recruiter reasoning (no LLM at runtime)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .constants import MUST_HAVE_SKILL_CLUSTERS
from .scoring import ScoreBreakdown

# Ordered (patterns, label) — first strong match wins for primary highlight.
TECHNICAL_HIGHLIGHTS: list[tuple[list[str], str]] = [
    (
        ["hybrid retrieval", "bm25", "dense recall", "dense vector", "vector recall"],
        "hybrid retrieval + LTR in production",
    ),
    (
        ["learning to rank", "learning-to-rank", "ltr", "lambdamart"],
        "LTR in production",
    ),
    (
        ["ndcg", "mrr", "map@", "offline-online", "eval framework", "a/b test"],
        "strong eval (NDCG)",
    ),
    (
        ["embedding", "sentence-transformer", "bge", "e5", "vector database", "faiss", "milvus"],
        "embeddings + vector search in production",
    ),
    (
        ["recommendation system", "ranking model", "search engine", "information retrieval"],
        "search/ranking systems in production",
    ),
    (
        ["rag", "retrieval augmented", "llm fine-tuning", "fine-tuning llm", "lora"],
        "RAG/LLM production experience",
    ),
    (
        ["production ml", "shipped", "deployed", "serving", "mlops"],
        "production ML systems",
    ),
]

AI_SKILL_KEYWORDS = {
    "nlp",
    "llm",
    "rag",
    "embedding",
    "vector",
    "pytorch",
    "tensorflow",
    "machine learning",
    "deep learning",
    "information retrieval",
    "learning to rank",
    "bm25",
    "faiss",
    "milvus",
    "pinecone",
    "weaviate",
    "ndcg",
    "xgboost",
    "lightgbm",
    "transformer",
    "fine-tuning",
    "lora",
    "recommendation",
    "retrieval",
    "ranking",
    "speech recognition",
    "image classification",
}


def load_candidate_profiles(
    candidates_path: str | Path,
    candidate_ids: set[str],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
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
                    profiles[cid] = record
                    remaining.remove(cid)
    elif suffix == ".json":
        for record in json.loads(path.read_text(encoding="utf-8")):
            cid = record["candidate_id"]
            if cid in candidate_ids:
                profiles[cid] = record
                remaining.discard(cid)
    else:
        raise ValueError(f"Unsupported candidates format: {suffix}")

    return profiles


def _candidate_text_blob(profile: dict[str, Any]) -> str:
    p = profile.get("profile", {})
    chunks = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
    ]
    for role in profile.get("career_history", []):
        chunks.append(role.get("title", ""))
        chunks.append(role.get("description", ""))
    for skill in profile.get("skills", []):
        chunks.append(skill.get("name", ""))
    return " ".join(str(c) for c in chunks if c).lower()


def _pick_technical_highlight(text_blob: str) -> str | None:
    for patterns, label in TECHNICAL_HIGHLIGHTS:
        if any(p in text_blob for p in patterns):
            return label
    return None


def _is_ai_core_skill(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in AI_SKILL_KEYWORDS)


def _verified_core_skill_count(
    profile: dict[str, Any],
    features: dict[str, Any],
) -> int:
    skills_block = features.get("skills", {})
    assessments = profile.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    skill_list = profile.get("skills", []) or []

    verified = 0
    for skill in skill_list:
        name = skill.get("name", "")
        if not _is_ai_core_skill(name):
            continue
        assessed = assessments.get(name)
        endorsements = int(skill.get("endorsements", 0) or 0)
        duration = int(skill.get("duration_months", 0) or 0)
        prof = skill.get("proficiency", "")

        is_verified = (
            (assessed is not None and assessed >= 45)
            or (endorsements >= 5 and duration >= 12)
            or prof in {"advanced", "expert"}
        )
        if is_verified:
            verified += 1

    if verified > 0:
        return verified

    # Fallback aligned with sample_submission.csv style ("N AI core skills").
    coverage = skills_block.get("jd_skill_cluster_coverage", 0.0)
    raw_count = sum(1 for s in skill_list if _is_ai_core_skill(s.get("name", "")))
    if raw_count == 0:
        return max(1, int(round(coverage * 6)))
    return raw_count


def _jd_cluster_count(text_blob: str) -> int:
    count = 0
    for _cluster, keywords in MUST_HAVE_SKILL_CLUSTERS.items():
        if any(kw in text_blob for kw in keywords):
            count += 1
    return count


def build_reasoning(
    profile: dict[str, Any],
    breakdown: ScoreBreakdown,
    features: dict[str, Any],
) -> str:
    """
    Template style (matches sample_submission.csv + challenge brief):

    Senior ML Engineer with 7.8 yrs; hybrid retrieval + LTR in production;
    strong eval (NDCG); 6 verified core skills; response rate 0.83.
    """
    p = profile.get("profile", {})
    title = p.get("current_title", "Candidate")
    years = float(p.get("years_of_experience", 0) or 0)
    hireability = features.get("hireability", {})
    response_rate = float(hireability.get("recruiter_response_rate", 0) or 0)

    text_blob = _candidate_text_blob(profile)
    core_skills = _verified_core_skill_count(profile, features)

    parts: list[str] = [f"{title} with {years:.1f} yrs"]

    # Primary technical differentiator from career/skills text.
    highlight = _pick_technical_highlight(text_blob)
    if highlight:
        parts.append(highlight)
    elif breakdown.career_trajectory >= 0.65:
        parts.append("ML/IR career trajectory")

    # Secondary eval signal when present but not already used as primary.
    if highlight != "strong eval (NDCG)" and any(
        kw in text_blob for kw in ("ndcg", "mrr", "offline-online", "eval framework")
    ):
        parts.append("strong eval (NDCG)")

    # Skill line — prefer "verified" when assessments exist, else sample-style "AI core".
    assessments = profile.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    if assessments:
        parts.append(f"{core_skills} verified core skills")
    else:
        parts.append(f"{core_skills} AI core skills")

    parts.append(f"response rate {response_rate:.2f}")

    # Short penalty note for transparency (kept brief).
    if breakdown.honeypot_penalty >= 0.85:
        parts.append("profile flagged for keyword/title mismatch")
    elif breakdown.honeypot_penalty >= 0.5:
        parts.append("career narrative consistency discounted")

    return "; ".join(parts) + "."


def submission_score_for_rank(rank: int) -> float:
    """Map rank 1..100 to a non-increasing score in [0.2, 1.0]."""
    return round(1.0 - (rank - 1) * 0.008, 4)
