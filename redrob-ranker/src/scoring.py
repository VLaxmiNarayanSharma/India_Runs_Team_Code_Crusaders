"""Stage 4: Interpretable hybrid scoring on retrieved candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .honeypot import HoneypotAssessment

# Career trajectory dominates over raw skill count.
DEFAULT_WEIGHTS = {
    "semantic_fit": 0.12,
    "career_trajectory": 0.38,
    "skill_depth_trust": 0.12,
    "behavioral_hireability": 0.18,
    "inconsistency_penalty": 0.10,
    "keyword_stuffing_penalty": 0.15,
    "honeypot_penalty": 0.20,
}


@dataclass
class ScoreBreakdown:
    candidate_id: str
    semantic_fit: float
    career_trajectory: float
    skill_depth_trust: float
    behavioral_hireability: float
    inconsistency_penalty: float
    keyword_stuffing_penalty: float
    honeypot_penalty: float
    honeypot_flags: list[str]
    retrieval_score: float
    total_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "semantic_fit": round(self.semantic_fit, 4),
            "career_trajectory": round(self.career_trajectory, 4),
            "skill_depth_trust": round(self.skill_depth_trust, 4),
            "behavioral_hireability": round(self.behavioral_hireability, 4),
            "inconsistency_penalty": round(self.inconsistency_penalty, 4),
            "keyword_stuffing_penalty": round(self.keyword_stuffing_penalty, 4),
            "honeypot_penalty": round(self.honeypot_penalty, 4),
            "honeypot_flags": self.honeypot_flags,
            "retrieval_score": round(self.retrieval_score, 4),
            "total_score": round(self.total_score, 4),
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_semantic_fit(
    retrieval_score: float,
    role_fit: dict[str, Any],
    skills: dict[str, Any],
) -> float:
    return _clamp(
        0.55 * retrieval_score
        + 0.25 * role_fit.get("title_similarity_to_jd", 0.0)
        + 0.20 * skills.get("jd_skill_cluster_coverage", 0.0)
    )


def compute_career_trajectory(
    role_fit: dict[str, Any],
    career_coherence: dict[str, Any],
) -> float:
    progression = max(0.0, career_coherence.get("title_progression_slope", 0.0))
    return _clamp(
        0.40 * role_fit.get("career_ml_ir_fraction", 0.0)
        + 0.30 * career_coherence.get("title_history_alignment", 0.0)
        + 0.15 * progression
        + 0.15 * role_fit.get("title_similarity_to_jd", 0.0)
    )


def compute_skill_depth_trust(
    skills: dict[str, Any],
    honeypot: HoneypotAssessment | None = None,
) -> float:
    depth = _clamp(skills.get("skill_depth_top5_avg", 0.0) / 12.0)
    verification = skills.get("skill_verification_avg", 0.0)
    coverage = skills.get("jd_skill_cluster_coverage", 0.0)
    gap = skills.get("skill_verification_gap", 0.0)

    shallow_breadth_penalty = 0.0
    skill_count = skills.get("skill_count", 0.0)
    if skill_count >= 10 and depth < 0.45:
        shallow_breadth_penalty = min(0.35, (skill_count - 8) / 25.0)

    base = 0.45 * depth + 0.30 * verification + 0.25 * coverage
    trust = _clamp(base - 0.20 * gap - shallow_breadth_penalty)

    if honeypot is not None:
        trust *= honeypot.skill_trust_multiplier
    return _clamp(trust)


def compute_behavioral_hireability(
    hireability: dict[str, Any],
    behavioral: dict[str, Any],
) -> float:
    return _clamp(
        0.22 * hireability.get("experience_fit", 0.0)
        + 0.14 * hireability.get("location_fit", 0.0)
        + 0.08 * hireability.get("work_mode_fit", 0.0)
        + 0.10 * hireability.get("open_to_work", 0.0)
        + 0.16 * hireability.get("recruiter_response_rate", 0.0)
        + 0.10 * hireability.get("notice_fit", 0.0)
        + 0.10 * hireability.get("recency_fit", 0.0)
        + 0.05 * behavioral.get("profile_completeness", 0.0)
        + 0.05 * behavioral.get("interview_completion_rate", 0.0)
    )


def compute_inconsistency_penalty(
    authenticity: dict[str, Any],
    career_coherence: dict[str, Any],
) -> float:
    return _clamp(
        0.45 * (1.0 - authenticity.get("authenticity_score", 1.0))
        + 0.25 * authenticity.get("copied_bio_penalty", 0.0)
        + 0.20 * authenticity.get("title_skill_mismatch", 0.0)
        + 0.10 * (1.0 - career_coherence.get("description_uniqueness", 1.0))
    )


def compute_keyword_stuffing_penalty(
    skills: dict[str, Any],
    authenticity: dict[str, Any],
) -> float:
    return _clamp(
        max(
            skills.get("keyword_stuffing_ratio", 0.0),
            authenticity.get("title_skill_mismatch", 0.0),
            authenticity.get("summary_title_gap", 0.0) * 0.7,
        )
    )


def score_candidate(
    candidate_id: str,
    features: dict[str, Any],
    retrieval_score: float,
    weights: dict[str, float] | None = None,
    honeypot: HoneypotAssessment | None = None,
) -> ScoreBreakdown:
    weights = weights or DEFAULT_WEIGHTS
    role_fit = features.get("role_fit", {})
    skills = features.get("skills", {})
    career = features.get("career_coherence", {})
    authenticity = features.get("authenticity", {})
    hireability = features.get("hireability", {})
    behavioral = features.get("behavioral", {})

    semantic = compute_semantic_fit(retrieval_score, role_fit, skills)
    career_traj = compute_career_trajectory(role_fit, career)
    skill_trust = compute_skill_depth_trust(skills, honeypot=honeypot)
    behavioral_hire = compute_behavioral_hireability(hireability, behavioral)
    inconsistency = compute_inconsistency_penalty(authenticity, career)
    stuffing = compute_keyword_stuffing_penalty(skills, authenticity)
    honeypot_penalty = honeypot.total_penalty if honeypot else 0.0
    honeypot_flags = list(honeypot.flags) if honeypot else []

    total = (
        weights["semantic_fit"] * semantic
        + weights["career_trajectory"] * career_traj
        + weights["skill_depth_trust"] * skill_trust
        + weights["behavioral_hireability"] * behavioral_hire
        - weights["inconsistency_penalty"] * inconsistency
        - weights["keyword_stuffing_penalty"] * stuffing
        - weights.get("honeypot_penalty", 0.0) * honeypot_penalty
    )

    return ScoreBreakdown(
        candidate_id=candidate_id,
        semantic_fit=semantic,
        career_trajectory=career_traj,
        skill_depth_trust=skill_trust,
        behavioral_hireability=behavioral_hire,
        inconsistency_penalty=inconsistency,
        keyword_stuffing_penalty=stuffing,
        honeypot_penalty=honeypot_penalty,
        honeypot_flags=honeypot_flags,
        retrieval_score=retrieval_score,
        total_score=total,
    )


def load_retrieved_candidates(path: str | Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scores[row["candidate_id"]] = float(row["final_retrieval_score"])
    return scores


def load_features_for_ids(
    features_path: str | Path,
    candidate_ids: set[str],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(candidate_ids)
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            record = json.loads(line)
            cid = record["candidate_id"]
            if cid in remaining:
                found[cid] = record
                remaining.remove(cid)
    if remaining:
        raise KeyError(f"Missing features for {len(remaining)} candidates, e.g. {next(iter(remaining))}")
    return found


def rank_candidates(
    retrieved_path: str | Path,
    features_path: str | Path,
    candidates_path: str | Path | None = None,
    weights: dict[str, float] | None = None,
) -> list[ScoreBreakdown]:
    from .candidate_loader import load_candidates_by_ids
    from .honeypot import assess_honeypot

    retrieval_scores = load_retrieved_candidates(retrieved_path)
    candidate_ids = set(retrieval_scores)
    features = load_features_for_ids(features_path, candidate_ids)

    raw_candidates: dict[str, dict] = {}
    if candidates_path is not None:
        raw_candidates = load_candidates_by_ids(candidates_path, candidate_ids)

    scored: list[ScoreBreakdown] = []
    for cid in retrieval_scores:
        honeypot = None
        if cid in raw_candidates:
            honeypot = assess_honeypot(raw_candidates[cid], features[cid])
        scored.append(
            score_candidate(
                cid,
                features[cid],
                retrieval_scores[cid],
                weights=weights,
                honeypot=honeypot,
            )
        )

    scored.sort(key=lambda x: (-x.total_score, x.candidate_id))
    return scored
