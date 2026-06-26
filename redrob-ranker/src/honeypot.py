"""Stage 5: Explicit anti-honeypot detection and penalties."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .candidate_features import (
    _is_ml_ir_title,
    _is_non_technical_title,
    _jaccard,
    _technical_keyword_density,
)

# Penalty magnitudes from challenge pseudocode.
PENALTY_LARGE = 0.85
PENALTY_MEDIUM = 0.55
SKILL_TRUST_DISCOUNT = 0.5

SKILL_COUNT_HIGH_THRESHOLD = 8
COPIED_BIO_JACCARD_THRESHOLD = 0.70
HIGH_ENDORSEMENT_THRESHOLD = 10
LOW_ASSESSMENT_THRESHOLD = 40.0


@dataclass
class HoneypotAssessment:
    candidate_id: str
    is_honeypot: bool
    flags: list[str] = field(default_factory=list)
    non_technical_keyword_stuffer: float = 0.0
    copied_bio: float = 0.0
    title_theme_mismatch: float = 0.0
    skill_trust_multiplier: float = 1.0
    total_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "is_honeypot": self.is_honeypot,
            "flags": self.flags,
            "non_technical_keyword_stuffer": round(self.non_technical_keyword_stuffer, 4),
            "copied_bio": round(self.copied_bio, 4),
            "title_theme_mismatch": round(self.title_theme_mismatch, 4),
            "skill_trust_multiplier": round(self.skill_trust_multiplier, 4),
            "total_penalty": round(self.total_penalty, 4),
        }


def _dominant_career_theme(history: list[dict[str, Any]]) -> str:
    """Infer whether career descriptions skew technical or non-technical."""
    if not history:
        return "unknown"

    technical_weight = 0.0
    non_technical_weight = 0.0
    for role in history:
        months = float(role.get("duration_months", 0) or 0)
        weight = max(1.0, months)
        title = role.get("title", "")
        desc = role.get("description", "")

        if _is_ml_ir_title(title) or _technical_keyword_density(desc) >= 0.35:
            technical_weight += weight
        elif _is_non_technical_title(title) or _technical_keyword_density(desc) < 0.15:
            non_technical_weight += weight

    if technical_weight > non_technical_weight * 1.2:
        return "technical"
    if non_technical_weight > technical_weight * 1.2:
        return "non_technical"
    return "mixed"


def _max_description_jaccard(descriptions: list[str]) -> float:
    if len(descriptions) < 2:
        return 0.0
    max_sim = 0.0
    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            max_sim = max(max_sim, _jaccard(descriptions[i], descriptions[j]))
    return max_sim


def _endorsement_assessment_gap(skills: list[dict[str, Any]], assessments: dict[str, float]) -> bool:
    gaps = 0
    for skill in skills:
        name = skill.get("name", "")
        endorsements = int(skill.get("endorsements", 0) or 0)
        assessed = assessments.get(name)
        if assessed is None:
            continue
        if endorsements >= HIGH_ENDORSEMENT_THRESHOLD and assessed < LOW_ASSESSMENT_THRESHOLD:
            gaps += 1
    return gaps >= 2


def assess_honeypot(
    candidate: dict[str, Any],
    features: dict[str, Any] | None = None,
) -> HoneypotAssessment:
    """
    Apply explicit honeypot rules from the challenge pseudocode.

    Rules:
    1. High skill count + non-technical career -> large penalty
    2. Copied role descriptions (Jaccard > 0.7) -> medium penalty
    3. Current title != dominant career theme -> medium penalty
    4. High endorsements + low assessments -> skill_trust *= 0.5
    """
    cid = candidate.get("candidate_id", "")
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", []) or []
    history = candidate.get("career_history", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}

    features = features or {}
    career = features.get("career_coherence", {})
    authenticity = features.get("authenticity", {})

    flags: list[str] = []
    non_technical_stuffer = 0.0
    copied_bio = 0.0
    title_mismatch = 0.0
    skill_trust_multiplier = 1.0

    skill_count = len(skills)
    current_title = profile.get("current_title", "")
    career_ml_frac = career.get("career_ml_ir_fraction", 0.0)
    career_not_technical = (
        _is_non_technical_title(current_title)
        or career_ml_frac < 0.25
    )

    # Rule 1: keyword-stuffer trap (Marketing Manager with 9 AI skills).
    if skill_count >= SKILL_COUNT_HIGH_THRESHOLD and career_not_technical:
        non_technical_stuffer = PENALTY_LARGE
        flags.append("non_technical_keyword_stuffer")

    # Rule 2: copied bios across jobs.
    descriptions = [r.get("description", "") for r in history if r.get("description")]
    max_jaccard = _max_description_jaccard(descriptions)
    if max_jaccard > COPIED_BIO_JACCARD_THRESHOLD:
        copied_bio = PENALTY_MEDIUM
        flags.append("copied_bio")

    # Also honor precomputed copied-bio signal from Stage 2 if present.
    if authenticity.get("copied_bio_penalty", 0.0) >= 0.5 and copied_bio == 0.0:
        copied_bio = PENALTY_MEDIUM
        if "copied_bio" not in flags:
            flags.append("copied_bio")

    # Rule 3: title theme mismatch vs dominant career history.
    dominant_theme = _dominant_career_theme(history)
    title_is_technical = _is_ml_ir_title(current_title) or (
        not _is_non_technical_title(current_title)
        and _technical_keyword_density(profile.get("summary", "")) >= 0.25
    )
    if dominant_theme == "technical" and not title_is_technical:
        title_mismatch = PENALTY_MEDIUM
        flags.append("title_theme_mismatch")
    elif dominant_theme == "non_technical" and title_is_technical and skill_count >= 6:
        title_mismatch = PENALTY_MEDIUM
        flags.append("title_theme_mismatch")

    # Rule 4: inflated endorsements vs low platform assessments.
    if _endorsement_assessment_gap(skills, assessments):
        skill_trust_multiplier *= SKILL_TRUST_DISCOUNT
        flags.append("endorsement_assessment_gap")

    total_penalty = min(1.0, non_technical_stuffer + copied_bio + title_mismatch)
    is_honeypot = total_penalty >= PENALTY_MEDIUM or "non_technical_keyword_stuffer" in flags

    return HoneypotAssessment(
        candidate_id=cid,
        is_honeypot=is_honeypot,
        flags=flags,
        non_technical_keyword_stuffer=non_technical_stuffer,
        copied_bio=copied_bio,
        title_theme_mismatch=title_mismatch,
        skill_trust_multiplier=skill_trust_multiplier,
        total_penalty=total_penalty,
    )
