"""Stage 2: Candidate canonical text and structured feature extraction."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    ML_IR_TITLE_KEYWORDS,
    MUST_HAVE_SKILL_CLUSTERS,
    NON_TECHNICAL_TITLE_KEYWORDS,
    PROFICIENCY_WEIGHT,
    ROLE_FAMILY_PATTERNS,
    TECHNICAL_DESCRIPTION_KEYWORDS,
)


def _safe_lower(value: str | None) -> str:
    return (value or "").lower().strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#.]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _months_since(value: str | None, today: date | None = None) -> float | None:
    dt = _parse_date(value)
    if dt is None:
        return None
    today = today or date.today()
    return max(0.0, (today - dt).days / 30.44)


def _detect_role_family_from_text(text: str) -> str | None:
    lower = text.lower()
    scores: dict[str, int] = {}
    for family, patterns in ROLE_FAMILY_PATTERNS.items():
        score = sum(1 for p in patterns if p in lower)
        if score:
            scores[family] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def _is_ml_ir_title(title: str) -> bool:
    lower = title.lower()
    return any(kw in lower for kw in ML_IR_TITLE_KEYWORDS)


def _is_non_technical_title(title: str) -> bool:
    lower = title.lower()
    return any(kw in lower for kw in NON_TECHNICAL_TITLE_KEYWORDS)


def _technical_keyword_density(text: str) -> float:
    lower = text.lower()
    if not lower:
        return 0.0
    hits = sum(1 for kw in TECHNICAL_DESCRIPTION_KEYWORDS if kw in lower)
    return min(1.0, hits / 8.0)


def build_canonical_text(candidate: dict[str, Any], top_n_skills: int = 12) -> str:
    """Build a single searchable text blob for retrieval/embedding."""
    profile = candidate.get("profile", {})
    parts: list[str] = []

    for key in ("headline", "summary", "current_title"):
        val = profile.get(key)
        if val:
            parts.append(str(val))

    for role in candidate.get("career_history", []):
        desc = role.get("description")
        if desc:
            parts.append(str(desc))

    skills = candidate.get("skills", [])
    ranked_skills = sorted(
        skills,
        key=lambda s: (
            PROFICIENCY_WEIGHT.get(s.get("proficiency", ""), 0.0)
            * math.log1p(s.get("endorsements", 0))
            * math.log1p(max(1, s.get("duration_months", 0)))
        ),
        reverse=True,
    )
    skill_names = [s.get("name", "") for s in ranked_skills[:top_n_skills] if s.get("name")]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    return "\n".join(parts).strip()


def _skill_depth_score(skill: dict[str, Any]) -> float:
    proficiency = PROFICIENCY_WEIGHT.get(skill.get("proficiency", ""), 0.0)
    endorsements = max(0, int(skill.get("endorsements", 0)))
    duration = max(0, int(skill.get("duration_months", 0)))
    return proficiency * math.log1p(endorsements) * math.log1p(max(1, duration))


def _match_jd_skill_clusters(text: str) -> dict[str, float]:
    lower = text.lower()
    scores: dict[str, float] = {}
    for cluster, keywords in MUST_HAVE_SKILL_CLUSTERS.items():
        matched = sum(1 for kw in keywords if kw in lower)
        if matched:
            scores[cluster] = min(1.0, matched / max(2, len(keywords) * 0.35))
    return scores


def _title_similarity_to_jd(title: str, jd: dict[str, Any]) -> float:
    jd_title = jd.get("role_title") or ""
    jd_family = jd.get("role_family") or ""
    title_lower = title.lower()

    title_sim = _sequence_similarity(title, jd_title)
    family_patterns = ROLE_FAMILY_PATTERNS.get(jd_family, [])
    family_hit = 1.0 if any(p in title_lower for p in family_patterns) else 0.0
    ml_ir_hit = 1.0 if _is_ml_ir_title(title) else 0.0

    return min(1.0, 0.45 * title_sim + 0.35 * family_hit + 0.20 * ml_ir_hit)


def _career_ml_ir_fraction(candidate: dict[str, Any]) -> float:
    history = candidate.get("career_history", [])
    if not history:
        return 0.0

    weighted_months = 0.0
    total_months = 0.0
    for role in history:
        months = float(role.get("duration_months", 0) or 0)
        if months <= 0:
            continue
        total_months += months
        title = role.get("title", "")
        desc = role.get("description", "")
        role_text = f"{title} {desc}"
        if _is_ml_ir_title(title) or _technical_keyword_density(desc) >= 0.35:
            weighted_months += months

    if total_months <= 0:
        return 0.0
    return weighted_months / total_months


def _skill_features(candidate: dict[str, Any], jd: dict[str, Any]) -> dict[str, float]:
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {}) or {}

    if not skills:
        return {
            "skill_depth_total": 0.0,
            "skill_depth_top5_avg": 0.0,
            "skill_count": 0.0,
            "skill_verification_avg": 0.0,
            "skill_verification_gap": 0.0,
            "jd_skill_cluster_coverage": 0.0,
            "keyword_stuffing_ratio": 0.0,
        }

    depths = [_skill_depth_score(s) for s in skills]
    depths_sorted = sorted(depths, reverse=True)
    top5_avg = sum(depths_sorted[:5]) / min(5, len(depths_sorted))

    verification_scores: list[float] = []
    gaps: list[float] = []
    for skill in skills:
        name = skill.get("name", "")
        claimed = PROFICIENCY_WEIGHT.get(skill.get("proficiency", ""), 0.0)
        assessed = assessments.get(name)
        if assessed is None:
            continue
        norm_assessed = assessed / 100.0
        verification_scores.append(norm_assessed)
        gaps.append(max(0.0, claimed - norm_assessed))

    canonical = build_canonical_text(candidate)
    cluster_hits = _match_jd_skill_clusters(canonical)
    coverage = len(cluster_hits) / max(1, len(jd.get("must_haves", {})))

    profile = candidate.get("profile", {})
    non_technical = _is_non_technical_title(profile.get("current_title", ""))
    shallow_skill_count = sum(
        1
        for s in skills
        if PROFICIENCY_WEIGHT.get(s.get("proficiency", ""), 0.0) >= 0.75
        and int(s.get("duration_months", 0) or 0) < 6
        and int(s.get("endorsements", 0) or 0) < 3
    )
    stuffing_ratio = 0.0
    if non_technical and len(skills) >= 8:
        stuffing_ratio = min(1.0, shallow_skill_count / max(1, len(skills)))

    return {
        "skill_depth_total": sum(depths),
        "skill_depth_top5_avg": top5_avg,
        "skill_count": float(len(skills)),
        "skill_verification_avg": (
            sum(verification_scores) / len(verification_scores) if verification_scores else 0.0
        ),
        "skill_verification_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "jd_skill_cluster_coverage": min(1.0, coverage),
        "keyword_stuffing_ratio": stuffing_ratio,
    }


def _career_coherence_features(candidate: dict[str, Any]) -> dict[str, float]:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    current_title = profile.get("current_title", "")

    if not history:
        return {
            "title_history_alignment": 0.0,
            "description_uniqueness": 0.0,
            "industry_consistency": 0.0,
            "title_progression_slope": 0.0,
            "career_ml_ir_fraction": 0.0,
        }

    descriptions = [r.get("description", "") for r in history if r.get("description")]
    uniqueness = 1.0
    if len(descriptions) > 1:
        sims = []
        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                sims.append(_sequence_similarity(descriptions[i], descriptions[j]))
        avg_sim = sum(sims) / len(sims)
        uniqueness = max(0.0, 1.0 - avg_sim)

    industries = [r.get("industry", "") for r in history if r.get("industry")]
    industry_consistency = 0.0
    if industries:
        most_common = Counter(industries).most_common(1)[0][1]
        industry_consistency = most_common / len(industries)

    titles = [r.get("title", "") for r in history]
    title_alignment_scores = [_jaccard(current_title, t) for t in titles]
    title_history_alignment = max(title_alignment_scores) if title_alignment_scores else 0.0

    # Penalize when current title theme mismatches dominant historical descriptions.
    dominant_desc = max(descriptions, key=len) if descriptions else ""
    theme_mismatch = 0.0
    if current_title and dominant_desc:
        title_technical = _is_ml_ir_title(current_title) or not _is_non_technical_title(current_title)
        desc_technical = _technical_keyword_density(dominant_desc) >= 0.35
        if title_technical != desc_technical:
            theme_mismatch = 1.0
    title_history_alignment = max(0.0, title_history_alignment - 0.5 * theme_mismatch)

    seniority_weights = {
        "intern": 0,
        "junior": 1,
        "associate": 1,
        "engineer": 2,
        "scientist": 2,
        "senior": 3,
        "lead": 4,
        "staff": 5,
        "principal": 6,
        "manager": 3,
    }

    def seniority_score(title: str) -> float:
        lower = title.lower()
        score = 1.0
        for key, weight in seniority_weights.items():
            if key in lower:
                score = max(score, float(weight))
        return score

    # History is usually most-recent first; reverse for progression slope.
    chronological = list(reversed(history))
    seniority_series = [seniority_score(r.get("title", "")) for r in chronological]
    progression_slope = 0.0
    if len(seniority_series) > 1:
        progression_slope = (seniority_series[-1] - seniority_series[0]) / (len(seniority_series) - 1)
        progression_slope = max(-1.0, min(1.0, progression_slope / 3.0))

    return {
        "title_history_alignment": title_history_alignment,
        "description_uniqueness": uniqueness,
        "industry_consistency": industry_consistency,
        "title_progression_slope": progression_slope,
        "career_ml_ir_fraction": _career_ml_ir_fraction(candidate),
    }


def _authenticity_features(candidate: dict[str, Any]) -> dict[str, float]:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    current_title = profile.get("current_title", "")
    summary = profile.get("summary", "")

    descriptions = [r.get("description", "") for r in history if r.get("description")]
    copied_bio_penalty = 0.0
    if len(descriptions) > 1:
        max_sim = 0.0
        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                max_sim = max(max_sim, _sequence_similarity(descriptions[i], descriptions[j]))
        if max_sim > 0.72:
            copied_bio_penalty = min(1.0, (max_sim - 0.72) / 0.28)

    title_skill_mismatch = 0.0
    skills = candidate.get("skills", [])
    if _is_non_technical_title(current_title) and len(skills) >= 8:
        advanced_ai_skills = sum(
            1
            for s in skills
            if any(
                kw in _safe_lower(s.get("name"))
                for kw in ("llm", "nlp", "embedding", "pytorch", "tensorflow", "rag", "vector")
            )
        )
        if advanced_ai_skills >= 5:
            title_skill_mismatch = min(1.0, advanced_ai_skills / len(skills))

    summary_title_gap = 0.0
    if summary and current_title:
        summary_technical = _technical_keyword_density(summary)
        title_technical = 0.0 if _is_non_technical_title(current_title) else 0.6
        if abs(summary_technical - title_technical) > 0.45:
            summary_title_gap = min(1.0, abs(summary_technical - title_technical))

    authenticity_score = max(
        0.0,
        1.0 - (0.45 * copied_bio_penalty + 0.35 * title_skill_mismatch + 0.20 * summary_title_gap),
    )

    return {
        "copied_bio_penalty": copied_bio_penalty,
        "title_skill_mismatch": title_skill_mismatch,
        "summary_title_gap": summary_title_gap,
        "authenticity_score": authenticity_score,
    }


def _hireability_features(candidate: dict[str, Any], jd: dict[str, Any]) -> dict[str, float]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    exp = float(profile.get("years_of_experience", 0) or 0)
    exp_band = jd.get("experience_band", {})
    min_years = exp_band.get("min_years")
    max_years = exp_band.get("max_years")

    experience_fit = 1.0
    if min_years is not None:
        if exp < min_years:
            experience_fit = max(0.0, 1.0 - (min_years - exp) / max(1.0, min_years))
        elif max_years is not None and exp > max_years:
            experience_fit = max(0.6, 1.0 - (exp - max_years) / max(2.0, max_years))

    location = jd.get("location", {})
    jd_cities = {c.lower() for c in location.get("cities", [])}
    jd_countries = {c.lower() for c in location.get("countries", [])}
    cand_location = _safe_lower(profile.get("location"))
    cand_country = _safe_lower(profile.get("country"))

    location_fit = 0.5
    if jd_cities and any(city in cand_location for city in jd_cities):
        location_fit = 1.0
    elif jd_countries and cand_country in jd_countries:
        location_fit = 0.85
    elif signals.get("willing_to_relocate"):
        location_fit = 0.75

    work_modes = set(jd.get("work_mode", {}).get("modes", []))
    preferred_mode = signals.get("preferred_work_mode")
    work_mode_fit = 0.7
    if work_modes and preferred_mode:
        if preferred_mode in work_modes or preferred_mode == "flexible":
            work_mode_fit = 1.0
        elif "remote" in work_modes and preferred_mode == "remote":
            work_mode_fit = 1.0
        else:
            work_mode_fit = 0.4

    notice = signals.get("notice_period_days")
    notice_fit = 1.0
    if notice is not None:
        notice_fit = max(0.0, 1.0 - max(0, int(notice) - 60) / 120.0)

    recency_months = _months_since(signals.get("last_active_date"))
    recency_fit = 0.5
    if recency_months is not None:
        recency_fit = max(0.0, 1.0 - recency_months / 12.0)

    return {
        "experience_years": exp,
        "experience_fit": experience_fit,
        "location_fit": location_fit,
        "work_mode_fit": work_mode_fit,
        "open_to_work": 1.0 if signals.get("open_to_work_flag") else 0.0,
        "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0) or 0),
        "notice_period_days": float(notice if notice is not None else 90),
        "notice_fit": notice_fit,
        "recency_fit": recency_fit,
    }


def _behavioral_features(candidate: dict[str, Any]) -> dict[str, float]:
    signals = candidate.get("redrob_signals", {})
    github = signals.get("github_activity_score", -1)
    github_norm = 0.0 if github is None or github < 0 else min(1.0, float(github) / 100.0)

    return {
        "profile_completeness": float(signals.get("profile_completeness_score", 0) or 0) / 100.0,
        "github_activity": github_norm,
        "saved_by_recruiters_30d": float(signals.get("saved_by_recruiters_30d", 0) or 0),
        "search_appearance_30d": float(signals.get("search_appearance_30d", 0) or 0),
        "interview_completion_rate": float(signals.get("interview_completion_rate", 0) or 0),
        "offer_acceptance_rate": max(0.0, float(signals.get("offer_acceptance_rate", 0) or 0)),
        "endorsements_received": float(signals.get("endorsements_received", 0) or 0),
        "verified_identity": float(
            bool(signals.get("verified_email")) + bool(signals.get("verified_phone"))
        )
        / 2.0,
    }


def extract_candidate_features(
    candidate: dict[str, Any],
    jd: dict[str, Any],
) -> dict[str, Any]:
    """Extract structured features for one candidate against parsed JD."""
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")

    role_fit = {
        "title_similarity_to_jd": _title_similarity_to_jd(current_title, jd),
        "career_ml_ir_fraction": _career_ml_ir_fraction(candidate),
        "current_role_family": _detect_role_family_from_text(
            f"{current_title} {profile.get('headline', '')}"
        ),
        "jd_role_family": jd.get("role_family"),
    }

    features: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id"),
        "canonical_text": build_canonical_text(candidate),
        "role_fit": role_fit,
        "skills": _skill_features(candidate, jd),
        "career_coherence": _career_coherence_features(candidate),
        "authenticity": _authenticity_features(candidate),
        "hireability": _hireability_features(candidate, jd),
        "behavioral": _behavioral_features(candidate),
    }
    return features


def iter_candidates_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_json(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def process_candidates(
    candidates_path: str | Path,
    jd_path: str | Path,
    out_path: str | Path,
    limit: int | None = None,
) -> Path:
    """Run Stage 2 over a JSONL or JSON candidate file."""
    jd = json.loads(Path(jd_path).read_text(encoding="utf-8"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates_path = Path(candidates_path)
    suffix = candidates_path.suffix.lower()

    if suffix == ".jsonl":
        source_iter: Iterable[dict[str, Any]] = iter_candidates_jsonl(candidates_path)
    elif suffix == ".json":
        source_iter = load_candidates_json(candidates_path)
    else:
        raise ValueError(f"Unsupported candidates format: {suffix}")

    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for candidate in source_iter:
            record = extract_candidate_features(candidate, jd)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    return out_path
