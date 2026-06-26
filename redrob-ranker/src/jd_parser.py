"""Stage 1: Parse a job description into structured requirements (rule-based, offline)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    MUST_HAVE_SKILL_CLUSTERS,
    ROLE_FAMILY_PATTERNS,
    SENIORITY_KEYWORDS,
    SOFT_SIGNAL_PATTERNS,
)


@dataclass
class ExperienceBand:
    min_years: float | None = None
    max_years: float | None = None
    raw_phrases: list[str] = field(default_factory=list)


@dataclass
class LocationConstraints:
    cities: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    willing_to_relocate_required: bool | None = None
    raw_phrases: list[str] = field(default_factory=list)


@dataclass
class WorkModeConstraints:
    modes: list[str] = field(default_factory=list)
    raw_phrases: list[str] = field(default_factory=list)


@dataclass
class ParsedJobDescription:
    source_path: str
    raw_text: str
    role_title: str | None = None
    role_family: str | None = None
    seniority: str | None = None
    must_haves: dict[str, list[str]] = field(default_factory=dict)
    nice_to_haves: dict[str, list[str]] = field(default_factory=dict)
    experience_band: ExperienceBand = field(default_factory=ExperienceBand)
    location: LocationConstraints = field(default_factory=LocationConstraints)
    work_mode: WorkModeConstraints = field(default_factory=WorkModeConstraints)
    soft_signals: dict[str, list[str]] = field(default_factory=dict)
    disqualifiers: dict[str, list[str]] = field(default_factory=dict)
    ideal_profile: dict[str, list[str]] = field(default_factory=dict)
    all_required_skills: list[str] = field(default_factory=list)
    canonical_jd_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jd_text(path: str | Path) -> str:
    """Load plain text from .txt, .md, or .docx."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job description not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8").strip()

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "Install python-docx to parse .docx files: pip install python-docx"
            ) from exc
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported job description format: {suffix}")


def _normalize(text: str, preserve_lines: bool = True) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2192", "->")
    if preserve_lines:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)
    return re.sub(r"\s+", " ", text).strip()


def _first_matching_line(text: str, patterns: list[str]) -> str | None:
    for line in text.splitlines():
        lower = line.lower().strip()
        if not lower:
            continue
        for pattern in patterns:
            if pattern in lower:
                return line.strip()
    return None


def _extract_role_title(text: str) -> str | None:
    match = re.search(
        r"job description:\s*(.+?)(?:\s+company:|\s+location:|\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" -")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    first = lines[0]
    if len(first) < 120 and not first.lower().startswith(("about", "what you", "must-have")):
        return first

    for line in lines[:8]:
        lower = line.lower()
        if any(kw in lower for kw in ("engineer", "scientist", "manager", "analyst", "designer")):
            return line
    return first


def _detect_role_family(text: str) -> str | None:
    lower = text.lower()
    scores: dict[str, int] = {}
    for family, patterns in ROLE_FAMILY_PATTERNS.items():
        score = sum(1 for p in patterns if p in lower)
        if score:
            scores[family] = score

    if not scores:
        return None

    # Prefer search/retrieval when tied with generic ML roles.
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    top_score = ranked[0][1]
    top_families = [f for f, s in ranked if s == top_score]
    priority = [
        "search_retrieval_ranking",
        "ml_engineer",
        "ai_engineer",
        "nlp_engineer",
        "data_scientist",
    ]
    for family in priority:
        if family in top_families:
            return family
    return ranked[0][0]


def _detect_seniority(text: str) -> str | None:
    lower = text.lower()
    for level in ("lead", "senior", "mid", "junior"):
        if any(kw in lower for kw in SENIORITY_KEYWORDS[level]):
            return level
    return None


def _split_sections(text: str) -> dict[str, str]:
    """Split JD into must-have / nice-to-have / logistics sections."""
    section_headers = {
        "must_have": [
            r"things you absolutely need",
            r"must[- ]have",
            r"required qualifications",
            r"requirements",
            r"qualifications",
        ],
        "nice_to_have": [
            r"things we.?d like you to have",
            r"nice[- ]to[- ]have",
            r"preferred",
            r"bonus",
        ],
        "disqualifiers": [
            r"things we explicitly do not want",
            r"disqualifiers we actually apply",
            r"disqualifiers",
        ],
        "logistics": [
            r"on location, comp, and logistics",
            r"logistics",
            r"notice period",
        ],
        "soft": [
            r"the vibe check",
            r"soft skills",
            r"who you are",
            r"what we value",
        ],
        "ideal": [r"how to read between the lines", r"ideal candidate"],
    }

    lines = text.splitlines()
    current = "body"
    buckets: dict[str, list[str]] = {
        "body": [],
        "must_have": [],
        "nice_to_have": [],
        "disqualifiers": [],
        "logistics": [],
        "soft": [],
        "ideal": [],
    }

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        matched = False
        for section, patterns in section_headers.items():
            if any(re.search(p, lower) for p in patterns):
                current = section
                matched = True
                break
        if not matched and stripped:
            buckets[current].append(stripped)

    return {k: "\n".join(v) for k, v in buckets.items()}


def _match_clusters(text: str, clusters: dict[str, list[str]]) -> dict[str, list[str]]:
    lower = text.lower()
    hits: dict[str, list[str]] = {}
    for cluster, keywords in clusters.items():
        matched = [kw for kw in keywords if kw in lower]
        if matched:
            hits[cluster] = matched
    return hits


def _extract_experience_band(text: str) -> ExperienceBand:
    lower = text.lower()
    phrases: list[str] = []
    min_years: float | None = None
    max_years: float | None = None

    priority_patterns = [
        r"experience required:\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*years?",
        r"what we mean by\s*[\"']?(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*years?",
        r"ideal candidate[^.]{0,120}?(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*years?",
    ]
    for pattern in priority_patterns:
        match = re.search(pattern, lower)
        if match:
            min_years = float(match.group(1))
            max_years = float(match.group(2))
            phrases.append(match.group(0))
            break

    if min_years is None:
        range_patterns = [
            r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:\+?\s*)?years?",
            r"(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*years?",
            r"(\d+(?:\.\d+)?)\s*\+\s*years?",
            r"at least\s*(\d+(?:\.\d+)?)\s*years?",
            r"minimum\s*of\s*(\d+(?:\.\d+)?)\s*years?",
        ]
        for pattern in range_patterns:
            for match in re.finditer(pattern, lower):
                groups = match.groups()
                phrase = match.group(0)
                phrases.append(phrase)
                if len(groups) == 2 and groups[1] is not None:
                    min_years = float(groups[0])
                    max_years = float(groups[1])
                elif len(groups) == 1:
                    val = float(groups[0])
                    min_years = val if min_years is None else min(min_years, val)

    return ExperienceBand(
        min_years=min_years,
        max_years=max_years,
        raw_phrases=sorted(set(phrases)),
    )


def _extract_location(text: str) -> LocationConstraints:
    lower = text.lower()
    cities: list[str] = []
    countries: list[str] = []
    phrases: list[str] = []

    city_patterns = [
        r"\b(bangalore|bengaluru|mumbai|delhi|hyderabad|pune|chennai|gurgaon|gurugram|noida)\b",
    ]
    for pattern in city_patterns:
        for match in re.finditer(pattern, lower):
            city = match.group(1).title()
            if city.lower() == "bengaluru":
                city = "Bangalore"
            cities.append(city)
            phrases.append(match.group(0))

    if "india" in lower:
        countries.append("India")
        phrases.append("india")

    relocate_required = None
    if re.search(r"willing to relocate|relocate preferred|relocation", lower):
        relocate_required = True
    if re.search(r"remote only|fully remote", lower):
        relocate_required = False

    return LocationConstraints(
        cities=sorted(set(cities)),
        countries=sorted(set(countries)),
        willing_to_relocate_required=relocate_required,
        raw_phrases=sorted(set(phrases)),
    )


def _extract_work_mode(text: str) -> WorkModeConstraints:
    lower = text.lower()
    modes: list[str] = []
    phrases: list[str] = []

    mode_map = {
        "remote": [r"\bremote\b", r"work from home", r"\bwfh\b"],
        "hybrid": [r"\bhybrid\b"],
        "onsite": [r"\bonsite\b", r"in[- ]office", r"on[- ]site"],
        "flexible": [r"\bflexible\b"],
    }

    for mode, patterns in mode_map.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                modes.append(mode)
                phrases.append(mode)
                break

    return WorkModeConstraints(modes=sorted(set(modes)), raw_phrases=sorted(set(phrases)))


def _flatten_skill_hits(hits: dict[str, list[str]]) -> list[str]:
    skills: list[str] = []
    for cluster, keywords in hits.items():
        skills.append(cluster)
        skills.extend(keywords)
    return sorted(set(skills))


def _build_canonical_jd_text(parsed: ParsedJobDescription) -> str:
    parts = [
        parsed.role_title or "",
        f"Role family: {parsed.role_family or 'unknown'}",
        f"Seniority: {parsed.seniority or 'unknown'}",
    ]
    if parsed.must_haves:
        parts.append("Must-haves: " + ", ".join(parsed.must_haves.keys()))
    if parsed.nice_to_haves:
        parts.append("Nice-to-haves: " + ", ".join(parsed.nice_to_haves.keys()))
    if parsed.experience_band.min_years is not None:
        band = f"{parsed.experience_band.min_years}"
        if parsed.experience_band.max_years is not None:
            band += f"-{parsed.experience_band.max_years}"
        parts.append(f"Experience: {band} years")
    if parsed.location.cities:
        parts.append("Location: " + ", ".join(parsed.location.cities))
    if parsed.work_mode.modes:
        parts.append("Work mode: " + ", ".join(parsed.work_mode.modes))
    if parsed.soft_signals:
        parts.append("Soft signals: " + ", ".join(parsed.soft_signals.keys()))
    parts.append(parsed.raw_text[:4000])
    return "\n".join(p for p in parts if p)


def parse_job_description(path: str | Path) -> ParsedJobDescription:
    """Parse a job description file into structured requirements."""
    path = Path(path)
    raw_text = _normalize(load_jd_text(path))
    sections = _split_sections(raw_text)

    must_text = sections["must_have"] or sections["body"]
    nice_text = sections["nice_to_have"]
    logistics_text = sections["logistics"] + "\n" + raw_text
    soft_text = sections["soft"] + "\n" + raw_text
    ideal_text = sections["ideal"]

    parsed = ParsedJobDescription(
        source_path=str(path.resolve()),
        raw_text=raw_text,
        role_title=_extract_role_title(raw_text),
        role_family=_detect_role_family(raw_text),
        seniority=_detect_seniority(raw_text),
        must_haves=_match_clusters(must_text, MUST_HAVE_SKILL_CLUSTERS),
        nice_to_haves=_match_clusters(nice_text, MUST_HAVE_SKILL_CLUSTERS),
        experience_band=_extract_experience_band(raw_text),
        location=_extract_location(logistics_text),
        work_mode=_extract_work_mode(logistics_text),
        soft_signals=_match_clusters(soft_text, SOFT_SIGNAL_PATTERNS),
        disqualifiers=_match_clusters(
            sections["disqualifiers"],
            {
                "consulting_only": ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"],
                "keyword_trap": ["langchain", "framework enthusiast", "hot framework"],
                "title_chaser": ["title-chaser", "switching companies every 1.5 years"],
                "non_production_research": ["pure research", "research-only", "academic labs"],
            },
        ),
        ideal_profile=_match_clusters(
            ideal_text,
            {
                "product_company_ml": ["product companies", "applied ml", "applied ml/ai"],
                "shipped_ranking_system": ["ranking", "search", "recommendation system"],
                "location_noida_pune": ["noida", "pune", "relocate"],
            },
        ),
    )

    parsed.all_required_skills = _flatten_skill_hits(parsed.must_haves)
    parsed.canonical_jd_text = _build_canonical_jd_text(parsed)
    return parsed


def save_parsed_jd(parsed: ParsedJobDescription, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parsed.to_dict(), indent=2), encoding="utf-8")
    return out_path


def load_parsed_jd(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
