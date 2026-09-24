"""Transparent resume-to-job comparison using the verified dataset fields."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .data_preprocessor import clean_text, parse_skill_list
    from .skill_extractor import extract_profile
except ImportError:
    from data_preprocessor import clean_text, parse_skill_list
    from skill_extractor import extract_profile


# These are initial, interpretable weights. They must be evaluated before being
# treated as final project parameters.
PROVISIONAL_WEIGHTS = {
    "text_similarity": 0.4,
    "skill_overlap": 0.4,
    "education_fit": 0.1,
    "experience_fit": 0.1,
}

_DEGREE_LEVELS = {
    "ph.d": 3,
    "phd": 3,
    "doctorate": 3,
    "master": 2,
    "masters": 2,
    "bachelor": 1,
    "bachelors": 1,
}


def text_similarity(resume_text: str, job_text: str) -> float:
    """Return TF-IDF cosine similarity in the range 0..1."""
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)
    if not resume_text or not job_text:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])


def skill_overlap(resume_skills: set[str], job_skills: set[str]) -> float:
    """Return the fraction of listed job skills found in the resume."""
    if not job_skills:
        return 0.0
    return len(resume_skills.intersection(job_skills)) / len(job_skills)


def _extract_required_years(job_description: str) -> float | None:
    matches = re.findall(
        r"(?:at least|minimum of|minimum|more than)?\s*(\d+)\+?\s*years?",
        job_description.casefold(),
    )
    return max((float(value) for value in matches), default=None)


def _extract_required_degree(job_description: str) -> int | None:
    found = []
    text = job_description.casefold()
    for degree, level in _DEGREE_LEVELS.items():
        if degree in text:
            found.append(level)
    return max(found, default=None)


def _education_result(candidate_level: int, required_level: int | None) -> tuple[float | None, str]:
    if required_level is None:
        return None, "unavailable"
    if candidate_level >= required_level:
        return 1.0, "matched"
    return 0.0, "not matched"


def _experience_result(candidate_years: float, required_years: float | None) -> tuple[float | None, str]:
    if required_years is None:
        return None, "unavailable"
    if candidate_years >= required_years:
        return 1.0, "matched"
    return 0.0, "not matched"


def score_resume_against_job(resume_text: str, job: dict) -> dict:
    """Return a score breakdown for one resume and one real job record."""
    profile = extract_profile(resume_text)
    job_description = clean_text(job.get("job_description", ""))
    job_skills = parse_skill_list(job.get("job_skill_set", []))
    candidate_skills = {skill.casefold() for skill in profile["skills"]}

    text_score = text_similarity(resume_text, job_description)
    skills_score = skill_overlap(candidate_skills, job_skills)
    education_score, education_status = _education_result(
        profile["education_level"], _extract_required_degree(job_description)
    )
    experience_score, experience_status = _experience_result(
        profile["experience_years"], _extract_required_years(job_description)
    )

    optional_values = {
        "education_fit": education_score,
        "experience_fit": experience_score,
    }
    total_weight = PROVISIONAL_WEIGHTS["text_similarity"] + PROVISIONAL_WEIGHTS["skill_overlap"]
    weighted_total = (
        PROVISIONAL_WEIGHTS["text_similarity"] * text_score
        + PROVISIONAL_WEIGHTS["skill_overlap"] * skills_score
    )
    for key, value in optional_values.items():
        if value is not None:
            total_weight += PROVISIONAL_WEIGHTS[key]
            weighted_total += PROVISIONAL_WEIGHTS[key] * value

    return {
        "job_id": job.get("job_id"),
        "job_title": job.get("job_title", ""),
        "category": job.get("category", ""),
        "match_score": round(weighted_total / total_weight * 100, 2),
        "text_similarity": round(text_score * 100, 2),
        "skill_overlap": round(skills_score * 100, 2),
        "education_fit": None if education_score is None else round(education_score * 100, 2),
        "education_status": education_status,
        "experience_fit": None if experience_score is None else round(experience_score * 100, 2),
        "experience_status": experience_status,
        "matched_skills": sorted(candidate_skills.intersection(job_skills)),
        "missing_skills": sorted(job_skills.difference(candidate_skills)),
    }
