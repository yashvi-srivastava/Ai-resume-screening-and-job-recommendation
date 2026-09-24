"""Rank real Hugging Face job records for an extracted resume."""

from __future__ import annotations

import pandas as pd

try:
    from .matcher import score_resume_against_job
except ImportError:
    from matcher import score_resume_against_job


REQUIRED_JOB_COLUMNS = {
    "job_id",
    "category",
    "job_title",
    "job_description",
    "job_skill_set",
}


def recommend_jobs(
    resume_text: str,
    jobs_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Score and rank jobs using the real job dataset schema."""
    missing = REQUIRED_JOB_COLUMNS.difference(jobs_df.columns)
    if missing:
        raise ValueError(f"Job dataset is missing columns: {sorted(missing)}")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if jobs_df.empty:
        return pd.DataFrame(
            columns=[
                "job_id",
                "job_title",
                "category",
                "match_score",
                "text_similarity",
                "skill_overlap",
                "education_fit",
                "education_status",
                "experience_fit",
                "experience_status",
                "matched_skills",
                "missing_skills",
            ]
        )

    results = [
        score_resume_against_job(resume_text, row.to_dict())
        for _, row in jobs_df.iterrows()
    ]
    return (
        pd.DataFrame(results)
        .sort_values(["match_score", "job_id"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def screen_resumes(
    resumes: dict[str, str],
    job: dict,
    threshold: float = 60.0,
) -> pd.DataFrame:
    """Rank multiple candidate texts against one real job record."""
    rows = []
    for candidate_name, resume_text in resumes.items():
        result = score_resume_against_job(resume_text, job)
        result["candidate"] = candidate_name
        result["shortlisted"] = result["match_score"] >= threshold
        rows.append(result)

    columns = [
        "candidate",
        "job_id",
        "job_title",
        "match_score",
        "shortlisted",
        "matched_skills",
        "missing_skills",
        "education_status",
        "experience_status",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("match_score", ascending=False)[columns].reset_index(drop=True)
