"""
demo.py
-------
Command-line demo of both halves of the system:
  A. Job recommendation  -> best jobs for ONE resume
  B. Resume screening    -> best candidates for ONE job

Run:  python demo.py
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from resume_parser import parse_resume  # noqa: E402
from recommender import recommend_jobs, screen_resumes  # noqa: E402

BASE = os.path.dirname(__file__)


def load_resumes() -> dict:
    folder = os.path.join(BASE, "data", "sample_resumes")
    resumes = {}
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        name = os.path.splitext(fname)[0].replace("_", " ").title()
        resumes[name] = parse_resume(path)
    return resumes


def main():
    jobs_df = pd.read_csv(os.path.join(BASE, "data", "sample_jobs.csv"))
    resumes = load_resumes()

    print("=" * 70)
    print("A. JOB RECOMMENDATION — best-fit jobs for Aisha Khan's resume")
    print("=" * 70)
    top_jobs = recommend_jobs(resumes["Aisha Khan"], jobs_df, top_n=3)
    print(top_jobs[["job_title", "match_score", "matched_skills"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("B. RESUME SCREENING — ranking all candidates for 'Data Scientist'")
    print("=" * 70)
    ds_job_row = jobs_df[jobs_df["title"] == "Data Scientist"].iloc[0]
    job = {
        "title": ds_job_row["title"],
        "description": ds_job_row["description"],
        "required_skills": ds_job_row["required_skills"].split(","),
        "min_years": ds_job_row["min_years"],
    }
    ranked_candidates = screen_resumes(resumes, job, threshold=60.0)
    print(ranked_candidates.to_string(index=False))


if __name__ == "__main__":
    main()
