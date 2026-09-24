import pandas as pd
import pytest

from src.data_preprocessor import preprocess_job_dataset
from src.recommender import recommend_jobs
from src.resume_parser import parse_resume


def _jobs():
    return pd.DataFrame(
        {
            "job_id": [1, 2],
            "category": ["INFORMATION-TECHNOLOGY", "SALES"],
            "job_title": ["Data Analyst", "Sales Representative"],
            "job_description": [
                "Analyze data with Python and SQL.",
                "Communicate with customers and manage accounts.",
            ],
            "job_skill_set": ["['Python', 'SQL']", "['Communication']"],
        }
    )


def test_valid_resume_returns_a_recommendation():
    results = recommend_jobs("Python and SQL data analyst", _jobs(), top_n=1)

    assert results.loc[0, "job_id"] == 1
    assert results.loc[0, "match_score"] >= 0


def test_resume_with_missing_information_still_processes():
    results = recommend_jobs("Python", _jobs(), top_n=2)

    assert len(results) == 2
    assert all(status == "unavailable" for status in results["education_status"])


def test_different_job_categories_are_retained():
    results = recommend_jobs("customer communication", _jobs(), top_n=2)

    assert set(results["category"]) == {"INFORMATION-TECHNOLOGY", "SALES"}


def test_invalid_file_is_rejected(tmp_path):
    invalid_file = tmp_path / "resume.csv"
    invalid_file.write_text("not a resume", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_resume(str(invalid_file))


def test_missing_dataset_values_are_cleaned_without_inventing_values():
    jobs = _jobs().copy()
    jobs.loc[0, "job_description"] = None
    jobs.loc[0, "job_skill_set"] = None

    processed = preprocess_job_dataset(jobs)

    assert processed.loc[0, "clean_description"] == ""
    assert processed.loc[0, "normalized_skills"] == set()


def test_no_matching_jobs_returns_empty_results():
    no_jobs = _jobs().iloc[0:0]

    results = recommend_jobs("Python", no_jobs, top_n=5)

    assert results.empty
    assert "job_id" in results.columns