import pandas as pd

from src.recommender import recommend_jobs


def test_recommend_jobs_ranks_real_job_records():
    jobs = pd.DataFrame(
        {
            "job_id": [2, 1],
            "category": ["SALES", "INFORMATION-TECHNOLOGY"],
            "job_title": ["Sales Representative", "Data Analyst"],
            "job_description": [
                "Communicate with customers.",
                "Requires Python and SQL for data analysis.",
            ],
            "job_skill_set": ["['Communication']", "['Python', 'SQL']"],
        }
    )

    results = recommend_jobs("Python and SQL data analyst", jobs, top_n=1)

    assert len(results) == 1
    assert results.loc[0, "job_id"] == 1
    assert results.loc[0, "job_title"] == "Data Analyst"
    assert results.loc[0, "matched_skills"] == ["python", "sql"]


def test_recommend_jobs_rejects_old_schema():
    jobs = pd.DataFrame({"title": ["Old schema"]})

    try:
        recommend_jobs("resume", jobs)
    except ValueError as error:
        assert "job_description" in str(error)
    else:
        raise AssertionError("Expected a missing-column error")
