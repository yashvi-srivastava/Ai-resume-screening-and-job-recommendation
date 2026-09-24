from src.matcher import score_resume_against_job


def test_matcher_uses_actual_job_dataset_fields():
    job = {
        "job_id": 10,
        "category": "INFORMATION-TECHNOLOGY",
        "job_title": "Data Analyst",
        "job_description": "Requires a Bachelor's degree and 2 years of experience with Python and SQL.",
        "job_skill_set": "['Python', 'SQL', 'Tableau']",
    }

    result = score_resume_against_job(
        "B.Tech graduate with 3 years of experience in Python and SQL.", job
    )

    assert result["job_id"] == 10
    assert result["job_title"] == "Data Analyst"
    assert result["matched_skills"] == ["python", "sql"]
    assert result["missing_skills"] == ["tableau"]
    assert result["education_status"] == "matched"
    assert result["experience_status"] == "matched"
    assert 0 <= result["match_score"] <= 100


def test_missing_job_requirements_are_unavailable():
    job = {
        "job_id": 11,
        "category": "SALES",
        "job_title": "Sales Representative",
        "job_description": "Communicate with customers and manage accounts.",
        "job_skill_set": "['Communication']",
    }

    result = score_resume_against_job("Communication skills.", job)

    assert result["education_status"] == "unavailable"
    assert result["experience_status"] == "unavailable"
    assert result["education_fit"] is None
    assert result["experience_fit"] is None
