from pathlib import Path

import pandas as pd

from src.data_loader import load_job_dataset
from src.recommender import recommend_jobs


SUPPLEMENTAL_PATH = Path(__file__).parents[1] / "data" / "additional_jobs.csv"


class FakeResponse:
    def __init__(self, rows):
        self.rows = rows
        self.ok = True
        self.status_code = 200
        self.headers = {}

    def json(self):
        return {"rows": self.rows}

    def raise_for_status(self):
        raise AssertionError("The fake source response should be successful")


def test_additional_jobs_are_appended_without_replacing_source_data(monkeypatch):
    original_row = {
        "job_id": 3902668440,
        "category": "HR",
        "job_title": "Existing Hugging Face Job",
        "job_description": "Existing source description.",
        "job_skill_set": "['Communication']",
    }

    def fake_get(*args, **kwargs):
        return FakeResponse([{"row": original_row}])

    monkeypatch.setattr("src.data_loader.requests.get", fake_get)
    jobs = load_job_dataset()

    assert len(jobs) == 18
    assert jobs["job_id"].is_unique
    assert original_row["job_id"] in set(jobs["job_id"])
    assert "Machine Learning Engineer" in set(jobs["job_title"])
    assert "SQL Developer" in set(jobs["job_title"])


def test_all_supplemental_records_have_compatible_fields():
    jobs = pd.read_csv(SUPPLEMENTAL_PATH)

    assert len(jobs) == 17
    assert jobs["job_id"].is_unique
    assert jobs["job_description"].str.len().min() > 150
    assert jobs["job_skill_set"].str.len().min() > 20


def test_additional_job_fields_work_with_recommendation_pipeline():
    jobs = pd.read_csv(SUPPLEMENTAL_PATH)
    selected = jobs[jobs["job_title"] == "Machine Learning Engineer"]
    results = recommend_jobs(
        "Machine Learning Engineer with Python, scikit-learn, Pandas, SQL, Docker, and model deployment experience.",
        selected,
        top_n=1,
    )

    assert results.loc[0, "job_id"] == 4000000001
    assert results.loc[0, "job_title"] == "Machine Learning Engineer"
    assert "python" in results.loc[0, "matched_skills"]
    assert jobs.loc[jobs["job_id"] == 4000000001, "job_description"].iloc[0].startswith("The Machine Learning Engineer")
