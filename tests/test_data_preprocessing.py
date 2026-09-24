import pandas as pd

from src.data_preprocessor import clean_text, parse_skill_list, preprocess_job_dataset


def test_clean_text_normalizes_whitespace():
    assert clean_text(" Python\n\n  SQL ") == "Python SQL"


def test_parse_skill_list_handles_job_list_text():
    assert parse_skill_list("['Python', 'SQL']") == {"python", "sql"}


def test_parse_skill_list_handles_resume_csv_text():
    assert parse_skill_list("Python, SQL, Python") == {"python", "sql"}


def test_preprocess_job_dataset_keeps_source_columns():
    jobs = pd.DataFrame(
        {
            "job_id": [1],
            "category": ["INFORMATION-TECHNOLOGY"],
            "job_title": ["Data Analyst"],
            "job_description": ["Work with Python."],
            "job_skill_set": ["['Python']"],
        }
    )
    processed = preprocess_job_dataset(jobs)
    assert processed.loc[0, "normalized_skills"] == {"python"}
    assert processed.loc[0, "clean_description"] == "Work with Python."