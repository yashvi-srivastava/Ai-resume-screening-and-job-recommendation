"""Load the verified resume and job datasets without changing source files."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests


RESUME_COLUMNS = {
    "id",
    "name",
    "years_experience",
    "highest_degree",
    "skills",
    "current_title",
    "has_portfolio",
    "raw_text",
    "label",
}
JOB_COLUMNS = {
    "job_id",
    "category",
    "job_title",
    "job_description",
    "job_skill_set",
}
JOB_DATASET_ID = "batuhanmtl/job-skill-set"
JOB_API_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_SUPPLEMENTAL_JOBS_PATH = Path(__file__).resolve().parent.parent / "data" / "additional_jobs.csv"


def _validate_columns(dataframe: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {sorted(missing)}")


def load_resume_dataset(zip_path: str | Path) -> pd.DataFrame:
    """Read the CSV inside the supplied ZIP archive without extracting it."""
    with zipfile.ZipFile(Path(zip_path)) as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError("Expected exactly one CSV file in the resume dataset ZIP")
        dataframe = pd.read_csv(io.BytesIO(archive.read(csv_files[0])))
    _validate_columns(dataframe, RESUME_COLUMNS, "Resume")
    return dataframe


def load_job_dataset(
    dataset_id: str = JOB_DATASET_ID,
    *,
    timeout_seconds: int = 60,
    supplemental_path: str | Path | None = DEFAULT_SUPPLEMENTAL_JOBS_PATH,
    retries: int = 3,
) -> pd.DataFrame:
    """Load Hugging Face jobs and optionally append local compatible records."""
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    while True:
        params = {
            "dataset": dataset_id,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": page_size,
        }
        for attempt in range(retries):
            response = requests.get(JOB_API_URL, params=params, timeout=timeout_seconds)
            if response.ok:
                break
            if attempt == retries - 1 or response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(delay)
        page = response.json().get("rows", [])
        rows.extend(item["row"] for item in page)
        if len(page) < page_size:
            break
        offset += page_size
    dataframe = pd.DataFrame(rows)
    _validate_columns(dataframe, JOB_COLUMNS, "Job")
    if supplemental_path is None:
        return dataframe

    supplemental = pd.read_csv(Path(supplemental_path))
    _validate_columns(supplemental, JOB_COLUMNS, "Supplemental job")
    if supplemental["job_id"].duplicated().any():
        raise ValueError("Supplemental job dataset contains duplicate job IDs")
    if set(dataframe["job_id"]).intersection(supplemental["job_id"]):
        raise ValueError("Supplemental job IDs overlap with Hugging Face job IDs")
    return pd.concat([dataframe, supplemental], ignore_index=True)
