"""Small, reusable preprocessing steps for the two verified datasets."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

import pandas as pd


def clean_text(value: object) -> str:
    """Normalize text while preserving words, numbers, and skill punctuation."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_skill(value: object) -> str:
    """Normalize one skill for case-insensitive comparison."""
    return clean_text(value).casefold()


def parse_skill_list(value: object) -> set[str]:
    """Parse a job skill-list string or a resume's comma-separated skills."""
    if value is None or pd.isna(value):
        return set()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, (list, tuple, set)):
                raise ValueError("Expected job_skill_set to contain a list")
            values: Iterable[object] = parsed
        else:
            values = text.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        raise TypeError("Skills must be a list-like value or comma-separated text")
    return {skill for item in values if (skill := normalize_skill(item))}


def preprocess_resume_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the resume data."""
    cleaned = dataframe.copy()
    cleaned["clean_text"] = cleaned["raw_text"].map(clean_text)
    cleaned["normalized_skills"] = cleaned["skills"].map(parse_skill_list)
    return cleaned


def preprocess_job_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the job data."""
    cleaned = dataframe.copy()
    cleaned["clean_description"] = cleaned["job_description"].map(clean_text)
    cleaned["normalized_skills"] = cleaned["job_skill_set"].map(parse_skill_list)
    return cleaned
