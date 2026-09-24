"""Evaluation helpers that do not invent recommendation ground truth."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def dataset_quality_report(dataframe: pd.DataFrame, id_column: str) -> dict:
    """Summarize missing values, duplicate rows, and duplicate identifiers."""
    comparable_rows = dataframe.map(repr).duplicated()
    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(comparable_rows.sum()),
        "duplicate_ids": int(dataframe[id_column].duplicated().sum()),
    }


def validate_match_results(results: pd.DataFrame) -> dict:
    """Check basic invariants for recommendation results."""
    scores = results["match_score"]
    return {
        "scores_in_range": bool(scores.between(0, 100).all()),
        "sorted_descending": bool(scores.is_monotonic_decreasing),
        "rows": len(results),
    }


def precision_at_k(recommended_ids: Iterable, relevant_ids: set, k: int) -> float:
    """Calculate Precision@K from explicitly reviewed relevant job IDs."""
    if k < 1:
        raise ValueError("k must be at least 1")
    top_ids = list(recommended_ids)[:k]
    if not top_ids:
        return 0.0
    return sum(job_id in relevant_ids for job_id in top_ids) / len(top_ids)


def recall_at_k(recommended_ids: Iterable, relevant_ids: set, k: int) -> float:
    """Calculate Recall@K from explicitly reviewed relevant job IDs."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        return 0.0
    top_ids = list(recommended_ids)[:k]
    return sum(job_id in relevant_ids for job_id in top_ids) / len(relevant_ids)