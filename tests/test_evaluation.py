import pandas as pd

from src.evaluation import (
    dataset_quality_report,
    precision_at_k,
    recall_at_k,
    validate_match_results,
)


def test_dataset_quality_report_counts_observed_problems():
    data = pd.DataFrame({"job_id": [1, 1], "title": ["A", "A"]})
    report = dataset_quality_report(data, "job_id")

    assert report["rows"] == 2
    assert report["duplicate_rows"] == 1
    assert report["duplicate_ids"] == 1


def test_match_results_have_valid_sorted_scores():
    results = pd.DataFrame({"match_score": [90.0, 50.0, 10.0]})

    assert validate_match_results(results) == {
        "scores_in_range": True,
        "sorted_descending": True,
        "rows": 3,
    }


def test_precision_and_recall_require_reviewed_ids():
    recommended = [10, 20, 30]
    relevant = {10, 30}

    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert recall_at_k(recommended, relevant, 2) == 0.5