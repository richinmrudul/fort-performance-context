import json
from pathlib import Path

import pandas as pd

from src.models.evaluate_track_b import (
    FEATURE_SETS,
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    fit_fold_preprocessor,
    split_folds,
    transform_with_preprocessor,
    validate_model_input_columns,
    evaluate_and_write,
)


def test_track_b_model_inputs_exclude_forbidden_columns():
    features = pd.read_csv("data/processed/track_b_features.csv")
    validate_model_input_columns(features)

    for columns in FEATURE_SETS.values():
        assert not (set(columns) & FORBIDDEN_MODEL_INPUT_COLUMNS)
        assert "observation_id" not in columns
        assert "athlete_id" not in columns
        assert not any("comment" in column.lower() for column in columns)


def test_fold_imputation_uses_training_medians_only():
    train = pd.DataFrame(
        {
            "numeric_feature": [1.0, 3.0, None],
            "category_feature": ["a", None, "b"],
        }
    )
    test = pd.DataFrame(
        {
            "numeric_feature": [None, 1000.0],
            "category_feature": [None, "unseen"],
        }
    )

    preprocessor = fit_fold_preprocessor(train, ["numeric_feature", "category_feature"])
    transformed = transform_with_preprocessor(test, preprocessor)

    assert preprocessor.numeric_medians["numeric_feature"] == 2.0
    assert transformed.loc[0, "numeric_feature"] == 2.0
    assert "__MISSING__" in preprocessor.categorical_levels["category_feature"]
    assert "category_feature=unseen" not in transformed.columns


def test_track_b_splits_do_not_use_random_rows():
    features = pd.read_csv("data/processed/track_b_features.csv")
    folds = split_folds(features)

    assert set(folds) == {"leave_one_athlete_out", "chronological_holdout"}
    assert len(folds["leave_one_athlete_out"]) == features["athlete_id"].nunique()
    assert len(folds["chronological_holdout"]) == 1

    for _, train_index, test_index in folds["leave_one_athlete_out"]:
        train_athletes = set(features.loc[train_index, "athlete_id"])
        test_athletes = set(features.loc[test_index, "athlete_id"])
        assert train_athletes.isdisjoint(test_athletes)

    _, train_index, test_index = folds["chronological_holdout"][0]
    assert len(train_index) + len(test_index) == len(features)
    assert set(train_index).isdisjoint(set(test_index))


def test_track_b_evaluation_writes_required_reports(tmp_path):
    metrics_path = tmp_path / "track_b_metrics.json"
    predictions_path = tmp_path / "track_b_predictions.csv"
    comparison_path = tmp_path / "model_comparison.json"

    result = evaluate_and_write(
        input_path=Path("data/processed/track_b_features.csv"),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        comparison_path=comparison_path,
        track_a_metrics_path=Path("reports/track_a_metrics.json"),
    )

    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert result.comparison_path.exists()

    predictions = pd.read_csv(predictions_path)
    expected_columns = {
        "observation_id",
        "athlete_id",
        "model_name",
        "feature_set",
        "evaluation_fold",
        "y_true_handgrip_kg",
        "y_pred_handgrip_kg",
        "y_true_pct_prior_median_baseline",
        "residual_kg",
    }
    assert expected_columns <= set(predictions.columns)
    assert not predictions["evaluation_fold"].str.contains("random", case=False).any()

    metrics = json.loads(metrics_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    assert metrics["row_order_policy"] == "source_row_order_only_no_random_splits"
    assert comparison["row_order_policy"] == "source_row_order_only_no_random_splits"


def test_every_feature_set_result_has_metrics_and_sample_counts(tmp_path):
    result = evaluate_and_write(
        input_path=Path("data/processed/track_b_features.csv"),
        metrics_path=tmp_path / "track_b_metrics.json",
        predictions_path=tmp_path / "track_b_predictions.csv",
        comparison_path=tmp_path / "model_comparison.json",
        track_a_metrics_path=Path("reports/track_a_metrics.json"),
    )

    for split_name, by_model in result.metrics["metrics"].items():
        assert split_name in {"leave_one_athlete_out", "chronological_holdout"}
        for by_feature_set in by_model.values():
            assert set(FEATURE_SETS) <= set(by_feature_set)
            for values in by_feature_set.values():
                assert values["sample_count"] > 0
                assert values["athlete_count"] > 0
                assert values["fold_count"] > 0
                assert "mae_kg" in values
                assert "rmse_kg" in values
                assert "mae_pct_prior_median_baseline" in values
