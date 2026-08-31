import json
from pathlib import Path

import pandas as pd

from src.features.build_track_a import (
    FORBIDDEN_SAME_ROW_FEATURE_COLUMNS,
    MIN_PRIOR_VALID_HANDGRIPS,
    MODEL_INPUT_COLUMNS,
    build_and_write as build_track_a_and_write,
    build_track_a_features,
    validate_track_a_features,
)
from src.models.evaluate_track_a import evaluate_and_write


def test_track_a_features_are_valid_prior_only_rows():
    observations = pd.read_csv("data/processed/observations.csv")
    features = build_track_a_features(observations)
    summary = validate_track_a_features(features)
    primary = features.loc[features["is_primary_modeling_row"]]

    assert summary["valid_handgrip_rows"] == int(observations["handgrip_kg"].notna().sum())
    assert features["handgrip_kg"].notna().all()
    assert (primary["prior_valid_handgrip_count"] >= MIN_PRIOR_VALID_HANDGRIPS).all()
    assert not primary["exclude_from_modeling_default"].astype(bool).any()


def test_track_a_model_inputs_exclude_forbidden_and_retrospective_fields():
    observations = pd.read_csv("data/processed/observations.csv")
    features = build_track_a_features(observations)

    assert not (FORBIDDEN_SAME_ROW_FEATURE_COLUMNS & set(features.columns))
    assert not any(column.startswith("retrospective_") for column in MODEL_INPUT_COLUMNS)
    assert "retrospective_descriptive_full_athlete_median_handgrip" not in MODEL_INPUT_COLUMNS
    assert set(MODEL_INPUT_COLUMNS) <= set(features.columns)


def test_track_a_artifacts_and_evaluation_outputs_are_written(tmp_path):
    feature_path = tmp_path / "track_a_features.csv"
    metrics_path = tmp_path / "track_a_metrics.json"
    predictions_path = tmp_path / "track_a_predictions.csv"

    build_result = build_track_a_and_write(
        Path("data/processed/observations.csv"),
        feature_path,
    )
    eval_result = evaluate_and_write(feature_path, metrics_path, predictions_path)

    assert build_result.output_path.exists()
    assert metrics_path.exists()
    assert predictions_path.exists()
    assert not eval_result.predictions.empty

    metrics = json.loads(metrics_path.read_text())
    predictions = pd.read_csv(predictions_path)
    assert metrics["row_order_policy"] == "source_row_order_only_no_random_splits"
    assert metrics["evaluation"]["chronological_holdout"]["random_split_used"] is False
    assert metrics["dataset_summary"]["primary_modeling_rows"] == build_result.summary["primary_modeling_rows"]
    assert set(
        [
            "observation_id",
            "athlete_id",
            "y_true_handgrip_kg",
            "y_pred_handgrip_kg",
            "y_true_pct_prior_median_baseline",
            "residual_kg",
            "evaluation_fold",
            "model_name",
        ]
    ) <= set(predictions.columns)
    assert "chronological_holdout:last_30pct_per_athlete_min_1" in set(predictions["evaluation_fold"])
    assert predictions["evaluation_fold"].str.contains("random", case=False).sum() == 0
