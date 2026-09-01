import json
import math
import builtins
from pathlib import Path

import pandas as pd
import pytest

from src.models import evaluate_track_b as track_b


def load_track_b_features():
    return pd.read_csv("data/processed/track_b_features.csv")


@pytest.fixture(scope="module")
def official_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("track_b_official")
    return track_b.evaluate_and_write(
        input_path=Path("data/processed/track_b_features.csv"),
        metrics_path=tmp_path / "track_b_metrics.json",
        predictions_path=tmp_path / "track_b_predictions.csv",
        comparison_path=tmp_path / "model_comparison.json",
        track_a_metrics_path=Path("reports/track_a_metrics.json"),
        bootstrap_path=tmp_path / "track_b_bootstrap.json",
        per_athlete_path=tmp_path / "track_b_per_athlete.csv",
        missingness_path=tmp_path / "track_b_missingness.json",
        feature_audit_path=tmp_path / "track_b_feature_audit.json",
        stability_path=tmp_path / "track_b_stability.json",
    )


def test_official_evaluation_requires_and_records_sklearn(official_result):
    result = official_result

    assert result.metrics["backend"]["official_backend"] == "scikit-learn"
    assert result.metrics["backend"]["numpy_fallback_used"] is False
    assert result.metrics["dependency_versions"]["scikit_learn"] == track_b.require_sklearn()["sklearn"].__version__
    assert "sklearn_ridge" in result.metrics["metrics"]["leave_one_athlete_out"]
    assert "sklearn_random_forest" in result.metrics["metrics"]["leave_one_athlete_out"]
    assert "ridge_numpy_fallback" not in set(result.predictions["model_name"])


def test_official_evaluation_fails_clearly_without_sklearn(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("blocked sklearn for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="requires scikit-learn"):
        track_b.require_sklearn()


def test_deterministic_estimator_parameters_are_recorded():
    sklearn_api = track_b.require_sklearn()
    params = track_b.estimator_params(sklearn_api)

    assert params["sklearn_ridge"]["alpha"] == 10.0
    assert params["sklearn_random_forest"]["n_estimators"] == 100
    assert params["sklearn_random_forest"]["max_depth"] == 3
    assert params["sklearn_random_forest"]["min_samples_leaf"] == 5
    assert params["sklearn_random_forest"]["random_state"] == track_b.SEED


def test_train_fold_only_numeric_imputation_scaling_and_categorical_vocabulary():
    sklearn_api = track_b.require_sklearn()
    train = pd.DataFrame(
        {
            "handgrip_kg": [10.0, 20.0, 30.0],
            "numeric_feature": [1.0, 3.0, None],
            "category_feature": ["a", None, "b"],
        }
    )
    test = pd.DataFrame(
        {
            "handgrip_kg": [40.0, 50.0],
            "numeric_feature": [None, 999.0],
            "category_feature": ["test_only", None],
        }
    )

    pipeline = track_b.fit_sklearn_pipeline(
        train,
        ["numeric_feature", "category_feature"],
        "sklearn_ridge",
        sklearn_api,
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipe = preprocessor.named_transformers_["numeric"]
    categorical_pipe = preprocessor.named_transformers_["categorical"]
    transformed_test = preprocessor.transform(test[["numeric_feature", "category_feature"]])

    assert numeric_pipe.named_steps["imputer"].statistics_[0] == 2.0
    assert numeric_pipe.named_steps["scaler"].mean_[0] == 2.0
    categories = categorical_pipe.named_steps["onehot"].categories_[0].tolist()
    assert categories == ["__MISSING__", "a", "b"]
    assert "test_only" not in categories
    assert transformed_test.shape[0] == len(test)


def test_forbidden_target_identifier_status_and_source_order_columns_fail(monkeypatch):
    features = load_track_b_features()
    track_b.validate_model_input_columns(features)

    bad_feature_sets = dict(track_b.FEATURE_SETS)
    bad_feature_sets["bad"] = ["prior_valid_handgrip_count", "handgrip_kg", "athlete_id", "raw_daily_row_index"]
    monkeypatch.setattr(track_b, "FEATURE_SETS", bad_feature_sets)

    with pytest.raises(ValueError, match="forbidden columns"):
        track_b.validate_model_input_columns(features)


def test_leave_one_athlete_out_and_chronological_isolation():
    features = track_b.primary_modeling_rows(load_track_b_features())
    folds = track_b.split_folds(features)
    track_b.assert_fold_isolation(features, folds)

    assert len(folds["leave_one_athlete_out"]) == features["athlete_id"].nunique()
    for _, train_index, test_index in folds["leave_one_athlete_out"]:
        assert set(features.loc[train_index, "athlete_id"]).isdisjoint(set(features.loc[test_index, "athlete_id"]))

    _, train_index, test_index = folds["chronological_holdout"][0]
    for athlete_id in features["athlete_id"].drop_duplicates():
        train_rows = features.loc[train_index][features.loc[train_index, "athlete_id"] == athlete_id]
        test_rows = features.loc[test_index][features.loc[test_index, "athlete_id"] == athlete_id]
        if not train_rows.empty and not test_rows.empty:
            assert train_rows["raw_daily_row_index"].max() < test_rows["raw_daily_row_index"].min()


def test_paired_track_a_track_b_row_alignment_and_delta_sign(official_result):
    result = official_result
    paired = track_b.paired_predictions(
        result.predictions,
        "leave_one_athlete_out",
        "sklearn_ridge",
        "prior_history_plus_match_context",
    )
    assert paired["observation_id"].is_unique
    delta = paired["track_b_abs_error"].mean() - paired["track_a_abs_error"].mean()
    reported = result.metrics["metrics"]["leave_one_athlete_out"]["sklearn_ridge"][
        "prior_history_plus_match_context"
    ]["mae_delta_vs_track_a_prior_mean_kg"]
    assert reported == pytest.approx(delta)
    assert reported > 0


def test_grouped_bootstrap_is_deterministic_and_has_schema():
    paired = pd.DataFrame(
        {
            "athlete_id": ["a", "a", "b", "b", "c", "c"],
            "track_a_abs_error": [1.0, 2.0, 2.0, 2.0, 0.5, 0.5],
            "track_b_abs_error": [0.5, 1.0, 3.0, 3.0, 0.25, 0.25],
        }
    )

    first = track_b.bootstrap_paired_mae(paired, iterations=200, seed=123)
    second = track_b.bootstrap_paired_mae(paired, iterations=200, seed=123)

    assert first == second
    assert first["resampling_unit"] == "athlete_cluster"
    assert first["athlete_cluster_count"] == 3
    assert set(first["paired_delta_mae_kg"]) == {"point_estimate", "ci_lower", "ci_upper"}
    assert first["paired_delta_mae_kg"]["point_estimate"] == pytest.approx(
        paired["track_b_abs_error"].mean() - paired["track_a_abs_error"].mean()
    )


def test_per_athlete_and_small_count_labels(official_result):
    result = official_result
    per_athlete = pd.read_csv(result.outputs["per_athlete"])

    assert len(per_athlete) == len(track_b.SKLEARN_ESTIMATORS) * len(track_b.FEATURE_SETS) * 2 * 15
    assert set(per_athlete["interpretability_bucket"]) == {
        "minimally_interpretable",
        "anecdotal_too_few_rows",
    }
    assert set(per_athlete["direction"]) <= {"improved", "unchanged", "worse"}
    assert result.outputs["per_athlete"].exists()


def test_feature_groups_missingness_and_feature_audit_are_complete(official_result):
    result = official_result
    missingness = json.loads(result.outputs["missingness"].read_text())
    audit = json.loads(result.outputs["feature_audit"].read_text())

    assert missingness["computed_before_imputation"] is True
    assert {row["feature_group"] for row in missingness["feature_group_coverage"]} == set(track_b.FEATURE_GROUPS)
    included = {row["feature_name"] for row in audit["included_model_features"]}
    expected = {column for columns in track_b.FEATURE_SETS.values() for column in columns}
    assert expected <= included
    excluded = {row["feature_name"]: row["exclusion_reason"] for row in audit["excluded_columns"]}
    assert excluded["handgrip_kg"] == "target_or_target_derived"
    assert excluded["athlete_id"] == "identifier_not_predictive_feature"
    assert result.outputs["missingness"].exists()


def test_outputs_are_deterministic_finite_and_raw_data_untouched(tmp_path):
    raw_path = Path("data/raw/REST.zip")
    before_hash = track_b.file_sha256(raw_path)
    kwargs = {
        "input_path": Path("data/processed/track_b_features.csv"),
        "track_a_metrics_path": Path("reports/track_a_metrics.json"),
    }
    first = track_b.evaluate_and_write(
        metrics_path=tmp_path / "first_metrics.json",
        predictions_path=tmp_path / "first_predictions.csv",
        comparison_path=tmp_path / "first_comparison.json",
        bootstrap_path=tmp_path / "first_bootstrap.json",
        per_athlete_path=tmp_path / "first_per_athlete.csv",
        missingness_path=tmp_path / "first_missingness.json",
        feature_audit_path=tmp_path / "first_feature_audit.json",
        stability_path=tmp_path / "first_stability.json",
        **kwargs,
    )
    second = track_b.evaluate_and_write(
        metrics_path=tmp_path / "second_metrics.json",
        predictions_path=tmp_path / "second_predictions.csv",
        comparison_path=tmp_path / "second_comparison.json",
        bootstrap_path=tmp_path / "second_bootstrap.json",
        per_athlete_path=tmp_path / "second_per_athlete.csv",
        missingness_path=tmp_path / "second_missingness.json",
        feature_audit_path=tmp_path / "second_feature_audit.json",
        stability_path=tmp_path / "second_stability.json",
        **kwargs,
    )

    assert (tmp_path / "first_metrics.json").read_bytes() == (tmp_path / "second_metrics.json").read_bytes()
    assert (tmp_path / "first_predictions.csv").read_bytes() == (tmp_path / "second_predictions.csv").read_bytes()
    assert before_hash == track_b.file_sha256(raw_path)
    assert first.metrics["backend"]["numpy_fallback_used"] is False
    assert second.metrics["backend"]["numpy_fallback_used"] is False

    for json_path in [
        tmp_path / "first_metrics.json",
        tmp_path / "first_comparison.json",
        tmp_path / "first_bootstrap.json",
        tmp_path / "first_missingness.json",
        tmp_path / "first_feature_audit.json",
        tmp_path / "first_stability.json",
    ]:
        payload = json.loads(json_path.read_text())
        track_b.assert_json_finite(payload)

    predictions = pd.read_csv(tmp_path / "first_predictions.csv")
    assert predictions["y_pred_handgrip_kg"].map(math.isfinite).all()
    assert not predictions["evaluation_fold"].str.contains("random", case=False).any()
