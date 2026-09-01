"""Hardened sklearn evaluation for exploratory Track B feature sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_track_a import MODEL_INPUT_COLUMNS as TRACK_A_PRIOR_ONLY_COLUMNS
from src.features.build_track_b import (
    ACTIGRAPHY_JOIN_ASSUMPTION,
    TRACK_B_OUTPUT,
)
from src.models.evaluate_track_a import (
    METRICS_OUTPUT as TRACK_A_METRICS_OUTPUT,
    NO_RANDOM_SPLIT_POLICY,
    chronological_holdout_rows,
)


TRACK_B_METRICS_OUTPUT = Path("reports/track_b_metrics.json")
TRACK_B_PREDICTIONS_OUTPUT = Path("reports/track_b_predictions.csv")
MODEL_COMPARISON_OUTPUT = Path("reports/model_comparison.json")
BOOTSTRAP_OUTPUT = Path("reports/track_b_bootstrap.json")
PER_ATHLETE_OUTPUT = Path("reports/track_b_per_athlete.csv")
MISSINGNESS_OUTPUT = Path("reports/track_b_missingness.json")
FEATURE_AUDIT_OUTPUT = Path("reports/track_b_feature_audit.json")
STABILITY_OUTPUT = Path("reports/track_b_stability.json")

SEED = 20260831
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
MIN_INTERPRETABLE_ATHLETE_ROWS = 3

TARGET_COLUMNS = ["handgrip_kg", "target_pct_prior_median_baseline"]
PROVENANCE_COLUMNS = [
    "observation_id",
    "athlete_id",
    "raw_daily_row_index",
    "athlete_row_index",
    "weekday",
    "duplicate_state",
    "exclude_from_modeling_default",
    "is_primary_modeling_row",
]
LABEL_OR_STATUS_COLUMNS = [
    "track_a_timing_label",
    "track_a_leakage_label",
    "actigraphy_join_assumption",
    "actigraphy_source_sequence_status",
    "actigraphy_weekday_window_id",
]
ACTIGRAPHY_METADATA_COLUMNS = [
    "actigraphy_row_count",
    "actigraphy_window_epoch_count",
    "actigraphy_expected_epoch_count",
    "actigraphy_partial_window_flag",
    "actigraphy_source_sequence_valid",
    "actigraphy_source_sequence_unexpected_transitions",
    "actigraphy_source_sequence_missing_samples",
    "actigraphy_shared_provisional_window",
    "actigraphy_shared_provisional_window_row_count",
]
FORBIDDEN_MODEL_INPUT_COLUMNS = set(
    TARGET_COLUMNS
    + PROVENANCE_COLUMNS
    + LABEL_OR_STATUS_COLUMNS
    + ACTIGRAPHY_METADATA_COLUMNS
    + [
        "retrospective_descriptive_full_athlete_median_handgrip",
        "retrospective_descriptive_performance_pct_full_athlete_median",
        "retrospective_descriptive_ui_only",
        "Comments (optional)",
    ]
)

FEATURE_GROUPS = {
    "prior_history": TRACK_A_PRIOR_ONLY_COLUMNS,
    "daily_sleep": [
        "objective_total_sleep_time",
        "objective_sleep_intervals",
        "objective_longest_interval_duration",
        "objective_sleep_period_mean",
        "objective_sleep_period_standard_deviation",
        "objective_sleep_period_msd",
        "sleep_onset_minutes_after_midnight",
        "sleep_offset_minutes_after_midnight",
        "longest_interval_start_minutes_after_midnight",
        "longest_interval_end_minutes_after_midnight",
    ],
    "caffeine": [
        "caffeine_mg",
        "caffeine_time_category",
        "coffee_count",
        "cola_count",
        "tea_count",
        "espresso_count",
        "caffeine_present",
    ],
    "wellness": [
        "sleepquality",
        "sleepdura_sr",
        "readiness",
        "fatigue",
        "soreness",
        "soreness_location_1",
        "soreness_location_2",
    ],
    "match_context": [
        "matchday_category",
        "matchday_present",
        "matchday_is_match_day",
        "matchday_relative_days",
    ],
    "actigraphy": [
        "actigraphy_vm_mean",
        "actigraphy_vm_median",
        "actigraphy_vm_std",
        "actigraphy_vm_max",
        "actigraphy_vm_sum",
        "actigraphy_x_mean",
        "actigraphy_y_mean",
        "actigraphy_z_mean",
        "actigraphy_all_algorithm_nonwear_invalid_fraction",
        "actigraphy_CK_sleep_fraction",
        "actigraphy_CK_wake_fraction",
        "actigraphy_CK_nonwear_fraction",
        "actigraphy_Oakley_sleep_fraction",
        "actigraphy_Oakley_wake_fraction",
        "actigraphy_Oakley_nonwear_fraction",
        "actigraphy_Sadeh_sleep_fraction",
        "actigraphy_Sadeh_wake_fraction",
        "actigraphy_Sadeh_nonwear_fraction",
        "actigraphy_fourier_HMM_sleep_fraction",
        "actigraphy_fourier_HMM_wake_fraction",
        "actigraphy_fourier_HMM_nonwear_fraction",
        "actigraphy_LSTM_sleep_fraction",
        "actigraphy_LSTM_wake_fraction",
        "actigraphy_LSTM_nonwear_fraction",
    ],
}

FEATURE_SETS = {
    "prior_history_only": FEATURE_GROUPS["prior_history"],
    "prior_history_plus_daily_sleep": FEATURE_GROUPS["prior_history"] + FEATURE_GROUPS["daily_sleep"],
    "prior_history_plus_caffeine": FEATURE_GROUPS["prior_history"] + FEATURE_GROUPS["caffeine"],
    "prior_history_plus_wellness": FEATURE_GROUPS["prior_history"] + FEATURE_GROUPS["wellness"],
    "prior_history_plus_match_context": FEATURE_GROUPS["prior_history"] + FEATURE_GROUPS["match_context"],
    "prior_history_plus_actigraphy": FEATURE_GROUPS["prior_history"] + FEATURE_GROUPS["actigraphy"],
    "all_track_b_exploratory": (
        FEATURE_GROUPS["prior_history"]
        + FEATURE_GROUPS["daily_sleep"]
        + FEATURE_GROUPS["caffeine"]
        + FEATURE_GROUPS["wellness"]
        + FEATURE_GROUPS["match_context"]
        + FEATURE_GROUPS["actigraphy"]
    ),
}

BASELINE_MODELS = {
    "track_a_prior_expanding_mean": "prior_expanding_handgrip_mean",
    "track_a_prior_expanding_median": "prior_expanding_handgrip_median",
}
SKLEARN_ESTIMATORS = ["sklearn_ridge", "sklearn_random_forest"]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    comparison: dict[str, Any]
    outputs: dict[str, Path]


def require_sklearn() -> dict[str, Any]:
    try:
        import sklearn
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
        raise RuntimeError(
            "Official Track B hardened evaluation requires scikit-learn. "
            "Install the declared modeling extra before generating reports."
        ) from exc

    return {
        "sklearn": sklearn,
        "ColumnTransformer": ColumnTransformer,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "Ridge": Ridge,
        "Pipeline": Pipeline,
        "FunctionTransformer": FunctionTransformer,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def load_features(input_path: Path = TRACK_B_OUTPUT) -> pd.DataFrame:
    return pd.read_csv(input_path)


def primary_modeling_rows(features: pd.DataFrame) -> pd.DataFrame:
    if not features["is_primary_modeling_row"].astype(bool).all():
        raise ValueError("Track B input must contain only primary Track A modeling rows")
    if len(features) != 84:
        raise ValueError(f"Expected 84 Track B modeling rows, found {len(features)}")
    return features.sort_values("raw_daily_row_index", kind="mergesort").reset_index(drop=True)


def validate_model_input_columns(features: pd.DataFrame) -> None:
    errors = []
    for feature_set, columns in FEATURE_SETS.items():
        missing = sorted(set(columns) - set(features.columns))
        forbidden = sorted(set(columns) & FORBIDDEN_MODEL_INPUT_COLUMNS)
        comments = sorted(column for column in columns if "comment" in column.lower())
        if missing:
            errors.append(f"{feature_set} missing columns: {missing}")
        if forbidden:
            errors.append(f"{feature_set} uses forbidden columns: {forbidden}")
        if comments:
            errors.append(f"{feature_set} uses comment columns: {comments}")
    if errors:
        raise ValueError("Track B model input validation failed: " + "; ".join(errors))


def split_folds(primary: pd.DataFrame) -> dict[str, list[tuple[str, pd.Index, pd.Index]]]:
    loo_folds = []
    for athlete_id in primary["athlete_id"].drop_duplicates():
        test_index = primary.index[primary["athlete_id"] == athlete_id]
        train_index = primary.index[primary["athlete_id"] != athlete_id]
        if len(train_index) and len(test_index):
            loo_folds.append((f"leave_one_athlete_out:{athlete_id}", train_index, test_index))

    holdout = chronological_holdout_rows(primary)
    holdout_ids = set(holdout["observation_id"])
    test_index = primary.index[primary["observation_id"].isin(holdout_ids)]
    train_index = primary.index[~primary["observation_id"].isin(holdout_ids)]
    chronological_folds = [
        ("chronological_holdout:last_30pct_per_athlete_min_1", train_index, test_index)
    ]
    return {
        "leave_one_athlete_out": loo_folds,
        "chronological_holdout": chronological_folds,
    }


def assert_fold_isolation(primary: pd.DataFrame, folds: dict[str, list[tuple[str, pd.Index, pd.Index]]]) -> None:
    for fold_label, train_index, test_index in folds["leave_one_athlete_out"]:
        train_athletes = set(primary.loc[train_index, "athlete_id"])
        test_athletes = set(primary.loc[test_index, "athlete_id"])
        if train_athletes & test_athletes:
            raise ValueError(f"Held-out athlete leaked into training fold {fold_label}")

    for fold_label, train_index, test_index in folds["chronological_holdout"]:
        for athlete_id in primary["athlete_id"].drop_duplicates():
            athlete_train = primary.loc[train_index][primary.loc[train_index, "athlete_id"] == athlete_id]
            athlete_test = primary.loc[test_index][primary.loc[test_index, "athlete_id"] == athlete_id]
            if athlete_train.empty or athlete_test.empty:
                continue
            if athlete_train["raw_daily_row_index"].max() >= athlete_test["raw_daily_row_index"].min():
                raise ValueError(f"Future chronological row leaked into training fold {fold_label}")


def feature_kinds(train: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(train[column]) or pd.api.types.is_bool_dtype(train[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def make_preprocessor(train: pd.DataFrame, feature_columns: list[str], sklearn_api: dict[str, Any]):
    numeric_columns, categorical_columns = feature_kinds(train, feature_columns)
    pipeline_cls = sklearn_api["Pipeline"]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                pipeline_cls(
                    [
                        ("imputer", sklearn_api["SimpleImputer"](strategy="median")),
                        ("scaler", sklearn_api["StandardScaler"]()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                pipeline_cls(
                    [
                        ("missing_normalizer", sklearn_api["FunctionTransformer"](normalize_missing_frame, validate=False)),
                        ("imputer", sklearn_api["SimpleImputer"](strategy="constant", fill_value="__MISSING__")),
                        ("onehot", make_one_hot_encoder(sklearn_api)),
                    ]
                ),
                categorical_columns,
            )
        )
    return sklearn_api["ColumnTransformer"](transformers=transformers)


def make_one_hot_encoder(sklearn_api: dict[str, Any]):
    encoder_cls = sklearn_api["OneHotEncoder"]
    try:
        return encoder_cls(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - supports older sklearn if needed.
        return encoder_cls(handle_unknown="ignore", sparse=False)


def normalize_missing_frame(values):
    frame = pd.DataFrame(values).copy()
    return frame.where(pd.notna(frame), np.nan)


def estimator_params(sklearn_api: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "sklearn_ridge": sklearn_api["Ridge"](alpha=10.0).get_params(deep=True),
        "sklearn_random_forest": sklearn_api["RandomForestRegressor"](
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        ).get_params(deep=True),
    }


def make_estimator(model_name: str, sklearn_api: dict[str, Any]):
    if model_name == "sklearn_ridge":
        return sklearn_api["Ridge"](alpha=10.0)
    if model_name == "sklearn_random_forest":
        return sklearn_api["RandomForestRegressor"](
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        )
    raise ValueError(f"Unknown sklearn model: {model_name}")


def fit_sklearn_pipeline(train: pd.DataFrame, feature_columns: list[str], model_name: str, sklearn_api: dict[str, Any]):
    pipeline_cls = sklearn_api["Pipeline"]
    pipeline = pipeline_cls(
        [
            ("preprocessor", make_preprocessor(train, feature_columns, sklearn_api)),
            ("model", make_estimator(model_name, sklearn_api)),
        ]
    )
    pipeline.fit(train[feature_columns], train["handgrip_kg"])
    return pipeline


def _rmse(values: pd.Series) -> float:
    return float(math.sqrt((values.pow(2)).mean()))


def metric_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    pct_pred = 100 * predictions["y_pred_handgrip_kg"] / predictions["prior_expanding_handgrip_median"]
    return {
        "sample_count": int(len(predictions)),
        "athlete_count": int(predictions["athlete_id"].nunique()),
        "fold_count": int(predictions["evaluation_fold"].nunique()),
        "mae_kg": float(predictions["residual_kg"].abs().mean()),
        "rmse_kg": _rmse(predictions["residual_kg"]),
        "mae_pct_prior_median_baseline": float(
            (predictions["y_true_pct_prior_median_baseline"] - pct_pred).abs().mean()
        ),
    }


def make_prediction_frame(
    frame: pd.DataFrame,
    y_pred: np.ndarray | pd.Series,
    model_name: str,
    estimator: str,
    feature_set: str,
    fold_label: str,
) -> pd.DataFrame:
    predictions = frame[
        [
            "observation_id",
            "athlete_id",
            "handgrip_kg",
            "target_pct_prior_median_baseline",
            "prior_expanding_handgrip_median",
        ]
    ].copy()
    predictions["y_pred_handgrip_kg"] = pd.Series(y_pred, index=frame.index).astype(float)
    predictions["residual_kg"] = predictions["handgrip_kg"] - predictions["y_pred_handgrip_kg"]
    predictions["model_name"] = model_name
    predictions["estimator"] = estimator
    predictions["feature_set"] = feature_set
    predictions["evaluation_fold"] = fold_label
    return predictions.rename(
        columns={
            "handgrip_kg": "y_true_handgrip_kg",
            "target_pct_prior_median_baseline": "y_true_pct_prior_median_baseline",
        }
    )


def baseline_predictions_for_folds(
    primary: pd.DataFrame,
    folds: dict[str, list[tuple[str, pd.Index, pd.Index]]],
) -> pd.DataFrame:
    frames = []
    for feature_set in FEATURE_SETS:
        for _, split_folds in folds.items():
            for fold_label, _, test_index in split_folds:
                test = primary.loc[test_index]
                for model_name, predictor_column in BASELINE_MODELS.items():
                    frames.append(
                        make_prediction_frame(
                            test,
                            test[predictor_column],
                            model_name,
                            "track_a_baseline",
                            feature_set,
                            fold_label,
                        )
                    )
    return pd.concat(frames, ignore_index=True)


def sklearn_predictions_for_folds(
    primary: pd.DataFrame,
    folds: dict[str, list[tuple[str, pd.Index, pd.Index]]],
    sklearn_api: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    fit_records = []
    for feature_set, columns in FEATURE_SETS.items():
        for model_name in SKLEARN_ESTIMATORS:
            for split_name, split_folds in folds.items():
                for fold_label, train_index, test_index in split_folds:
                    train = primary.loc[train_index]
                    test = primary.loc[test_index]
                    pipeline = fit_sklearn_pipeline(train, columns, model_name, sklearn_api)
                    y_pred = pipeline.predict(test[columns])
                    frames.append(
                        make_prediction_frame(
                            test,
                            y_pred,
                            model_name,
                            model_name,
                            feature_set,
                            fold_label,
                        )
                    )
                    transformed_train = pipeline.named_steps["preprocessor"].transform(train[columns])
                    fit_records.append(
                        {
                            "feature_set": feature_set,
                            "estimator": model_name,
                            "evaluation_scheme": split_name,
                            "fold": fold_label,
                            "train_sample_count": int(len(train)),
                            "test_sample_count": int(len(test)),
                            "encoded_feature_count": int(transformed_train.shape[1]),
                        }
                    )
    return pd.concat(frames, ignore_index=True), fit_records


def add_split_family(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.assign(
        evaluation_scheme=predictions["evaluation_fold"].str.split(":", n=1).str[0]
    )


def metrics_by_split_model_feature_set(predictions: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    grouped = add_split_family(predictions)
    for (scheme, model_name, feature_set), group in grouped.groupby(
        ["evaluation_scheme", "model_name", "feature_set"],
        sort=True,
    ):
        results.setdefault(scheme, {}).setdefault(model_name, {})[feature_set] = metric_summary(group)
    return results


def per_fold_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    grouped = add_split_family(predictions)
    for (scheme, fold, model_name, feature_set), group in grouped.groupby(
        ["evaluation_scheme", "evaluation_fold", "model_name", "feature_set"],
        sort=True,
    ):
        rows.append(
            {
                "evaluation_scheme": scheme,
                "evaluation_fold": fold,
                "model_name": model_name,
                "feature_set": feature_set,
                **metric_summary(group),
            }
        )
    return rows


def paired_predictions(
    predictions: pd.DataFrame,
    evaluation_scheme: str,
    model_name: str,
    feature_set: str,
    baseline_model_name: str = "track_a_prior_expanding_mean",
) -> pd.DataFrame:
    with_scheme = add_split_family(predictions)
    model = with_scheme[
        (with_scheme["evaluation_scheme"] == evaluation_scheme)
        & (with_scheme["model_name"] == model_name)
        & (with_scheme["feature_set"] == feature_set)
    ].copy()
    baseline = with_scheme[
        (with_scheme["evaluation_scheme"] == evaluation_scheme)
        & (with_scheme["model_name"] == baseline_model_name)
        & (with_scheme["feature_set"] == feature_set)
    ].copy()
    key = ["observation_id", "athlete_id", "evaluation_fold", "feature_set"]
    model = model.sort_values(key, kind="mergesort").reset_index(drop=True)
    baseline = baseline.sort_values(key, kind="mergesort").reset_index(drop=True)
    if model[key].to_dict("list") != baseline[key].to_dict("list"):
        raise ValueError(f"Paired Track A/Track B rows do not align for {evaluation_scheme} {model_name} {feature_set}")
    return pd.DataFrame(
        {
            "observation_id": model["observation_id"],
            "athlete_id": model["athlete_id"],
            "evaluation_fold": model["evaluation_fold"],
            "feature_set": model["feature_set"],
            "model_name": model_name,
            "track_b_abs_error": model["residual_kg"].abs(),
            "track_a_abs_error": baseline["residual_kg"].abs(),
        }
    )


def validate_predictions(predictions: pd.DataFrame) -> None:
    numeric_columns = ["y_true_handgrip_kg", "y_pred_handgrip_kg", "y_true_pct_prior_median_baseline", "residual_kg"]
    if not np.isfinite(predictions[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite prediction values detected")
    identity_columns = ["observation_id", "model_name", "feature_set", "evaluation_fold"]
    if predictions.duplicated(identity_columns).any():
        raise ValueError("Prediction identities are not unique within evaluation/configuration")
    if predictions["evaluation_fold"].str.contains("random", case=False).any():
        raise ValueError("Random split label detected")


def bootstrap_paired_mae(
    paired: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    athletes = sorted(paired["athlete_id"].unique())
    rng = np.random.default_rng(seed)
    track_a_values = []
    track_b_values = []
    delta_values = []
    for _ in range(iterations):
        sampled_athletes = rng.choice(athletes, size=len(athletes), replace=True)
        sampled_parts = [paired.loc[paired["athlete_id"] == athlete] for athlete in sampled_athletes]
        sampled = pd.concat(sampled_parts, ignore_index=True)
        track_a_mae = float(sampled["track_a_abs_error"].mean())
        track_b_mae = float(sampled["track_b_abs_error"].mean())
        track_a_values.append(track_a_mae)
        track_b_values.append(track_b_mae)
        delta_values.append(track_b_mae - track_a_mae)

    alpha = 1.0 - confidence_level
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    def summarize(values: list[float], point: float) -> dict[str, float]:
        return {
            "point_estimate": point,
            "ci_lower": float(np.percentile(values, lower_pct)),
            "ci_upper": float(np.percentile(values, upper_pct)),
        }

    point_track_a = float(paired["track_a_abs_error"].mean())
    point_track_b = float(paired["track_b_abs_error"].mean())
    return {
        "confidence_level": confidence_level,
        "bootstrap_iterations": iterations,
        "seed": seed,
        "resampling_unit": "athlete_cluster",
        "athlete_cluster_count": int(len(athletes)),
        "row_count": int(len(paired)),
        "track_a_mae_kg": summarize(track_a_values, point_track_a),
        "track_b_mae_kg": summarize(track_b_values, point_track_b),
        "paired_delta_mae_kg": summarize(delta_values, point_track_b - point_track_a),
        "limitation": (
            "Descriptive clustered bootstrap over fixed out-of-fold predictions; "
            "small athlete count and no refitting inside bootstrap replicates."
        ),
    }


def build_bootstrap_report(predictions: pd.DataFrame) -> dict[str, Any]:
    comparisons = []
    for scheme in ["leave_one_athlete_out", "chronological_holdout"]:
        for model_name in SKLEARN_ESTIMATORS:
            for feature_set in FEATURE_SETS:
                paired = paired_predictions(predictions, scheme, model_name, feature_set)
                comparisons.append(
                    {
                        "evaluation_scheme": scheme,
                        "estimator": model_name,
                        "feature_set": feature_set,
                        **bootstrap_paired_mae(paired),
                    }
                )
    return {
        "analysis_label": "descriptive_cluster_bootstrap_fixed_oof_predictions",
        "baseline_model": "track_a_prior_expanding_mean",
        "delta_definition": "Track B MAE - Track A MAE; negative means Track B improved",
        "comparisons": comparisons,
    }


def direction(delta: float, tolerance: float = 1e-12) -> str:
    if delta < -tolerance:
        return "improved"
    if delta > tolerance:
        return "worse"
    return "unchanged"


def build_per_athlete_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in ["leave_one_athlete_out", "chronological_holdout"]:
        for model_name in SKLEARN_ESTIMATORS:
            for feature_set in FEATURE_SETS:
                paired = paired_predictions(predictions, scheme, model_name, feature_set)
                for athlete_id, athlete_rows in paired.groupby("athlete_id", sort=True):
                    track_a_mae = float(athlete_rows["track_a_abs_error"].mean())
                    track_b_mae = float(athlete_rows["track_b_abs_error"].mean())
                    delta = track_b_mae - track_a_mae
                    rows.append(
                        {
                            "athlete_id": athlete_id,
                            "evaluation_scheme": scheme,
                            "estimator": model_name,
                            "feature_set": feature_set,
                            "evaluated_row_count": int(len(athlete_rows)),
                            "interpretability_bucket": (
                                "minimally_interpretable"
                                if len(athlete_rows) >= MIN_INTERPRETABLE_ATHLETE_ROWS
                                else "anecdotal_too_few_rows"
                            ),
                            "track_a_mae_kg": track_a_mae,
                            "track_b_mae_kg": track_b_mae,
                            "paired_mae_delta_kg": delta,
                            "direction": direction(delta),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["evaluation_scheme", "estimator", "feature_set", "athlete_id"],
        kind="mergesort",
    )


def build_missingness_report(features: pd.DataFrame) -> dict[str, Any]:
    rows = []
    group_rows = []
    athlete_rows = []
    for group_name, columns in FEATURE_GROUPS.items():
        group = features[columns]
        rows_with_any = int(group.notna().any(axis=1).sum())
        rows_complete = int(group.notna().all(axis=1).sum())
        athletes_with_any = sorted(features.loc[group.notna().any(axis=1), "athlete_id"].unique().tolist())
        athletes_no_coverage = sorted(set(features["athlete_id"]) - set(athletes_with_any))
        group_rows.append(
            {
                "feature_group": group_name,
                "feature_count": len(columns),
                "rows_with_at_least_one_feature_present": rows_with_any,
                "rows_with_complete_group_coverage": rows_complete,
                "athletes_with_any_coverage_count": len(athletes_with_any),
                "athletes_with_no_coverage_count": len(athletes_no_coverage),
                "athletes_with_no_coverage": athletes_no_coverage,
            }
        )
        for column in columns:
            present = features[column].notna()
            rows.append(
                {
                    "feature_name": column,
                    "feature_group": group_name,
                    "non_missing_count": int(present.sum()),
                    "missing_count": int((~present).sum()),
                    "missing_percent": round(float((~present).mean() * 100), 2),
                    "athletes_with_any_coverage": int(features.loc[present, "athlete_id"].nunique()),
                    "athletes_with_no_coverage": int(features["athlete_id"].nunique() - features.loc[present, "athlete_id"].nunique()),
                    "near_constant": bool(features[column].nunique(dropna=True) <= 1),
                }
            )
        for athlete_id, athlete_frame in features.groupby("athlete_id", sort=True):
            athlete_group = athlete_frame[columns]
            athlete_rows.append(
                {
                    "athlete_id": athlete_id,
                    "feature_group": group_name,
                    "row_count": int(len(athlete_frame)),
                    "rows_with_at_least_one_feature_present": int(athlete_group.notna().any(axis=1).sum()),
                    "rows_with_complete_group_coverage": int(athlete_group.notna().all(axis=1).sum()),
                    "missing_cell_percent": round(float(athlete_group.isna().to_numpy().mean() * 100), 2),
                }
            )
    return {
        "dataset_rows": int(len(features)),
        "athlete_count": int(features["athlete_id"].nunique()),
        "computed_before_imputation": True,
        "feature_missingness": rows,
        "feature_group_coverage": group_rows,
        "missingness_by_athlete": athlete_rows,
        "diagnostic_notes": [
            "Sparse or uneven feature coverage may plausibly weaken Track B models.",
            "Near-constant fields and timing-unverified fields are tracked as data-quality concerns.",
            "No missing values are inferred from outcomes or future observations.",
        ],
    }


def build_feature_audit(features: pd.DataFrame) -> dict[str, Any]:
    all_included = []
    for feature_set, columns in FEATURE_SETS.items():
        train = features[columns]
        numeric_columns, categorical_columns = feature_kinds(train, columns)
        for column in columns:
            group = next(group for group, group_columns in FEATURE_GROUPS.items() if column in group_columns)
            all_included.append(
                {
                    "feature_name": column,
                    "feature_set": feature_set,
                    "feature_group": group,
                    "included": True,
                    "handling": "numeric_median_impute_scale" if column in numeric_columns else "categorical_missing_category_one_hot",
                    "timing_verification_status": (
                        "prior_only_prospective_assumption" if group == "prior_history" else "timing_unverified"
                    ),
                }
            )
    included_columns = sorted({row["feature_name"] for row in all_included})
    excluded = []
    for column in features.columns:
        if column in included_columns:
            continue
        if column in TARGET_COLUMNS:
            reason = "target_or_target_derived"
        elif column in ["athlete_id", "observation_id"]:
            reason = "identifier_not_predictive_feature"
        elif column in ["raw_daily_row_index", "athlete_row_index"]:
            reason = "source_order_field_not_predictive_feature"
        elif column in LABEL_OR_STATUS_COLUMNS:
            reason = "status_or_leakage_label_unavailable_at_prediction_time"
        elif column in ACTIGRAPHY_METADATA_COLUMNS:
            reason = "actigraphy_join_or_validation_metadata_not_context_signal"
        elif column in PROVENANCE_COLUMNS:
            reason = "provenance_or_split_control"
        elif column.startswith("retrospective_"):
            reason = "retrospective_descriptive_baseline_prohibited"
        elif "comment" in column.lower():
            reason = "free_text_comment_prohibited"
        else:
            reason = "not_in_declared_track_b_feature_sets"
        excluded.append({"feature_name": column, "included": False, "exclusion_reason": reason})
    return {
        "analysis_label": "track_b_model_feature_audit",
        "included_model_features": all_included,
        "excluded_columns": sorted(excluded, key=lambda row: row["feature_name"]),
        "leakage_controls": [
            "Targets, identifiers, row indices, status labels, and retrospective baselines are excluded.",
            "Numeric imputation, scaling, categorical missing imputation, and one-hot encoding are fit inside training folds only.",
            "Track B daily and actigraphy features remain timing-unverified.",
        ],
    }


def bootstrap_lookup(bootstrap_report: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (entry["evaluation_scheme"], entry["estimator"], entry["feature_set"]): entry
        for entry in bootstrap_report["comparisons"]
    }


def build_stability_report(
    predictions: pd.DataFrame,
    per_athlete: pd.DataFrame,
    bootstrap_report: dict[str, Any],
) -> dict[str, Any]:
    metrics = metrics_by_split_model_feature_set(predictions)
    add_mae_deltas(metrics)
    boot = bootstrap_lookup(bootstrap_report)
    rows = []
    for feature_set in FEATURE_SETS:
        for estimator in SKLEARN_ESTIMATORS:
            loo = metrics["leave_one_athlete_out"][estimator][feature_set]
            chrono = metrics["chronological_holdout"][estimator][feature_set]
            loo_boot = boot[("leave_one_athlete_out", estimator, feature_set)]["paired_delta_mae_kg"]
            chrono_boot = boot[("chronological_holdout", estimator, feature_set)]["paired_delta_mae_kg"]
            athlete_slice = per_athlete[
                (per_athlete["estimator"] == estimator)
                & (per_athlete["feature_set"] == feature_set)
                & (per_athlete["interpretability_bucket"] == "minimally_interpretable")
            ]
            improved_athletes = athlete_slice[athlete_slice["direction"] == "improved"]["athlete_id"].nunique()
            total_athletes = athlete_slice["athlete_id"].nunique()
            loo_direction = direction(loo_boot["point_estimate"])
            chrono_direction = direction(chrono_boot["point_estimate"])
            consistent = loo_direction == "improved" and chrono_direction == "improved"
            if consistent:
                classification = "consistently_improved_but_uncertain"
            elif loo_direction == "worse" and chrono_direction == "worse":
                classification = "consistently_worse"
            elif improved_athletes <= 2:
                classification = "isolated_descriptive_improvement"
            else:
                classification = "directionally_mixed"
            rows.append(
                {
                    "feature_set": feature_set,
                    "estimator": estimator,
                    "ridge_or_forest": estimator,
                    "loo_mae_delta_kg": loo["mae_delta_vs_track_a_prior_mean_kg"],
                    "chronological_mae_delta_kg": chrono["mae_delta_vs_track_a_prior_mean_kg"],
                    "loo_bootstrap_delta_ci": loo_boot,
                    "chronological_bootstrap_delta_ci": chrono_boot,
                    "minimally_interpretable_athletes": int(total_athletes),
                    "athletes_improved_count": int(improved_athletes),
                    "athletes_improved_proportion": 0.0 if total_athletes == 0 else float(improved_athletes / total_athletes),
                    "improvement_direction_consistent_across_schemes": consistent,
                    "classification": classification,
                    "small_row_dependency_note": "Per-athlete rows below 3 are labeled anecdotal and excluded from this stability proportion.",
                }
            )
    return {
        "analysis_label": "feature_set_stability",
        "delta_definition": "Track B MAE - Track A MAE; negative means Track B improved",
        "stability_rows": rows,
    }


def add_mae_deltas(metrics: dict[str, dict[str, dict[str, Any]]]) -> None:
    for scheme, by_model in metrics.items():
        baseline = by_model["track_a_prior_expanding_mean"]
        for model_name, by_feature_set in by_model.items():
            for feature_set, values in by_feature_set.items():
                baseline_values = baseline[feature_set]
                values["track_a_prior_mean_mae_kg"] = baseline_values["mae_kg"]
                values["mae_delta_vs_track_a_prior_mean_kg"] = values["mae_kg"] - baseline_values["mae_kg"]
                values["delta_definition"] = "Track B MAE - Track A MAE; negative means Track B improved"


def load_track_a_metrics(path: Path = TRACK_A_METRICS_OUTPUT) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_contextual_rows(metrics: dict[str, Any], scheme: str) -> list[dict[str, Any]]:
    rows = []
    for estimator in SKLEARN_ESTIMATORS:
        for feature_set, values in metrics["metrics"][scheme][estimator].items():
            if feature_set == "prior_history_only":
                continue
            rows.append({"estimator": estimator, "feature_set": feature_set, **values})
    return sorted(rows, key=lambda row: row["mae_kg"])


def build_model_comparison(track_a_metrics: dict[str, Any], track_b_metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for scheme in ["leave_one_athlete_out", "chronological_holdout"]:
        track_a_prior_mean = track_a_metrics["metrics"][scheme]["prior_expanding_mean"]
        best_contextual = best_contextual_rows(track_b_metrics, scheme)[0]
        comparisons[scheme] = {
            "track_a_prior_mean_baseline": {
                "model_name": "prior_expanding_mean",
                "feature_set": "track_a",
                **track_a_prior_mean,
            },
            "best_track_b_contextual_model": best_contextual,
            "delta_mae_kg_vs_track_a_prior_mean": best_contextual["mae_kg"] - track_a_prior_mean["mae_kg"],
            "delta_definition": "Track B MAE - Track A MAE; negative means Track B improved",
            "improved_over_track_a_prior_mean": best_contextual["mae_kg"] < track_a_prior_mean["mae_kg"],
        }
    return {
        "comparison_scope": "Hardened Track B sklearn models compared with Track A prior-only baseline",
        "caution": (
            "Track B features are timing-unverified and actigraphy alignment is weekday-only; "
            "results do not prove causality or deployable pre-workout prediction."
        ),
        "row_order_policy": NO_RANDOM_SPLIT_POLICY,
        "baseline_identity": "Track A prior_expanding_mean on the exact same prediction rows",
        "comparisons": comparisons,
    }


def assert_json_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_json_finite(item)
    elif isinstance(value, list):
        for item in value:
            assert_json_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON report contains NaN or Infinity")


def build_metrics(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    sklearn_api: dict[str, Any],
    fit_records: list[dict[str, Any]],
    bootstrap_report: dict[str, Any],
) -> dict[str, Any]:
    metrics = metrics_by_split_model_feature_set(predictions)
    add_mae_deltas(metrics)
    return {
        "track": "Track B",
        "analysis_label": "hardened_exploratory_timing_unverified",
        "generation_command": "python3 -m src.models.evaluate_track_b",
        "row_order_policy": NO_RANDOM_SPLIT_POLICY,
        "dataset_summary": {
            "primary_modeling_rows": int(len(features)),
            "primary_athletes": int(features["athlete_id"].nunique()),
            "minimum_prior_valid_handgrips": 3,
            "actigraphy_join_assumption": ACTIGRAPHY_JOIN_ASSUMPTION,
        },
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn_api["sklearn"].__version__,
            "pandas": pd.__version__,
        },
        "backend": {
            "official_backend": "scikit-learn",
            "numpy_fallback_used": False,
            "preprocessing_backend": "sklearn Pipeline + ColumnTransformer",
        },
        "seed": SEED,
        "feature_sets": FEATURE_SETS,
        "estimators": {
            "sklearn_ridge": {
                "class_name": "sklearn.linear_model.Ridge",
                "parameters": estimator_params(sklearn_api)["sklearn_ridge"],
            },
            "sklearn_random_forest": {
                "class_name": "sklearn.ensemble.RandomForestRegressor",
                "parameters": estimator_params(sklearn_api)["sklearn_random_forest"],
            },
        },
        "baseline_model": "track_a_prior_expanding_mean",
        "delta_definition": "Track B MAE - Track A MAE; negative means Track B improved",
        "forbidden_model_input_columns": sorted(FORBIDDEN_MODEL_INPUT_COLUMNS),
        "evaluation": {
            "leave_one_athlete_out": {
                "feasible": int(features["athlete_id"].nunique()) >= 2,
                "athlete_count": int(features["athlete_id"].nunique()),
                "random_split_used": False,
            },
            "chronological_holdout": {
                "assumption": "source row order is treated as temporary temporal order",
                "split": "last 30 percent of eligible rows per athlete, minimum one row",
                "random_split_used": False,
            },
            "fit_records": fit_records,
        },
        "metrics": metrics,
        "bootstrap_summary": {
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "seed": SEED,
            "resampling_unit": "athlete_cluster",
            "athlete_cluster_count": int(features["athlete_id"].nunique()),
            "detail_report": BOOTSTRAP_OUTPUT.as_posix(),
        },
        "per_fold_metrics": per_fold_metrics(predictions),
        "cautions": [
            "Only 84 Track B rows are available.",
            "Daily recovery, caffeine, wellness, match-context, and actigraphy timing is unresolved.",
            "Actigraphy features use provisional athlete plus weekday alignment, not verified athlete-day windows.",
            "Bootstrap intervals are descriptive and conditional on fixed out-of-fold predictions.",
            "No random row split is used.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    assert_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_predictions(path: Path, predictions: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = predictions[
        [
            "observation_id",
            "athlete_id",
            "model_name",
            "estimator",
            "feature_set",
            "evaluation_fold",
            "y_true_handgrip_kg",
            "y_pred_handgrip_kg",
            "y_true_pct_prior_median_baseline",
            "residual_kg",
        ]
    ].sort_values(["evaluation_fold", "model_name", "feature_set", "observation_id"], kind="mergesort")
    ordered.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def write_per_athlete(path: Path, per_athlete: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_athlete.to_csv(path, index=False)
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_and_write(
    input_path: Path = TRACK_B_OUTPUT,
    metrics_path: Path = TRACK_B_METRICS_OUTPUT,
    predictions_path: Path = TRACK_B_PREDICTIONS_OUTPUT,
    comparison_path: Path = MODEL_COMPARISON_OUTPUT,
    track_a_metrics_path: Path = TRACK_A_METRICS_OUTPUT,
    bootstrap_path: Path = BOOTSTRAP_OUTPUT,
    per_athlete_path: Path = PER_ATHLETE_OUTPUT,
    missingness_path: Path = MISSINGNESS_OUTPUT,
    feature_audit_path: Path = FEATURE_AUDIT_OUTPUT,
    stability_path: Path = STABILITY_OUTPUT,
) -> EvaluationResult:
    sklearn_api = require_sklearn()
    features = primary_modeling_rows(load_features(input_path))
    validate_model_input_columns(features)
    folds = split_folds(features)
    assert_fold_isolation(features, folds)

    baseline_predictions = baseline_predictions_for_folds(features, folds)
    sklearn_predictions, fit_records = sklearn_predictions_for_folds(features, folds, sklearn_api)
    predictions = pd.concat([baseline_predictions, sklearn_predictions], ignore_index=True)
    validate_predictions(predictions)

    bootstrap_report = build_bootstrap_report(predictions)
    metrics = build_metrics(features, predictions, sklearn_api, fit_records, bootstrap_report)
    comparison = build_model_comparison(load_track_a_metrics(track_a_metrics_path), metrics)
    per_athlete = build_per_athlete_report(predictions)
    missingness = build_missingness_report(features)
    feature_audit = build_feature_audit(features)
    stability = build_stability_report(predictions, per_athlete, bootstrap_report)

    outputs = {
        "metrics": write_json(metrics_path, metrics),
        "predictions": write_predictions(predictions_path, predictions),
        "comparison": write_json(comparison_path, comparison),
        "bootstrap": write_json(bootstrap_path, bootstrap_report),
        "per_athlete": write_per_athlete(per_athlete_path, per_athlete),
        "missingness": write_json(missingness_path, missingness),
        "feature_audit": write_json(feature_audit_path, feature_audit),
        "stability": write_json(stability_path, stability),
    }
    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
        comparison=comparison,
        outputs=outputs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hardened exploratory Track B sklearn models.")
    parser.add_argument("--input", default=str(TRACK_B_OUTPUT), help="Path to Track B features CSV")
    parser.add_argument("--metrics-output", default=str(TRACK_B_METRICS_OUTPUT), help="Path for metrics JSON")
    parser.add_argument("--predictions-output", default=str(TRACK_B_PREDICTIONS_OUTPUT), help="Path for predictions CSV")
    parser.add_argument("--comparison-output", default=str(MODEL_COMPARISON_OUTPUT), help="Path for comparison JSON")
    parser.add_argument("--track-a-metrics", default=str(TRACK_A_METRICS_OUTPUT), help="Path to Track A metrics JSON")
    args = parser.parse_args()

    result = evaluate_and_write(
        input_path=Path(args.input),
        metrics_path=Path(args.metrics_output),
        predictions_path=Path(args.predictions_output),
        comparison_path=Path(args.comparison_output),
        track_a_metrics_path=Path(args.track_a_metrics),
    )
    print("Track B hardened evaluation")
    print(f"metrics: {result.outputs['metrics']}")
    print(f"predictions: {result.outputs['predictions']}")
    print(f"comparison: {result.outputs['comparison']}")
    print(f"primary_modeling_rows: {result.metrics['dataset_summary']['primary_modeling_rows']}")
    print(f"sklearn_version: {result.metrics['dependency_versions']['scikit_learn']}")


if __name__ == "__main__":
    main()
