"""Evaluate exploratory Track B feature sets against Track A baselines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

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

DAILY_SLEEP_COLUMNS = [
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
]
CAFFEINE_COLUMNS = [
    "caffeine_mg",
    "caffeine_time_category",
    "coffee_count",
    "cola_count",
    "tea_count",
    "espresso_count",
    "caffeine_present",
]
WELLNESS_COLUMNS = [
    "sleepquality",
    "sleepdura_sr",
    "readiness",
    "fatigue",
    "soreness",
    "soreness_location_1",
    "soreness_location_2",
]
MATCH_CONTEXT_COLUMNS = [
    "matchday_category",
    "matchday_present",
    "matchday_is_match_day",
    "matchday_relative_days",
]
ACTIGRAPHY_MODEL_COLUMNS = [
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
]

FEATURE_SETS = {
    "prior_history_only": TRACK_A_PRIOR_ONLY_COLUMNS,
    "prior_history_plus_daily_sleep": TRACK_A_PRIOR_ONLY_COLUMNS + DAILY_SLEEP_COLUMNS,
    "prior_history_plus_caffeine": TRACK_A_PRIOR_ONLY_COLUMNS + CAFFEINE_COLUMNS,
    "prior_history_plus_wellness": TRACK_A_PRIOR_ONLY_COLUMNS + WELLNESS_COLUMNS,
    "prior_history_plus_match_context": TRACK_A_PRIOR_ONLY_COLUMNS + MATCH_CONTEXT_COLUMNS,
    "prior_history_plus_actigraphy": TRACK_A_PRIOR_ONLY_COLUMNS + ACTIGRAPHY_MODEL_COLUMNS,
    "all_track_b_exploratory": (
        TRACK_A_PRIOR_ONLY_COLUMNS
        + DAILY_SLEEP_COLUMNS
        + CAFFEINE_COLUMNS
        + WELLNESS_COLUMNS
        + MATCH_CONTEXT_COLUMNS
        + ACTIGRAPHY_MODEL_COLUMNS
    ),
}


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, object]
    predictions: pd.DataFrame
    comparison: dict[str, object]
    metrics_path: Path
    predictions_path: Path
    comparison_path: Path


@dataclass(frozen=True)
class FoldPreprocessor:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: dict[str, float]
    categorical_levels: dict[str, list[str]]


def load_features(input_path: Path = TRACK_B_OUTPUT) -> pd.DataFrame:
    return pd.read_csv(input_path)


def primary_modeling_rows(features: pd.DataFrame) -> pd.DataFrame:
    if not features["is_primary_modeling_row"].astype(bool).all():
        raise ValueError("Track B input must contain only primary Track A modeling rows")
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


def fit_fold_preprocessor(train: pd.DataFrame, feature_columns: list[str]) -> FoldPreprocessor:
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(train[column]) or pd.api.types.is_bool_dtype(train[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    medians = {}
    for column in numeric_columns:
        numeric = pd.to_numeric(train[column], errors="coerce")
        median = numeric.median()
        medians[column] = 0.0 if pd.isna(median) else float(median)

    levels = {}
    for column in categorical_columns:
        values = train[column].astype("string").fillna("__MISSING__")
        levels[column] = sorted(values.unique().tolist())

    return FoldPreprocessor(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_medians=medians,
        categorical_levels=levels,
    )


def transform_with_preprocessor(frame: pd.DataFrame, preprocessor: FoldPreprocessor) -> pd.DataFrame:
    parts = []
    if preprocessor.numeric_columns:
        numeric = frame[preprocessor.numeric_columns].apply(pd.to_numeric, errors="coerce")
        for column, median in preprocessor.numeric_medians.items():
            numeric[column] = numeric[column].fillna(median)
        parts.append(numeric.astype(float))

    for column in preprocessor.categorical_columns:
        values = frame[column].astype("string").fillna("__MISSING__")
        encoded = pd.DataFrame(index=frame.index)
        for level in preprocessor.categorical_levels[column]:
            encoded[f"{column}={level}"] = values.eq(level).astype(float)
        if "__MISSING__" not in preprocessor.categorical_levels[column]:
            encoded[f"{column}=__MISSING__"] = values.eq("__MISSING__").astype(float)
        parts.append(encoded)

    if not parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(parts, axis=1)


def _rmse(values: pd.Series) -> float:
    return float(math.sqrt((values.pow(2)).mean()))


def metric_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    if predictions.empty:
        return {
            "sample_count": 0,
            "athlete_count": 0,
            "fold_count": 0,
            "mae_kg": float("nan"),
            "rmse_kg": float("nan"),
            "mae_pct_prior_median_baseline": float("nan"),
        }
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
    baseline_specs = {
        "prior_expanding_median": "prior_expanding_handgrip_median",
        "prior_expanding_mean": "prior_expanding_handgrip_mean",
    }
    for feature_set in FEATURE_SETS:
        for _, split_folds in folds.items():
            for fold_label, _, test_index in split_folds:
                test = primary.loc[test_index]
                for model_name, predictor_column in baseline_specs.items():
                    frames.append(
                        make_prediction_frame(
                            test,
                            test[predictor_column],
                            model_name,
                            feature_set,
                            fold_label,
                        )
                    )
    return pd.concat(frames, ignore_index=True)


def fit_numpy_ridge(x_train: pd.DataFrame, y_train: pd.Series, alpha: float = 10.0) -> np.ndarray:
    x = x_train.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x_design.T @ x_design + penalty) @ x_design.T @ y


def predict_numpy_ridge(x_test: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    x = x_test.to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    return x_design @ coef


def numpy_ridge_predictions_for_folds(
    primary: pd.DataFrame,
    folds: dict[str, list[tuple[str, pd.Index, pd.Index]]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = []
    fold_records = []
    for feature_set, columns in FEATURE_SETS.items():
        for split_name, split_folds in folds.items():
            for fold_label, train_index, test_index in split_folds:
                train = primary.loc[train_index]
                test = primary.loc[test_index]
                preprocessor = fit_fold_preprocessor(train, columns)
                x_train = transform_with_preprocessor(train, preprocessor)
                x_test = transform_with_preprocessor(test, preprocessor)
                coef = fit_numpy_ridge(x_train, train["handgrip_kg"])
                y_pred = predict_numpy_ridge(x_test, coef)
                frames.append(
                    make_prediction_frame(
                        test,
                        y_pred,
                        "ridge_numpy_fallback",
                        feature_set,
                        fold_label,
                    )
                )
                fold_records.append(
                    {
                        "feature_set": feature_set,
                        "split": split_name,
                        "fold": fold_label,
                        "train_sample_count": int(len(train)),
                        "test_sample_count": int(len(test)),
                        "encoded_feature_count": int(x_train.shape[1]),
                    }
                )

    return pd.concat(frames, ignore_index=True), {
        "model_name": "ridge_numpy_fallback",
        "evaluated": True,
        "reason": "sklearn is not installed; used deterministic ridge regression implemented with NumPy",
        "alpha": 10.0,
        "fold_preprocessing": "numeric medians and categorical levels are fit on training folds only",
        "folds": fold_records,
    }


def sklearn_predictions_for_folds(
    primary: pd.DataFrame,
    folds: dict[str, list[tuple[str, pd.Index, pd.Index]]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError:
        return numpy_ridge_predictions_for_folds(primary, folds)

    frames = []
    model_status = {
        "sklearn_available": True,
        "models": {
            "ridge_regularized_linear": {"evaluated": True, "alpha": 10.0},
            "random_forest_small": {
                "evaluated": True,
                "n_estimators": 100,
                "max_depth": 3,
                "min_samples_leaf": 5,
            },
        },
        "fold_preprocessing": "SimpleImputer and OneHotEncoder are fit inside each training fold only",
    }

    for feature_set, columns in FEATURE_SETS.items():
        sample = primary[columns]
        numeric_columns = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(sample[column]) or pd.api.types.is_bool_dtype(sample[column])
        ]
        categorical_columns = [column for column in columns if column not in numeric_columns]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_columns,
                ),
            ]
        )
        models = {
            "ridge_regularized_linear": Ridge(alpha=10.0),
            "random_forest_small": RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=5,
                random_state=0,
            ),
        }
        for model_name, model in models.items():
            for _, split_folds in folds.items():
                for fold_label, train_index, test_index in split_folds:
                    train = primary.loc[train_index]
                    test = primary.loc[test_index]
                    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
                    pipeline.fit(train[columns], train["handgrip_kg"])
                    frames.append(
                        make_prediction_frame(
                            test,
                            pipeline.predict(test[columns]),
                            model_name,
                            feature_set,
                            fold_label,
                        )
                    )

    return pd.concat(frames, ignore_index=True), model_status


def metrics_by_split_model_feature_set(predictions: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    results: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    split_family = predictions["evaluation_fold"].str.split(":", n=1).str[0]
    grouped = predictions.assign(evaluation_split_family=split_family)
    for (split_name, model_name, feature_set), group in grouped.groupby(
        ["evaluation_split_family", "model_name", "feature_set"],
        sort=True,
    ):
        results.setdefault(split_name, {}).setdefault(model_name, {})[feature_set] = metric_summary(group)
    return results


def per_fold_metrics(predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for (fold, model, feature_set), group in predictions.groupby(
        ["evaluation_fold", "model_name", "feature_set"],
        sort=True,
    ):
        split_family = fold.split(":", 1)[0]
        rows.append(
            {
                "evaluation_split_family": split_family,
                "evaluation_fold": fold,
                "model_name": model,
                "feature_set": feature_set,
                **metric_summary(group),
            }
        )
    return rows


def load_track_a_metrics(path: Path = TRACK_A_METRICS_OUTPUT) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_metric_rows(metrics: dict[str, object], split_name: str) -> list[dict[str, object]]:
    rows = []
    split_metrics = metrics["metrics"].get(split_name, {})
    for model_name, by_feature_set in split_metrics.items():
        if "mae_kg" in by_feature_set:
            rows.append({"model_name": model_name, "feature_set": "track_a", **by_feature_set})
            continue
        for feature_set, values in by_feature_set.items():
            rows.append({"model_name": model_name, "feature_set": feature_set, **values})
    return sorted(rows, key=lambda row: row["mae_kg"])


def build_model_comparison(track_a_metrics: dict[str, object], track_b_metrics: dict[str, object]) -> dict[str, object]:
    comparisons = {}
    for split_name in ["leave_one_athlete_out", "chronological_holdout"]:
        track_a_rows = best_metric_rows(track_a_metrics, split_name)
        track_b_rows = best_metric_rows(track_b_metrics, split_name)
        track_a_prior_mean = next(
            row for row in track_a_rows if row["model_name"] == "prior_expanding_mean"
        )
        best_track_b_contextual = next(
            row for row in track_b_rows if row["model_name"] not in {"prior_expanding_median", "prior_expanding_mean"}
        )
        comparisons[split_name] = {
            "track_a_prior_mean_baseline": track_a_prior_mean,
            "best_track_b_contextual_model": best_track_b_contextual,
            "delta_mae_kg_vs_track_a_prior_mean": best_track_b_contextual["mae_kg"]
            - track_a_prior_mean["mae_kg"],
            "improved_over_track_a_prior_mean": best_track_b_contextual["mae_kg"]
            < track_a_prior_mean["mae_kg"],
        }

    return {
        "comparison_scope": "Track B exploratory models compared with Track A prior-only baselines",
        "caution": (
            "Track B features are timing-unverified and actigraphy alignment is weekday-only; "
            "improvements do not prove causality or deployable pre-workout prediction."
        ),
        "row_order_policy": NO_RANDOM_SPLIT_POLICY,
        "comparisons": comparisons,
    }


def build_metrics(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    model_status: dict[str, object],
) -> dict[str, object]:
    return {
        "track": "Track B",
        "analysis_label": "exploratory_timing_unverified",
        "row_order_policy": NO_RANDOM_SPLIT_POLICY,
        "dataset_summary": {
            "primary_modeling_rows": int(len(features)),
            "primary_athletes": int(features["athlete_id"].nunique()),
            "minimum_prior_valid_handgrips": 3,
            "actigraphy_join_assumption": ACTIGRAPHY_JOIN_ASSUMPTION,
        },
        "feature_sets": FEATURE_SETS,
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
            "model_status": model_status,
        },
        "metrics": metrics_by_split_model_feature_set(predictions),
        "per_fold_metrics": per_fold_metrics(predictions),
        "cautions": [
            "Only 84 Track B rows are available.",
            "Daily recovery, caffeine, wellness, match-context, and actigraphy timing is unresolved.",
            "Actigraphy features use provisional athlete plus weekday alignment, not verified athlete-day windows.",
            "No random row split is used.",
        ],
    }


def write_outputs(
    metrics: dict[str, object],
    predictions: pd.DataFrame,
    comparison: dict[str, object],
    metrics_path: Path,
    predictions_path: Path,
    comparison_path: Path,
) -> tuple[Path, Path, Path]:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    output_predictions = predictions[
        [
            "observation_id",
            "athlete_id",
            "model_name",
            "feature_set",
            "evaluation_fold",
            "y_true_handgrip_kg",
            "y_pred_handgrip_kg",
            "y_true_pct_prior_median_baseline",
            "residual_kg",
        ]
    ].copy()
    output_predictions.to_csv(predictions_path, index=False)
    return metrics_path, predictions_path, comparison_path


def evaluate_and_write(
    input_path: Path = TRACK_B_OUTPUT,
    metrics_path: Path = TRACK_B_METRICS_OUTPUT,
    predictions_path: Path = TRACK_B_PREDICTIONS_OUTPUT,
    comparison_path: Path = MODEL_COMPARISON_OUTPUT,
    track_a_metrics_path: Path = TRACK_A_METRICS_OUTPUT,
) -> EvaluationResult:
    features = primary_modeling_rows(load_features(input_path))
    validate_model_input_columns(features)
    folds = split_folds(features)
    baseline_predictions = baseline_predictions_for_folds(features, folds)
    contextual_predictions, model_status = sklearn_predictions_for_folds(features, folds)
    predictions = pd.concat([baseline_predictions, contextual_predictions], ignore_index=True)
    metrics = build_metrics(features, predictions, model_status)
    comparison = build_model_comparison(load_track_a_metrics(track_a_metrics_path), metrics)
    written_metrics, written_predictions, written_comparison = write_outputs(
        metrics,
        predictions,
        comparison,
        metrics_path,
        predictions_path,
        comparison_path,
    )
    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
        comparison=comparison,
        metrics_path=written_metrics,
        predictions_path=written_predictions,
        comparison_path=written_comparison,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exploratory Track B handgrip models.")
    parser.add_argument("--input", default=str(TRACK_B_OUTPUT), help="Path to Track B features CSV")
    parser.add_argument("--metrics-output", default=str(TRACK_B_METRICS_OUTPUT), help="Path for metrics JSON")
    parser.add_argument(
        "--predictions-output",
        default=str(TRACK_B_PREDICTIONS_OUTPUT),
        help="Path for predictions CSV",
    )
    parser.add_argument(
        "--comparison-output",
        default=str(MODEL_COMPARISON_OUTPUT),
        help="Path for Track A vs Track B comparison JSON",
    )
    parser.add_argument(
        "--track-a-metrics",
        default=str(TRACK_A_METRICS_OUTPUT),
        help="Path to Track A metrics JSON",
    )
    args = parser.parse_args()

    result = evaluate_and_write(
        input_path=Path(args.input),
        metrics_path=Path(args.metrics_output),
        predictions_path=Path(args.predictions_output),
        comparison_path=Path(args.comparison_output),
        track_a_metrics_path=Path(args.track_a_metrics),
    )
    print("Track B evaluation")
    print(f"metrics: {result.metrics_path}")
    print(f"predictions: {result.predictions_path}")
    print(f"comparison: {result.comparison_path}")
    print(f"primary_modeling_rows: {result.metrics['dataset_summary']['primary_modeling_rows']}")


if __name__ == "__main__":
    main()
