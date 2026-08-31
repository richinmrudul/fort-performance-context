"""Evaluate conservative Track A handgrip baseline models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from src.features.build_track_a import MODEL_INPUT_COLUMNS, TRACK_A_OUTPUT, validate_track_a_features
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.features.build_track_a import MODEL_INPUT_COLUMNS, TRACK_A_OUTPUT, validate_track_a_features


METRICS_OUTPUT = Path("reports/track_a_metrics.json")
PREDICTIONS_OUTPUT = Path("reports/track_a_predictions.csv")
NO_RANDOM_SPLIT_POLICY = "source_row_order_only_no_random_splits"
BASELINE_MODEL_NAME = "prior_expanding_median"


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, object]
    predictions: pd.DataFrame
    metrics_path: Path
    predictions_path: Path


def load_features(input_path: Path = TRACK_A_OUTPUT) -> pd.DataFrame:
    return pd.read_csv(input_path)


def primary_modeling_rows(features: pd.DataFrame) -> pd.DataFrame:
    validate_track_a_features(features)
    primary = features.loc[features["is_primary_modeling_row"].astype(bool)].copy()
    return primary.sort_values(["raw_daily_row_index"], kind="mergesort").reset_index(drop=True)


def _rmse(values: pd.Series) -> float:
    return float(math.sqrt((values.pow(2)).mean()))


def metric_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    if predictions.empty:
        return {
            "sample_count": 0,
            "athlete_count": 0,
            "mae_kg": float("nan"),
            "rmse_kg": float("nan"),
            "mae_pct_prior_median_baseline": float("nan"),
        }
    pct_pred = 100 * predictions["y_pred_handgrip_kg"] / predictions["prior_expanding_handgrip_median"]
    return {
        "sample_count": int(len(predictions)),
        "athlete_count": int(predictions["athlete_id"].nunique()),
        "mae_kg": float(predictions["residual_kg"].abs().mean()),
        "rmse_kg": _rmse(predictions["residual_kg"]),
        "mae_pct_prior_median_baseline": float(
            (predictions["y_true_pct_prior_median_baseline"] - pct_pred).abs().mean()
        ),
    }


def make_predictions(frame: pd.DataFrame, model_name: str, split_label: str, predictor: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    predictions = frame[
        [
            "observation_id",
            "athlete_id",
            "handgrip_kg",
            "target_pct_prior_median_baseline",
            "prior_expanding_handgrip_median",
        ]
    ].copy()
    predictions["y_pred_handgrip_kg"] = predictor(frame).astype(float)
    predictions["residual_kg"] = predictions["handgrip_kg"] - predictions["y_pred_handgrip_kg"]
    predictions["evaluation_fold"] = split_label
    predictions["model_name"] = model_name
    predictions = predictions.rename(
        columns={
            "handgrip_kg": "y_true_handgrip_kg",
            "target_pct_prior_median_baseline": "y_true_pct_prior_median_baseline",
        }
    )
    return predictions


def leave_one_athlete_out_predictions(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    athlete_ids = list(primary["athlete_id"].drop_duplicates())
    feasible = len(athlete_ids) >= 2
    note = "feasible" if feasible else "too few athletes for leave-one-athlete-out evaluation"
    if not feasible:
        return pd.DataFrame(), {"feasible": False, "reason": note, "athlete_count": len(athlete_ids)}

    frames = []
    for athlete_id in athlete_ids:
        fold_rows = primary.loc[primary["athlete_id"] == athlete_id]
        fold_label = f"leave_one_athlete_out:{athlete_id}"
        frames.append(
            make_predictions(
                fold_rows,
                BASELINE_MODEL_NAME,
                fold_label,
                lambda rows: rows["prior_expanding_handgrip_median"],
            )
        )
        frames.append(
            make_predictions(
                fold_rows,
                "prior_expanding_mean",
                fold_label,
                lambda rows: rows["prior_expanding_handgrip_mean"],
            )
        )

    return pd.concat(frames, ignore_index=True), {"feasible": True, "reason": note, "athlete_count": len(athlete_ids)}


def chronological_holdout_rows(primary: pd.DataFrame, holdout_fraction: float = 0.30) -> pd.DataFrame:
    holdout_parts = []
    for _, athlete_rows in primary.groupby("athlete_id", sort=False):
        athlete_rows = athlete_rows.sort_values("raw_daily_row_index", kind="mergesort")
        holdout_count = max(1, math.ceil(len(athlete_rows) * holdout_fraction))
        holdout_parts.append(athlete_rows.tail(holdout_count))
    if not holdout_parts:
        return primary.iloc[0:0].copy()
    return pd.concat(holdout_parts, ignore_index=True).sort_values("raw_daily_row_index", kind="mergesort")


def chronological_holdout_predictions(primary: pd.DataFrame) -> pd.DataFrame:
    holdout = chronological_holdout_rows(primary)
    if holdout.empty:
        return holdout
    frames = [
        make_predictions(
            holdout,
            BASELINE_MODEL_NAME,
            "chronological_holdout:last_30pct_per_athlete_min_1",
            lambda rows: rows["prior_expanding_handgrip_median"],
        ),
        make_predictions(
            holdout,
            "prior_expanding_mean",
            "chronological_holdout:last_30pct_per_athlete_min_1",
            lambda rows: rows["prior_expanding_handgrip_mean"],
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def sklearn_regression_predictions(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return pd.DataFrame(), {
            "model_name": "ridge_prior_only_numeric",
            "evaluated": False,
            "reason": "sklearn is not installed in this environment",
        }

    holdout = chronological_holdout_rows(primary)
    train_ids = set(holdout["observation_id"])
    train = primary.loc[~primary["observation_id"].isin(train_ids)].copy()
    if train.empty or holdout.empty:
        return pd.DataFrame(), {
            "model_name": "ridge_prior_only_numeric",
            "evaluated": False,
            "reason": "too few chronological rows for a train/holdout regression fit",
        }

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[MODEL_INPUT_COLUMNS], train["handgrip_kg"])
    predictions = make_predictions(
        holdout,
        "ridge_prior_only_numeric",
        "chronological_holdout:last_30pct_per_athlete_min_1",
        lambda rows: pd.Series(model.predict(rows[MODEL_INPUT_COLUMNS]), index=rows.index),
    )
    return predictions, {
        "model_name": "ridge_prior_only_numeric",
        "evaluated": True,
        "training_sample_count": int(len(train)),
        "training_athlete_count": int(train["athlete_id"].nunique()),
        "feature_columns": MODEL_INPUT_COLUMNS,
    }


def metrics_by_split(predictions: pd.DataFrame) -> dict[str, dict[str, dict[str, float | int]]]:
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    if predictions.empty:
        return results
    split_families = predictions["evaluation_fold"].str.split(":", n=1).str[0]
    grouped_predictions = predictions.assign(evaluation_split_family=split_families)
    for (split_family, model_name), group in grouped_predictions.groupby(
        ["evaluation_split_family", "model_name"],
        sort=True,
    ):
        results.setdefault(split_family, {})[model_name] = metric_summary(group)
    return results


def build_metrics(features: pd.DataFrame, primary: pd.DataFrame, predictions: pd.DataFrame, loo_status: dict[str, object], ridge_status: dict[str, object]) -> dict[str, object]:
    return {
        "track": "Track A",
        "row_order_policy": NO_RANDOM_SPLIT_POLICY,
        "dataset_summary": {
            "feature_rows_with_valid_handgrip": int(len(features)),
            "primary_modeling_rows": int(len(primary)),
            "primary_athletes": int(primary["athlete_id"].nunique()),
            "excluded_valid_handgrip_rows": int((~features["is_primary_modeling_row"].astype(bool)).sum()),
            "minimum_prior_valid_handgrips": 3,
        },
        "feature_manifest": {
            "model_input_columns": MODEL_INPUT_COLUMNS,
            "target_columns": ["handgrip_kg", "target_pct_prior_median_baseline"],
            "provenance_columns": [
                "observation_id",
                "athlete_id",
                "raw_daily_row_index",
                "athlete_row_index",
                "weekday",
                "duplicate_state",
                "exclude_from_modeling_default",
            ],
            "leakage_labels": {
                "prior_valid_handgrip_count": "prior_only_source_row_order_assumed_temporal",
                "prior_expanding_handgrip_mean": "prior_only_source_row_order_assumed_temporal",
                "prior_expanding_handgrip_median": "prior_only_source_row_order_assumed_temporal",
                "prior_expanding_handgrip_std": "prior_only_source_row_order_assumed_temporal",
                "retrospective_descriptive_baselines": "prohibited_as_model_inputs",
                "same_row_recovery_lifestyle_activity": "prohibited_until_pre_outcome_timing_is_verified",
            },
        },
        "evaluation": {
            "leave_one_athlete_out": loo_status,
            "chronological_holdout": {
                "assumption": "source row order is treated as temporary temporal order",
                "split": "last 30 percent of eligible rows per athlete, minimum one row",
                "random_split_used": False,
            },
            "ridge_regression": ridge_status,
        },
        "metrics": metrics_by_split(predictions),
    }


def write_outputs(metrics: dict[str, object], predictions: pd.DataFrame, metrics_path: Path, predictions_path: Path) -> tuple[Path, Path]:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    output_predictions = predictions[
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
    ].copy()
    output_predictions.to_csv(predictions_path, index=False)
    return metrics_path, predictions_path


def evaluate_and_write(
    input_path: Path = TRACK_A_OUTPUT,
    metrics_path: Path = METRICS_OUTPUT,
    predictions_path: Path = PREDICTIONS_OUTPUT,
) -> EvaluationResult:
    features = load_features(input_path)
    primary = primary_modeling_rows(features)
    loo_predictions, loo_status = leave_one_athlete_out_predictions(primary)
    holdout_predictions = chronological_holdout_predictions(primary)
    ridge_predictions, ridge_status = sklearn_regression_predictions(primary)
    predictions = pd.concat(
        [frame for frame in [loo_predictions, holdout_predictions, ridge_predictions] if not frame.empty],
        ignore_index=True,
    )
    metrics = build_metrics(features, primary, predictions, loo_status, ridge_status)
    written_metrics, written_predictions = write_outputs(metrics, predictions, metrics_path, predictions_path)
    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
        metrics_path=written_metrics,
        predictions_path=written_predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Track A prior-only handgrip baselines.")
    parser.add_argument("--input", default=str(TRACK_A_OUTPUT), help="Path to Track A features CSV")
    parser.add_argument("--metrics-output", default=str(METRICS_OUTPUT), help="Path for metrics JSON")
    parser.add_argument("--predictions-output", default=str(PREDICTIONS_OUTPUT), help="Path for predictions CSV")
    args = parser.parse_args()

    result = evaluate_and_write(Path(args.input), Path(args.metrics_output), Path(args.predictions_output))
    print("Track A evaluation")
    print(f"metrics: {result.metrics_path}")
    print(f"predictions: {result.predictions_path}")
    print(f"primary_modeling_rows: {result.metrics['dataset_summary']['primary_modeling_rows']}")


if __name__ == "__main__":
    main()
