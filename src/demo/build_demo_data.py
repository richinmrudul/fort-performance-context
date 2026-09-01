"""Build privacy-safe static data for the Fort product demo.

The adapter reads existing processed/model outputs only. It does not train,
score, mutate raw data, or infer calendar dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TRACK_A_FEATURES = Path("data/processed/track_a_features.csv")
TRACK_B_FEATURES = Path("data/processed/track_b_features.csv")
TRACK_A_PREDICTIONS = Path("reports/track_a_predictions.csv")
TRACK_A_METRICS = Path("reports/track_a_metrics.json")
TRACK_B_TIMING_AUDIT = Path("reports/track_b_timing_audit.json")
MODEL_COMPARISON = Path("reports/model_comparison.json")
DEFAULT_OUTPUT = Path("frontend/public/data/demo.json")

SOURCE_PATHS = (
    TRACK_A_FEATURES,
    TRACK_B_FEATURES,
    TRACK_A_PREDICTIONS,
    TRACK_A_METRICS,
    TRACK_B_TIMING_AUDIT,
    MODEL_COMPARISON,
)

MIN_REQUIRED_PRIOR_FALLBACK = 3
EXPECTED_MODEL_NAME = "prior_expanding_mean"
CHRONOLOGICAL_FOLD_PREFIX = "chronological_holdout:"

TRACK_A_REQUIRED_COLUMNS = {
    "observation_id",
    "athlete_id",
    "raw_daily_row_index",
    "athlete_row_index",
    "weekday",
    "handgrip_kg",
    "prior_valid_handgrip_count",
    "prior_expanding_handgrip_mean",
    "prior_expanding_handgrip_median",
    "prior_expanding_handgrip_std",
    "is_primary_modeling_row",
}

TRACK_B_REQUIRED_COLUMNS = TRACK_A_REQUIRED_COLUMNS | {
    "objective_total_sleep_time",
    "caffeine_mg",
    "caffeine_time_category",
    "sleepquality",
    "sleepdura_sr",
    "readiness",
    "fatigue",
    "soreness",
    "matchday_category",
    "actigraphy_vm_mean",
    "actigraphy_row_count",
    "actigraphy_LSTM_sleep_fraction",
}

PREDICTION_REQUIRED_COLUMNS = {
    "observation_id",
    "athlete_id",
    "y_true_handgrip_kg",
    "y_pred_handgrip_kg",
    "residual_kg",
    "evaluation_fold",
    "model_name",
}

PRIVATE_IDENTIFIER_COLUMNS = {"observation_id", "athlete_id"}


@dataclass(frozen=True)
class SourceFrames:
    track_a: pd.DataFrame
    track_b: pd.DataFrame
    predictions: pd.DataFrame
    track_a_metrics: dict[str, Any]
    timing_audit: dict[str, Any]
    model_comparison: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_sources() -> SourceFrames:
    for path in SOURCE_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"required source output is missing: {path.as_posix()}")

    track_a = pd.read_csv(TRACK_A_FEATURES)
    track_b = pd.read_csv(TRACK_B_FEATURES)
    predictions = pd.read_csv(TRACK_A_PREDICTIONS)
    validate_columns(track_a, TRACK_A_REQUIRED_COLUMNS, TRACK_A_FEATURES.as_posix())
    validate_columns(track_b, TRACK_B_REQUIRED_COLUMNS, TRACK_B_FEATURES.as_posix())
    validate_columns(predictions, PREDICTION_REQUIRED_COLUMNS, TRACK_A_PREDICTIONS.as_posix())
    return SourceFrames(
        track_a=track_a,
        track_b=track_b,
        predictions=predictions,
        track_a_metrics=load_json(TRACK_A_METRICS),
        timing_audit=load_json(TRACK_B_TIMING_AUDIT),
        model_comparison=load_json(MODEL_COMPARISON),
    )


def finite_number(value: Any, digits: int | None = None) -> float | int | None:
    if pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if digits is not None:
        number = round(number, digits)
    if number.is_integer():
        return int(number)
    return number


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def assert_finite_json(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("demo JSON contains a non-finite float")


def minimum_prior_count(metrics: dict[str, Any]) -> int:
    return int(
        metrics.get("dataset_summary", {}).get(
            "minimum_prior_valid_handgrips",
            MIN_REQUIRED_PRIOR_FALLBACK,
        )
    )


def expected_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.loc[predictions["model_name"] == EXPECTED_MODEL_NAME].copy()
    if rows.empty:
        raise ValueError("Track A predictions do not contain prior_expanding_mean rows")
    return rows


def select_prediction_for_observation(predictions: pd.DataFrame, observation_id: str) -> pd.Series:
    rows = expected_prediction_rows(predictions)
    matches = rows.loc[rows["observation_id"] == observation_id].copy()
    if matches.empty:
        raise ValueError(f"selected observation has no {EXPECTED_MODEL_NAME} Track A prediction")
    unique_predictions = matches["y_pred_handgrip_kg"].dropna().unique()
    if len(unique_predictions) != 1:
        raise ValueError("selected observation has inconsistent Track A expected values")
    matches["fold_rank"] = matches["evaluation_fold"].astype(str).str.startswith(CHRONOLOGICAL_FOLD_PREFIX).map(
        {True: 0, False: 1}
    )
    return matches.sort_values(["fold_rank", "evaluation_fold"], kind="mergesort").iloc[0]


def select_default_session(sources: SourceFrames) -> tuple[pd.Series, pd.Series]:
    min_prior = minimum_prior_count(sources.track_a_metrics)
    candidates = sources.track_b.loc[
        sources.track_b["is_primary_modeling_row"].astype(bool)
        & sources.track_b["handgrip_kg"].notna()
        & sources.track_b["prior_expanding_handgrip_mean"].notna()
        & (sources.track_b["prior_valid_handgrip_count"] >= min_prior)
    ].copy()
    if candidates.empty:
        raise ValueError("no eligible Track B row has sufficient prior history for the demo")

    prediction_ids = set(expected_prediction_rows(sources.predictions)["observation_id"])
    candidates = candidates.loc[candidates["observation_id"].isin(prediction_ids)]
    if candidates.empty:
        raise ValueError("eligible rows cannot be joined to Track A predictions")

    selected = candidates.sort_values(["raw_daily_row_index", "observation_id"], kind="mergesort").iloc[-1]
    prediction = select_prediction_for_observation(sources.predictions, str(selected["observation_id"]))
    if str(prediction["athlete_id"]) != str(selected["athlete_id"]):
        raise ValueError("Track A prediction athlete does not match selected row")
    if not math.isclose(float(prediction["y_true_handgrip_kg"]), float(selected["handgrip_kg"]), abs_tol=1e-9):
        raise ValueError("Track A prediction actual value does not match selected row")
    if not math.isclose(
        float(prediction["y_pred_handgrip_kg"]),
        float(selected["prior_expanding_handgrip_mean"]),
        abs_tol=1e-9,
    ):
        raise ValueError("Track A expected value does not match prior_expanding_handgrip_mean")
    return selected, prediction


def privacy_maps(track_b: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    athlete_ids = sorted(str(value) for value in track_b["athlete_id"].dropna().unique())
    athlete_map = {athlete_id: f"athlete-{index:02d}" for index, athlete_id in enumerate(athlete_ids, start=1)}
    athlete_label = {
        athlete_id: f"Athlete {index:02d}" for index, athlete_id in enumerate(athlete_ids, start=1)
    }
    return athlete_map, athlete_label


def trend_rows(track_a: pd.DataFrame, athlete_id: str, selected_observation_id: str) -> list[dict[str, Any]]:
    rows = track_a.loc[
        (track_a["athlete_id"] == athlete_id) & track_a["handgrip_kg"].notna()
    ].sort_values(["athlete_row_index", "raw_daily_row_index"], kind="mergesort")
    output = []
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        expected = finite_number(row.prior_expanding_handgrip_mean, 2)
        prior_count = int(row.prior_valid_handgrip_count)
        output.append(
            {
                "sequence": index,
                "label": f"Observation {index}",
                "actual_kg": finite_number(row.handgrip_kg, 2),
                "expected_kg": expected,
                "difference_kg": (
                    finite_number(float(row.handgrip_kg) - float(row.prior_expanding_handgrip_mean), 2)
                    if expected is not None
                    else None
                ),
                "prior_observation_count": prior_count,
                "history_state": history_state(prior_count, expected),
                "is_selected": str(row.observation_id) == selected_observation_id,
            }
        )
    return output


def history_state(prior_count: int, expected: float | int | None) -> str:
    if prior_count == 0:
        return "no_prior_observations"
    if expected is None or prior_count < MIN_REQUIRED_PRIOR_FALLBACK:
        return "limited_history"
    return "expected_available"


def timing_status(timing_audit: dict[str, Any], group: str) -> str:
    return str(timing_audit["feature_group_classifications"][group]["status"])


def context_item(
    key: str,
    label: str,
    group: str,
    source_column: str,
    value: Any,
    unit: str | None,
    timing_audit: dict[str, Any],
    digits: int | None = None,
) -> dict[str, Any]:
    numeric = finite_number(value, digits) if not isinstance(value, str) else value
    available = numeric is not None and numeric != ""
    return {
        "key": key,
        "label": label,
        "group": group,
        "source_column": source_column,
        "value": numeric,
        "unit": unit,
        "available": available,
        "display_state": "shown_descriptively" if available else "not_available",
        "timing_status": timing_status(timing_audit, group),
        "timing_label": "Timing not verified" if available else "Not available for this observation",
        "interpretation": "Shown for reference, not used in the trusted prediction" if available else "Not available for this observation",
    }


def build_context(row: pd.Series, timing_audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        context_item(
            "objective_sleep",
            "Objective sleep",
            "objective_sleep",
            "objective_total_sleep_time",
            row["objective_total_sleep_time"],
            "min",
            timing_audit,
            0,
        ),
        context_item(
            "caffeine",
            "Caffeine",
            "caffeine",
            "caffeine_mg",
            row["caffeine_mg"],
            "mg",
            timing_audit,
            0,
        ),
        context_item(
            "caffeine_timing",
            "Caffeine timing",
            "caffeine",
            "caffeine_time_category",
            row["caffeine_time_category"] if pd.notna(row["caffeine_time_category"]) else None,
            None,
            timing_audit,
        ),
        context_item(
            "readiness_survey",
            "Readiness survey item",
            "subjective_wellness",
            "readiness",
            row["readiness"],
            "/10",
            timing_audit,
            0,
        ),
        context_item(
            "fatigue",
            "Fatigue survey item",
            "subjective_wellness",
            "fatigue",
            row["fatigue"],
            "/10",
            timing_audit,
            0,
        ),
        context_item(
            "soreness",
            "Soreness survey item",
            "subjective_wellness",
            "soreness",
            row["soreness"],
            "/10",
            timing_audit,
            0,
        ),
        context_item(
            "match_context",
            "Match context",
            "match_context",
            "matchday_category",
            row["matchday_category"] if pd.notna(row["matchday_category"]) else None,
            None,
            timing_audit,
        ),
        context_item(
            "actigraphy_vm",
            "Actigraphy vector magnitude",
            "actigraphy",
            "actigraphy_vm_mean",
            row["actigraphy_vm_mean"],
            "mean",
            timing_audit,
            1,
        ),
        context_item(
            "actigraphy_sleep_fraction",
            "LSTM sleep fraction",
            "actigraphy",
            "actigraphy_LSTM_sleep_fraction",
            row["actigraphy_LSTM_sleep_fraction"],
            "fraction",
            timing_audit,
            3,
        ),
    ]


def source_inventory() -> list[dict[str, Any]]:
    return [
        {
            "path": path.as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in SOURCE_PATHS
        if path.exists()
    ]


def build_demo_payload() -> dict[str, Any]:
    sources = load_sources()
    selected, prediction = select_default_session(sources)
    athlete_map, athlete_labels = privacy_maps(sources.track_b)
    athlete_id = str(selected["athlete_id"])
    actual = float(selected["handgrip_kg"])
    expected = float(prediction["y_pred_handgrip_kg"])
    percent = actual / expected * 100
    difference_kg = actual - expected

    track_a_chrono = sources.track_a_metrics["metrics"]["chronological_holdout"][EXPECTED_MODEL_NAME]
    model_comparison = sources.model_comparison["comparisons"]["chronological_holdout"]

    payload = {
        "schema_version": "2026-09-01.1",
        "generated_by": "src.demo.build_demo_data",
        "demo_mode": "read_only_static_frontend",
        "selection_rule": {
            "name": "latest_eligible_observation_with_sufficient_prior_history",
            "minimum_prior_observations": minimum_prior_count(sources.track_a_metrics),
            "sort_order": ["raw_daily_row_index", "observation_id"],
            "selection_direction": "last",
            "not_selected_by": ["highest_performance", "largest_residual", "best_model_result"],
        },
        "athlete_index": [
            {
                "athlete_key": athlete_map[athlete],
                "label": athlete_labels[athlete],
                "eligible_session_count": int(
                    sources.track_b.loc[
                        (sources.track_b["athlete_id"] == athlete)
                        & sources.track_b["is_primary_modeling_row"].astype(bool)
                    ].shape[0]
                ),
            }
            for athlete in sorted(athlete_map)
        ],
        "selected_session": {
            "athlete_key": athlete_map[athlete_id],
            "athlete_label": athlete_labels[athlete_id],
            "session_key": f"{athlete_map[athlete_id]}-selected",
            "session_label": "Most recent eligible observation",
            "date_label": None,
            "date_state": "no_defensible_date",
            "x_axis_label": "Observation",
            "weekday_label": str(selected["weekday"]) if pd.notna(selected["weekday"]) else None,
            "actual_kg": finite_number(actual, 2),
            "track_a_expected_kg": finite_number(expected, 2),
            "expected_label": "Expected from prior personal history",
            "actual_percent_of_expected": finite_number(percent, 1),
            "difference_kg": finite_number(difference_kg, 2),
            "difference_percent": finite_number(percent - 100, 1),
            "comparison_direction": "above" if difference_kg > 0 else "below" if difference_kg < 0 else "at",
            "prior_observation_count": int(selected["prior_valid_handgrip_count"]),
            "history_state": history_state(int(selected["prior_valid_handgrip_count"]), expected),
            "track_a_prediction_source": {
                "model_name": EXPECTED_MODEL_NAME,
                "evaluation_fold": str(prediction["evaluation_fold"]),
                "source_file": TRACK_A_PREDICTIONS.as_posix(),
            },
        },
        "trend": {
            "athlete_key": athlete_map[athlete_id],
            "axis_policy": "observation_sequence_no_calendar_dates",
            "observations": trend_rows(sources.track_a, athlete_id, str(selected["observation_id"])),
        },
        "context": {
            "headline": "Context recorded for this study observation",
            "timing_summary": "Timing relative to performance was not available",
            "prediction_use": "Shown for reference, not used in the trusted prediction",
            "items": build_context(selected, sources.timing_audit),
        },
        "model_card": {
            "track_a_summary": "Track A uses prior personal handgrip history only.",
            "expected_performance_definition": "prior_expanding_mean row-level prediction from Track A",
            "track_a_chronological_mae_kg": finite_number(track_a_chrono["mae_kg"], 3),
            "track_a_chronological_sample_count": int(track_a_chrono["sample_count"]),
            "track_a_chronological_athlete_count": int(track_a_chrono["athlete_count"]),
            "track_b_summary": "Contextual Track B features were evaluated experimentally and did not improve generalization in this dataset.",
            "track_b_best_contextual_delta_mae_kg_vs_track_a": finite_number(
                model_comparison["delta_mae_kg_vs_track_a_prior_mean"],
                3,
            ),
            "timing_audit_decision": sources.timing_audit["final_decision"],
            "row_order_policy": sources.track_a_metrics["row_order_policy"],
            "future_data_requirements": [
                "timestamped performance",
                "survey submission time",
                "timestamped caffeine",
                "sleep start/end with timezone",
                "exact calendar-date actigraphy alignment",
            ],
        },
        "missing_states": {
            "no_prior_observations": "More personal history is needed",
            "limited_history": "Limited prior history",
            "missing_actual": "Not available for this observation",
            "missing_expected": "Expected value not available",
            "missing_context": "Not available for this observation",
            "timing_unverified": "Timing not verified",
            "no_defensible_date": "Observation sequence shown; no defensible calendar date",
        },
        "provenance": {
            "sources": source_inventory(),
            "private_identifiers_excluded": sorted(PRIVATE_IDENTIFIER_COLUMNS),
            "raw_rest_mutated": False,
            "model_outputs_mutated": False,
            "backend_required": False,
        },
    }
    cleaned = json_clean(payload)
    assert_finite_json(cleaned)
    return cleaned


def write_demo_data(output_path: Path = DEFAULT_OUTPUT) -> Path:
    payload = build_demo_payload()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static data for the read-only Fort product demo.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to write demo JSON")
    args = parser.parse_args()
    output = write_demo_data(Path(args.output))
    print(f"wrote {output.as_posix()}")


if __name__ == "__main__":
    main()
