"""Build leakage-aware Track A modeling features for REST handgrip baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


OBSERVATIONS_INPUT = Path("data/processed/observations.csv")
TRACK_A_OUTPUT = Path("data/processed/track_a_features.csv")
MIN_PRIOR_VALID_HANDGRIPS = 3

PROVENANCE_COLUMNS = [
    "observation_id",
    "athlete_id",
    "raw_daily_row_index",
    "athlete_row_index",
    "weekday",
    "duplicate_state",
    "exclude_from_modeling_default",
]

PRIOR_ONLY_FEATURE_COLUMNS = [
    "prior_valid_handgrip_count",
    "prior_expanding_handgrip_mean",
    "prior_expanding_handgrip_median",
    "prior_expanding_handgrip_std",
]

TARGET_COLUMNS = [
    "handgrip_kg",
    "target_pct_prior_median_baseline",
]

MODEL_INPUT_COLUMNS = PRIOR_ONLY_FEATURE_COLUMNS.copy()

RETROSPECTIVE_DESCRIPTIVE_PREFIXES = (
    "retrospective_",
    "retrospective_descriptive_",
)

FORBIDDEN_SAME_ROW_FEATURE_COLUMNS = {
    "sleep_onset",
    "sleep_offset",
    "total_sleep_time",
    "sleep_intervals",
    "longest_interval_start",
    "longest_interval_end",
    "longest_interval_duration",
    "sleep period mean",
    "sleep period standard deviation",
    "sleep period msd",
    "screentime",
    "Comments (optional)",
    "energy",
    "cola",
    "coffee",
    "tea",
    "espresso",
    "caffeine_time",
    "caffeine_mg",
    "sleepquality",
    "matchday",
    "fatigue",
    "soreness",
    "soreness_location_1",
    "soreness_location_2",
    "readiness",
    "sleepdura_sr",
}


@dataclass(frozen=True)
class TrackABuildResult:
    features: pd.DataFrame
    output_path: Path
    summary: dict[str, int]


def load_observations(input_path: Path = OBSERVATIONS_INPUT) -> pd.DataFrame:
    return pd.read_csv(input_path)


def build_track_a_features(observations: pd.DataFrame) -> pd.DataFrame:
    required_columns = set(PROVENANCE_COLUMNS + PRIOR_ONLY_FEATURE_COLUMNS + ["handgrip_kg"])
    missing_columns = sorted(required_columns - set(observations.columns))
    if missing_columns:
        raise ValueError(f"observations are missing required columns: {missing_columns}")

    features = observations.loc[observations["handgrip_kg"].notna(), PROVENANCE_COLUMNS + PRIOR_ONLY_FEATURE_COLUMNS + ["handgrip_kg"]].copy()
    features["target_pct_prior_median_baseline"] = (
        100 * features["handgrip_kg"] / features["prior_expanding_handgrip_median"]
    )
    features["is_primary_modeling_row"] = (
        ~features["exclude_from_modeling_default"].astype(bool)
        & (features["prior_valid_handgrip_count"] >= MIN_PRIOR_VALID_HANDGRIPS)
        & features["prior_expanding_handgrip_median"].notna()
        & features["target_pct_prior_median_baseline"].notna()
    )
    features["track_a_timing_label"] = "prior_only_source_row_order_assumed_temporal"
    features["track_a_leakage_label"] = "no_same_row_recovery_lifestyle_activity_or_retrospective_baseline_inputs"

    ordered_columns = (
        PROVENANCE_COLUMNS
        + TARGET_COLUMNS
        + PRIOR_ONLY_FEATURE_COLUMNS
        + [
            "is_primary_modeling_row",
            "track_a_timing_label",
            "track_a_leakage_label",
        ]
    )
    return features[ordered_columns].reset_index(drop=True)


def validate_track_a_features(features: pd.DataFrame) -> dict[str, int]:
    errors = []
    if features["handgrip_kg"].isna().any():
        errors.append("Track A feature rows include invalid handgrip targets")

    primary = features.loc[features["is_primary_modeling_row"]]
    if (primary["prior_valid_handgrip_count"] < MIN_PRIOR_VALID_HANDGRIPS).any():
        errors.append("primary Track A rows include fewer than 3 prior valid handgrips")
    if primary["exclude_from_modeling_default"].astype(bool).any():
        errors.append("primary Track A rows include default-excluded duplicate/conflict rows")

    forbidden_present = sorted(FORBIDDEN_SAME_ROW_FEATURE_COLUMNS & set(features.columns))
    if forbidden_present:
        errors.append(f"forbidden same-row fields are present: {forbidden_present}")

    retrospective_inputs = [
        column
        for column in MODEL_INPUT_COLUMNS
        if column.startswith(RETROSPECTIVE_DESCRIPTIVE_PREFIXES)
    ]
    if retrospective_inputs:
        errors.append(f"retrospective descriptive model inputs are present: {retrospective_inputs}")

    if errors:
        raise ValueError("Track A validation failed: " + "; ".join(errors))

    return {
        "valid_handgrip_rows": int(len(features)),
        "primary_modeling_rows": int(primary.shape[0]),
        "primary_athletes": int(primary["athlete_id"].nunique()),
        "excluded_valid_handgrip_rows": int((~features["is_primary_modeling_row"]).sum()),
    }


def write_track_a_features(features: pd.DataFrame, output_path: Path = TRACK_A_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path


def build_and_write(input_path: Path = OBSERVATIONS_INPUT, output_path: Path = TRACK_A_OUTPUT) -> TrackABuildResult:
    observations = load_observations(input_path)
    features = build_track_a_features(observations)
    summary = validate_track_a_features(features)
    written_path = write_track_a_features(features, output_path)
    return TrackABuildResult(features=features, output_path=written_path, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track A prior-only handgrip features.")
    parser.add_argument("--input", default=str(OBSERVATIONS_INPUT), help="Path to observations.csv")
    parser.add_argument("--output", default=str(TRACK_A_OUTPUT), help="Path for Track A features CSV")
    args = parser.parse_args()

    result = build_and_write(Path(args.input), Path(args.output))
    print("Track A feature validation")
    print(f"output: {result.output_path}")
    for key, value in result.summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
