"""Build exploratory Track B REST features with explicit timing labels."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.data.audit_rest import csv_members, read_csv_from_zip, validate_actigraphy_sequence
from src.data.build_observations import DAILY_MEMBER, RAW_ZIP
from src.features.build_track_a import (
    MODEL_INPUT_COLUMNS as TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS,
    build_track_a_features,
)


OBSERVATIONS_INPUT = Path("data/processed/observations.csv")
TRACK_B_OUTPUT = Path("data/processed/track_b_features.csv")
TRACK_B_SUMMARY_OUTPUT = Path("reports/track_b_feature_summary.json")

TIMING_PRIOR_ONLY = "prior_only_prospective_assumption"
TIMING_UNVERIFIED = "timing_unverified"
TIMING_UNSAFE = "unsafe_for_prospective_modeling"
ACTIGRAPHY_JOIN_ASSUMPTION = "weekday_only_unverified"

OBJECTIVE_SLEEP_COLUMNS = {
    "total_sleep_time": "objective_total_sleep_time",
    "sleep_intervals": "objective_sleep_intervals",
    "longest_interval_duration": "objective_longest_interval_duration",
    "sleep period mean": "objective_sleep_period_mean",
    "sleep period standard deviation": "objective_sleep_period_standard_deviation",
    "sleep period msd": "objective_sleep_period_msd",
}

SLEEP_TIMING_COLUMNS = {
    "sleep_onset": "sleep_onset_minutes_after_midnight",
    "sleep_offset": "sleep_offset_minutes_after_midnight",
    "longest_interval_start": "longest_interval_start_minutes_after_midnight",
    "longest_interval_end": "longest_interval_end_minutes_after_midnight",
}

CAFFEINE_COUNT_COLUMNS = {
    "coffee": "coffee_count",
    "cola": "cola_count",
    "tea": "tea_count",
    "espresso": "espresso_count",
}

WELLNESS_COLUMNS = [
    "sleepquality",
    "sleepdura_sr",
    "readiness",
    "fatigue",
    "soreness",
    "soreness_location_1",
    "soreness_location_2",
]

ALGORITHM_COLUMNS = ["CK", "Oakley", "Sadeh", "fourier_HMM", "LSTM"]

DAILY_FEATURE_COLUMNS = (
    list(OBJECTIVE_SLEEP_COLUMNS.values())
    + list(SLEEP_TIMING_COLUMNS.values())
    + [
        "caffeine_mg",
        "caffeine_time_category",
        "coffee_count",
        "cola_count",
        "tea_count",
        "espresso_count",
        "caffeine_present",
    ]
    + WELLNESS_COLUMNS
    + [
        "matchday_category",
        "matchday_present",
        "matchday_is_match_day",
        "matchday_relative_days",
    ]
)

ACTIGRAPHY_FEATURE_COLUMNS = [
    "actigraphy_vm_mean",
    "actigraphy_vm_median",
    "actigraphy_vm_std",
    "actigraphy_vm_max",
    "actigraphy_vm_sum",
    "actigraphy_x_mean",
    "actigraphy_y_mean",
    "actigraphy_z_mean",
    "actigraphy_all_algorithm_nonwear_invalid_fraction",
    "actigraphy_row_count",
    "actigraphy_window_epoch_count",
    "actigraphy_expected_epoch_count",
    "actigraphy_partial_window_flag",
    "actigraphy_source_sequence_valid",
    "actigraphy_source_sequence_status",
    "actigraphy_source_sequence_unexpected_transitions",
    "actigraphy_source_sequence_missing_samples",
    "actigraphy_join_assumption",
    "actigraphy_weekday_window_id",
    "actigraphy_shared_provisional_window",
    "actigraphy_shared_provisional_window_row_count",
]

for _algorithm in ALGORITHM_COLUMNS:
    ACTIGRAPHY_FEATURE_COLUMNS.extend(
        [
            f"actigraphy_{_algorithm}_sleep_fraction",
            f"actigraphy_{_algorithm}_wake_fraction",
            f"actigraphy_{_algorithm}_nonwear_fraction",
        ]
    )

TRACK_B_FEATURE_COLUMNS = (
    TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS + DAILY_FEATURE_COLUMNS + ACTIGRAPHY_FEATURE_COLUMNS
)

FORBIDDEN_TRACK_B_COLUMNS = {
    "Comments (optional)",
    "retrospective_descriptive_full_athlete_median_handgrip",
    "retrospective_descriptive_performance_pct_full_athlete_median",
    "retrospective_descriptive_ui_only",
}


@dataclass(frozen=True)
class TrackBBuildResult:
    features: pd.DataFrame
    output_path: Path
    summary_path: Path
    summary: dict[str, object]


def parse_time_to_minutes(series: pd.Series) -> pd.Series:
    timedeltas = pd.to_timedelta(series, errors="coerce")
    return timedeltas.dt.total_seconds().div(60)


def parse_count_with_four_or_more(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    normalized = normalized.str.replace(",", ".", regex=False)
    normalized = normalized.str.replace(r"^4\s+or\s+more$", "4", regex=True)
    return pd.to_numeric(normalized, errors="coerce")


def encode_matchday(value: object) -> dict[str, object]:
    if pd.isna(value) or str(value).strip() == "":
        return {
            "matchday_category": pd.NA,
            "matchday_present": False,
            "matchday_is_match_day": False,
            "matchday_relative_days": pd.NA,
        }
    text = str(value).strip()
    match = re.fullmatch(r"MD(?:(?P<sign>[+-])(?P<days>\d+))?", text)
    relative_days = pd.NA
    is_match_day = text == "MD"
    if match and match.group("days"):
        sign = -1 if match.group("sign") == "-" else 1
        relative_days = sign * int(match.group("days"))
    elif is_match_day:
        relative_days = 0
    return {
        "matchday_category": text,
        "matchday_present": True,
        "matchday_is_match_day": is_match_day,
        "matchday_relative_days": relative_days,
    }


def build_daily_features(zip_path: Path = RAW_ZIP) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        daily = read_csv_from_zip(zf, DAILY_MEMBER)

    features = pd.DataFrame({"raw_daily_row_index": range(len(daily))})
    for source, output in OBJECTIVE_SLEEP_COLUMNS.items():
        features[output] = pd.to_numeric(daily[source], errors="coerce")
    for source, output in SLEEP_TIMING_COLUMNS.items():
        features[output] = parse_time_to_minutes(daily[source])
    for source, output in CAFFEINE_COUNT_COLUMNS.items():
        features[output] = parse_count_with_four_or_more(daily[source])

    features["caffeine_mg"] = pd.to_numeric(daily["caffeine_mg"], errors="coerce")
    features["caffeine_time_category"] = daily["caffeine_time"].astype("string")
    features["caffeine_present"] = (
        features["caffeine_mg"].fillna(0).gt(0)
        | features[list(CAFFEINE_COUNT_COLUMNS.values())].fillna(0).gt(0).any(axis=1)
        | features["caffeine_time_category"].notna()
    )

    for column in WELLNESS_COLUMNS:
        features[column] = pd.to_numeric(daily[column], errors="coerce")

    matchday = pd.DataFrame([encode_matchday(value) for value in daily["matchday"]])
    features = pd.concat([features, matchday], axis=1)
    return features


def sequence_status(validation: dict[str, object]) -> tuple[bool, str]:
    is_valid = (
        validation["total_rows"] == validation["expected_rows_given_observed_duration"]
        and validation["duplicate_sequential_samples"] == 0
        and validation["missing_sequential_samples_inferable"] == 0
        and validation["unexpected_weekday_time_transitions"] == 0
    )
    return is_valid, "valid_30s_sequence" if is_valid else "invalid_sequence"


def summarize_algorithm(group: pd.DataFrame, algorithm: str) -> dict[str, float]:
    labels = group[algorithm].astype("string").str.lower()
    denominator = labels.notna().sum()
    if denominator == 0:
        return {
            f"actigraphy_{algorithm}_sleep_fraction": pd.NA,
            f"actigraphy_{algorithm}_wake_fraction": pd.NA,
            f"actigraphy_{algorithm}_nonwear_fraction": pd.NA,
        }
    return {
        f"actigraphy_{algorithm}_sleep_fraction": float(labels.eq("s").sum() / denominator),
        f"actigraphy_{algorithm}_wake_fraction": float(labels.eq("w").sum() / denominator),
        f"actigraphy_{algorithm}_nonwear_fraction": float(labels.eq("n").sum() / denominator),
    }


def build_actigraphy_weekday_features(zip_path: Path = RAW_ZIP) -> pd.DataFrame:
    return _build_actigraphy_weekday_features_cached(zip_path.as_posix()).copy()


@lru_cache(maxsize=4)
def _build_actigraphy_weekday_features_cached(zip_path: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(Path(zip_path)) as zf:
        members = sorted(member for member in csv_members(zf) if member.startswith("REST/actigraphy/"))
        for member in members:
            athlete_id = Path(member).stem
            actigraphy = read_csv_from_zip(zf, member)
            validation = validate_actigraphy_sequence(actigraphy[["weekday", "time"]])
            is_valid, status = sequence_status(validation)
            actigraphy = actigraphy.reset_index(drop=True)
            actigraphy["provisional_sequence_index"] = actigraphy.index
            actigraphy["provisional_elapsed_seconds"] = actigraphy["provisional_sequence_index"] * 30

            for weekday, group in actigraphy.groupby("weekday", sort=False):
                row = {
                    "athlete_id": athlete_id,
                    "weekday": weekday,
                    "actigraphy_vm_mean": float(group["vm"].mean()),
                    "actigraphy_vm_median": float(group["vm"].median()),
                    "actigraphy_vm_std": float(group["vm"].std(ddof=1)),
                    "actigraphy_vm_max": float(group["vm"].max()),
                    "actigraphy_vm_sum": float(group["vm"].sum()),
                    "actigraphy_x_mean": float(group["x"].mean()),
                    "actigraphy_y_mean": float(group["y"].mean()),
                    "actigraphy_z_mean": float(group["z"].mean()),
                    "actigraphy_row_count": int(len(group)),
                    "actigraphy_window_epoch_count": int(len(group)),
                    "actigraphy_expected_epoch_count": int(len(group)),
                    "actigraphy_partial_window_flag": False,
                    "actigraphy_source_sequence_valid": bool(is_valid),
                    "actigraphy_source_sequence_status": status,
                    "actigraphy_source_sequence_unexpected_transitions": int(
                        validation["unexpected_weekday_time_transitions"]
                    ),
                    "actigraphy_source_sequence_missing_samples": int(
                        validation["missing_sequential_samples_inferable"]
                    ),
                    "actigraphy_join_assumption": ACTIGRAPHY_JOIN_ASSUMPTION,
                    "actigraphy_weekday_window_id": f"{athlete_id}|{weekday}|{ACTIGRAPHY_JOIN_ASSUMPTION}",
                }
                algorithm_values = group[ALGORITHM_COLUMNS].astype("string").apply(lambda col: col.str.lower())
                row["actigraphy_all_algorithm_nonwear_invalid_fraction"] = float(
                    (algorithm_values.isna() | algorithm_values.eq("n")).sum().sum()
                    / algorithm_values.size
                )
                for algorithm in ALGORITHM_COLUMNS:
                    row.update(summarize_algorithm(group, algorithm))
                rows.append(row)

    return pd.DataFrame(rows)


def build_manifest(features: pd.DataFrame) -> dict[str, object]:
    metadata = []
    for column in TRACK_B_FEATURE_COLUMNS:
        if column in TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS:
            group = "prior_history"
            timing = TIMING_PRIOR_ONLY
            notes = "Track A model-safe prior-only field; relies on source row order as temporal."
        elif column in DAILY_FEATURE_COLUMNS:
            if column.startswith("objective_"):
                group = "daily_sleep_objective"
            elif "minutes_after_midnight" in column:
                group = "sleep_timing"
            elif column.startswith(("coffee_", "cola_", "tea_", "espresso_", "caffeine_")):
                group = "caffeine"
            elif column.startswith("matchday_"):
                group = "match_context"
            else:
                group = "wellness"
            timing = TIMING_UNVERIFIED
            notes = "Same-row daily-response signal; collection timing relative to handgrip is unverified."
        elif column in ACTIGRAPHY_FEATURE_COLUMNS:
            group = "actigraphy_weekday_window"
            timing = TIMING_UNVERIFIED
            notes = "Provisional athlete + weekday aggregation only; not a verified athlete-day window."
        else:
            group = "unknown"
            timing = TIMING_UNSAFE
            notes = "Unexpected feature column."

        missing_count = int(features[column].isna().sum())
        metadata.append(
            {
                "feature_name": column,
                "source_group": group,
                "timing_status": timing,
                "missing_count": missing_count,
                "missing_percent": round(float(features[column].isna().mean() * 100), 2),
                "dtype": str(features[column].dtype),
                "notes": notes,
            }
        )

    return {
        "dataset": "track_b_features",
        "row_count": int(len(features)),
        "column_count": int(features.shape[1]),
        "feature_count": len(TRACK_B_FEATURE_COLUMNS),
        "track_a_prior_only_feature_count": len(TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS),
        "track_b_added_feature_count": len(TRACK_B_FEATURE_COLUMNS)
        - len(TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS),
        "timing_policy": "Track B exploratory features are timing-unverified except prior-only history.",
        "actigraphy_join_assumption": ACTIGRAPHY_JOIN_ASSUMPTION,
        "features": metadata,
    }


def build_track_b_features(
    observations: pd.DataFrame,
    zip_path: Path = RAW_ZIP,
) -> tuple[pd.DataFrame, dict[str, object]]:
    track_a = build_track_a_features(observations)
    primary = track_a.loc[track_a["is_primary_modeling_row"].astype(bool)].copy()

    daily_features = build_daily_features(zip_path)
    actigraphy_features = build_actigraphy_weekday_features(zip_path)

    features = primary.merge(daily_features, on="raw_daily_row_index", how="left", validate="one_to_one")
    features = features.merge(
        actigraphy_features,
        on=["athlete_id", "weekday"],
        how="left",
        validate="many_to_one",
    )

    shared_counts = features.groupby("actigraphy_weekday_window_id", dropna=False)["observation_id"].transform("size")
    features["actigraphy_shared_provisional_window_row_count"] = shared_counts.astype("Int64")
    features["actigraphy_shared_provisional_window"] = shared_counts.gt(1)

    ordered_columns = list(track_a.columns) + [
        column for column in TRACK_B_FEATURE_COLUMNS if column not in track_a.columns
    ]
    features = features[ordered_columns].reset_index(drop=True)
    summary = build_manifest(features)
    validate_track_b_features(features, summary, track_a)
    return features, summary


def validate_track_b_features(
    features: pd.DataFrame,
    summary: dict[str, object],
    track_a_features: pd.DataFrame | None = None,
) -> dict[str, object]:
    errors = []
    if track_a_features is not None:
        expected_rows = int(track_a_features["is_primary_modeling_row"].astype(bool).sum())
        if len(features) != expected_rows:
            errors.append(f"expected {expected_rows} Track B rows, found {len(features)}")
    if not features["is_primary_modeling_row"].astype(bool).all():
        errors.append("Track B includes non-primary Track A rows")
    missing_track_a_prior = sorted(set(TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS) - set(features.columns))
    if missing_track_a_prior:
        errors.append(f"missing Track A prior-only fields: {missing_track_a_prior}")
    forbidden_present = sorted(FORBIDDEN_TRACK_B_COLUMNS & set(features.columns))
    if forbidden_present:
        errors.append(f"forbidden Track B columns are present: {forbidden_present}")
    if "Comments (optional)" in features.columns or any("comment" in col.lower() for col in features.columns):
        errors.append("free-text comment fields are present")

    manifest_features = {entry["feature_name"] for entry in summary["features"]}
    missing_manifest = sorted(set(TRACK_B_FEATURE_COLUMNS) - manifest_features)
    if missing_manifest:
        errors.append(f"features missing manifest entries: {missing_manifest}")
    if any(not entry.get("timing_status") for entry in summary["features"]):
        errors.append("manifest entries missing timing status")

    actigraphy_columns = [column for column in ACTIGRAPHY_FEATURE_COLUMNS if column in features.columns]
    if actigraphy_columns and not features["actigraphy_join_assumption"].eq(ACTIGRAPHY_JOIN_ASSUMPTION).all():
        errors.append("actigraphy rows lack the weekday-only unverified assumption flag")
    if not features["observation_id"].is_unique:
        errors.append("Track B observation rows were collapsed or duplicated unexpectedly")

    if errors:
        raise ValueError("Track B validation failed: " + "; ".join(errors))
    return {
        "rows": int(len(features)),
        "columns": int(features.shape[1]),
        "features_in_manifest": len(summary["features"]),
        "shared_provisional_actigraphy_windows": int(features["actigraphy_shared_provisional_window"].sum()),
    }


def write_track_b_features(features: pd.DataFrame, output_path: Path = TRACK_B_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path


def write_summary(summary: dict[str, object], output_path: Path = TRACK_B_SUMMARY_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_and_write(
    input_path: Path = OBSERVATIONS_INPUT,
    output_path: Path = TRACK_B_OUTPUT,
    summary_path: Path = TRACK_B_SUMMARY_OUTPUT,
    zip_path: Path = RAW_ZIP,
) -> TrackBBuildResult:
    observations = pd.read_csv(input_path)
    features, summary = build_track_b_features(observations, zip_path)
    written_output = write_track_b_features(features, output_path)
    written_summary = write_summary(summary, summary_path)
    return TrackBBuildResult(
        features=features,
        output_path=written_output,
        summary_path=written_summary,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exploratory Track B features.")
    parser.add_argument("--input", default=str(OBSERVATIONS_INPUT), help="Path to observations.csv")
    parser.add_argument("--output", default=str(TRACK_B_OUTPUT), help="Path for Track B features CSV")
    parser.add_argument(
        "--summary-output",
        default=str(TRACK_B_SUMMARY_OUTPUT),
        help="Path for Track B feature summary JSON",
    )
    parser.add_argument("--zip", default=str(RAW_ZIP), help="Path to raw REST.zip")
    args = parser.parse_args()

    result = build_and_write(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_path=Path(args.summary_output),
        zip_path=Path(args.zip),
    )
    validation = validate_track_b_features(result.features, result.summary)
    print("Track B feature validation")
    print(f"output: {result.output_path}")
    print(f"summary: {result.summary_path}")
    for key, value in validation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
