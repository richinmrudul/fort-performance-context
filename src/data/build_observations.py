"""Build canonical REST daily-response observations.

This scaffold reads the raw REST ZIP in place and writes one canonical
observation row per raw daily-response row. It does not extract the archive,
collapse duplicates, join same-row recovery/lifestyle/activity features, or
create model-ready tables.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from src.data.audit_rest import csv_members, read_csv_from_zip, validate_actigraphy_sequence
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from audit_rest import csv_members, read_csv_from_zip, validate_actigraphy_sequence


RAW_ZIP = Path("data/raw/REST.zip")
DAILY_MEMBER = "REST/daily_responses.csv"
QUESTIONNAIRE_MEMBER = "REST/initial_questionnaire.csv"
PROCESSED_DIR = Path("data/processed")
CSV_OUTPUT = PROCESSED_DIR / "observations.csv"
PARQUET_OUTPUT = PROCESSED_DIR / "observations.parquet"
EXPECTED_ROW_COUNT = 370
EXPECTED_VALID_HANDGRIPS = 142
EXPECTED_PRIOR_ELIGIBILITY = {1: 127, 2: 112, 3: 97, 5: 69}

DUPLICATE_KEY_COLUMNS = ["id", "weekday", "sleep_onset", "sleep_offset", "total_sleep_time"]
CAFFEINE_VARIANT_COLUMNS = {
    "cola",
    "coffee",
    "tea",
    "espresso",
    "caffeine_time",
    "caffeine_mg",
    "screentime",
    "energy",
    "Comments (optional)",
}
TARGET_COLUMNS = {"handgrip_kg"}
CONTEXT_COLUMNS = {
    "sleepquality",
    "matchday",
    "fatigue",
    "soreness",
    "soreness_location_1",
    "soreness_location_2",
    "readiness",
    "sleepdura_sr",
    "sleep_intervals",
    "longest_interval_start",
    "longest_interval_end",
    "longest_interval_duration",
    "sleep period mean",
    "sleep period standard deviation",
    "sleep period msd",
}
CORRECTION_PATTERN = r"feil|wrong|forrige|previous|skrev"


@dataclass(frozen=True)
class BuildResult:
    observations: pd.DataFrame
    output_path: Path
    output_format: str


def parse_handgrip(series: pd.Series) -> pd.Series:
    """Parse REST comma-decimal handgrip values as kilograms."""
    return pd.to_numeric(series.astype("string").str.replace(",", ".", regex=False), errors="coerce")


def make_observation_id(source_archive_path: str, source_member_path: str, raw_daily_row_index: int) -> str:
    payload = (
        f"source_archive_path={source_archive_path}|"
        f"source_member_path={source_member_path}|"
        f"raw_daily_row_index={raw_daily_row_index}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"rest_daily_{digest}"


def make_duplicate_group_id(group_key: Iterable[object]) -> str:
    parts = ["<NA>" if pd.isna(value) else str(value) for value in group_key]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"rest_dup_{digest}"


def has_correction_hint(frame: pd.DataFrame) -> bool:
    if "Comments (optional)" not in frame:
        return False
    return bool(frame["Comments (optional)"].astype("string").str.contains(CORRECTION_PATTERN, case=False, na=False).any())


def varying_columns(frame: pd.DataFrame) -> set[str]:
    varying = set()
    for column in frame.columns:
        distinct_values = frame[column].astype("string").fillna("<NA>").nunique(dropna=False)
        if distinct_values > 1:
            varying.add(column)
    return varying


def classify_duplicate_group(group: pd.DataFrame) -> str:
    if len(group) <= 1:
        return "not_duplicate"
    if group.duplicated(keep=False).all():
        return "exact_duplicate"
    if has_correction_hint(group):
        return "correction_hint"

    varying = varying_columns(group)
    if varying & TARGET_COLUMNS:
        return "conflicting_target"
    if varying & CONTEXT_COLUMNS:
        return "conflicting_context"
    if varying and varying <= CAFFEINE_VARIANT_COLUMNS:
        return "same_sleep_window_caffeine_variant"
    return "unresolved_duplicate"


def build_duplicate_fields(daily: pd.DataFrame) -> pd.DataFrame:
    duplicate_frame = pd.DataFrame(index=daily.index)
    duplicate_frame["duplicate_heuristic_group_id"] = pd.NA
    duplicate_frame["duplicate_group_size"] = 1
    duplicate_frame["duplicate_state"] = "not_duplicate"

    grouped = daily.groupby(DUPLICATE_KEY_COLUMNS, dropna=False, sort=False)
    for group_key, index in grouped.groups.items():
        group = daily.loc[index]
        if len(group) <= 1:
            continue
        duplicate_frame.loc[index, "duplicate_heuristic_group_id"] = make_duplicate_group_id(group_key)
        duplicate_frame.loc[index, "duplicate_group_size"] = len(group)
        duplicate_frame.loc[index, "duplicate_state"] = classify_duplicate_group(group)

    duplicate_frame["exclude_from_modeling_default"] = duplicate_frame["duplicate_state"] != "not_duplicate"
    return duplicate_frame


def actigraphy_sequence_statuses(zf: zipfile.ZipFile, actigraphy_members: list[str]) -> dict[str, str]:
    statuses = {}
    for member in actigraphy_members:
        athlete_id = Path(member).stem
        actigraphy = read_csv_from_zip(zf, member, usecols=["weekday", "time"])
        validation = validate_actigraphy_sequence(actigraphy)
        is_valid = (
            validation["total_rows"] == validation["expected_rows_given_observed_duration"]
            and validation["duplicate_sequential_samples"] == 0
            and validation["missing_sequential_samples_inferable"] == 0
            and validation["unexpected_weekday_time_transitions"] == 0
        )
        statuses[athlete_id] = "valid_30s_sequence" if is_valid else "invalid_sequence"
    return statuses


def add_prior_handgrip_fields(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    result["prior_valid_handgrip_count"] = 0
    result["prior_expanding_handgrip_mean"] = pd.NA
    result["prior_expanding_handgrip_median"] = pd.NA
    result["prior_expanding_handgrip_std"] = pd.NA

    for _, group in result.groupby("athlete_id", sort=False):
        prior_values: list[float] = []
        for index, value in group["handgrip_kg"].items():
            valid_prior = pd.Series(prior_values, dtype="float64")
            result.at[index, "prior_valid_handgrip_count"] = len(prior_values)
            if prior_values:
                result.at[index, "prior_expanding_handgrip_mean"] = float(valid_prior.mean())
                result.at[index, "prior_expanding_handgrip_median"] = float(valid_prior.median())
            if len(prior_values) >= 2:
                result.at[index, "prior_expanding_handgrip_std"] = float(valid_prior.std(ddof=1))
            if pd.notna(value):
                prior_values.append(float(value))

    for threshold in EXPECTED_PRIOR_ELIGIBILITY:
        result[f"eligible_prior_{threshold}"] = (
            result["has_valid_handgrip"] & (result["prior_valid_handgrip_count"] >= threshold)
        )
    return result


def add_retrospective_descriptive_fields(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    medians = result.groupby("athlete_id", sort=False)["handgrip_kg"].transform("median")
    result["retrospective_descriptive_full_athlete_median_handgrip"] = medians
    result["retrospective_descriptive_performance_pct_full_athlete_median"] = (
        result["handgrip_kg"] / medians * 100
    )
    result["retrospective_descriptive_ui_only"] = True
    return result


def build_observations(zip_path: Path = RAW_ZIP) -> pd.DataFrame:
    source_archive_path = zip_path.as_posix()
    with zipfile.ZipFile(zip_path) as zf:
        members = csv_members(zf)
        daily = read_csv_from_zip(zf, DAILY_MEMBER)
        questionnaire = read_csv_from_zip(zf, QUESTIONNAIRE_MEMBER, sep=";")
        actigraphy_members = sorted(member for member in members if member.startswith("REST/actigraphy/"))
        actigraphy_ids = {Path(member).stem for member in actigraphy_members}
        actigraphy_status = actigraphy_sequence_statuses(zf, actigraphy_members)

    observations = pd.DataFrame(index=daily.index)
    observations["source_archive_path"] = source_archive_path
    observations["source_member_path"] = DAILY_MEMBER
    observations["raw_daily_row_index"] = range(len(daily))
    observations["athlete_id"] = daily["id"].astype("string")
    observations["athlete_row_index"] = observations.groupby("athlete_id", sort=False).cumcount()
    observations["weekday"] = daily["weekday"].astype("string")
    observations["observation_id"] = [
        make_observation_id(source_archive_path, DAILY_MEMBER, int(row_index))
        for row_index in observations["raw_daily_row_index"]
    ]

    observations["raw_handgrip_kg"] = daily["handgrip_kg"].astype("string")
    observations["handgrip_kg"] = parse_handgrip(daily["handgrip_kg"])
    raw_present = daily["handgrip_kg"].notna() & daily["handgrip_kg"].astype("string").str.strip().ne("")
    observations["handgrip_parse_status"] = "missing"
    observations.loc[observations["handgrip_kg"].notna(), "handgrip_parse_status"] = "valid"
    observations.loc[raw_present & observations["handgrip_kg"].isna(), "handgrip_parse_status"] = "parse_error"
    observations["has_valid_handgrip"] = observations["handgrip_parse_status"] == "valid"

    observations = pd.concat([observations, build_duplicate_fields(daily)], axis=1)

    questionnaire_ids = set(questionnaire["id"].dropna().astype(str))
    observations["questionnaire_linked"] = observations["athlete_id"].astype(str).isin(questionnaire_ids)
    observations["questionnaire_link_status"] = observations["questionnaire_linked"].map({True: "linked", False: "unlinked"})
    observations["actigraphy_linked"] = observations["athlete_id"].astype(str).isin(actigraphy_ids)
    observations["actigraphy_link_status"] = observations["actigraphy_linked"].map({True: "linked", False: "unlinked"})
    observations["actigraphy_sequence_validation_status"] = observations["athlete_id"].astype(str).map(
        actigraphy_status
    )
    observations.loc[
        observations["actigraphy_linked"] & observations["actigraphy_sequence_validation_status"].isna(),
        "actigraphy_sequence_validation_status",
    ] = "not_checked"
    observations.loc[
        ~observations["actigraphy_linked"],
        "actigraphy_sequence_validation_status",
    ] = "missing_file"

    observations["source_row_order_temporal_assumption"] = True
    observations = add_prior_handgrip_fields(observations)
    observations = add_retrospective_descriptive_fields(observations)

    return observations.reset_index(drop=True)


def validate_observations(observations: pd.DataFrame) -> dict[str, object]:
    summary = {
        "rows": len(observations),
        "unique_observation_ids": int(observations["observation_id"].nunique()),
        "valid_handgrips": int(observations["has_valid_handgrip"].sum()),
        "eligible_prior_1": int(observations["eligible_prior_1"].sum()),
        "eligible_prior_2": int(observations["eligible_prior_2"].sum()),
        "eligible_prior_3": int(observations["eligible_prior_3"].sum()),
        "eligible_prior_5": int(observations["eligible_prior_5"].sum()),
        "duplicate_rows": int((observations["duplicate_state"] != "not_duplicate").sum()),
        "duplicate_groups": int(observations["duplicate_heuristic_group_id"].dropna().nunique()),
        "questionnaire_linked_rows": int(observations["questionnaire_linked"].sum()),
        "actigraphy_linked_rows": int(observations["actigraphy_linked"].sum()),
        "actigraphy_valid_sequence_rows": int(
            (observations["actigraphy_sequence_validation_status"] == "valid_30s_sequence").sum()
        ),
    }

    errors = []
    if summary["rows"] != EXPECTED_ROW_COUNT:
        errors.append(f"expected {EXPECTED_ROW_COUNT} rows, found {summary['rows']}")
    if summary["unique_observation_ids"] != summary["rows"]:
        errors.append("observation_id values are not unique")
    if summary["valid_handgrips"] != EXPECTED_VALID_HANDGRIPS:
        errors.append(
            f"expected {EXPECTED_VALID_HANDGRIPS} valid handgrips, found {summary['valid_handgrips']}"
        )
    for threshold, expected in EXPECTED_PRIOR_ELIGIBILITY.items():
        key = f"eligible_prior_{threshold}"
        if summary[key] != expected:
            errors.append(f"expected {expected} {key} rows, found {summary[key]}")
    if list(observations["raw_daily_row_index"]) != list(range(len(observations))):
        errors.append("raw_daily_row_index does not preserve source row order")
    if errors:
        raise ValueError("Observation validation failed: " + "; ".join(errors))
    return summary


def parquet_supported() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def write_observations(observations: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if parquet_supported():
        output_path = output_dir / PARQUET_OUTPUT.name
        observations.to_parquet(output_path, index=False)
        return output_path, "parquet"

    output_path = output_dir / CSV_OUTPUT.name
    observations.to_csv(output_path, index=False)
    return output_path, "csv"


def build_and_write(zip_path: Path = RAW_ZIP, output_dir: Path = PROCESSED_DIR) -> BuildResult:
    observations = build_observations(zip_path)
    validate_observations(observations)
    output_path, output_format = write_observations(observations, output_dir)
    return BuildResult(observations=observations, output_path=output_path, output_format=output_format)


def print_validation_summary(summary: dict[str, object], output_path: Path, output_format: str) -> None:
    print("REST canonical observations validation")
    print(f"output: {output_path} ({output_format})")
    print(f"rows: {summary['rows']}")
    print(f"unique_observation_ids: {summary['unique_observation_ids']}")
    print(f"valid_handgrips: {summary['valid_handgrips']}")
    print(
        "prior_eligibility: "
        f">=1 {summary['eligible_prior_1']}, "
        f">=2 {summary['eligible_prior_2']}, "
        f">=3 {summary['eligible_prior_3']}, "
        f">=5 {summary['eligible_prior_5']}"
    )
    print(
        "links: "
        f"questionnaire {summary['questionnaire_linked_rows']}, "
        f"actigraphy {summary['actigraphy_linked_rows']}, "
        f"valid_actigraphy_sequence {summary['actigraphy_valid_sequence_rows']}"
    )
    print(
        "duplicates: "
        f"rows {summary['duplicate_rows']}, "
        f"heuristic_groups {summary['duplicate_groups']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical REST daily-response observations.")
    parser.add_argument("--zip", default=str(RAW_ZIP), help="Path to raw REST.zip")
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR), help="Directory for observations output")
    args = parser.parse_args()

    result = build_and_write(Path(args.zip), Path(args.output_dir))
    summary = validate_observations(result.observations)
    print_validation_summary(summary, result.output_path, result.output_format)


if __name__ == "__main__":
    main()
