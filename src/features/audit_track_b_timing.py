"""Temporal-feasibility audit for REST Track B contextual features.

This module is intentionally not a modeling script. It reads the existing raw
ZIP and processed Track B table, classifies timing evidence, and writes a
deterministic JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.audit_rest import csv_members, read_csv_from_zip, validate_actigraphy_sequence
from src.data.build_observations import DAILY_MEMBER, QUESTIONNAIRE_MEMBER, RAW_ZIP


TRACK_B_INPUT = Path("data/processed/track_b_features.csv")
REPORT_OUTPUT = Path("reports/track_b_timing_audit.json")
AUDIT_VERSION = "2026-09-01.1"

VERIFIED_PRE = "verified_pre_performance"
VERIFIED_POST = "verified_post_performance"
DATE_ONLY_AMBIGUOUS = "date_only_ambiguous"
ORDERING_UNVERIFIED = "ordering_unverified"
TIMEZONE_UNVERIFIED = "timezone_unverified"
ATHLETE_DAY_UNVERIFIED = "athlete_day_alignment_unverified"
TIMESTAMP_UNAVAILABLE = "timestamp_unavailable"
NOT_APPLICABLE = "not_applicable"

FINAL_METADATA_INSUFFICIENT = "metadata_insufficient"
FINAL_TIMING_SAFE_SUBSET = "timing_safe_subset_exists"

OBJECTIVE_SLEEP_FIELDS = [
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

CAFFEINE_FIELDS = [
    "caffeine_mg",
    "caffeine_time_category",
    "coffee_count",
    "cola_count",
    "tea_count",
    "espresso_count",
    "caffeine_present",
]

WELLNESS_FIELDS = [
    "sleepquality",
    "sleepdura_sr",
    "readiness",
    "fatigue",
    "soreness",
    "soreness_location_1",
    "soreness_location_2",
]

ACTIGRAPHY_PREFIXES = ("actigraphy_",)
PRIOR_HISTORY_FIELDS = [
    "prior_valid_handgrip_count",
    "prior_expanding_handgrip_mean",
    "prior_expanding_handgrip_median",
    "prior_expanding_handgrip_std",
]


@dataclass(frozen=True)
class TimingPoint:
    """A timestamp with the identity needed for prospective alignment."""

    athlete_id: str
    timestamp: datetime | None
    source_field: str
    timezone_verified: bool = True
    calendar_date_verified: bool = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_member_sha256(zf: zipfile.ZipFile, member: str) -> str:
    return hashlib.sha256(zf.read(member)).hexdigest()


def finite_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [finite_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [finite_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def assert_json_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_json_finite(item)
    elif isinstance(value, list):
        for item in value:
            assert_json_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON report contains a non-finite float")


def classify_temporal_relation(feature: TimingPoint, cutoff: TimingPoint | None) -> str:
    if cutoff is None or cutoff.timestamp is None:
        return TIMESTAMP_UNAVAILABLE
    if feature.athlete_id != cutoff.athlete_id:
        return ATHLETE_DAY_UNVERIFIED
    if feature.timestamp is None:
        return TIMESTAMP_UNAVAILABLE
    if not feature.calendar_date_verified or not cutoff.calendar_date_verified:
        return DATE_ONLY_AMBIGUOUS
    if not feature.timezone_verified or not cutoff.timezone_verified:
        return TIMEZONE_UNVERIFIED
    if feature.timestamp <= cutoff.timestamp:
        return VERIFIED_PRE
    return VERIFIED_POST


def classify_caffeine_timing(label: object, handgrip_cutoff: TimingPoint | None) -> str:
    if pd.isna(label) or str(label).strip() == "":
        return TIMESTAMP_UNAVAILABLE
    if handgrip_cutoff is None or handgrip_cutoff.timestamp is None:
        return ORDERING_UNVERIFIED
    return ORDERING_UNVERIFIED


def classify_actigraphy_alignment(join_keys: set[str]) -> str:
    required = {"athlete_id", "calendar_date"}
    if required <= join_keys:
        return VERIFIED_PRE
    if {"athlete_id", "relative_day"} <= join_keys:
        return ORDERING_UNVERIFIED
    if {"athlete_id", "weekday"} <= join_keys:
        return ATHLETE_DAY_UNVERIFIED
    return ATHLETE_DAY_UNVERIFIED


def row_has_defensible_cutoff(row: pd.Series) -> bool:
    candidate_fields = [
        "handgrip_timestamp",
        "handgrip_measured_at",
        "performance_timestamp",
        "survey_submitted_at",
        "recorded_at",
    ]
    return any(field in row.index and pd.notna(row[field]) for field in candidate_fields)


def build_safe_cohort(
    rows: pd.DataFrame,
    feature_classifications: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cutoff_mask = rows.apply(row_has_defensible_cutoff, axis=1)
    safe_groups = [
        group
        for group, classification in sorted(feature_classifications.items())
        if classification["status"] == VERIFIED_PRE and group != "target_handgrip"
    ]
    safe_rows = rows.loc[cutoff_mask].copy()
    if not safe_groups:
        safe_rows = safe_rows.iloc[0:0]
    return {
        "row_count": int(len(safe_rows)),
        "athlete_count": int(safe_rows["athlete_id"].nunique()) if "athlete_id" in safe_rows else 0,
        "surviving_feature_groups": safe_groups,
        "excluded_feature_groups": [
            {
                "feature_group": group,
                "status": classification["status"],
                "exclusion_reason": classification["exclusion_reason"],
            }
            for group, classification in sorted(feature_classifications.items())
            if group != "target_handgrip" and group not in safe_groups
        ],
        "exact_timing_evidence_used": [],
    }


def raw_hash_inventory(zip_path: Path, zf: zipfile.ZipFile) -> dict[str, Any]:
    members = [
        member
        for member in csv_members(zf)
        if member in {DAILY_MEMBER, QUESTIONNAIRE_MEMBER} or member.startswith("REST/actigraphy/")
    ]
    return {
        "archive": {
            "path": "data/raw/REST.zip",
            "sha256": file_sha256(zip_path),
        },
        "members": [
            {
                "member": member,
                "bytes": int(zf.getinfo(member).file_size),
                "sha256": zip_member_sha256(zf, member),
            }
            for member in sorted(members)
        ],
    }


def processed_hash_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if path.exists():
            rows.append(
                {
                    "path": path.as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    return rows


def schema_inventory(zf: zipfile.ZipFile) -> dict[str, Any]:
    daily = read_csv_from_zip(zf, DAILY_MEMBER)
    questionnaire = read_csv_from_zip(zf, QUESTIONNAIRE_MEMBER, sep=";")
    actigraphy_member = sorted(member for member in csv_members(zf) if member.startswith("REST/actigraphy/"))[0]
    actigraphy = read_csv_from_zip(zf, actigraphy_member, nrows=5)
    return {
        "daily_responses": {
            "member": DAILY_MEMBER,
            "rows": int(len(daily)),
            "columns": list(daily.columns),
            "timestamp_fields_found": [],
            "date_fields_found": [],
            "time_of_day_fields": [
                "sleep_onset",
                "sleep_offset",
                "longest_interval_start",
                "longest_interval_end",
            ],
            "weekday_field": "weekday",
            "handgrip_field": "handgrip_kg",
        },
        "initial_questionnaire": {
            "member": QUESTIONNAIRE_MEMBER,
            "rows": int(len(questionnaire)),
            "columns": list(questionnaire.columns),
            "timestamp_fields_found": [],
            "date_fields_found": [],
        },
        "actigraphy": {
            "example_member": actigraphy_member,
            "columns": list(actigraphy.columns),
            "timestamp_fields_found": [],
            "date_fields_found": [],
            "time_of_day_field": "time",
            "weekday_field": "weekday",
            "athlete_id_source": "actigraphy filename stem",
        },
    }


def actigraphy_sequence_inventory(zf: zipfile.ZipFile) -> dict[str, Any]:
    members = sorted(member for member in csv_members(zf) if member.startswith("REST/actigraphy/"))
    examples = []
    statuses = set()
    for member in members:
        df = read_csv_from_zip(zf, member, usecols=["weekday", "time"])
        validation = validate_actigraphy_sequence(df)
        valid = (
            validation["total_rows"] == validation["expected_rows_given_observed_duration"]
            and validation["duplicate_sequential_samples"] == 0
            and validation["missing_sequential_samples_inferable"] == 0
            and validation["unexpected_weekday_time_transitions"] == 0
        )
        status = "valid_30s_sequence" if valid else "invalid_sequence"
        statuses.add(status)
        if len(examples) < 3:
            examples.append(
                {
                    "member": member,
                    "rows": int(len(df)),
                    "first_weekday_time": f"{df['weekday'].iloc[0]} {df['time'].iloc[0]}",
                    "last_weekday_time": f"{df['weekday'].iloc[-1]} {df['time'].iloc[-1]}",
                    "sequence_status": status,
                }
            )
    return {
        "athlete_file_count": int(len(members)),
        "sequence_statuses": sorted(statuses),
        "examples": examples,
        "alignment_support": "weekday-only alignment; no calendar date or target-day key",
        "status": ATHLETE_DAY_UNVERIFIED,
    }


def feature_group_classifications() -> dict[str, dict[str, Any]]:
    return {
        "target_handgrip": {
            "status": TIMESTAMP_UNAVAILABLE,
            "source_fields": ["REST/daily_responses.csv:handgrip_kg"],
            "evidence_type": "no documented timestamp",
            "exclusion_reason": "handgrip has a value field but no measurement timestamp, sequence number, event order, or collection metadata",
        },
        "prior_history": {
            "status": ORDERING_UNVERIFIED,
            "source_fields": PRIOR_HISTORY_FIELDS,
            "evidence_type": "inferred source-row ordering only",
            "exclusion_reason": "prior features depend on source row order, which is not a documented pre-handgrip cutoff",
        },
        "objective_sleep": {
            "status": DATE_ONLY_AMBIGUOUS,
            "source_fields": OBJECTIVE_SLEEP_FIELDS,
            "evidence_type": "same daily row plus time-of-day sleep fields; no calendar date and no handgrip time",
            "exclusion_reason": "sleep interval cannot be proven to end before handgrip",
        },
        "caffeine": {
            "status": ORDERING_UNVERIFIED,
            "source_fields": CAFFEINE_FIELDS,
            "evidence_type": "same daily row; caffeine_time labels only",
            "exclusion_reason": "morning/afternoon/evening labels cannot be ordered relative to unknown handgrip time",
        },
        "subjective_wellness": {
            "status": DATE_ONLY_AMBIGUOUS,
            "source_fields": WELLNESS_FIELDS,
            "evidence_type": "same daily survey grouping only",
            "exclusion_reason": "survey recording time is not proven before handgrip",
        },
        "actigraphy": {
            "status": ATHLETE_DAY_UNVERIFIED,
            "source_fields": ["REST/actigraphy/*.csv:weekday", "REST/actigraphy/*.csv:time"],
            "evidence_type": "athlete plus weekday-only provisional join",
            "exclusion_reason": "weekday-only matching does not prove exact athlete-day or previous-night alignment",
        },
        "match_context": {
            "status": DATE_ONLY_AMBIGUOUS,
            "source_fields": ["REST/daily_responses.csv:matchday"],
            "evidence_type": "same daily row; relative match labels",
            "exclusion_reason": "match context does not establish prediction cutoff or feature availability before handgrip",
        },
    }


def objective_sleep_inventory() -> list[dict[str, Any]]:
    raw_source = {
        "objective_total_sleep_time": "total_sleep_time",
        "objective_sleep_intervals": "sleep_intervals",
        "objective_longest_interval_duration": "longest_interval_duration",
        "objective_sleep_period_mean": "sleep period mean",
        "objective_sleep_period_standard_deviation": "sleep period standard deviation",
        "objective_sleep_period_msd": "sleep period msd",
        "sleep_onset_minutes_after_midnight": "sleep_onset",
        "sleep_offset_minutes_after_midnight": "sleep_offset",
        "longest_interval_start_minutes_after_midnight": "longest_interval_start",
        "longest_interval_end_minutes_after_midnight": "longest_interval_end",
    }
    return [
        {
            "feature": feature,
            "raw_source_field": raw_source[feature],
            "source_timestamp_or_date": "weekday plus time-of-day only; no calendar date or timezone",
            "sleep_interval_semantics": "field names indicate sleep interval summaries, but no data dictionary proves interval date or relation to performance",
            "athlete_day_join_key": ["athlete_id", "raw_daily_row_index"],
            "definitively_ends_before_handgrip": False,
            "timezone_assumptions": "timezone unavailable",
            "ambiguity": "handgrip timestamp unavailable; date and timezone unavailable",
            "status": DATE_ONLY_AMBIGUOUS,
        }
        for feature in OBJECTIVE_SLEEP_FIELDS
    ]


def caffeine_inventory(track_b: pd.DataFrame) -> dict[str, Any]:
    labels = sorted(str(value) for value in track_b["caffeine_time_category"].dropna().unique())
    return {
        "fields": CAFFEINE_FIELDS,
        "observed_time_labels": labels,
        "classification": "ambiguous",
        "status": ORDERING_UNVERIFIED,
        "usable_ordering_from_labels": False,
        "reason": "labels such as Morning, Afternoon, and Evening are not interpretable before/after handgrip without an independent handgrip time",
    }


def prediction_cutoff_findings(track_b: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for _, row in track_b.iterrows():
        rows.append(
            {
                "observation_id": row["observation_id"],
                "athlete_id": row["athlete_id"],
                "defensible_prediction_cutoff_exists": False,
                "status": TIMESTAMP_UNAVAILABLE,
                "reason": "no source field provides handgrip measurement time or a pre-handgrip survey submission time",
            }
        )
    return {
        "rows_checked": int(len(rows)),
        "rows_with_defensible_cutoff": 0,
        "rows_without_defensible_cutoff": int(len(rows)),
        "cutoff_policy": "cutoff cannot be fabricated from row order, file order, file modification time, ingestion time, result time, or assumed time of day",
        "rows": rows,
    }


def build_report(zip_path: Path = RAW_ZIP, track_b_path: Path = TRACK_B_INPUT) -> dict[str, Any]:
    track_b = pd.read_csv(track_b_path)
    with zipfile.ZipFile(zip_path) as zf:
        raw_hashes = raw_hash_inventory(zip_path, zf)
        schema = schema_inventory(zf)
        actigraphy_sequences = actigraphy_sequence_inventory(zf)

    classifications = feature_group_classifications()
    cutoff = prediction_cutoff_findings(track_b)
    safe_cohort = build_safe_cohort(track_b, classifications)
    decision = FINAL_TIMING_SAFE_SUBSET if safe_cohort["row_count"] > 0 else FINAL_METADATA_INSUFFICIENT
    report = {
        "audit_version": AUDIT_VERSION,
        "raw_and_processed_data_hashes": {
            "raw": raw_hashes,
            "processed_and_reports": processed_hash_inventory(
                [
                    track_b_path,
                    Path("reports/track_b_feature_audit.json"),
                    Path("reports/track_b_missingness.json"),
                    Path("reports/track_b_feature_summary.json"),
                    Path("reports/track_b_metrics.json"),
                    Path("reports/track_b_predictions.csv"),
                    Path("reports/track_b_bootstrap.json"),
                    Path("reports/track_b_per_athlete.csv"),
                    Path("reports/track_b_stability.json"),
                ]
            ),
        },
        "row_and_athlete_counts": {
            "track_b_rows": int(len(track_b)),
            "track_b_athletes": int(track_b["athlete_id"].nunique()),
        },
        "source_timing_inventory": {
            "schemas": schema,
            "handgrip_timing": {
                "documented_timestamp": False,
                "inferred_ordering": "source row order exists but is not proof",
                "file_or_row_order": "present but explicitly not used as timing evidence",
                "date_only_association": "weekday label only; no calendar date",
                "no_timing_evidence": True,
                "status": TIMESTAMP_UNAVAILABLE,
            },
            "actigraphy_sequence": actigraphy_sequences,
        },
        "feature_group_classifications": classifications,
        "objective_sleep_fields": objective_sleep_inventory(),
        "caffeine_findings": caffeine_inventory(track_b),
        "subjective_wellness_findings": {
            "fields": WELLNESS_FIELDS,
            "status": DATE_ONLY_AMBIGUOUS,
            "exploratory_only": True,
            "reason": "same-row survey grouping does not prove survey answers were recorded before handgrip",
        },
        "actigraphy_findings": {
            "available_identifiers": ["athlete_id from filename", "weekday", "time of day"],
            "exact_athlete_day_alignment": False,
            "relative_day_alignment": False,
            "weekday_only_alignment": True,
            "no_defensible_alignment": True,
            "status": ATHLETE_DAY_UNVERIFIED,
        },
        "prediction_cutoff_findings": cutoff,
        "timing_safe_cohort_summary": safe_cohort,
        "exact_missing_metadata": [
            "handgrip measurement timestamp",
            "survey submission timestamp",
            "daily response calendar date",
            "daily response timezone",
            "field-level event ordering within daily response",
            "objective sleep interval calendar dates",
            "proof objective sleep interval ended before handgrip",
            "caffeine intake timestamp",
            "subjective wellness recording timestamp",
            "actigraphy calendar date mapping",
            "actigraphy-to-target-day join key",
        ],
        "final_decision": decision,
    }
    cleaned = finite_json_value(report)
    assert_json_finite(cleaned)
    return cleaned


def write_report(report: dict[str, Any], output_path: Path = REPORT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Track B timing feasibility without modeling.")
    parser.add_argument("--zip", default=str(RAW_ZIP), help="Path to raw REST.zip")
    parser.add_argument("--track-b", default=str(TRACK_B_INPUT), help="Path to processed Track B CSV")
    parser.add_argument("--output", default=str(REPORT_OUTPUT), help="Path for timing audit JSON")
    args = parser.parse_args()

    report = build_report(Path(args.zip), Path(args.track_b))
    output = write_report(report, Path(args.output))
    print("Track B temporal-feasibility audit")
    print(f"output: {output}")
    print(f"track_b_rows: {report['row_and_athlete_counts']['track_b_rows']}")
    print(f"track_b_athletes: {report['row_and_athlete_counts']['track_b_athletes']}")
    print(f"rows_with_defensible_cutoff: {report['prediction_cutoff_findings']['rows_with_defensible_cutoff']}")
    print(f"timing_safe_cohort_rows: {report['timing_safe_cohort_summary']['row_count']}")
    print(f"final_decision: {report['final_decision']}")


if __name__ == "__main__":
    main()
