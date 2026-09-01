import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features import audit_track_b_timing as audit


def test_no_timestamp_is_fabricated_for_official_rows():
    report = audit.build_report()

    assert report["prediction_cutoff_findings"]["rows_checked"] == 84
    assert report["prediction_cutoff_findings"]["rows_with_defensible_cutoff"] == 0
    assert {
        row["defensible_prediction_cutoff_exists"]
        for row in report["prediction_cutoff_findings"]["rows"]
    } == {False}


def test_row_and_file_order_are_not_treated_as_time():
    row = pd.Series(
        {
            "observation_id": "row-1",
            "athlete_id": "athlete-a",
            "raw_daily_row_index": 1,
            "athlete_row_index": 0,
            "source_member_path": "REST/daily_responses.csv",
        }
    )

    assert audit.row_has_defensible_cutoff(row) is False


def test_same_day_data_remains_ambiguous_without_ordering_evidence():
    feature = audit.TimingPoint(
        athlete_id="athlete-a",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        source_field="daily_date",
        calendar_date_verified=False,
    )
    cutoff = audit.TimingPoint(
        athlete_id="athlete-a",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        source_field="handgrip_date",
    )

    assert audit.classify_temporal_relation(feature, cutoff) == audit.DATE_ONLY_AMBIGUOUS


def test_caffeine_daypart_labels_are_not_ordered_against_unknown_handgrip_time():
    for label in ["Morning", "Afternoon", "Evening"]:
        assert audit.classify_caffeine_timing(label, None) == audit.ORDERING_UNVERIFIED


def test_weekday_only_actigraphy_matching_is_rejected():
    assert audit.classify_actigraphy_alignment({"athlete_id", "weekday"}) == audit.ATHLETE_DAY_UNVERIFIED


def test_athlete_boundaries_are_preserved():
    feature = audit.TimingPoint(
        athlete_id="athlete-a",
        timestamp=datetime(2026, 1, 1, 8, tzinfo=timezone.utc),
        source_field="feature_at",
    )
    cutoff = audit.TimingPoint(
        athlete_id="athlete-b",
        timestamp=datetime(2026, 1, 1, 9, tzinfo=timezone.utc),
        source_field="handgrip_at",
    )

    assert audit.classify_temporal_relation(feature, cutoff) == audit.ATHLETE_DAY_UNVERIFIED


def test_exact_supported_athlete_day_matches_are_recognized():
    feature = audit.TimingPoint(
        athlete_id="athlete-a",
        timestamp=datetime(2026, 1, 1, 8, tzinfo=timezone.utc),
        source_field="feature_at",
    )
    cutoff = audit.TimingPoint(
        athlete_id="athlete-a",
        timestamp=datetime(2026, 1, 1, 9, tzinfo=timezone.utc),
        source_field="handgrip_at",
    )

    assert audit.classify_temporal_relation(feature, cutoff) == audit.VERIFIED_PRE


def test_ambiguous_fields_cannot_enter_safe_cohort():
    rows = pd.DataFrame(
        {
            "observation_id": ["one"],
            "athlete_id": ["athlete-a"],
            "handgrip_timestamp": ["2026-01-01T09:00:00Z"],
        }
    )
    classifications = {
        "caffeine": {
            "status": audit.ORDERING_UNVERIFIED,
            "exclusion_reason": "unknown order",
        }
    }

    cohort = audit.build_safe_cohort(rows, classifications)

    assert cohort["row_count"] == 0
    assert cohort["surviving_feature_groups"] == []
    assert cohort["excluded_feature_groups"][0]["feature_group"] == "caffeine"


def test_audit_is_deterministic(tmp_path):
    first = audit.build_report()
    second = audit.build_report()

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    audit.write_report(first, first_path)
    audit.write_report(second, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_raw_files_are_never_written(tmp_path):
    raw_path = Path("data/raw/REST.zip")
    before = audit.file_sha256(raw_path)

    report = audit.build_report()
    audit.write_report(report, tmp_path / "report.json")

    assert audit.file_sha256(raw_path) == before


def test_report_contains_only_finite_valid_json_values(tmp_path):
    report = audit.build_report()
    output = audit.write_report(report, tmp_path / "report.json")
    loaded = json.loads(output.read_text())

    audit.assert_json_finite(loaded)

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif isinstance(value, float):
            yield value

    assert all(math.isfinite(value) for value in walk(loaded))
