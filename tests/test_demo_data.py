import json
import math
import re
from pathlib import Path

import pandas as pd

from src.demo import build_demo_data as demo


def hash_file(path: Path) -> str:
    return demo.file_sha256(path)


def walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)
    else:
        yield value


def test_source_reports_are_read_without_mutation(tmp_path):
    paths = demo.SOURCE_PATHS
    before = {path: hash_file(path) for path in paths}

    demo.write_demo_data(tmp_path / "demo.json")

    assert {path: hash_file(path) for path in paths} == before


def test_raw_rest_archive_is_not_read_or_mutated(tmp_path):
    raw_rest = Path("data/raw/REST.zip")
    assert raw_rest not in demo.SOURCE_PATHS
    if not raw_rest.exists():
        return
    before = hash_file(raw_rest)

    demo.write_demo_data(tmp_path / "demo.json")

    assert hash_file(raw_rest) == before


def test_selected_session_follows_documented_latest_eligible_rule():
    sources = demo.load_sources()
    selected, prediction = demo.select_default_session(sources)
    min_prior = demo.minimum_prior_count(sources.track_a_metrics)
    prediction_ids = set(demo.expected_prediction_rows(sources.predictions)["observation_id"])
    candidates = sources.track_b.loc[
        sources.track_b["is_primary_modeling_row"].astype(bool)
        & sources.track_b["handgrip_kg"].notna()
        & sources.track_b["prior_expanding_handgrip_mean"].notna()
        & (sources.track_b["prior_valid_handgrip_count"] >= min_prior)
        & sources.track_b["observation_id"].isin(prediction_ids)
    ].sort_values(["raw_daily_row_index", "observation_id"], kind="mergesort")

    assert selected["observation_id"] == candidates.iloc[-1]["observation_id"]
    assert prediction["model_name"] == demo.EXPECTED_MODEL_NAME


def test_formulas_match_source_values():
    sources = demo.load_sources()
    selected, prediction = demo.select_default_session(sources)
    payload = demo.build_demo_payload()
    session = payload["selected_session"]

    actual = float(selected["handgrip_kg"])
    expected = float(prediction["y_pred_handgrip_kg"])

    assert session["actual_kg"] == round(actual, 2)
    assert session["track_a_expected_kg"] == round(expected, 2)
    assert session["actual_percent_of_expected"] == round(actual / expected * 100, 1)
    assert session["difference_kg"] == round(actual - expected, 2)
    assert session["difference_percent"] == round(actual / expected * 100 - 100, 1)
    assert session["prior_observation_count"] == int(selected["prior_valid_handgrip_count"])


def test_no_fabricated_rows_or_metrics_appear():
    sources = demo.load_sources()
    selected, _ = demo.select_default_session(sources)
    payload = demo.build_demo_payload()
    trend = payload["trend"]["observations"]
    source_history = sources.track_a.loc[
        (sources.track_a["athlete_id"] == selected["athlete_id"])
        & sources.track_a["handgrip_kg"].notna()
    ]

    assert len(trend) == len(source_history)
    assert all(row["label"] == f"Observation {row['sequence']}" for row in trend)
    assert payload["selected_session"]["date_label"] is None
    assert payload["selected_session"]["date_state"] == "no_defensible_date"


def test_track_a_prediction_identity_matches_actual_row():
    sources = demo.load_sources()
    selected, prediction = demo.select_default_session(sources)

    assert prediction["athlete_id"] == selected["athlete_id"]
    assert math.isclose(prediction["y_true_handgrip_kg"], selected["handgrip_kg"])
    assert math.isclose(prediction["y_pred_handgrip_kg"], selected["prior_expanding_handgrip_mean"])


def test_early_rows_show_limited_history_state():
    payload = demo.build_demo_payload()
    trend = payload["trend"]["observations"]

    assert trend[0]["expected_kg"] is None
    assert trend[0]["history_state"] == "no_prior_observations"
    assert any(row["history_state"] == "limited_history" for row in trend)


def test_missing_context_remains_missing():
    payload = demo.build_demo_payload()
    assert payload["missing_states"]["missing_context"] == "Not available for this observation"
    assert all(item["value"] != 0 or item["available"] for item in payload["context"]["items"])


def test_timing_ambiguous_context_remains_labeled_ambiguous():
    payload = demo.build_demo_payload()
    statuses = {item["timing_status"] for item in payload["context"]["items"]}

    assert "date_only_ambiguous" in statuses
    assert "ordering_unverified" in statuses
    assert "athlete_day_alignment_unverified" in statuses
    assert {item["timing_label"] for item in payload["context"]["items"]} == {"Timing not verified"}


def test_no_contextual_field_enters_expected_performance_calculation():
    payload = demo.build_demo_payload()
    source = payload["selected_session"]["track_a_prediction_source"]

    assert source["source_file"] == demo.TRACK_A_PREDICTIONS.as_posix()
    assert source["model_name"] == demo.EXPECTED_MODEL_NAME
    assert payload["model_card"]["expected_performance_definition"] == (
        "prior_expanding_mean row-level prediction from Track A"
    )


def test_output_is_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    demo.write_demo_data(first)
    demo.write_demo_data(second)

    assert first.read_bytes() == second.read_bytes()


def test_json_contains_only_finite_values():
    payload = demo.build_demo_payload()

    for value in walk(payload):
        assert not isinstance(value, float) or math.isfinite(value)


def test_private_paths_and_raw_identifiers_are_excluded():
    payload = demo.build_demo_payload()
    text = json.dumps(payload, sort_keys=True)

    assert "/Users/" not in text
    assert "data/raw" not in text
    assert "rest_daily_" not in text
    assert not re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        text,
    )
