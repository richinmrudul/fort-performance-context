import json
from pathlib import Path

import pandas as pd

from src.features.build_track_a import MODEL_INPUT_COLUMNS as TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS
from src.features.build_track_a import build_track_a_features
from src.features.build_track_b import (
    ACTIGRAPHY_FEATURE_COLUMNS,
    ACTIGRAPHY_JOIN_ASSUMPTION,
    FORBIDDEN_TRACK_B_COLUMNS,
    TRACK_B_FEATURE_COLUMNS,
    build_and_write,
    build_track_b_features,
    validate_track_b_features,
)


def test_track_b_uses_track_a_primary_anchor_and_preserves_prior_fields():
    observations = pd.read_csv("data/processed/observations.csv")
    track_a = build_track_a_features(observations)
    features, summary = build_track_b_features(observations)
    validation = validate_track_b_features(features, summary, track_a)

    primary = track_a.loc[track_a["is_primary_modeling_row"].astype(bool)]
    assert validation["rows"] == len(primary)
    assert features["observation_id"].tolist() == primary["observation_id"].tolist()
    assert set(TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS) <= set(features.columns)
    assert features[TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS].equals(
        primary[TRACK_A_PRIOR_ONLY_FEATURE_COLUMNS].reset_index(drop=True)
    )


def test_track_b_excludes_retrospective_baselines_and_free_text_comments():
    observations = pd.read_csv("data/processed/observations.csv")
    features, summary = build_track_b_features(observations)

    assert not (FORBIDDEN_TRACK_B_COLUMNS & set(features.columns))
    assert "Comments (optional)" not in features.columns
    assert not any("comment" in column.lower() for column in features.columns)
    validate_track_b_features(features, summary)


def test_track_b_manifest_covers_every_feature_with_timing_status():
    observations = pd.read_csv("data/processed/observations.csv")
    features, summary = build_track_b_features(observations)

    manifest = {entry["feature_name"]: entry for entry in summary["features"]}
    assert set(TRACK_B_FEATURE_COLUMNS) <= set(manifest)
    assert all(entry["timing_status"] for entry in manifest.values())
    assert all(
        manifest[column]["source_group"] == "actigraphy_weekday_window"
        for column in ACTIGRAPHY_FEATURE_COLUMNS
    )
    assert summary["row_count"] == len(features)
    assert summary["feature_count"] == len(TRACK_B_FEATURE_COLUMNS)


def test_track_b_actigraphy_join_is_labeled_unverified_and_preserves_shared_rows():
    observations = pd.read_csv("data/processed/observations.csv")
    features, summary = build_track_b_features(observations)

    assert features["actigraphy_join_assumption"].eq(ACTIGRAPHY_JOIN_ASSUMPTION).all()
    assert features["actigraphy_source_sequence_valid"].all()
    assert features["actigraphy_shared_provisional_window"].any()

    shared = features.loc[features["actigraphy_shared_provisional_window"]]
    assert shared["actigraphy_weekday_window_id"].duplicated(keep=False).any()
    assert features["observation_id"].is_unique
    assert summary["actigraphy_join_assumption"] == ACTIGRAPHY_JOIN_ASSUMPTION


def test_track_b_artifacts_are_written(tmp_path):
    feature_path = tmp_path / "track_b_features.csv"
    summary_path = tmp_path / "track_b_feature_summary.json"

    result = build_and_write(
        Path("data/processed/observations.csv"),
        feature_path,
        summary_path,
    )

    assert result.output_path.exists()
    assert result.summary_path.exists()

    written_features = pd.read_csv(feature_path)
    written_summary = json.loads(summary_path.read_text())
    assert len(written_features) == result.summary["row_count"]
    assert written_summary["feature_count"] == len(TRACK_B_FEATURE_COLUMNS)
