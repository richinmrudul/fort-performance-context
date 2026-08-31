from pathlib import Path

import pandas as pd

from src.data.build_observations import (
    EXPECTED_PRIOR_ELIGIBILITY,
    EXPECTED_ROW_COUNT,
    EXPECTED_VALID_HANDGRIPS,
    build_and_write,
    build_observations,
    make_observation_id,
    validate_observations,
)


def test_build_observations_contract_counts_and_flags():
    observations = build_observations()
    summary = validate_observations(observations)

    assert summary["rows"] == EXPECTED_ROW_COUNT
    assert summary["unique_observation_ids"] == EXPECTED_ROW_COUNT
    assert summary["valid_handgrips"] == EXPECTED_VALID_HANDGRIPS
    for threshold, expected in EXPECTED_PRIOR_ELIGIBILITY.items():
        assert summary[f"eligible_prior_{threshold}"] == expected

    assert observations["raw_daily_row_index"].tolist() == list(range(EXPECTED_ROW_COUNT))
    assert observations["observation_id"].is_unique
    assert observations["has_valid_handgrip"].sum() == observations["handgrip_kg"].notna().sum()
    assert set(observations["handgrip_parse_status"].unique()) == {"missing", "valid"}
    assert observations["questionnaire_linked"].all()
    assert observations["actigraphy_linked"].all()
    assert set(observations["actigraphy_sequence_validation_status"].unique()) == {"valid_30s_sequence"}
    assert observations["source_row_order_temporal_assumption"].all()
    assert observations["retrospective_descriptive_ui_only"].all()


def test_observation_ids_are_deterministic_and_independent_of_derived_values():
    first = build_observations()
    second = build_observations()

    pd.testing.assert_series_equal(first["observation_id"], second["observation_id"])
    assert first.loc[0, "observation_id"] == make_observation_id(
        "data/raw/REST.zip",
        "REST/daily_responses.csv",
        0,
    )


def test_duplicate_rows_are_preserved_and_flagged():
    observations = build_observations()

    assert len(observations) == EXPECTED_ROW_COUNT
    assert observations["duplicate_group_size"].ge(1).all()
    assert (observations["duplicate_state"] != "not_duplicate").sum() > 0
    assert observations.loc[
        observations["duplicate_state"] != "not_duplicate",
        "exclude_from_modeling_default",
    ].all()
    assert observations["duplicate_heuristic_group_id"].dropna().nunique() == 19


def test_build_and_write_creates_only_observations_artifact(tmp_path):
    result = build_and_write(output_dir=tmp_path)

    assert result.output_path.exists()
    assert result.output_path.parent == tmp_path
    assert result.output_path.name in {"observations.csv", "observations.parquet"}
    assert sorted(path.name for path in Path(tmp_path).iterdir()) == [result.output_path.name]

    if result.output_path.suffix == ".csv":
        written = pd.read_csv(result.output_path)
    else:
        written = pd.read_parquet(result.output_path)
    assert len(written) == EXPECTED_ROW_COUNT
    assert written["observation_id"].tolist() == result.observations["observation_id"].tolist()
