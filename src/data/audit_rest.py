"""Read-only modeling-readiness audit for the REST dataset.

This script reads data/raw/REST.zip in place and prints reproducible counts and
checks used by docs/data_audit.md. It does not extract, clean, overwrite, or
produce modeling data.
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


RAW_ZIP = Path("data/raw/REST.zip")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def read_csv_from_zip(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with zf.open(name) as handle:
        return pd.read_csv(handle, **kwargs)


def csv_members(zf: zipfile.ZipFile) -> list[str]:
    return [
        info.filename
        for info in zf.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and not Path(info.filename).name.startswith("._")
        and info.filename.lower().endswith(".csv")
    ]


def file_inventory(zf: zipfile.ZipFile) -> pd.DataFrame:
    rows = []
    for info in zf.infolist():
        name = info.filename
        kind = "directory" if info.is_dir() else Path(name).suffix.lower().lstrip(".") or "none"
        rows.append(
            {
                "path": name,
                "bytes": info.file_size,
                "archive_modified": f"{info.date_time[0]:04d}-{info.date_time[1]:02d}-{info.date_time[2]:02d} "
                f"{info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}",
                "format": kind,
            }
        )
    return pd.DataFrame(rows)


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": [int(df[col].isna().sum()) for col in df.columns],
            "missing_pct": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
        }
    )


def parse_handgrip(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(",", ".", regex=False), errors="coerce")


def id_validation(daily: pd.DataFrame, questionnaire: pd.DataFrame, actigraphy_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = {
        "daily_responses": daily["id"],
        "initial_questionnaire": questionnaire["id"],
        "actigraphy_files": pd.Series(actigraphy_ids, dtype="object"),
    }
    non_null_sets = {name: set(values.dropna().astype(str)) for name, values in sources.items()}
    shared = set.intersection(*non_null_sets.values())
    all_ids = set.union(*non_null_sets.values())
    summary_rows = []
    diff_rows = []
    for name, values in sources.items():
        ids = non_null_sets[name]
        summary_rows.append(
            {
                "source": name,
                "rows_or_files": len(values),
                "non_null_unique_ids": len(ids),
                "null_id_count": int(values.isna().sum()),
            }
        )
        diff_rows.append(
            {
                "source": name,
                "ids_shared_by_all_sources": len(shared),
                "ids_missing_from_source": ", ".join(sorted(all_ids - ids)) or "(none)",
                "extra_ids_only_in_source": ", ".join(sorted(ids - set.union(*(s for n, s in non_null_sets.items() if n != name)))) or "(none)",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(diff_rows)


def prior_handgrip_eligibility(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for athlete_id, group in daily.groupby("id", sort=True):
        prior_valid = 0
        counts = {1: 0, 2: 0, 3: 0, 5: 0}
        valid_total = 0
        for value in group["handgrip_kg_numeric"]:
            if pd.notna(value):
                valid_total += 1
                for threshold in counts:
                    if prior_valid >= threshold:
                        counts[threshold] += 1
                prior_valid += 1
        rows.append(
            {
                "athlete_id": athlete_id,
                "valid_handgrip_rows": valid_total,
                "eligible_with_1_prior": counts[1],
                "eligible_with_2_prior": counts[2],
                "eligible_with_3_prior": counts[3],
                "eligible_with_5_prior": counts[5],
            }
        )
    by_athlete = pd.DataFrame(rows)
    summary = []
    for threshold in [1, 2, 3, 5]:
        col = f"eligible_with_{threshold}_prior"
        summary.append(
            {
                "prior_valid_handgrip_threshold": threshold,
                "eligible_target_rows": int(by_athlete[col].sum()),
                "athletes_contributing": int((by_athlete[col] > 0).sum()),
            }
        )
    return pd.DataFrame(summary), by_athlete


def weekday_time_to_seconds(weekday: str, time_value: str) -> int:
    weekday_offset = WEEKDAYS.index(weekday) * 24 * 60 * 60
    return weekday_offset + int(pd.to_timedelta(time_value).total_seconds())


def add_seconds_to_weekday_time(weekday: str, time_value: str, seconds: int) -> tuple[str, str]:
    start = weekday_time_to_seconds(weekday, time_value)
    week_seconds = 7 * 24 * 60 * 60
    updated = (start + seconds) % week_seconds
    day_index, second_of_day = divmod(updated, 24 * 60 * 60)
    hours, remainder = divmod(second_of_day, 60 * 60)
    minutes, secs = divmod(remainder, 60)
    return WEEKDAYS[day_index], f"{hours:02d}:{minutes:02d}:{secs:02d}"


def observed_transition_delta_seconds(prev_weekday: str, prev_time: str, next_weekday: str, next_time: str) -> int:
    prev_seconds = weekday_time_to_seconds(prev_weekday, prev_time)
    next_seconds = weekday_time_to_seconds(next_weekday, next_time)
    delta = next_seconds - prev_seconds
    if delta < 0:
        delta += 7 * 24 * 60 * 60
    return delta


def validate_actigraphy_sequence(df: pd.DataFrame, expected_interval_seconds: int = 30) -> dict[str, object]:
    duplicate_samples = 0
    missing_samples = 0
    unexpected_transitions = 0
    examples = []
    elapsed_seconds = 0
    for index in range(1, len(df)):
        prev = df.iloc[index - 1]
        current = df.iloc[index]
        prev_label = (str(prev["weekday"]), str(prev["time"]))
        current_label = (str(current["weekday"]), str(current["time"]))
        if current_label == prev_label:
            duplicate_samples += 1
            delta = 0
        else:
            delta = observed_transition_delta_seconds(*prev_label, *current_label)
        if delta == expected_interval_seconds:
            elapsed_seconds += delta
            continue
        if delta > expected_interval_seconds and delta % expected_interval_seconds == 0:
            missing_samples += int(delta / expected_interval_seconds) - 1
            elapsed_seconds += delta
        else:
            unexpected_transitions += 1
        if len(examples) < 5:
            expected_label = add_seconds_to_weekday_time(*prev_label, expected_interval_seconds)
            examples.append(
                {
                    "row_index": index,
                    "previous": f"{prev_label[0]} {prev_label[1]}",
                    "observed": f"{current_label[0]} {current_label[1]}",
                    "expected": f"{expected_label[0]} {expected_label[1]}",
                    "delta_seconds": delta,
                }
            )
    expected_rows = int(elapsed_seconds / expected_interval_seconds) + 1 if len(df) else 0
    return {
        "total_rows": len(df),
        "expected_rows_given_observed_duration": expected_rows,
        "duplicate_sequential_samples": duplicate_samples,
        "missing_sequential_samples_inferable": missing_samples,
        "unexpected_weekday_time_transitions": unexpected_transitions,
        "transition_examples": examples,
    }


def actigraphy_summary(zf: zipfile.ZipFile, actigraphy_files: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    missing_rows = []
    sequence_rows = []
    reference_sequence = None
    for name in actigraphy_files:
        athlete_id = Path(name).stem
        df = read_csv_from_zip(zf, name)
        first = f"{df['weekday'].iloc[0]} {df['time'].iloc[0]}" if len(df) else None
        last = f"{df['weekday'].iloc[-1]} {df['time'].iloc[-1]}" if len(df) else None
        weekday_counts = df["weekday"].value_counts().reindex(WEEKDAYS).fillna(0).astype(int).to_dict()
        missing_by_col = df.isna().sum()
        sequence = list(zip(df["weekday"].astype(str), df["time"].astype(str)))
        if reference_sequence is None:
            reference_sequence = sequence
        sequence_check = validate_actigraphy_sequence(df)
        rows.append(
            {
                "athlete_id": athlete_id,
                "rows": len(df),
                "first_observation": first,
                "last_observation": last,
                "unique_weekdays": int(df["weekday"].nunique()),
                "weekday_counts": weekday_counts,
                "time_min": df["time"].min(),
                "time_max": df["time"].max(),
            }
        )
        sequence_rows.append(
            {
                "athlete_id": athlete_id,
                **{key: value for key, value in sequence_check.items() if key != "transition_examples"},
                "sequence_matches_first_actigraphy_file": sequence == reference_sequence,
            }
        )
        for col, count in missing_by_col.items():
            if count:
                missing_rows.append({"athlete_id": athlete_id, "column": col, "missing": int(count)})
    return pd.DataFrame(rows), pd.DataFrame(sequence_rows), pd.DataFrame(missing_rows)


def duplicate_summary(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact = daily[daily.duplicated(keep=False)].copy()
    key_cols = ["id", "weekday", "sleep_onset", "sleep_offset", "total_sleep_time"]
    repeated = (
        daily.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="rows")
        .query("rows > 1")
        .sort_values(["id", "weekday", "rows"], ascending=[True, True, False])
    )

    examples = []
    for _, group_key in repeated.iterrows():
        mask = pd.Series(True, index=daily.index)
        for col in key_cols:
            value = group_key[col]
            mask &= daily[col].isna() if pd.isna(value) else daily[col].eq(value)
        group = daily[mask]
        varying = [
            col
            for col in daily.columns
            if group[col].astype("string").fillna("<NA>").nunique(dropna=False) > 1
        ]
        caffeine_cols = ["cola", "coffee", "tea", "espresso", "caffeine_time", "caffeine_mg"]
        context_cols = [
            "sleepquality",
            "handgrip_kg",
            "matchday",
            "fatigue",
            "soreness",
            "soreness_location_1",
            "soreness_location_2",
            "readiness",
            "sleepdura_sr",
        ]
        objective_sleep_cols = [
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
        ]
        non_caffeine_varying = [col for col in varying if col not in caffeine_cols]
        context_varying = [col for col in varying if col in context_cols]
        correction_hint = group["Comments (optional)"].astype("string").str.contains(
            "feil|wrong|forrige|previous|skrev", case=False, na=False
        ).any()
        if group.duplicated(keep=False).all():
            classification = "exact duplicate"
        elif correction_hint:
            classification = "unresolved"
        elif not context_varying and all(col not in objective_sleep_cols for col in varying):
            classification = "strong likely duplicate"
        elif context_varying:
            classification = "conflicting same-context rows"
        else:
            classification = "unresolved"
        examples.append(
            {
                "id": group_key["id"],
                "weekday": group_key["weekday"],
                "rows": int(group_key["rows"]),
                "classification": classification,
                "varying_columns": ", ".join(varying),
            }
        )
    return exact, repeated, pd.DataFrame(examples)


def print_table(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
    print(f"\n## {title}")
    if df.empty:
        print("(none)")
        return
    shown = df if max_rows is None else df.head(max_rows)
    print(markdown_table(shown))
    if max_rows is not None and len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows")


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    string_rows = []
    for _, row in df.iterrows():
        string_rows.append([format_cell(row[col]) for col in columns])
    widths = []
    for index, col in enumerate(columns):
        widths.append(max(len(str(col)), *(len(row[index]) for row in string_rows)))
    header = "| " + " | ".join(str(col).ljust(widths[index]) for index, col in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in string_rows
    ]
    return "\n".join([header, separator, *body])


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the raw REST dataset.")
    parser.add_argument("--zip", default=str(RAW_ZIP), help="Path to REST.zip")
    parser.add_argument("--max-examples", type=int, default=20, help="Maximum duplicate examples to print")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    with zipfile.ZipFile(zip_path) as zf:
        inventory = file_inventory(zf)
        members = csv_members(zf)
        daily = read_csv_from_zip(zf, "REST/daily_responses.csv")
        questionnaire = read_csv_from_zip(zf, "REST/initial_questionnaire.csv", sep=";")
        actigraphy_files = sorted(name for name in members if name.startswith("REST/actigraphy/"))
        actigraphy_ids = [Path(name).stem for name in actigraphy_files]

        daily["handgrip_kg_numeric"] = parse_handgrip(daily["handgrip_kg"])

        print(f"# REST dataset audit: {zip_path}")
        print_table("Archive inventory", inventory)

        id_summary, id_diffs = id_validation(daily, questionnaire, actigraphy_ids)
        print_table("Athlete ID validation summary", id_summary)
        print_table("Athlete ID set comparison", id_diffs)

        print_table("Daily rows per athlete", daily["id"].value_counts().rename_axis("athlete_id").reset_index(name="rows"))
        print_table("Daily schema and missingness", missingness(daily.drop(columns=["handgrip_kg_numeric"])))
        handgrip = (
            daily.groupby("id")["handgrip_kg_numeric"]
            .agg(rows="size", available="count", missing=lambda s: int(s.isna().sum()), min="min", median="median", max="max")
            .reset_index()
        )
        print_table("Handgrip availability by athlete", handgrip)
        eligibility_summary, eligibility_by_athlete = prior_handgrip_eligibility(daily)
        print_table("Prospective personal-baseline target eligibility summary", eligibility_summary)
        print_table("Prospective personal-baseline target eligibility by athlete", eligibility_by_athlete)

        caffeine_cols = ["cola", "coffee", "tea", "espresso", "caffeine_time", "caffeine_mg"]
        print_table("Caffeine missingness", missingness(daily[caffeine_cols]))
        for col in caffeine_cols:
            counts = daily[col].value_counts(dropna=False).rename_axis(col).reset_index(name="rows")
            print_table(f"Caffeine distribution: {col}", counts)

        weekday_counts = daily["weekday"].value_counts(dropna=False).rename_axis("weekday").reset_index(name="rows")
        print_table("Daily weekday distribution", weekday_counts)
        sequence_counts = daily.groupby("id")["weekday"].apply(lambda s: " -> ".join(s.astype(str))).reset_index(name="weekday_sequence")
        print_table("Daily weekday sequence by athlete", sequence_counts)

        exact, repeated, examples = duplicate_summary(daily.drop(columns=["handgrip_kg_numeric"]))
        print(f"\n## Duplicate overview")
        print(f"Exact duplicate rows, counting all copies: {len(exact)}")
        print(f"Duplicate-looking groups by id + weekday + sleep window: {len(repeated)}")
        print_table("Duplicate-looking group classifications", examples, max_rows=args.max_examples)

        comments = daily[daily["Comments (optional)"].notna()][["id", "weekday", "Comments (optional)"]]
        print_table("Non-empty daily comments", comments, max_rows=50)

        print_table("Questionnaire schema and missingness", missingness(questionnaire))
        print_table("Questionnaire athlete IDs", questionnaire[["id"]])

        act_summary, act_sequence, act_missing = actigraphy_summary(zf, actigraphy_files)
        print_table("Actigraphy per-athlete summary", act_summary)
        print_table("Actigraphy sequential cadence validation", act_sequence)
        first_act = read_csv_from_zip(zf, actigraphy_files[0])
        print_table("Actigraphy schema and missingness from first file", missingness(first_act))
        alg_cols = ["CK", "Oakley", "Sadeh", "fourier_HMM", "LSTM"]
        alg_values = []
        for name in actigraphy_files:
            df = read_csv_from_zip(zf, name, usecols=alg_cols)
            for col in alg_cols:
                values = Counter(df[col].dropna().astype(str))
                alg_values.append({"athlete_id": Path(name).stem, "column": col, "values": dict(values)})
        print_table("Actigraphy sleep/wake algorithm values", pd.DataFrame(alg_values), max_rows=40)
        print_table("Actigraphy missing values", act_missing)


if __name__ == "__main__":
    main()
