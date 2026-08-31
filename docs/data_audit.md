# REST Modeling-Readiness Data Audit

Audit date: 2026-08-31

Scope: read-only audit of `data/raw/REST.zip`. This audit does not extract or
modify raw REST data, create a final cleaned modeling dataset, train models, or
build product code.

## Executive Summary

The REST archive can support a leakage-aware prototype only if we keep two
analysis tracks separate.

Track A, confirmed-safe/conservative, can use linked static questionnaire
variables and raw handgrip target rows. Under current raw evidence, no
per-observation daily or actigraphy feature is definitely pre-outcome because
the raw files do not encode calendar dates or handgrip collection time.

Track B, exploratory, can use source row order, weekday/time labels, daily sleep
summaries, actigraphy windows, caffeine, wellness, and match context, but every
such feature must be marked timing-unverified until collection timing is
confirmed.

Do not recommend a train/test split yet. The raw data has consistent ordering
signals, but not enough verified temporal semantics.

## Dataset Inventory

Local raw files:

| Path | Format | Notes |
| --- | --- | --- |
| `data/raw/REST.zip` | ZIP archive | Raw REST archive. |
| `data/raw/.gitkeep` | empty file | Repository placeholder. |

Analytic archive contents:

| Archive path | Format | Notes |
| --- | --- | --- |
| `REST/daily_responses.csv` | comma-delimited CSV | 370 daily response rows. |
| `REST/initial_questionnaire.csv` | semicolon-delimited CSV | 22 questionnaire rows. |
| `REST/actigraphy/*.csv` | comma-delimited CSV | 21 athlete-level files. |

Non-analytic archive contents include `REST/.DS_Store` and `__MACOSX/...`
metadata sidecars.

## Athlete ID Validation

Null IDs were excluded from set comparisons.

| Source | Rows/files | Non-null unique IDs | Null ID count |
| --- | ---: | ---: | ---: |
| daily_responses | 370 | 21 | 0 |
| initial_questionnaire | 22 | 21 | 1 |
| actigraphy_files | 21 | 21 | 0 |

| Source | IDs shared by all sources | IDs missing from source | Extra IDs only in source |
| --- | ---: | --- | --- |
| daily_responses | 21 | none | none |
| initial_questionnaire | 21 | none | none |
| actigraphy_files | 21 | none | none |

The questionnaire has one row with a missing `id`. It has no raw key that can
be linked to an athlete without external study metadata.

## Daily Response Audit

Total rows: 370

Unique non-null athlete IDs: 21

Rows per athlete:

| Athlete rows | Athlete count |
| ---: | ---: |
| 20 | 1 |
| 19 | 2 |
| 18 | 6 |
| 17 | 12 |

Daily schema and missingness:

| Column | Inferred dtype | Missing | Missing % |
| --- | --- | ---: | ---: |
| `sleep_onset` | object/time-like | 38 | 10.27 |
| `sleep_offset` | object/time-like | 38 | 10.27 |
| `total_sleep_time` | float64 | 38 | 10.27 |
| `sleep_intervals` | float64 | 38 | 10.27 |
| `longest_interval_start` | object/time-like | 38 | 10.27 |
| `longest_interval_end` | object/time-like | 38 | 10.27 |
| `longest_interval_duration` | float64 | 38 | 10.27 |
| `sleep period mean` | float64 | 38 | 10.27 |
| `sleep period standard deviation` | float64 | 38 | 10.27 |
| `sleep period msd` | float64 | 38 | 10.27 |
| `id` | object | 0 | 0.00 |
| `screentime` | object | 160 | 43.24 |
| `Comments (optional)` | object | 330 | 89.19 |
| `energy` | float64 | 337 | 91.08 |
| `cola` | float64 | 309 | 83.51 |
| `coffee` | object | 263 | 71.08 |
| `tea` | float64 | 364 | 98.38 |
| `espresso` | float64 | 309 | 83.51 |
| `caffeine_time` | object | 162 | 43.78 |
| `sleepquality` | float64 | 97 | 26.22 |
| `handgrip_kg` | object/comma-decimal numeric | 228 | 61.62 |
| `matchday` | object | 200 | 54.05 |
| `fatigue` | float64 | 92 | 24.86 |
| `soreness` | float64 | 97 | 26.22 |
| `soreness_location_1` | float64 | 139 | 37.57 |
| `soreness_location_2` | float64 | 242 | 65.41 |
| `readiness` | float64 | 90 | 24.32 |
| `sleepdura_sr` | float64 | 90 | 24.32 |
| `caffeine_mg` | float64 | 0 | 0.00 |
| `weekday` | object | 0 | 0.00 |

`handgrip_kg` is stored as comma-decimal text, for example `46,8`, and must be
parsed before numeric use.

Weekday counts:

| Weekday | Rows |
| --- | ---: |
| Friday | 65 |
| Saturday | 65 |
| Sunday | 64 |
| Tuesday | 47 |
| Wednesday | 44 |
| Thursday | 43 |
| Monday | 42 |

Daily row order is consistent with weekday progression if same-weekday repeats
are allowed. This is still only an exploratory ordering signal, not proof that
source row order is the true temporal order of performance outcomes.

## Target Availability And Prospective Baseline Eligibility

Valid handgrip values: 142 of 370 daily rows.

Athletes with at least one valid handgrip: 15.

Athletes with no valid handgrip: 6.

The following eligibility counts use only prior valid handgrip values within
each athlete, preserving source row order. They do not use future handgrip
values. These counts are valid for prospective modeling only if source row order
is confirmed to represent temporal order.

| Prior valid handgrips required | Eligible target rows | Athletes contributing |
| ---: | ---: | ---: |
| 1 | 127 | 15 |
| 2 | 112 | 15 |
| 3 | 97 | 15 |
| 5 | 69 | 13 |

Eligibility by athlete:

| Athlete ID | Valid handgrip rows | >=1 prior | >=2 prior | >=3 prior | >=5 prior |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2068e7ec-51a8-441a-8053-2be7533e613f | 12 | 11 | 10 | 9 | 7 |
| 687552ea-5963-45df-8680-c4c6058eab87 | 10 | 9 | 8 | 7 | 5 |
| 6a2ca03c-e61b-4589-a931-58f5e25aacd8 | 4 | 3 | 2 | 1 | 0 |
| 86fed448-f51d-4005-aef8-bd9413933cbe | 11 | 10 | 9 | 8 | 6 |
| 8d47422f-2512-44a3-a6d6-8feb726f4759 | 9 | 8 | 7 | 6 | 4 |
| 957de0b1-7ee9-437c-b47b-dffd09b6653a | 13 | 12 | 11 | 10 | 8 |
| 9cee6de4-93b3-46a7-a018-90f60d705ebb | 9 | 8 | 7 | 6 | 4 |
| a2d89dcf-6bd1-445b-bc9c-7ac6b852a9fc | 9 | 8 | 7 | 6 | 4 |
| c887cb7e-63cf-4dc2-9fd2-8429137ba44b | 10 | 9 | 8 | 7 | 5 |
| e362b878-9920-4dcf-9ab7-687a194a821c | 12 | 11 | 10 | 9 | 7 |
| e6e4e3b9-5cfa-42c6-a4cd-0f96561de3eb | 9 | 8 | 7 | 6 | 4 |
| ec3fc0a0-ab15-4c0c-b1ad-5243c6e5636f | 9 | 8 | 7 | 6 | 4 |
| efa2dbe6-24c0-4500-8b7b-1d5d14e1d271 | 10 | 9 | 8 | 7 | 5 |
| f2b08928-a410-4d1d-9f19-1f8c063a6311 | 4 | 3 | 2 | 1 | 0 |
| fd0c9e61-cb99-4d60-ab37-b4b3f5ac60d2 | 11 | 10 | 9 | 8 | 6 |
| six no-handgrip athletes | 0 each | 0 | 0 | 0 | 0 |

## Questionnaire Audit

The questionnaire contains baseline demographics and athlete sleep behavior
items. All non-null questionnaire IDs are present in daily responses and
actigraphy.

Key schema groups:

| Group | Raw fields |
| --- | --- |
| Demographics | age, height, weight |
| Sensor context | placement of sensor and dominant hand |
| Baseline lifestyle | nicotine, contraceptive use, sleep medication |
| Baseline sleep behavior | ASBQ items |
| Free text | optional comments |

Missingness: all questionnaire columns are complete except `Comments
(optional)` with 21 missing rows and `id` with one missing row.

## Actigraphy Audit

Files per athlete: 1.

Athlete files: 21.

Rows per athlete: 48,480.

Total actigraphy rows: 1,018,080.

Columns:

| Column | Inferred meaning |
| --- | --- |
| `Unnamed: 0` | Original row index/export index. |
| `x`, `y`, `z` | Accelerometer-axis activity/count-like values; unit not documented in raw file. |
| `vm` | Vector magnitude or activity magnitude; exact formula/unit not documented. |
| `CK`, `Oakley`, `Sadeh`, `fourier_HMM`, `LSTM` | Algorithm labels with observed values `s`, `w`, `n`; likely sleep, wake, non-wear/no classification, but this requires data-dictionary confirmation. |
| `weekday` | Weekday label only. |
| `time` | Time of day only. |

Sequential cadence validation was corrected to use file row order and expected
30-second transitions across midnight and weekday rollovers. It does not invent
calendar dates.

| Metric | Result |
| --- | --- |
| Total rows per file | 48,480 |
| Expected rows given observed duration | 48,480 |
| Duplicate sequential samples | 0 in every file |
| Inferable missing sequential samples | 0 in every file |
| Unexpected weekday/time transitions | 0 in every file |
| Sequence consistent across athlete files | yes; all 21 match the first actigraphy file's weekday/time sequence |

Observed weekday/time coverage is identical in all athlete files: first row is
Friday 14:00:00 and final row is Monday 09:59:30. This is a repeated
weekday/time clock sequence, not a calendar date sequence.

Wear/non-wear and missingness:

- `x`, `y`, `z`, and `vm` have no missing values in the checked files.
- `CK`, `Oakley`, `Sadeh`, and `fourier_HMM` have no missing values.
- `LSTM` has 180 missing rows total across 13 athletes.
- Rows where algorithms report `n` vary greatly by athlete, suggesting
  substantial non-wear or invalid-classification periods for some athletes.

## Temporal Alignment Findings

What is supported by raw data:

- The same 21 non-null athlete IDs are present in daily responses, questionnaire
  rows, and actigraphy filenames.
- Actigraphy row order and weekday/time labels form a complete, consistent
  30-second sequence across all athlete files.
- Daily response source row order follows weekday progression with same-day
  repeats allowed.

What is not supported by raw data:

- Calendar dates for daily rows.
- Calendar dates for actigraphy rows.
- A real athlete-day key.
- Whether daily `weekday` means sleep-night day, survey day, performance day,
  match day, or another convention.
- Whether daily objective sleep summaries were computed before handgrip
  performance or after the fact.
- Which actigraphy window maps to each daily row, especially when repeated
  same-weekday rows exist.
- Whether caffeine, wellness, and match-context fields were entered before or
  after handgrip measurement.

## Duplicate Analysis

Duplicate detection here is explicitly heuristic. The grouping key `id +
weekday + sleep_onset + sleep_offset + total_sleep_time` is not asserted to be a
true athlete-day key. It is only a way to find rows that appear to refer to the
same context.

All original rows should be preserved until a row-resolution policy is approved.

| Confidence class | Groups | Interpretation |
| --- | ---: | --- |
| Exact duplicate | 1 | All fields identical within the duplicate-looking group. |
| Strong likely duplicate | 5 | Same context with differences mainly in caffeine/screentime/energy/free text and no target/wellness conflict. |
| Conflicting same-context rows | 12 | Same heuristic context but conflicting target, wellness, match, or subjective fields. |
| Unresolved | 1 | Includes a correction/date-entry clue and should not be collapsed automatically. |

The exact duplicate pair is athlete `2068e7ec-51a8-441a-8053-2be7533e613f`,
Wednesday, sleep window `00:06:00` to `11:01:00`.

Examples:

| Class | Example |
| --- | --- |
| Strong likely duplicate | `e362...821c`, Friday: same sleep and handgrip, caffeine fields differ. |
| Conflicting same-context rows | `efa2...d271`, Saturday: same heuristic context but `handgrip_kg` differs. |
| Conflicting same-context rows | `6a2c...acd8`, Friday: missing sleep context, handgrip/fatigue/self-reported sleep duration differ. |
| Unresolved | `c887...44b`, Saturday: comment says a previous form was for the 19th but entered as the 20th. |

## Candidate Feature Timing Confidence

| Category | Feature group | Fields/examples | Current use |
| --- | --- | --- | --- |
| A. Definitely pre-outcome | Baseline questionnaire fields with non-null athlete ID | age, height, weight, sensor placement, nicotine, contraceptive use, sleep medication, ASBQ items | Track A allowed. |
| A. Definitely pre-outcome | Athlete identifier for within-athlete baselines | `id` | Track A allowed as grouping key, not as a causal predictor. |
| B. Probably pre-outcome but timing not fully verified | Prior-only handgrip history under source-row-order assumption | counts/rolling summaries of prior valid `handgrip_kg` only | Track B until row order is confirmed; suitable for Track A only after temporal order is documented. |
| B. Probably pre-outcome but timing not fully verified | Objective daily sleep summaries | `sleep_onset`, `sleep_offset`, `total_sleep_time`, `sleep_intervals`, longest interval, sleep period stats | Track B only. |
| B. Probably pre-outcome but timing not fully verified | Actigraphy-derived previous-night sleep or prior-day activity | `x`, `y`, `z`, `vm`, algorithm labels over a verified pre-outcome window | Track B only until day/window mapping is confirmed. |
| C. Timing-ambiguous | Subjective sleep/wellness | `sleepquality`, `sleepdura_sr`, `readiness`, `fatigue`, `soreness`, soreness locations | Exploratory only; may have been entered before or after performance. |
| C. Timing-ambiguous | Caffeine amount/timing | `cola`, `coffee`, `tea`, `espresso`, `caffeine_time`, `caffeine_mg` | Exploratory only; afternoon/evening values and duplicates require timing rules. |
| C. Timing-ambiguous | Structured match context | `matchday` | Exploratory only until definition and entry timing are verified. |
| D. Unsafe / clearly post-outcome for a prospective model | Free-text comments | `Comments (optional)` | Do not use as prospective model features; comments include explanations and correction clues. |
| D. Unsafe / clearly post-outcome for a prospective model | Full-athlete handgrip summary used as baseline | mean/median from all athlete handgrip rows | Unsafe for prospective prediction because it uses future outcomes. |

## Personal Normalization Leakage Risk

| Baseline type | Retrospective UI visualization | Prospective prediction | Exploratory analysis |
| --- | --- | --- | --- |
| Full-athlete mean/median baseline | Suitable if labeled retrospective; useful for showing an athlete's overall study-relative performance. | Unsafe; leaks future handgrip values into earlier predictions. | Acceptable only for descriptive analysis with clear leakage labeling. |
| Prior-only expanding baseline | Suitable, though early rows have sparse history. | Potentially suitable after source row order and duplicate policy are verified. | Good candidate for exploratory prototype work. |
| Prior-only rolling baseline | Suitable for trend-aware displays. | Potentially suitable after source row order, window size, minimum history, and duplicate handling are verified. | Good candidate for sensitivity analysis. |

No final normalized target should be created yet. The only prior-handgrip
calculation in this audit is eligibility counting.

## Analysis Tracks

### Track A: Confirmed-Safe / Conservative

Purpose: support a defensible minimum prototype under current evidence.

Candidate safe feature set:

- Non-null linked questionnaire fields.
- Athlete ID as a grouping key.
- Raw valid `handgrip_kg` as the target.

Current limitation: Track A cannot yet test whether recovery, recent activity,
or pre-workout lifestyle signals improve prediction because those
per-observation fields are not confirmed pre-outcome.

### Track B: Exploratory

Purpose: explore the likely modeling signal while preserving timing warnings.

Exploratory feature set:

- Prior-only handgrip history using source row order.
- Daily objective sleep summaries.
- Actigraphy sleep/activity windows using a documented exploratory windowing
  assumption.
- Caffeine amount and coarse timing.
- Subjective wellness: `sleepquality`, `sleepdura_sr`, `readiness`, `fatigue`,
  `soreness`, soreness locations.
- Structured match context: `matchday`.

Track B outputs must not be described as leakage-safe until timing and
athlete-day alignment are confirmed.

## Unresolved Questions

1. What calendar dates correspond to daily rows and actigraphy rows?
2. What does daily `weekday` represent?
3. When was handgrip measured relative to caffeine intake and survey entry?
4. Were objective daily sleep summaries generated before handgrip, after
   actigraphy processing, or by another system?
5. Which actigraphy algorithm should be authoritative for sleep windows?
6. What do `soreness_location_1` and `soreness_location_2` encode?
7. How should repeated same-weekday rows be resolved?
8. Can the missing-ID questionnaire row be linked?

## Recommended Next Investigation

Before building the canonical preprocessing pipeline, obtain or reconstruct a
data dictionary/protocol that defines:

- collection timing for handgrip, daily survey fields, caffeine, and wellness;
- calendar date mapping or an authoritative ordinal day index;
- duplicate/correction semantics;
- actigraphy sleep/wake label meanings and non-wear policy.

## Exact Recommended Next Implementation Step

Create a preprocessing design contract, not the final cleaned dataset. The next
code step should define a canonical observation-index specification with:

- immutable raw provenance fields: source file, source row number, athlete ID;
- parsed-but-not-modeled primitive fields such as numeric handgrip and
  weekday/time labels;
- explicit flags for duplicate-looking groups, missing questionnaire IDs, and
  actigraphy non-wear/missingness;
- separate Track A and Track B feature manifests with leakage labels.

This prepares the project to build a canonical preprocessing pipeline without
training models or silently resolving uncertain observations.

## Reproduction

Run:

```bash
python3 src/data/audit_rest.py
```

The script reads `data/raw/REST.zip` in place and reproduces the important
counts and checks in this report.
