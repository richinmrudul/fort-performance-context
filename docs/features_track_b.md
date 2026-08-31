# Track B Exploratory Features

Updated: 2026-08-31

Track B starts from Track A's conservative primary modeling anchor. The output
table keeps only rows with a valid handgrip target, at least 3 prior valid
handgrips, and no default duplicate/conflict exclusion. It preserves the Track A
prior-only history fields and then adds exploratory recovery, activity,
caffeine, wellness, and match-context signals.

Track B is a product-hypothesis dataset, not a confirmed prospective modeling
dataset. Same-row daily response fields and actigraphy summaries are
timing-unverified because the raw REST files do not prove when those values were
captured relative to handgrip testing.

## Artifacts

Generated files:

- `data/processed/track_b_features.csv`
- `reports/track_b_feature_summary.json`

Exact generated shape:

- Rows: 84
- Columns: 80
- Manifest feature entries: 68
- Track A prior-history features: 4
- Added exploratory Track B features: 64
- Athletes represented: 15

Feature groups in the manifest:

| Source group | Feature count | Timing status |
| --- | ---: | --- |
| `prior_history` | 4 | `prior_only_prospective_assumption` |
| `daily_sleep_objective` | 6 | `timing_unverified` |
| `sleep_timing` | 4 | `timing_unverified` |
| `caffeine` | 7 | `timing_unverified` |
| `wellness` | 7 | `timing_unverified` |
| `match_context` | 4 | `timing_unverified` |
| `actigraphy_weekday_window` | 36 | `timing_unverified` |

## Included Signals

Track A prior-only fields are preserved:

- `prior_valid_handgrip_count`
- `prior_expanding_handgrip_mean`
- `prior_expanding_handgrip_median`
- `prior_expanding_handgrip_std`

Daily objective sleep summaries are parsed as numeric fields:

- `objective_total_sleep_time`
- `objective_sleep_intervals`
- `objective_longest_interval_duration`
- `objective_sleep_period_mean`
- `objective_sleep_period_standard_deviation`
- `objective_sleep_period_msd`

Sleep timing fields are parsed to minutes after midnight:

- `sleep_onset_minutes_after_midnight`
- `sleep_offset_minutes_after_midnight`
- `longest_interval_start_minutes_after_midnight`
- `longest_interval_end_minutes_after_midnight`

Caffeine fields include amount, category, parsed drink counts, and presence.
The drink-count parser maps `4 or more` to `4` as a conservative lower bound.

Wellness fields include subjective sleep, readiness, fatigue, soreness, and
soreness locations. Match context is encoded conservatively with the raw matchday
category, a present flag, a match-day flag, and a relative-day numeric value
only when the source value has a single unambiguous `MD`, `MD-#`, or `MD+#`
form.

Free-text daily comments are intentionally excluded. They may contain
correction hints or post-hoc explanations and are not model features in Track B.

## Actigraphy Join

The actigraphy features use a provisional weekday-only alignment:

1. Each athlete actigraphy file is read from `data/raw/REST.zip` in place.
2. File row order and weekday/time labels are used to validate the sequential
   30-second pseudo-time sequence.
3. Rows are aggregated by athlete and weekday label only.
4. Track B observations join to those summaries by `athlete_id` and `weekday`.

This is not a verified athlete-day or pre-handgrip window. Every row carries:

- `actigraphy_join_assumption = weekday_only_unverified`
- `actigraphy_source_sequence_valid`
- `actigraphy_source_sequence_status`
- `actigraphy_weekday_window_id`
- `actigraphy_shared_provisional_window`

All 84 Track B rows link to actigraphy files with `valid_30s_sequence` source
status. Because the join is weekday-only, 46 of 84 rows share the same
provisional athlete-weekday actigraphy window with at least one other retained
observation. Those rows are preserved and flagged; they are not collapsed.

Actigraphy weekday row counts in the Track B table are 5,760, 6,960, or 8,640
epochs, depending on how often that weekday appears in the repeated
weekday/time sequence. These are matched weekday rows, not verified daily
windows.

## Missingness And Non-Wear Concerns

Highest missingness among exploratory daily features:

| Feature | Missing | Missing % |
| --- | ---: | ---: |
| `tea_count` | 81 | 96.43 |
| `cola_count` | 77 | 91.67 |
| `espresso_count` | 68 | 80.95 |
| `coffee_count` | 57 | 67.86 |
| `soreness_location_2` | 46 | 54.76 |
| `caffeine_time_category` | 36 | 42.86 |
| `sleepquality` | 21 | 25.00 |
| `soreness_location_1` | 16 | 19.05 |

Objective sleep and parsed sleep-timing features each have 7 missing values
(8.33 percent) in the Track B table. `caffeine_mg`, `readiness`, and
`sleepdura_sr` have no missing values among the 84 retained rows.

Actigraphy aggregate features have no missing values after the provisional join,
but non-wear or invalid-classification fractions vary widely. The
all-algorithm non-wear/invalid fraction ranges from 0.0000 to 0.8598, with a
median of 0.1229. This should be treated as a major quality concern for any
future activity/recovery modeling.

## Product Interpretation

If timing metadata is later confirmed, these Track B groups are plausible Fort
product signals:

- Objective sleep duration, continuity, and sleep timing.
- Recent activity and sleep/wake actigraphy summaries with documented windows.
- Caffeine amount and intake timing before performance.
- Subjective readiness, fatigue, soreness, and sleep quality.
- Match proximity and match-day context.

Until then, Track B must not be used for model evaluation, product efficacy
claims, or prospective-readiness claims. Its purpose is to expose candidate
signals, missingness, and alignment risks for follow-up data collection and
metadata confirmation.
