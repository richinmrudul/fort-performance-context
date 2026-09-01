# Track B Temporal-Feasibility Audit

Updated: 2026-09-01

## Decision

The existing public REST dataset cannot support a defensible pre-workout
contextual model using features provably available before the handgrip
performance target.

Final machine-readable decision:

`metadata_insufficient`

Generated report:

- `reports/track_b_timing_audit.json`

Generation command:

```bash
python3 -m src.features.audit_track_b_timing
```

This audit does not train, tune, compare, or rerun Track B models. Previous
Track B modeling reports remain unchanged.

## Core Finding

The 84 Track B rows have valid handgrip targets and exploratory contextual
features, but none has a defensible prediction cutoff. A cutoff would need to
represent the latest time a prospective prediction could have been made before
handgrip measurement. The REST public files do not provide handgrip measurement
timestamps, survey submission timestamps, within-row event order, calendar
dates, or timezone metadata.

Source row order, file order, ZIP metadata, ingestion time, same-day grouping,
weekday labels, and assumed times of day are not valid cutoff evidence.

## Handgrip Timing

Source field:

- `REST/daily_responses.csv:handgrip_kg`

Timing evidence classification:

| Evidence type | Finding |
| --- | --- |
| documented timestamp | none |
| inferred ordering | source row order exists but is not proof |
| file or row ordering | present, not valid timing evidence |
| date-only association | weekday label only; no calendar date |
| timing evidence | unavailable |

Status: `timestamp_unavailable`

Handgrip timing cannot be established from the public REST dataset.

## Objective Sleep

Objective sleep fields include `total_sleep_time`, `sleep_intervals`,
`longest_interval_duration`, `sleep period mean`, `sleep period standard
deviation`, `sleep period msd`, and time-of-day interval fields such as
`sleep_onset` and `sleep_offset`.

These fields are carried on the same daily response row as handgrip, with
athlete identity and row index available. They do not include calendar dates,
timezone, a verified target-day key, or proof that the interval ended before
handgrip.

Status: `date_only_ambiguous`

No objective sleep field is pre-performance safe under the public metadata.

## Caffeine

Fields:

- `caffeine_mg`
- `caffeine_time`
- `coffee`
- `cola`
- `tea`
- `espresso`

Observed daypart labels include values such as Morning, Afternoon, and Evening.
Those labels do not establish usable ordering because handgrip time is unknown.
Morning cannot be assumed to precede handgrip without independent handgrip
timing.

Classification: ambiguous

Status: `ordering_unverified`

## Subjective Wellness

Fields include subjective sleep quality, self-reported sleep duration,
readiness, fatigue, soreness, and soreness locations. They are same-row daily
survey fields with no recording timestamp or verified within-row ordering
relative to handgrip.

Status: `date_only_ambiguous`

These fields remain exploratory and timing-ambiguous.

## Actigraphy

Actigraphy files provide athlete identity through filename stems and rows with
weekday plus time-of-day labels. The existing Track B feature builder joins
actigraphy to Track B observations by athlete and weekday only.

Available support:

| Alignment type | Supported |
| --- | --- |
| exact athlete-day alignment | no |
| relative-day alignment | no |
| weekday-only alignment | yes |
| defensible target-day alignment | no |

Status: `athlete_day_alignment_unverified`

Weekday-only matching is not a defensible calendar alignment and cannot prove
previous-day or previous-night activity before a specific handgrip observation.

## Timing-Safe Cohort

Rows checked: 84

Rows with defensible prediction cutoff: 0

Timing-safe cohort rows: 0

Timing-safe cohort athletes: 0

Surviving contextual feature groups: none

Excluded feature groups:

| Feature group | Status | Reason |
| --- | --- | --- |
| objective sleep | `date_only_ambiguous` | interval cannot be proven before handgrip |
| caffeine | `ordering_unverified` | daypart labels cannot be ordered against unknown handgrip time |
| subjective wellness | `date_only_ambiguous` | survey recording time is not proven before handgrip |
| actigraphy | `athlete_day_alignment_unverified` | weekday-only join lacks exact athlete-day alignment |
| match context | `date_only_ambiguous` | does not establish cutoff or feature availability |
| prior history | `ordering_unverified` | relies on source row order rather than documented time |

Missingness alone did not make rows unsafe. Ambiguous temporal ordering did.

## Product Interpretation

Track B contextual features can still be useful for exploratory analysis,
missingness review, and future data-collection planning. They should not be used
for a pre-workout prospective contextual performance estimate, efficacy claim,
or athlete-facing recommendation until the dataset includes target timestamps,
feature timestamps, calendar dates, timezone semantics, and exact athlete-day
alignment.
