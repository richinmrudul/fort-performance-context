# REST Preprocessing Design Contract

Contract date: 2026-08-31

Scope: design contract for a future preprocessing pipeline for the REST-based
Fort Performance Context prototype. This document does not implement the
pipeline, create cleaned modeling data, train models, build application code, or
modify raw REST files.

Source of truth:

- `docs/data_audit.md`
- `src/data/audit_rest.py`
- Raw archive to be read in place: `data/raw/REST.zip`

The prototype explores whether recovery, recent activity, and pre-workout
lifestyle signals can improve prediction of an athlete's strength performance
relative to their own personal baseline. Current evidence supports two separate
analysis tracks:

- Track A: conservative / defensible.
- Track B: exploratory / product hypothesis.

Track B results must never be presented as equally certain as Track A results.

## 1. Canonical Observation Grain

The initial canonical observation grain is one row per raw row in
`REST/daily_responses.csv`.

This grain is not a verified athlete-day. The raw data does not currently
provide calendar dates, a verified athlete-day key, or verified semantics for
daily `weekday`.

Requirements:

- Preserve every original raw daily-response row initially.
- Preserve source row order globally and per athlete.
- Do not collapse exact duplicates or duplicate-looking rows by default.
- Do not create cleaned modeling data as part of this contract.

Required observation index fields:

| Field | Type | Meaning | Construction |
| --- | --- | --- | --- |
| `raw_daily_row_index` | integer | Zero-based physical row position in `REST/daily_responses.csv`, excluding the header. | Assign from CSV read order before filtering, sorting, parsing, deduplication, or joins. |
| `athlete_row_index` | integer | Zero-based row position within one athlete's daily-response rows. | Group by raw `id` and assign in source row order. |
| `observation_id` | string | Stable unique identifier for one raw daily-response observation. | Deterministically hash immutable provenance fields. |

Recommended deterministic `observation_id` construction:

1. Use immutable raw provenance only:
   `source_archive_path`, `source_member_path`, and `raw_daily_row_index`.
2. Canonicalize as a UTF-8 string with explicit field names and separators:
   `source_archive_path=data/raw/REST.zip|source_member_path=REST/daily_responses.csv|raw_daily_row_index=<n>`.
3. Compute SHA-256.
4. Store as `rest_daily_<first_16_hex_chars>`.

This ID is deterministic, stable across preprocessing runs, unique for the raw
daily-response observation, and independent of mutable derived values such as
parsed handgrip, duplicate class, linked features, or model eligibility flags.

## 2. Raw Provenance Fields

Every canonical observation must carry these provenance fields.

| Field | Type | Meaning | Source | Raw or derived | Required | Allowed values |
| --- | --- | --- | --- | --- | --- | --- |
| `observation_id` | string | Stable unique row identifier. | Deterministic construction from immutable provenance. | Derived | yes | `rest_daily_` plus 16 lowercase SHA-256 hex characters. |
| `source_archive_path` | string | Path to the raw archive read by preprocessing. | Pipeline input. | Raw provenance | yes | For this prototype, `data/raw/REST.zip`. |
| `source_member_path` | string | Archive member containing the source row. | ZIP member name. | Raw provenance | yes | For canonical observations, `REST/daily_responses.csv`. |
| `raw_daily_row_index` | integer | Zero-based source row position in daily responses. | CSV read order. | Raw provenance | yes | Integer `>= 0`; unique within `source_member_path`. |
| `athlete_id` | string | Athlete identifier from daily `id`. | Raw daily column `id`. | Raw | yes | Non-null for audited daily rows; value must not be rewritten. |
| `athlete_row_index` | integer | Zero-based row position within athlete in source row order. | Derived from `athlete_id` and row order. | Derived | yes | Integer `>= 0`; unique within `athlete_id`. |
| `weekday` | string | Raw weekday label from daily responses. It is not a verified calendar date. | Raw daily column `weekday`. | Raw | yes | `Monday`, `Tuesday`, `Wednesday`, `Thursday`, `Friday`, `Saturday`, `Sunday`, or explicit parse error if future data differs. |
| `raw_handgrip_value` | string or null | Original raw daily handgrip text. | Raw daily column `handgrip_kg`. | Raw | yes | Raw value, including comma-decimal text such as `46,8`, or null. |
| `parsed_handgrip_kg` | float or null | Numeric handgrip parsed from `raw_handgrip_value`. | `handgrip_kg` with comma decimal converted to dot decimal. | Derived primitive | yes | Float in kg or null. |
| `handgrip_parse_status` | categorical string | Status of handgrip parsing. | Derived from raw and parsed handgrip. | Derived | yes | `valid`, `missing`, `parse_error`. |
| `duplicate_heuristic_group_id` | string or null | Identifier for rows grouped by a duplicate-looking heuristic. | Derived from `athlete_id`, `weekday`, `sleep_onset`, `sleep_offset`, `total_sleep_time`. | Derived | yes | Null when no group has more than one row; otherwise deterministic hash of the heuristic key. |
| `duplicate_classification` | categorical string | Duplicate policy state for the row. | Derived heuristic. | Derived | yes | `exact_duplicate`, `same_sleep_window_caffeine_variant`, `conflicting_target`, `conflicting_context`, `correction_hint`, `unresolved_duplicate`, `not_duplicate`. |
| `questionnaire_link_status` | categorical string | Whether the daily row can link to questionnaire data by athlete ID. | Daily `id` compared with non-null questionnaire `id`. | Derived | yes | `linked`, `missing_questionnaire_id`, `unlinked`, `not_attempted`. |
| `actigraphy_link_status` | categorical string | Whether the athlete has an actigraphy file by ID. | Daily `id` compared with actigraphy filename stems. | Derived | yes | `linked`, `unlinked`, `not_attempted`. |
| `actigraphy_sequence_validation_status` | categorical string | Status of the athlete's actigraphy 30-second sequence validation. | Actigraphy audit/checks. | Derived | yes | `valid_30s_sequence`, `invalid_sequence`, `missing_file`, `not_checked`. |
| `source_row_order_temporal_assumption` | boolean | Whether later processing is treating source row order as temporal. | Pipeline configuration. | Derived metadata | yes | `true` or `false`. Must default to `false` unless explicitly enabled. |

The duplicate heuristic key `athlete_id + weekday + sleep_onset + sleep_offset
+ total_sleep_time` is only a heuristic. It is not a verified athlete-day key.

## 3. Target And Baseline Fields

These fields may be created later by the preprocessing pipeline. This contract
defines their permitted semantics only; final baseline values must not be
calculated yet.

| Field | Type | Meaning | Descriptive-only | Allowed in prospective modeling | Depends on source-row-order temporal assumption | Uses only prior observations | Leakage risk | Future observations allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `handgrip_kg` | float or null | Modeling target copied from `parsed_handgrip_kg` after parse validation. | no | yes, as outcome only | no | not applicable | Low as a target; unsafe if used as a same-row feature. | no |
| `retrospective_personal_baseline` | float or null | Full-athlete mean or median handgrip across all valid rows for the athlete. | yes, unless explicitly labeled otherwise | no | no | no | High for prospective modeling because it uses future outcomes. | yes, by definition |
| `prior_expanding_baseline` | float or null | Expanding mean or median of valid prior handgrip rows for the athlete. | no | only if temporal order is documented or explicitly assumed for an exploratory run | yes until source row order is verified | yes | Moderate while row order is unverified; low after temporal order and duplicate policy are verified. | no |
| `prior_rolling_baseline` | float or null | Rolling mean or median over a configured count or window of prior valid handgrip rows. | no | only if temporal order, window size, and duplicate policy are documented | yes until source row order is verified | yes | Moderate while row order is unverified; low after temporal order and window policy are verified. | no |
| `performance_pct_baseline` | float or null | Percent of baseline, for example `handgrip_kg / baseline * 100`. | depends on baseline source | allowed only when the chosen baseline is prospective-valid | inherits chosen baseline | inherits chosen baseline | High if based on retrospective baseline; otherwise inherits prior baseline risk. | only if based on retrospective baseline, and then descriptive-only |
| `handgrip_history_count` | integer | Number of valid prior handgrip observations available before the current row. | no | only if temporal order is documented or explicitly assumed for exploratory run | yes until source row order is verified | yes | Moderate while row order is unverified. | no |
| `eligible_prior_1` | boolean | Current row has at least one prior valid handgrip. | no | only if source order is valid for the chosen track | yes until source row order is verified | yes | Moderate while row order is unverified. | no |
| `eligible_prior_2` | boolean | Current row has at least two prior valid handgrips. | no | only if source order is valid for the chosen track | yes until source row order is verified | yes | Moderate while row order is unverified. | no |
| `eligible_prior_3` | boolean | Current row has at least three prior valid handgrips. | no | only if source order is valid for the chosen track | yes until source row order is verified | yes | Moderate while row order is unverified. | no |
| `eligible_prior_5` | boolean | Current row has at least five prior valid handgrips. | no | only if source order is valid for the chosen track | yes until source row order is verified | yes | Moderate while row order is unverified. | no |

Full-athlete mean or median baselines may be used only for
retrospective/descriptive UI unless the output is explicitly labeled as
leakage-containing and excluded from prospective modeling.

Future observations are never allowed in prospective modeling features,
baselines, target normalization, eligibility flags, imputation, scaling, or
model selection.

## 4. Duplicate Handling Policy

Duplicate handling is conservative: do not collapse duplicate-looking rows by
default.

The heuristic duplicate grouping key is:

`athlete_id + weekday + sleep_onset + sleep_offset + total_sleep_time`

This key is not a verified athlete-day key. It is only a way to identify rows
that appear to refer to the same sleep/daily context.

| State | Criteria | Confidence | Original row preserved | Included in Track A by default | May be included in Track B | Aggregation allowed later | Manual review required |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `exact_duplicate` | All raw daily fields are identical within the heuristic group. | High that rows are duplicate copies; not high that one can be deleted silently. | yes | no | yes, with duplicate flag or optional sensitivity analysis | yes, only with documented rule and audit trail | recommended before aggregation |
| `same_sleep_window_caffeine_variant` | Same heuristic group; sleep context and target/wellness/context fields do not conflict; variation is limited to caffeine, screentime, energy, or free text. | Medium to high duplicate likelihood; caffeine timing semantics unresolved. | yes | no | yes, exploratory only | yes, after a documented caffeine-resolution rule | yes before aggregation |
| `conflicting_target` | Same heuristic group with more than one non-null or differing raw/parsed handgrip value. | High conflict; true target unknown. | yes | no | yes only for sensitivity analysis with explicit conflict flag | no automatic aggregation | yes |
| `conflicting_context` | Same heuristic group with conflicts in wellness, matchday, self-reported sleep duration, soreness, readiness, fatigue, or objective sleep fields. | Medium to high conflict. | yes | no | yes only with explicit conflict flag | no automatic aggregation unless a rule is approved | yes |
| `correction_hint` | Same heuristic group or row comment contains a correction/date-entry clue, such as text indicating a prior form, wrong day, or entered correction. | Medium; requires interpretation. | yes | no | yes only with explicit unresolved/correction flag | no automatic aggregation | yes |
| `unresolved_duplicate` | Same heuristic group but criteria do not cleanly identify one of the above states. | Low to medium. | yes | no | yes only with explicit unresolved flag | no automatic aggregation | yes |
| `not_duplicate` | Row is not part of a repeated heuristic group. | High for this heuristic only. | yes | yes if all other Track A criteria pass | yes | not applicable | no |

Rows excluded from a modeling subset must remain in canonical outputs and must
be explainable through duplicate flags and model-eligibility flags.

## 5. Track A Feature Manifest

Track A is the conservative / defensible analysis track. It includes only data
that can currently be justified without unsupported per-observation timing
assumptions.

Allowed fields:

- `athlete_id` as a grouping key, split key, and baseline grouping key.
- Valid `handgrip_kg` as the outcome.
- Linked questionnaire/static fields with non-null athlete ID, including audited
  groups such as demographics, sensor context, baseline lifestyle, baseline
  sleep behavior, and questionnaire comments only if treated as static metadata
  and not as prospective predictive text.
- Provenance and quality flags needed to audit inclusion or exclusion.
- Prior-only handgrip history only if source row order is explicitly treated as
  temporal for the run. Until temporal order is verified, these fields must be
  labeled assumption-dependent.

Prohibited fields:

- Same-row daily lifestyle, recovery, wellness, activity, caffeine, and match
  variables unless their timing is later verified as pre-outcome.
- Same-row objective daily sleep summaries unless their timing and meaning are
  later verified as pre-outcome.
- Actigraphy-derived features until a documented alignment/windowing rule and
  pre-outcome timing rule exist.
- Free-text daily comments as prospective model features.
- Any field with `timing_class = unsafe_post_outcome`.
- Any feature with `uses_future_information = true`.
- Retrospective full-athlete baselines for prospective modeling.

Assumptions:

- Athlete IDs link daily rows, questionnaire rows, and actigraphy files.
- Questionnaire fields with non-null athlete ID are static or baseline-like.
- Source row order is not temporal by default. If enabled, the run must set
  `source_row_order_temporal_assumption = true`.

Required quality flags:

- `has_valid_handgrip = true`
- `questionnaire_linked = true` for questionnaire features
- `duplicate_conflict_flag = false`
- `timing_verified = true` for any non-static feature
- `source_order_assumed_temporal = true` if prior handgrip history is included

Duplicate exclusions:

- Exclude `exact_duplicate`, `same_sleep_window_caffeine_variant`,
  `conflicting_target`, `conflicting_context`, `correction_hint`, and
  `unresolved_duplicate` by default.
- Include `not_duplicate` rows only if all other Track A criteria pass.

Minimum target-history requirements:

- For models using only static questionnaire features and raw handgrip outcome:
  no prior target history is required beyond a valid current target.
- For models using prior-only handgrip baselines or history: require at least
  one prior valid handgrip at minimum; prefer reporting sensitivity for
  thresholds 1, 2, 3, and 5 because the audit already counted these eligibility
  levels.

## 6. Track B Feature Manifest

Track B is exploratory / timing-unverified product research. It may include
fields that are plausible pre-workout signals but not yet verified as
pre-outcome. Track B outputs must carry timing and leakage labels and must not
be used for strong product claims until timing is verified.

| Feature group | Fields/examples | Why potentially useful | Timing uncertainty | Leakage risk | Evidence type | Product-claim status |
| --- | --- | --- | --- | --- | --- | --- |
| Objective daily sleep summaries | `sleep_onset`, `sleep_offset`, `total_sleep_time`, `sleep_intervals`, `longest_interval_start`, `longest_interval_end`, `longest_interval_duration`, `sleep period mean`, `sleep period standard deviation`, `sleep period msd` | Sleep duration and continuity may relate to recovery and strength readiness. | Unknown whether daily `weekday` is sleep night, survey day, performance day, or another convention; unknown whether summaries were computed before handgrip. | Medium to high until timing is verified. | Observational / derived from source summaries. | Exclude from product claims until timing is verified. |
| Actigraphy-derived sleep/activity | Future windowed features from `x`, `y`, `z`, `vm`, `CK`, `Oakley`, `Sadeh`, `fourier_HMM`, `LSTM` | Objective movement and sleep/wake signals may explain recovery and recent load. | No calendar-date mapping; no daily-row alignment rule; algorithm label meanings require confirmation. | High until alignment and pre-outcome windows are documented. | Sensor-derived. | Exclude from product claims until timing and alignment are verified. |
| Caffeine amount | `cola`, `coffee`, `tea`, `espresso`, `caffeine_mg` | Stimulant intake may affect handgrip performance and readiness. | Entry timing and intake timing relative to handgrip are unverified; duplicates may differ mainly by caffeine. | Medium to high; possible same-row post-outcome reporting. | Observational / self-reported. | Exclude from product claims until timing is verified. |
| Caffeine timing | `caffeine_time` | Timing may determine whether caffeine could plausibly affect performance. | Unknown whether field is before handgrip, after handgrip, or day-level recall. | High if recorded after performance or ambiguously tied to the day. | Observational / self-reported. | Exclude from product claims until timing is verified. |
| Subjective sleep and wellness | `sleepquality`, `sleepdura_sr`, `readiness`, `fatigue`, `soreness`, `soreness_location_1`, `soreness_location_2` | Perceived recovery and soreness may predict daily strength variation. | Unknown whether responses were entered before or after handgrip. | Medium to high; subjective values could be influenced by performance. | Subjective. | Exclude from product claims until timing is verified. |
| Match context | `matchday` | Match day or match proximity may affect fatigue, motivation, and performance. | Raw definition and entry timing are unverified. | Medium. | Observational / structured self-report. | Exclude from product claims until timing is verified. |
| Prior-only handgrip history | `prior_expanding_baseline`, `prior_rolling_baseline`, `handgrip_history_count`, eligibility flags | Captures athlete-specific strength baseline and recent performance trend. | Depends on treating daily source row order as temporal. | Medium until source row order is verified; unsafe if future rows are used. | Outcome-history derived. | May support exploratory hypotheses, but claims must state the row-order assumption. |

Track B may include duplicate-looking rows only with explicit duplicate flags.
Sensitivity analyses should report results with and without unresolved or
conflicting duplicate states.

## 7. Actigraphy Feature Contract

Future actigraphy-derived features must satisfy these constraints before they
can be joined to daily observations or included in Track B model tables.

Requirements:

- Validate the sequential 30-second structure before feature extraction for
  every athlete file.
- Use file row order and raw `weekday`/`time` only; do not invent calendar
  dates.
- Define an explicit pseudo-time/windowing convention before aggregation.
- Record the exact source window for every derived feature.
- Produce non-wear or invalid-classification summaries for every window.
- Flag partial windows.
- Do not perform an actigraphy-to-daily join until a documented
  alignment/windowing rule exists.
- Treat sleep/wake algorithm labels `s`, `w`, and `n` as unresolved until a data
  dictionary confirms their meanings.

Required window provenance for every actigraphy feature:

- `actigraphy_source_member_path`
- `actigraphy_athlete_id`
- `window_id`
- `window_start_weekday`
- `window_start_time`
- `window_end_weekday`
- `window_end_time`
- `window_epoch_count`
- `expected_epoch_count`
- `partial_window_flag`
- `nonwear_fraction`
- `algorithm_used`, when sleep/wake features are derived

Candidate future features, not yet implemented:

- `total_activity`
- `active_epoch_fraction`
- `inactivity_fraction`
- `morning_activity`
- `afternoon_activity`
- `evening_activity`
- `late_night_activity`
- `movement_variability`
- `sleep_fraction_ck`
- `wake_fraction_ck`
- `sleep_fraction_oakley`
- `wake_fraction_oakley`
- `sleep_fraction_sadeh`
- `wake_fraction_sadeh`
- `sleep_fraction_fourier_hmm`
- `wake_fraction_fourier_hmm`
- `sleep_fraction_lstm`
- `wake_fraction_lstm`
- `nonwear_fraction`

## 8. Timing And Leakage Metadata

Every candidate feature must have metadata before it can enter any model table.

Timing classes:

| Timing class | Meaning |
| --- | --- |
| `definitely_pre_outcome` | Source evidence confirms the feature is known before the handgrip outcome. |
| `probably_pre_outcome_unverified` | Feature is plausibly before the handgrip outcome, but current raw evidence does not prove it. |
| `timing_ambiguous` | Feature may be before or after the outcome, or its day/window convention is unclear. |
| `unsafe_post_outcome` | Feature clearly uses post-outcome information or explanations. |
| `unknown` | Timing has not been assessed. |

Required feature metadata:

| Metadata field | Type | Meaning |
| --- | --- | --- |
| `feature_name` | string | Name of the candidate feature. |
| `timing_class` | categorical string | One of the timing classes above. |
| `uses_future_information` | boolean | True if the feature uses any future observation relative to the modeled row. |
| `depends_on_temporal_order_assumption` | boolean | True if the feature is valid only when source row order is treated as temporal. |
| `allowed_track_a` | boolean | True only if the feature satisfies Track A rules. |
| `allowed_track_b` | boolean | True if the feature may be used in exploratory Track B with labels. |
| `source_fields` | list of strings | Raw or derived fields used to create the feature. |
| `timing_evidence` | string | Short explanation or citation to protocol/audit evidence. |
| `leakage_note` | string | Human-readable leakage risk. |

The preprocessing pipeline must fail schema validation if a model-table feature
lacks this metadata. Track A generation must fail if any included feature has
`uses_future_information = true`, `timing_class = unsafe_post_outcome`, or
`allowed_track_a = false`.

## 9. Quality Flags

Later preprocessing must produce these canonical quality flags.

| Flag | Type | Meaning |
| --- | --- | --- |
| `has_valid_handgrip` | boolean | `parsed_handgrip_kg` is non-null and `handgrip_parse_status = valid`. |
| `handgrip_parse_error` | boolean | Raw handgrip is present but cannot be parsed to numeric kg. |
| `duplicate_flag` | boolean | Row belongs to any repeated heuristic duplicate group. |
| `duplicate_conflict_flag` | boolean | Row is classified as `conflicting_target`, `conflicting_context`, `correction_hint`, or `unresolved_duplicate`. |
| `questionnaire_linked` | boolean | `athlete_id` matches one non-null questionnaire `id`. |
| `questionnaire_id_missing` | boolean | Questionnaire source row has missing `id`; this should be true only for the unlinked questionnaire record, not daily rows unless represented in a separate audit table. |
| `actigraphy_linked` | boolean | `athlete_id` matches one actigraphy file stem. |
| `actigraphy_sequence_valid` | boolean | Linked actigraphy file has valid uniform 30-second sequence. |
| `timing_verified` | boolean | All included non-static feature timing is verified as pre-outcome. |
| `timing_ambiguous` | boolean | At least one included feature has `timing_class = timing_ambiguous`, `probably_pre_outcome_unverified`, or `unknown`. |
| `source_order_assumed_temporal` | boolean | The run treats daily source row order as temporal. |
| `usable_track_a` | boolean | Row passes the Track A inclusion policy for the selected Track A manifest. |
| `usable_track_b` | boolean | Row passes the Track B inclusion policy for the selected Track B manifest. |

Quality flags are inclusion metadata, not row-deletion instructions. All
exclusions from modeling subsets must be reproducible from flags.

## 10. Preprocessing Acceptance Criteria

Before any modeling begins, the future preprocessing pipeline must satisfy all
of the following:

- Raw files remain untouched.
- Every output row maps back to raw provenance.
- `observation_id` values are deterministic and unique for canonical daily
  observations.
- No silent row deletion occurs.
- All exclusions are explainable through flags and manifest rules.
- All duplicate-looking rows are preserved before any optional aggregation.
- Any optional aggregation is separately documented and produces an audit trail
  linking input observation IDs to output rows.
- All baseline fields obey leakage rules.
- Track A and Track B manifests are enforced by code, not only by convention.
- No feature marked `unsafe_post_outcome` may appear in Track A.
- No feature with `uses_future_information = true` may appear in any
  prospective model table.
- Row counts are reproducible.
- Parsing is reproducible, including comma-decimal handgrip parsing.
- Outputs are deterministic across repeated runs on the same raw archive.
- Output schemas are validated.
- Unit or validation tests cover observation IDs, row counts, handgrip parsing,
  duplicate classification, baseline leakage constraints, and manifest
  enforcement.
- Missing values are handled explicitly; no implicit imputation is allowed.
- Every row removed from any modeling subset is audited with reason flags.

## 11. Recommended Future Output Tables

These tables describe intended future outputs only. Do not create them as part
of this documentation task.

| Table | Grain | Primary/unique key | Purpose | Allowed joins |
| --- | --- | --- | --- | --- |
| `survey_events` | One row per raw daily-response row. | `observation_id`; also unique on `source_member_path + raw_daily_row_index`. | Preserve raw daily survey/performance events, parsed primitive values, provenance, duplicate status, and quality flags. | Join to `athlete_static` by `athlete_id`; join to actigraphy only after an approved alignment rule. |
| `athlete_static` | One row per non-null questionnaire athlete ID. | `athlete_id`. | Store baseline questionnaire/static attributes and questionnaire provenance. | Join to `survey_events`, `canonical_observations`, and model tables by `athlete_id`. Missing-ID questionnaire row must remain in an audit output or exclusion log. |
| `actigraphy_daily_or_windowed` | One row per athlete per documented pseudo-time window. | `actigraphy_athlete_id + window_id`. | Store validated actigraphy-derived window summaries and source-window metadata. | Join to daily observations only through a documented alignment/windowing table or rule; never by weekday alone. |
| `canonical_observations` | One row per raw daily-response row enriched with approved primitive links and metadata. | `observation_id`. | Main canonical preprocessing output before track-specific filtering. | Join to `athlete_static` by `athlete_id`; join to actigraphy features only after documented alignment. |
| `track_a_model_table` | One row per Track A eligible modeled observation. | `observation_id` if no aggregation; otherwise a documented model-row ID with source observation IDs. | Conservative model-ready subset with only Track A allowed fields. | May join only to Track A-approved static/baseline fields. |
| `track_b_model_table` | One row per Track B eligible modeled observation. | `observation_id` if no aggregation; otherwise a documented model-row ID with source observation IDs. | Exploratory model-ready subset with timing-unverified features and warnings. | May join to Track B-approved daily and actigraphy features through documented rules. |

## 12. Implementation Boundary And Next Step

### Verified Assumptions

- Daily responses contain 370 rows and 21 non-null athlete IDs.
- Questionnaire data contains 22 rows, 21 non-null athlete IDs, and one missing
  `id`.
- Actigraphy contains 21 athlete-level CSV files, one per athlete.
- The same 21 non-null athlete IDs are shared across daily responses,
  questionnaire rows, and actigraphy filenames.
- There are 142 usable handgrip observations after comma-decimal parsing.
- Actigraphy files have a verified uniform 30-second sequence by file row order.
- All athlete actigraphy files share the same weekday/time sequence.
- Daily response source row order follows weekday progression with same-weekday
  repeats allowed, but this is not proof of true temporal order.

### Unresolved Assumptions

- Calendar dates for daily rows.
- Calendar dates for actigraphy rows.
- A true athlete-day key.
- What daily `weekday` represents.
- Whether handgrip was measured before or after caffeine, wellness, survey
  entry, and match-context fields.
- Whether objective daily sleep summaries were generated before handgrip.
- Which actigraphy window maps to each daily row.
- Which actigraphy sleep/wake algorithm is authoritative.
- What `soreness_location_1` and `soreness_location_2` encode.
- How repeated same-weekday rows should be resolved.
- Whether the missing-ID questionnaire row can be linked.

### Track A Summary

Track A may use linked static questionnaire fields, athlete ID as a grouping
key, and valid raw handgrip as the outcome. Prior-only handgrip history may
enter Track A only when source row order is explicitly treated as temporal for
the run and all leakage metadata marks the fields as prior-only and
future-free. Same-row lifestyle, recovery, activity, caffeine, subjective
wellness, match context, and actigraphy features are excluded until timing is
verified.

### Track B Summary

Track B may explore objective daily sleep summaries, actigraphy-derived
sleep/activity windows, caffeine amount/timing, `sleepquality`,
`sleepdura_sr`, `readiness`, `fatigue`, `soreness`, `matchday`, and prior-only
handgrip history. These features must be labeled timing-unverified or
assumption-dependent until protocol evidence resolves timing and alignment.

### Leakage Rules

- Never use future observations in prospective model features.
- Never use full-athlete mean or median baselines in prospective modeling.
- Never include `unsafe_post_outcome` features in Track A.
- Prior-only baselines must be computed strictly from earlier valid handgrip
  observations within athlete.
- Prior-only fields depend on temporal source-row-order validity until an
  authoritative date or ordinal day index exists.
- Any feature without timing/leakage metadata is ineligible for model tables.

### Duplicate Policy Summary

- Preserve all original daily rows.
- Use duplicate classifications as flags, not deletion commands.
- Treat `athlete_id + weekday + sleep-window values` as heuristic only.
- Exclude duplicate-looking and conflicting rows from Track A by default.
- Allow Track B inclusion only with explicit flags and sensitivity reporting.
- Require manual review before aggregation or conflict resolution.

### Acceptance Checklist

- [ ] Raw files remain untouched.
- [ ] Every output row maps back to raw provenance.
- [ ] Deterministic `observation_id` is implemented and tested.
- [ ] No silent row deletion occurs.
- [ ] Duplicate-looking rows are preserved before optional aggregation.
- [ ] All exclusions are explainable via flags.
- [ ] Baseline fields obey prior-only and future-information rules.
- [ ] Track A and Track B manifests are enforced.
- [ ] No `unsafe_post_outcome` feature appears in Track A.
- [ ] Row counts, parsing, and outputs are reproducible.
- [ ] Schema validation is implemented.
- [ ] Unit or validation tests cover core contract rules.
- [ ] Missing values are explicitly handled.
- [ ] Rows removed from modeling subsets are audited.

### Exact Recommended Next Implementation Step

Build the preprocessing pipeline that implements this contract exactly.

Do not jump to modeling.
