# Track A Modeling

Updated: 2026-08-31

Track A is the conservative REST prototype modeling path. It uses only valid
handgrip outcomes, athlete grouping, row provenance, duplicate/default-exclusion
flags, and prior-only handgrip history computed within athlete source row order.

## Allowed Inputs

Model inputs are limited to:

- `prior_valid_handgrip_count`
- `prior_expanding_handgrip_mean`
- `prior_expanding_handgrip_median`
- `prior_expanding_handgrip_std`

`athlete_id` is used only as a grouping, fold, and baseline context key.
Provenance fields are preserved for audit but are not predictors.

Leakage labels:

| Feature group | Track A label |
| --- | --- |
| Prior handgrip count/mean/median/std | Prior-only, source-row-order-assumed temporal |
| Same-row sleep, caffeine, wellness, matchday, questionnaire, or actigraphy-derived fields | Prohibited until pre-outcome timing is verified |
| Retrospective full-athlete baseline fields | Prohibited as prospective model inputs because they use future outcomes |

## Exclusions

Track A excludes same-row recovery, lifestyle, wellness, match context,
questionnaire, and actigraphy-derived fields. The raw files do not prove those
signals were collected before the handgrip outcome, so including them could let
the model learn from post-outcome or same-outcome context. This run also excludes
rows where `exclude_from_modeling_default` is true, including duplicate-looking
or conflicting rows.

## Dataset Counts

Input observations: 370 rows.

Rows with valid `handgrip_kg`: 142.

Primary Track A modeling rows: 84.

Primary athletes: 15.

Excluded valid handgrip rows: 58. These are excluded because they have fewer
than 3 prior valid handgrips, fail the prior-median target requirement, or are
default-excluded by duplicate/conflict policy.

## Evaluation

Temporary temporal assumption: source row order is treated as within-athlete
time order only for prior-history construction and chronological holdout.
No random row split is used.

Evaluated predictors:

- `prior_expanding_median`: predicts the prior expanding median handgrip.
- `prior_expanding_mean`: predicts the prior expanding mean handgrip.
- `ridge_prior_only_numeric`: evaluated only if `sklearn` is installed.

Splits:

- Leave-one-athlete-out is feasible with 15 athletes. For the prior median/mean
  predictors, each athlete is labeled as the held-out fold.
- Chronological holdout uses the last 30 percent of eligible rows per athlete,
  with a minimum of one row per athlete.

Metrics are written to `reports/track_a_metrics.json`, and row-level
predictions are written to `reports/track_a_predictions.csv`.

## Results

The key baseline comparison is against `prior_expanding_median`, the naive
athlete prior median baseline. Current metrics:

| Split | Model | n | Athletes | MAE kg | RMSE kg | MAE percent of prior median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Leave-one-athlete-out | prior expanding median | 84 | 15 | 1.702 | 2.483 | 4.963 |
| Leave-one-athlete-out | prior expanding mean | 84 | 15 | 1.625 | 2.365 | 4.745 |
| Chronological holdout | prior expanding median | 33 | 15 | 1.624 | 2.234 | 4.774 |
| Chronological holdout | prior expanding mean | 33 | 15 | 1.456 | 1.973 | 4.327 |

In the current environment, `sklearn` is not installed, so ridge regression is
recorded as skipped in the metrics JSON. The evaluated Track A result is
therefore a direct comparison of prior expanding median versus prior expanding
mean. Prior expanding mean is modestly better than the naive prior median on
these rows, but this does not yet establish a richer predictive model. Track A
can establish an athlete's own performance baseline and quantify deviations
from that baseline, but it does not yet show that recovery, lifestyle, or
activity context improves prediction.

## Unresolved Assumptions

- Source row order is not verified calendar time.
- Daily `weekday` is not a verified athlete-day or date key.
- Handgrip collection time is unknown.
- Same-row sleep, caffeine, wellness, matchday, and questionnaire timing is
  unknown relative to handgrip.
- Actigraphy alignment to daily rows is not documented.
- Duplicate-looking rows remain preserved in canonical observations and require
  manual review before any aggregation policy.

## Product Story

Track A supports a defensible prototype claim that Fort can calculate personal
handgrip baselines and evaluate performance relative to prior athlete history.
It does not support claims that sleep, caffeine, wellness, match context,
questionnaire variables, or actigraphy improve readiness prediction. Those
claims belong to a future Track B analysis only after timing and alignment are
verified.
