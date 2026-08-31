# Track B Exploratory Modeling

Updated: 2026-08-31

Track B evaluates whether exploratory recovery, recent activity, caffeine,
wellness, and match-context features improve handgrip prediction beyond the
Track A prior-only baselines. These results are timing-unverified product
research. They do not prove causality and do not yet prove pre-workout
deployability.

## Dataset And Splits

Input table: `data/processed/track_b_features.csv`

Rows: 84 primary Track A-eligible observations from 15 athletes.

Targets:

- `handgrip_kg`
- `target_pct_prior_median_baseline`

Evaluation follows Track A:

- No random row split.
- Leave-one-athlete-out across 15 athletes.
- Chronological holdout using the last 30 percent of eligible rows per athlete,
  minimum one row, under the source-row-order temporal assumption.

Forbidden columns are excluded from model inputs, including observation IDs,
athlete IDs, row indices, target fields, retrospective descriptive baselines,
free-text comments, leakage labels, manifest/status labels, and actigraphy
join/status metadata.

## Feature Sets

The evaluator tests these ablations:

| Feature set | Contents |
| --- | --- |
| `prior_history_only` | Track A prior handgrip history fields |
| `prior_history_plus_daily_sleep` | Prior history plus objective sleep summaries and sleep timing |
| `prior_history_plus_caffeine` | Prior history plus caffeine amount, timing category, drink counts, and caffeine-present flag |
| `prior_history_plus_wellness` | Prior history plus subjective sleep, readiness, fatigue, and soreness fields |
| `prior_history_plus_match_context` | Prior history plus conservatively encoded match context |
| `prior_history_plus_actigraphy` | Prior history plus provisional weekday-only actigraphy activity and sleep/wake fractions |
| `all_track_b_exploratory` | All allowed Track B exploratory modeling features |

Numeric missing values are imputed from training-fold medians only.
Categorical missing values are imputed with an explicit `__MISSING__` category.
Categorical levels are learned inside each training fold.

## Models

Always evaluated:

- `prior_expanding_median`
- `prior_expanding_mean`

In this environment, `sklearn` is not installed. The evaluator therefore used a
transparent fallback model:

- `ridge_numpy_fallback`, a low-complexity ridge regression implemented with
  NumPy and `alpha = 10.0`

If `sklearn` is installed in a future environment, the evaluator can also run:

- `ridge_regularized_linear`
- `random_forest_small` with conservative depth and leaf-size settings

## Results

Track A baseline comparison:

| Split | Model | MAE kg | RMSE kg | MAE % baseline | n | Folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Leave-one-athlete-out | Track A `prior_expanding_mean` | 1.625 | 2.365 | 4.745 | 84 | 15 |
| Leave-one-athlete-out | Track A `prior_expanding_median` | 1.702 | 2.483 | 4.963 | 84 | 15 |
| Chronological holdout | Track A `prior_expanding_mean` | 1.456 | 1.973 | 4.327 | 33 | 1 |
| Chronological holdout | Track A `prior_expanding_median` | 1.624 | 2.234 | 4.774 | 33 | 1 |

Best contextual Track B fallback models:

| Split | Feature set | MAE kg | RMSE kg | MAE % baseline | Delta MAE vs Track A prior mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Leave-one-athlete-out | `prior_history_plus_match_context` | 1.690 | 2.263 | 4.968 | +0.064 |
| Chronological holdout | `prior_history_plus_caffeine` | 1.807 | 2.260 | 5.385 | +0.351 |

Full fallback ridge ablation ranking:

| Split | Feature set | MAE kg | RMSE kg | MAE % baseline |
| --- | --- | ---: | ---: | ---: |
| Leave-one-athlete-out | `prior_history_plus_match_context` | 1.690 | 2.263 | 4.968 |
| Leave-one-athlete-out | `prior_history_only` | 1.751 | 2.322 | 5.134 |
| Leave-one-athlete-out | `prior_history_plus_wellness` | 1.784 | 2.377 | 5.330 |
| Leave-one-athlete-out | `prior_history_plus_caffeine` | 1.792 | 2.366 | 5.294 |
| Leave-one-athlete-out | `prior_history_plus_actigraphy` | 1.831 | 2.399 | 5.418 |
| Leave-one-athlete-out | `prior_history_plus_daily_sleep` | 2.068 | 2.691 | 6.192 |
| Leave-one-athlete-out | `all_track_b_exploratory` | 2.387 | 3.221 | 7.242 |
| Chronological holdout | `prior_history_plus_caffeine` | 1.807 | 2.260 | 5.385 |
| Chronological holdout | `prior_history_only` | 1.832 | 2.266 | 5.510 |
| Chronological holdout | `prior_history_plus_match_context` | 1.986 | 2.365 | 6.118 |
| Chronological holdout | `prior_history_plus_daily_sleep` | 2.015 | 2.528 | 6.157 |
| Chronological holdout | `prior_history_plus_wellness` | 2.042 | 2.497 | 6.251 |
| Chronological holdout | `prior_history_plus_actigraphy` | 2.162 | 2.544 | 6.693 |
| Chronological holdout | `all_track_b_exploratory` | 2.316 | 2.836 | 7.186 |

## Interpretation

No Track B contextual feature group improved MAE over the Track A
`prior_expanding_mean` baseline in either evaluation split.

The most useful model-supported contextual signal in this run is weak and
split-dependent:

- Match context is the best contextual ablation in leave-one-athlete-out, but
  its MAE is still 0.064 kg worse than Track A prior mean.
- Caffeine is the best contextual ablation in chronological holdout, but its
  MAE is still 0.351 kg worse than Track A prior mean.

Daily sleep, actigraphy, wellness, and the full exploratory feature set do not
show reliable improvement in this small fallback-model evaluation. The all-in
feature set performs worst in both splits, which is consistent with too many
timing-unverified and partly sparse features for 84 rows.

## Product Implications

Track A remains the defensible baseline for this prototype: estimate expected
handgrip from an athlete's own prior performance history and report observed
percent of baseline after the performance is known.

Track B signals remain candidates for future product work, not deployable
pre-workout predictors. Fort may consider using recovery, caffeine, wellness,
match context, and actigraphy signals in an internal expected-performance model
only after collection timing and actigraphy alignment are confirmed.

For athlete-facing UI, the safer current direction is to show observed percent
of baseline and contextual signals after performance, not a hidden pre-workout
estimate that depends on timing-unverified same-row features.

## Cautions

- The sample is small: 84 rows from 15 athletes.
- Source row order is assumed temporal for prior history and chronological
  holdout, but calendar dates are not verified.
- Daily response timing relative to handgrip is unresolved.
- Actigraphy is joined by athlete and weekday only; it is not a verified
  athlete-day or pre-handgrip window.
- These results are predictive comparisons only. They do not establish
  causal effects of sleep, caffeine, soreness, match context, or activity.

Detailed outputs:

- `reports/track_b_metrics.json`
- `reports/track_b_predictions.csv`
- `reports/model_comparison.json`
