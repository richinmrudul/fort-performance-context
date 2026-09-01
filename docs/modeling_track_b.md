# Track B Hardened Exploratory Modeling

Updated: 2026-08-31

Track B evaluates whether timing-unverified contextual features add predictive
value beyond Track A personal handgrip history. This is an evaluation-hardening
result, not a search for a positive finding. Track B remains exploratory because
daily response timing and actigraphy alignment are not verified as pre-handgrip.

## Environment

Generation command:

```bash
python3 -m src.models.evaluate_track_b
```

Recorded dependency versions:

| Dependency | Version |
| --- | --- |
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| pandas | 2.3.2 |
| scikit-learn | 1.5.2 |

The official hardened evaluator requires scikit-learn. It does not silently
write official reports with the older NumPy fallback.

## Dataset And Splits

Input table: `data/processed/track_b_features.csv`

Rows: 84 primary Track A-eligible observations from 15 athletes.

Targets:

- `handgrip_kg`
- `target_pct_prior_median_baseline`

Evaluation schemes:

- Leave-one-athlete-out: 15 folds, 84 evaluated rows.
- Chronological holdout: last 30 percent per athlete, minimum one row, 33
  evaluated rows.

No random row split is used.

## Leakage Controls

The evaluator excludes IDs, row-order fields, targets, retrospective descriptive
baselines, free-text comments, leakage/status labels, split labels, output
columns, and actigraphy join/status metadata from model inputs.

Preprocessing is fold-local through scikit-learn `Pipeline` and
`ColumnTransformer`:

- numeric imputation uses training-fold medians only;
- scaling is fit on training rows only;
- categorical missing values become `__MISSING__`;
- one-hot vocabularies are learned from training rows only;
- unseen test categories are ignored without refitting;
- model fitting uses only training rows.

Regression tests cover held-out athlete isolation, chronological no-future-row
isolation, paired Track A/Track B row alignment, forbidden-column failures, and
fold-local preprocessing.

Detailed machine-readable audit:

- `reports/track_b_feature_audit.json`

## Feature Sets And Estimators

Feature sets:

- `prior_history_only`
- `prior_history_plus_daily_sleep`
- `prior_history_plus_caffeine`
- `prior_history_plus_wellness`
- `prior_history_plus_match_context`
- `prior_history_plus_actigraphy`
- `all_track_b_exploratory`

Estimators:

- `sklearn_ridge`: `sklearn.linear_model.Ridge(alpha=10.0)`
- `sklearn_random_forest`:
  `RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=5,
  random_state=20260831)`

Track A `prior_expanding_mean` is the benchmark. Deltas are always:

`Track B MAE - Track A MAE`

Negative means Track B improved; positive means Track B was worse.

## Main Results

Best contextual sklearn model in each scheme:

| Scheme | Estimator | Feature set | Track A MAE | Track B MAE | Delta MAE |
| --- | --- | --- | ---: | ---: | ---: |
| Leave-one-athlete-out | Ridge | `prior_history_plus_match_context` | 1.625 | 1.780 | +0.155 |
| Chronological holdout | Ridge | `prior_history_plus_caffeine` | 1.456 | 1.843 | +0.387 |

Random Forest did not improve on Track A:

| Scheme | Best RF feature set | Track B MAE | Delta MAE |
| --- | --- | ---: | ---: |
| Leave-one-athlete-out | `prior_history_only` | 2.194 | +0.569 |
| Chronological holdout | `prior_history_only` | 1.854 | +0.398 |

Every sklearn Ridge and Random Forest feature-set comparison had a positive
MAE delta versus Track A `prior_expanding_mean`.

## Bootstrap Uncertainty

Bootstrap method:

- compute out-of-fold predictions first;
- resample athletes as clusters with replacement;
- retain all evaluated rows for each sampled athlete;
- compute Track A MAE, Track B MAE, and paired MAE delta on the same resampled
  observations;
- fixed seed: 20260831;
- iterations: 2,000;
- confidence level: 95 percent.

Representative intervals for the best contextual models:

| Scheme | Estimator / feature set | Track A MAE CI | Track B MAE CI | Paired delta CI |
| --- | --- | --- | --- | --- |
| Leave-one-athlete-out | Ridge `prior_history_plus_match_context` | 1.269 to 2.030 | 1.450 to 2.163 | -0.064 to +0.391 |
| Chronological holdout | Ridge `prior_history_plus_caffeine` | 1.029 to 1.854 | 1.407 to 2.235 | +0.074 to +0.781 |

These are descriptive uncertainty estimates conditional on the observed
athletes, feature construction, evaluation design, and fixed out-of-fold
predictions. They do not represent causal effects and do not capture every
source of model-training uncertainty because models are not refit inside each
bootstrap replicate.

Detailed report:

- `reports/track_b_bootstrap.json`

## Per-Athlete Findings

Per-athlete results are split into:

- `minimally_interpretable`: at least 3 evaluated rows;
- `anecdotal_too_few_rows`: fewer than 3 evaluated rows.

Some athletes show repeated descriptive improvement under both schemes for
specific estimator/feature-set combinations, but those cases are isolated. They
do not reverse the cohort-level result and are often estimator- or feature-set
specific. The repeated minimally interpretable improvements are concentrated in
two athlete IDs already present in processed modeling output:

- `687552ea-5963-45df-8680-c4c6058eab87`
- `e362b878-9920-4dcf-9ab7-687a194a821c`

These should not be described as stable individual benefit. They are descriptive
error-pattern observations in a small, timing-unverified dataset.

Detailed report:

- `reports/track_b_per_athlete.csv`

## Stability

The stability summary checks each feature set by estimator and scheme using MAE
deltas, bootstrap intervals, and the proportion of minimally interpretable
athletes improved.

All 14 estimator/feature-set stability rows are classified as
`consistently_worse` by point-estimate direction across leave-one-athlete-out
and chronological evaluation. No feature group is consistently improved across
both schemes.

Detailed report:

- `reports/track_b_stability.json`

## Missingness

Missingness is measured before imputation on the 84 modeling rows.

Group coverage:

| Feature group | Rows with any coverage | Complete rows | Athletes with any coverage |
| --- | ---: | ---: | ---: |
| prior history | 84 | 84 | 15 |
| daily sleep | 77 | 77 | 14 |
| caffeine | 84 | 0 | 15 |
| wellness | 84 | 26 | 15 |
| match context | 84 | 72 | 15 |
| actigraphy | 84 | 84 | 15 |

Highest feature-level missingness:

| Feature | Missing % |
| --- | ---: |
| `tea_count` | 96.43 |
| `cola_count` | 91.67 |
| `espresso_count` | 80.95 |
| `coffee_count` | 67.86 |
| `soreness_location_2` | 54.76 |
| `caffeine_time_category` | 42.86 |
| `sleepquality` | 25.00 |

Weak Track B performance could plausibly relate to sparse coverage, uneven
coverage between athletes, near-constant fields, groups dominated by imputation,
limited chronological rows, and unresolved timing. These are data-quality
explanations, not established causes.

Detailed report:

- `reports/track_b_missingness.json`

## Product Implications

The hardened evaluation supports evidence of no meaningful Track B improvement
under this evaluation. It is stronger than merely saying there is no positive
result, but it is not proof that contextual features can never help with better
timing metadata, larger data, or verified actigraphy windows.

For the current product story, Track A remains the defensible performance
baseline. Athlete-facing UI should emphasize observed percent of personal
baseline and contextual signals after performance. A hidden pre-workout expected
performance estimate should use Track B-style signals only after timing and
alignment are confirmed.

No causal conclusion is supported, and no athlete intervention should be
recommended from these results.

Detailed outputs:

- `reports/track_b_metrics.json`
- `reports/track_b_predictions.csv`
- `reports/model_comparison.json`
