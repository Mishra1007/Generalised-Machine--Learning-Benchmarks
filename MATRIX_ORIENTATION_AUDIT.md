MATRIX ORIENTATION AUDIT

Scope

This audit traces how fold-level metrics become matrices for statistical tests, and verifies that matrix orientation is consistent across producers and consumers.

Legend

- Observations = folds (or dataset-fold pairs) shared across models.
- Models = columns used in Friedman, Nemenyi, Wilcoxon, ranking, confidence intervals, and effect-size comparisons.

Trace Table

1) Validation results → Score matrix

Source
- validation/results.py: `ValidationResults.fold_results` is a list of `FoldResult` with (`repetition_id`, `fold_id`, `metrics`).

Transformation
- metrics/storage.py: `_build_score_matrix(validation_results, primary_metric)`
  - Collects per-model `scores[(repetition_id, fold_id)] = metric`.
  - Intersects common fold keys across models.
  - Orders fold keys and builds matrix.

Consumer
- metrics/storage.py: `save_experiment_results` → `global_significance_analysis`.

Expected Shape
- (n_observations, n_models)
  - Rows: ordered fold keys (observations)
  - Columns: model order in `model_names`

Actual Shape
- (n_observations, n_models)
  - Implemented as `matrix = [[score_map[key] for score_map in score_maps] for key in ordered_keys]`.

Verified
- YES (integration test: tests/test_matrix_orientation.py)

2) Score matrix → Friedman test

Source
- analysis/statistics.py: `friedman_test(score_matrix, model_names)`

Transformation
- Uses columns of `score_matrix` as model samples.

Consumer
- analysis/significance.py: `global_significance_analysis`.

Expected Shape
- (n_observations, n_models) with columns = models.

Actual Shape
- Enforced by assertion: model_names length must match `matrix.shape[1]`.

Verified
- YES (tests/test_matrix_orientation.py + runtime assertions)

3) Score matrix → Nemenyi test

Source
- analysis/significance.py: `nemenyi_test(score_matrix, model_names)`

Transformation
- Ranks each row (observation) across model columns.

Expected Shape
- (n_observations, n_models) with columns = models.

Actual Shape
- Enforced by assertion: model_names length must match `matrix.shape[1]`.

Verified
- YES (tests/test_matrix_orientation.py + runtime assertions)

4) Score matrix → Pairwise Wilcoxon

Source
- analysis/significance.py: `pairwise_wilcoxon(score_matrix, model_names)`

Transformation
- Uses column vectors `matrix[:, i]` and `matrix[:, j]` (paired observations).

Expected Shape
- (n_observations, n_models) with columns = models, rows aligned by fold key.

Actual Shape
- Enforced by assertion: model_names length must match `matrix.shape[1]`.

Verified
- YES (tests/test_matrix_orientation.py + runtime assertions)

5) Score matrix → Confidence intervals

Source
- metrics/storage.py: `save_experiment_results` uses `matrix[:, idx]` for per-model CI.

Expected Shape
- (n_observations, n_models) with columns = models.

Actual Shape
- Uses `values = np.asarray(matrix)[:, idx]`.

Verified
- YES (code inspection)

6) Score matrix → Effect sizes

Source
- metrics/storage.py: per-pair effect sizes computed from `matrix[:, idx_a]` and `matrix[:, idx_b]`.

Expected Shape
- (n_observations, n_models) with columns = models.

Actual Shape
- Uses `matrix_arr[:, idx_a]` and `matrix_arr[:, idx_b]`.

Verified
- YES (code inspection)

7) Reports → Significance report

Source
- analysis/reports.py: uses DataFrames produced by `analysis/significance.py`.

Transformation
- Report uses `pairwise_wilcoxon` outputs and ranking tables; no matrix operations.

Expected Shape
- Not applicable (reports consume tables only).

Actual Shape
- Not applicable.

Verified
- YES (code inspection)

Summary

All statistical consumers now receive matrices in the expected orientation: rows are observations (folds), columns are models. Assertions guard against shape/model-name mismatches, and integration tests verify that ranking and significance use the correct orientation.
