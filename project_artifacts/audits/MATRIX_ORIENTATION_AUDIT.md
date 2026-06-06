# Matrix Orientation Audit

Phase 3 forensic audit for statistical benchmark matrix orientation.

## Canonical Contract

All benchmark score matrices passed into statistical procedures use:

- Rows = aligned observations: fold keys or repetition-fold keys.
- Columns = models: one column per model in `model_names`.
- Observation alignment key = `(repetition_id, fold_id)`.

No transpose or reshape operation exists in the audited statistical path. The only matrix transformations are row-wise rank conversion, column-wise aggregation, and paired column slicing.

## Orientation Trace

| Pipeline | Source | Transformation | Consumer | Expected Shape | Actual Shape | Verified |
|---|---|---|---|---|---|---|
| Fold metrics to score matrix | `metrics/storage.py::_build_score_matrix`; source rows are per-model `FoldResult` entries keyed by `(repetition_id, fold_id)` | Builds per-model maps, intersects common fold keys, sorts keys, emits `[[score_map[key] for score_map in score_maps] for key in ordered_keys]`; no transpose; no reshape | `metrics/storage.py::save_experiment_results`; `analysis.significance.global_significance_analysis` | Rows = aligned observations; columns = models | Rows = sorted common fold keys; columns = `model_names` order | YES |
| Friedman | Score matrix from storage or caller | Validated by `validate_score_matrix`; slices `matrix[:, idx]` for each model column; no aggregation before test | `analysis/statistics.py::friedman_test`; SciPy `friedmanchisquare` | Rows = observations; columns = models | Columns are passed as model samples with identical row order | YES |
| Pairwise Wilcoxon | Score matrix from storage or caller | Validated by `validate_score_matrix`; pairwise slices `matrix[:, i]` and `matrix[:, j]`; `wilcoxon_signed_rank` validates equal-length paired vectors | `analysis/significance.py::pairwise_wilcoxon`; `analysis/statistics.py::wilcoxon_signed_rank` | Paired aligned observations | Same row indices used for both model vectors | YES |
| Nemenyi post-hoc | Score matrix from storage or caller | Validated by `validate_score_matrix`; converts each observation row with `stats.rankdata(-row)` when higher is better; averages ranks with `ranks_per_dataset.mean(axis=0)` | `analysis/significance.py::nemenyi_test` | Rank matrix generated from identical observations | Rank rows = original observation rows; rank columns = models | YES |
| Model ranking | Score matrix from storage or caller | Validated by `validate_score_matrix`; aggregates `matrix.mean(axis=0)`; converts column means to ranks | `analysis/statistics.py::rank_models` | Rows = observations; columns = models | Means and ranks are per model column | YES |
| Confidence intervals | Score matrix from storage | Slices `matrix[:, idx]` for each model; computes mean CI and bootstrap CI on one model column at a time | `metrics/storage.py::save_experiment_results`; `analysis/statistics.py::mean_confidence_interval`; `bootstrap_confidence_interval` | One aligned observation vector per model | Values are model columns from canonical matrix | YES |
| Effect sizes | Score matrix from storage pairwise results | Slices `matrix_arr[:, idx_a]` and `matrix_arr[:, idx_b]`; calls `effect_size_summary(..., paired=True)`; paired vectors validate equal length and finite values | `metrics/storage.py::save_experiment_results`; `analysis/effect_size.py` | Aligned paired differences | Same observation rows compared between model columns | YES |
| Significance report | Analysis tables produced by significance functions | Formats existing DataFrames; no matrix operation; no transpose; no reshape | `analysis/reports.py::build_significance_report`; `write_significance_artifacts` | Not applicable after matrix consumers have produced tables | Table-only reporting | YES |

## Procedure Details

### Source Shape

`metrics/storage.py::_build_score_matrix` consumes `validation_results`, a mapping:

- Key = model name.
- Value = `ValidationResults` containing `fold_results`.
- Each fold result contributes one metric score at observation key `(repetition_id, fold_id)`.

The emitted score matrix is:

- Rows = sorted common observation keys across every retained model.
- Columns = models in input iteration order.

Duplicate observation keys for a model are now rejected.

### Transformations

- Transpose: none in audited path.
- Reshape: none in audited path.
- Rank conversion: `nemenyi_test` ranks each observation row across model columns; `rank_models` ranks column means.
- Aggregation: `rank_models` uses `mean(axis=0)`; Nemenyi uses `ranks_per_dataset.mean(axis=0)`; confidence intervals aggregate one model column at a time.
- Paired slicing: Wilcoxon and effect sizes use matching row indices from two model columns.

### Consumer Expectations

- Friedman expects rows = observations and columns = models. Verified by matching SciPy reference using `matrix[:, 0]`, `matrix[:, 1]`, `matrix[:, 2]`.
- Wilcoxon expects paired aligned observations. Verified by shuffled source fold order producing aligned columns before pairwise testing.
- Nemenyi expects a rank matrix generated from identical observations. Verified by row-wise reference `stats.rankdata`.
- Confidence intervals expect a single model's observation vector. Verified by code path slicing model columns.
- Effect sizes expect aligned paired differences. Enforced by `paired=True` in storage and paired vector validation.

## Assertions Added

- `analysis/statistics.py::validate_score_matrix` enforces 2D matrices, minimum observation/model counts, finite scores, and `len(model_names) == matrix.shape[1]`.
- `analysis/statistics.py::validate_paired_observations` enforces 1D equal-length finite paired vectors.
- `friedman_test`, `rank_models`, `nemenyi_test`, `pairwise_wilcoxon`, and `global_significance_analysis` call the shared score-matrix validator.
- `wilcoxon_signed_rank` and paired effect-size paths call paired-observation validation.
- `_build_score_matrix` rejects duplicate observation keys.

## Validation Evidence

Synthetic benchmark datasets in `tests/test_matrix_orientation.py` verify:

- Model alignment by column order.
- Fold alignment by sorted `(repetition_id, fold_id)` keys despite shuffled per-model fold order.
- Rejection of duplicate fold keys.
- Friedman input shape and SciPy-equivalent column slicing.
- Wilcoxon paired alignment.
- Nemenyi row-wise rank generation.
- Ranking aggregation over observation rows.
- Effect-size paired differences and equal-length validation.

Focused validation command:

```powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_matrix_orientation.py tests/test_statistics.py tests/test_effect_sizes.py tests/test_multiple_comparisons.py
```

Result: 17 passed.
