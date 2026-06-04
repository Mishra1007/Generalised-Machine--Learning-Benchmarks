# Nemenyi Validation

Phase 4 scientific validation audit for `analysis/significance.py::nemenyi_test`.

## Scope

Reviewed:

- `nemenyi_test`
- `nemenyi_critical_difference`
- Nemenyi q-alpha critical values
- Row-wise rank generation
- Pairwise significance decisions

CBS and benchmarking architecture were not modified.

## Mathematical Formulation

For `k` models evaluated on `N` aligned observations, each observation row is ranked across model columns. The best model receives rank 1 when `higher_is_better=True`; ties receive average ranks.

Average rank for model `j`:

```text
R_j = (1 / N) * sum_i r_i,j
```

Standard error of average-rank differences:

```text
SE = sqrt(k * (k + 1) / (6N))
```

Nemenyi critical difference:

```text
CD = q_alpha * sqrt(k * (k + 1) / (6N))
```

Two models are significant at level `alpha` when the Studentized range upper-tail probability is at most `alpha`:

```text
Q = abs(R_i - R_j) / SE * sqrt(2)
p = studentized_range.sf(Q, k, infinity)
significant = p <= alpha
```

This is equivalent to comparing `abs(R_i - R_j)` against `CD`, where:

```text
q_alpha = studentized_range.ppf(1 - alpha, k, infinity) / sqrt(2)
```

## Critical Value Audit

Previous implementation:

- Used one fixed q-alpha per alpha.
- Did not vary q-alpha by number of models.
- Used values that matched Bonferroni-Dunn/Nemenyi fragments for some cases but were not valid all-pairs Nemenyi values across `k`.

Current implementation:

- Computes q-alpha from SciPy's Studentized range distribution.
- Divides by `sqrt(2)`, matching Demsar's published Nemenyi critical-value convention.
- Varies correctly with alpha and number of models.
- Supports any valid `0 < alpha < 1`, not only tabulated alpha values.

Reference checkpoints from Demsar 2006, Table 5(a), rounded to three decimals:

| k models | alpha=0.05 | alpha=0.10 |
|---:|---:|---:|
| 3 | 2.343 | 2.052 |
| 5 | 2.728 | 2.459 |
| 10 | 3.164 | 2.920 |

The test suite compares computed values against these rounded published values with tolerance for table rounding.

## Implementation Details

`nemenyi_q_alpha(k, alpha)`:

- Validates `k >= 2`.
- Validates `0 < alpha < 1`.
- Computes `stats.studentized_range.ppf(1 - alpha, k, np.inf) / sqrt(2)`.

`nemenyi_critical_difference(k, n, alpha)`:

- Validates `k >= 2` and `n >= 1`.
- Computes `q_alpha * sqrt(k * (k + 1) / (6n))`.

`nemenyi_test(score_matrix, model_names, alpha, higher_is_better)`:

- Validates score matrix orientation through `validate_score_matrix`.
- Generates ranks row by row using `stats.rankdata`.
- Computes average ranks by column.
- Computes pairwise p-values with the Studentized range survival function.
- Produces both `p_value_matrix` and boolean `significance_matrix`.
- Marks `significant_vs_best` by p-value comparison against the best-ranked model.

## Reference Comparison

`scikit-posthocs` was checked and is not installed in the project virtual environment. Its documentation states that `posthoc_nemenyi_friedman` consumes block-design matrices with rows as blocks and columns as groups, and that its statistics use upper quantiles of the Studentized range distribution.

The local validation therefore compares against an independent reference calculation using SciPy's Studentized range distribution, which is the same statistical distribution documented by scikit-posthocs and Demsar for the Nemenyi all-pairs procedure.

Covered comparisons:

- Average ranks against direct row-wise `stats.rankdata` reference.
- Critical difference against Demsar Table 5(a).
- Demsar Table 6 example: `k=4`, `N=14`, `alpha=.05`, CD approximately `1.25`.
- Pairwise p-values against direct Studentized range survival-function reference.
- Significance decisions against `p <= alpha`.

## Edge Cases

Validated in `tests/test_nemenyi.py`:

- 3 models.
- 5 models.
- 10 models.
- Tied ranks.
- Equal performance.
- Small sample count with `N=4`.

## Limitations

- This is the asymptotic Nemenyi all-pairs procedure after Friedman. It uses the Studentized range distribution with infinite degrees of freedom, consistent with Demsar's table.
- The implementation does not apply tie correction to Nemenyi p-values. This matches scikit-posthocs documentation for `posthoc_nemenyi_friedman`, which states that the function does not test for ties.
- The method is conservative for all-pairs comparisons. If the scientific question is comparison against a single control model, Bonferroni-Dunn or Holm-style control comparisons may be more powerful.

## References

- Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30. https://www.jmlr.org/papers/v7/demsar06a.html
- Demsar 2006 PDF, Section 3.2.2 and Table 5(a): https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
- scikit-posthocs `posthoc_nemenyi_friedman` documentation: https://scikit-posthocs.readthedocs.io/en/stable/generated/scikit_posthocs.posthoc_nemenyi_friedman.html
- SciPy `studentized_range` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.studentized_range.html

## Validation Command

```powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_nemenyi.py tests/test_statistics.py tests/test_matrix_orientation.py
```

Result: 19 passed.
