# Phase 4 Remediation Report

## Objective

Validate the Nemenyi post-hoc implementation against accepted statistical references.

## Findings

- The Phase 3 implementation had correct matrix orientation and rank direction.
- The previous Nemenyi q-alpha constants were scientifically incomplete because they did not vary with the number of models.
- Demsar 2006 Table 5(a) requires q-alpha values for the all-pairs Nemenyi test to vary by `k`.
- The correct all-pairs critical value is based on the Studentized range distribution divided by `sqrt(2)`.
- The significance decision should be based on the Studentized range upper-tail probability or, equivalently, the critical difference from the same q-alpha source.

## Changes Made

| File | Change | Status |
|---|---|---|
| `analysis/significance.py` | Added `nemenyi_q_alpha(k, alpha)` using `stats.studentized_range.ppf(1 - alpha, k, np.inf) / sqrt(2)` | Complete |
| `analysis/significance.py` | Updated critical difference calculation to use k-dependent q-alpha | Complete |
| `analysis/significance.py` | Added Studentized range p-value matrix to `nemenyi_test` output | Complete |
| `analysis/significance.py` | Updated significance decisions to use `p <= alpha` | Complete |
| `analysis/__init__.py` | Exported `nemenyi_q_alpha` | Complete |
| `tests/test_nemenyi.py` | Added reference tests for q values, CD, rank generation, p-values, significance decisions, ties, equal performance, 3/5/10 models, and small samples | Complete |
| `NEMENYI_VALIDATION.md` | Documented formulation, implementation, references, limitations, and validation results | Complete |

## Reference Basis

- Demsar 2006, Section 3.2.2 and Table 5(a), for the Nemenyi CD formula and rounded q-alpha checkpoints.
- scikit-posthocs documentation for `posthoc_nemenyi_friedman`, which describes the same block-design orientation and Studentized range basis.
- SciPy `studentized_range` for full-precision distribution quantiles and survival probabilities.

`scikit-posthocs` is not installed in the project virtual environment, so tests compare against an independent SciPy Studentized range reference calculation rather than importing that package.

## Validation

Command:

```powershell
.\.venv311\Scripts\python.exe -m pytest tests/test_nemenyi.py tests/test_statistics.py tests/test_matrix_orientation.py
```

Result:

- 19 passed.
- Warnings were from third-party SciPy/Matplotlib internals and existing small-sample Wilcoxon synthetic cases.

## Acceptance Criteria

- Results match accepted reference formulation: PASSED.
- Every critical value has documented provenance: PASSED.
- No undocumented approximations: PASSED.
- Edge cases covered by tests: PASSED.
- CBS unchanged: PASSED.
- Benchmarking architecture unchanged: PASSED.
