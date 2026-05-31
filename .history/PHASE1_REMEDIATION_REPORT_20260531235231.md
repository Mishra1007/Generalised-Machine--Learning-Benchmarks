PHASE1 REMEDIATION REPORT

Objective

Implement paired Cohen's d for paired model comparisons and provide tests validating both paired and independent computations.

Summary of Changes

1. `analysis/effect_size.py`
   - Added `paired: bool = False` parameter to `cohens_d`.
   - Paired behaviour: when `paired=True`, compute Cohen's d as the mean of differences divided by the sample standard deviation of differences (ddof=1). If differences have zero variance or inputs are invalid, return `NaN`.
   - Preserved legacy behaviour: when `paired=False` (default), the original pooled (independent-samples) Cohen's d is returned.
   - Updated `effect_size_summary(x, y, paired=False)` to accept a `paired` flag which is passed to `cohens_d` (backwards-compatible default remains independent).

2. `tests/test_effect_sizes.py`
   - New tests that validate:
     - Independent-samples Cohen's d matches manual pooled calculation.
     - Paired Cohen's d returns `NaN` for zero-variance differences and matches manual calculation for non-constant differences.
     - `effect_size_summary(..., paired=True)` propagates the paired option and returns expected keys.
   - Tests are runnable directly via `python tests/test_effect_sizes.py` (the script inserts the repository root into `sys.path` so no package installation is required).

Validation

- Executed `python tests/test_effect_sizes.py` in the repository; all tests passed locally.

Compatibility & API

- Backwards compatibility preserved: default behaviour of `cohens_d(x, y)` is unchanged (independent pooled d). Code that expects the previous API will continue to work.
- New usage for paired comparisons: call `cohens_d(x, y, paired=True)` or use `effect_size_summary(x, y, paired=True)` to get paired Cohen's d in the summary.

Notes on edge cases

- For paired comparisons where differences have zero sample variance, Cohen's d is undefined; the implementation returns `NaN`. This choice surfaces the issue to downstream code/reporting rather than silently reporting 0.0.
- Independent-sample pooled Cohen's d keeps prior behavior: if pooled variance is zero or invalid, function returns 0.0 (legacy behavior preserved).

Recommended Next Steps

- Update places where effect sizes are computed for paired tests (e.g., in `metrics/storage.py` where pairwise effect sizes are appended) to call `effect_size_summary(..., paired=True)` when fold-wise paired data are provided. This completes methodological correctness across the reporting pipeline.
- Add unit tests for the reporting pipeline to ensure the paired option is used when appropriate and that artifacts (CSV) contain correct values.

Files Modified

- Modified: `analysis/effect_size.py`
- Added: `tests/test_effect_sizes.py`
- Added: `PHASE1_REMEDIATION_REPORT.md` (this file)

Acceptance

- Changes implement the requested paired Cohen's d and preserve legacy API.
- New tests validate paired and independent calculations and passed locally.

End of Phase 1 remediation.
