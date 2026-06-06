PHASE2 REMEDIATION REPORT

Objective

Implement Holm-Bonferroni correction for all pairwise Wilcoxon comparisons and ensure reporting includes raw and adjusted p-values with corrected significance decisions.

Summary of Changes

1. `analysis/significance.py`
   - Added `holm_bonferroni(p_values, alpha=0.05)` helper that returns Holm-adjusted p-values and rejection flags.
   - Updated `pairwise_wilcoxon(...)` to:
     - Keep raw p-values in `p_value` and `p_value_raw`.
     - Add `p_value_adj` with Holm-adjusted values.
     - Set `significant` and `decision` based on Holm correction.

2. `analysis/reports.py`
   - Updated report generation to include adjusted p-values when available.

3. `tests/test_multiple_comparisons.py`
   - Added tests that:
     - Validate Holm-adjusted p-values against a published reference example.
     - Cross-check against `statsmodels.stats.multitest.multipletests` if available (optional import).
     - Ensure pairwise Wilcoxon tables include raw and adjusted p-values and that significance flags match Holm correction.

Validation

- Executed `python tests/test_multiple_comparisons.py` locally; all tests passed.

Backward Compatibility

- Existing APIs are preserved. `pairwise_wilcoxon` now emits additional columns but keeps `p_value` intact as the raw p-value.

Acceptance

- All required columns are present in comparison tables.
- Holm-Bonferroni correction is applied to all pairwise comparisons.
- Tests validate correctness and (when available) cross-check against statsmodels.

Files Modified

- Modified: `analysis/significance.py`
- Modified: `analysis/reports.py`
- Added: `tests/test_multiple_comparisons.py`
- Added: `PHASE2_REMEDIATION_REPORT.md` (this file)

End of Phase 2 remediation.
