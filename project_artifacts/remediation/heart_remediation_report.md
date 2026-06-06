# Heart Dataset Remediation Report

This report documents the deduplication remediation applied to the Heart dataset (`datasets/heart.csv`) 
via `DatasetSanitizer.remediate_heart()`.

## Before / After Summary

| Metric | Before | After |
|---|---|---|
| Total rows | 1,025 | 302 |
| Unique rows | 302 | 302 |
| Duplicate rows | 723 | 0 |
| Duplication rate | 70.54% | 0.00% |

## Remediation Applied

### Row Deduplication

- **Total rows before**: 1,025
- **Exact duplicate rows detected**: 723 (70.54%)
- **Unique rows retained**: 302
- The original Cleveland heart disease dataset contains 303 records; this dataset had 302 unique rows 
  (one record from the original may have been dropped during prior curation).

## Class Distribution

### Before Deduplication

| Target Class | Count |
|---|---|
| 0 | 499 |
| 1 | 526 |

### After Deduplication

| Target Class | Count |
|---|---|
| 0 | 138 |
| 1 | 164 |

## Cross-Validation Leakage Impact

Prior to deduplication, a standard 5-fold CV on the duplicated dataset showed **~97.56% fold contamination** 
(see `duplicate_audit_report.md`). After deduplication, each fold will contain genuinely unseen samples, 
producing valid generalization estimates.

## Integration

Deduplication is automatically applied when loading the Heart dataset via:
```python
X, y, metadata = load_dataset("datasets/heart.csv", target_column="target")
```
The logic is implemented in `DatasetSanitizer.remediate_heart()` and invoked by `DatasetLoader.load_csv()`.
