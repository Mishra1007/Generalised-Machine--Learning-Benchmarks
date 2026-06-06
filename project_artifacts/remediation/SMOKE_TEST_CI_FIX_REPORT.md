SMOKE TEST CI FIX REPORT

Root Cause

- The smoke dataset had only 6 samples.
- Train/test split left ~4 training samples.
- Cross-validation requested 5 folds, violating sample/class count constraints.
- CV validation failed, so Logistic Regression produced no results.
- Visualization attempted to plot an empty results set and crashed (min() on empty sequence).

Fixes Applied

1. Smoke dataset enlarged to 60 rows with balanced classes.
2. Added CV safety checks to reject invalid folds early.
3. Visualization now skips plotting when model summaries are empty.
4. Smoke test assertions now verify non-empty outputs and critical artifacts.

Validation Evidence (Local)

- `python tests\test_cv_validation.py` → PASS
- `python tests\test_visualization_empty.py` → PASS
- `python tests\test_failed_model_handling.py` → PASS
- `python tests\smoke_test.py` → PASS

Notes

- CI verification must be run in GitHub Actions to confirm environment parity.
