# Reproducibility Report

Generated: 2026-06-04T13:14:10.221392Z
Experiment ID: exp-20260604T131307Z-f27406ef
Run ID: run-b9aa921b6d
Framework Version: phase6.1a
Git Commit: 9368762

## Environment Summary
Python: 3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]
OS: Windows-10-10.0.26200-SP0
CPU: 12 logical cores
NumPy: 1.26.4
Pandas: 1.5.3
Scikit-learn: 1.7.1
SciPy: 1.16.1

## Dataset Summary
Dataset: Edu-Primary
Source: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\datasets\Edu-Primary.csv
Size: 649
Feature Count: 32
Target: G3
Fingerprint: 524abd618019820939ee08e3e4a279097574553aed731311c7e4602139d8eb79
Class Distribution: {11: 104, 10: 97, 13: 82, 12: 72, 14: 63, 15: 49, 16: 36, 9: 35, 8: 35, 17: 29, 18: 15, 0: 15, 7: 10, 6: 3, 19: 2, 1: 1, 5: 1}

## Model Summary
Models: ['DecisionTree', 'GradientBoosting', 'LogisticRegression', 'RandomForest', 'SVM']
Hyperparameters: {'DecisionTree': {}, 'GradientBoosting': {}, 'LogisticRegression': {}, 'RandomForest': {}, 'SVM': {}}
Search Space: {'DecisionTree': {}, 'GradientBoosting': {}, 'LogisticRegression': {}, 'RandomForest': {}, 'SVM': {}}
Final Configuration: {'DecisionTree': {}, 'GradientBoosting': {}, 'LogisticRegression': {}, 'RandomForest': {}, 'SVM': {}}

## Validation Protocol
Validation Strategy: {'n_splits': 2, 'n_repetitions': 3, 'train_test_split': {'test_size': 0.3, 'train_size': None}}
CV Fold Count: 2
Repetition Count: 3
Random Seed: 42
Train/Test Split: {'test_size': 0.3, 'train_size': None}
Deterministic: False

## Reproduction Instructions
1. Use the stored experiment_manifest.json as the single source of truth.
2. Restore the dataset from the recorded source and verify the fingerprint.
3. Reuse the recorded random seed, validation strategy, and model configuration.
4. Recreate outputs using the stored benchmark artifacts and fold metadata.

## Stored Artifacts
- config: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\config.json
- metadata: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\metadata.json
- raw_results: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\raw_results.csv
- normalized_results: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\normalized_results.csv
- cbs_scores: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\cbs_scores.csv
- metrics_json: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\metrics.json
- plots_dir: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\plots
