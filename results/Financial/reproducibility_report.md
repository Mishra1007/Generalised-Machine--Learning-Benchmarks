# Reproducibility Report

Generated: 2026-06-04T17:33:44.318273Z
Experiment ID: exp-20260604T125748Z-eccb915b
Run ID: run-a692a380f4
Framework Version: phase6.1a
Git Commit: 2bef6f4

## Environment Summary
Python: 3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]
OS: Windows-10-10.0.26200-SP0
CPU: 12 logical cores
NumPy: 1.26.4
Pandas: 1.5.3
Scikit-learn: 1.7.1
SciPy: 1.16.1

## Dataset Summary
Dataset: Financial
Source: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\datasets\Financial.csv
Size: 50000
Feature Count: 26
Target: Payment_Behaviour
Fingerprint: 332d96f83fc2d8a20755d7194687b286a4fc8d59f1033f8cc3751c85a9b456c3
Class Distribution: {'Low_spent_Small_value_payments': 12694, 'High_spent_Medium_value_payments': 8922, 'High_spent_Large_value_payments': 6844, 'Low_spent_Medium_value_payments': 6837, 'High_spent_Small_value_payments': 5651, 'Low_spent_Large_value_payments': 5252, '!@9#%8': 3800}

## Model Summary
Models: ['DecisionTree', 'GradientBoosting', 'LogisticRegression', 'RandomForest', 'SVM']
Hyperparameters: {'DecisionTree': {}, 'GradientBoosting': {}, 'LogisticRegression': {}, 'RandomForest': {}, 'SVM': {}}
Search Space: {'DecisionTree': {}, 'GradientBoosting': {}, 'LogisticRegression': {}, 'RandomForest': {}, 'SVM': {}}
Final Configuration: {'DecisionTree': {}, 'GradientBoosting': {}, 'LogisticRegression': {}, 'RandomForest': {}, 'SVM': {}}

## Validation Protocol
Validation Strategy: {'n_splits': 5, 'n_repetitions': 3, 'train_test_split': {'test_size': 0.3, 'train_size': None}}
CV Fold Count: 5
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
- config: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\config.json
- metadata: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\metadata.json
- raw_results: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\raw_results.csv
- normalized_results: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\normalized_results.csv
- cbs_scores: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\cbs_scores.csv
- metrics_json: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\metrics.json
- plots_dir: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\plots
