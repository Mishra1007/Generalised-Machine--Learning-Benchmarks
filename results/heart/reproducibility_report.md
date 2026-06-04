# Reproducibility Report

Generated: 2026-06-04T12:38:40.017481Z
Experiment ID: exp-20260604T123820Z-8e680d8d
Run ID: run-0aba70e20a
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
Dataset: heart
Source: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\datasets\heart.csv
Size: 1025
Feature Count: 13
Target: target
Fingerprint: 80cb6173b72ed1015eb5489caad87f322772c8718f58bf0aefe3d8bf6bb03a72
Class Distribution: {1: 526, 0: 499}

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
- config: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\config.json
- metadata: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\metadata.json
- raw_results: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\raw_results.csv
- normalized_results: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\normalized_results.csv
- cbs_scores: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\cbs_scores.csv
- metrics_json: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\metrics.json
- plots_dir: C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\plots
