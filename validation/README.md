# Validation Framework

This directory contains the cross-validation and benchmarking validation framework.

## Purpose

Provides reproducible, research-grade validation for fair model comparison:
- Stratified 5-fold cross-validation
- Repeated 3 times for statistical robustness
- Fixed random seed (42) for reproducibility
- Complete fold-level metric logging
- Identical evaluation conditions for all models

## Structure

```
validation/
├── cross_validator.py       # Main validation engine
├── fold_manager.py          # Fold generation and tracking
├── results.py               # Result storage and aggregation
└── __init__.py              # Public API exports
```

## Core Components

### 1. **CrossValidator** (`cross_validator.py`)

Main validation engine orchestrating the entire CV process.

```python
from validation import CrossValidator
from sklearn.ensemble import RandomForestClassifier

# Initialize validator (5-fold CV × 3 reps, seed=42)
validator = CrossValidator(n_splits=5, n_repetitions=3, random_state=42)

# Validate single model
results = validator.validate(
    X_train, y_train, 
    model=RandomForestClassifier(random_state=42),
    model_name='RandomForest',
    dataset_name='iris'
)

# Get summary
summary = results.get_summary()
print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
```

**Features**:
- Stratified k-fold splitting
- Multiple repetitions for robustness
- Prevents data leakage
- Complete fold-level logging
- Metrics collection per fold
- Timing information

### 2. **FoldManager** (`fold_manager.py`)

Manages fold generation and tracking.

```python
from validation import FoldManager

fold_manager = FoldManager(n_splits=5, n_repetitions=3, random_state=42)

# Generate folds
for rep_id, fold_id, train_idx, test_idx in fold_manager.generate_folds(X, y):
    X_train = X[train_idx]
    X_test = X[test_idx]
    # Train and evaluate...

# Validate integrity
fold_manager.validate_fold_indices(X, y)

# Get fold statistics
fold_info = fold_manager.get_all_fold_info(X, y)
```

**Features**:
- Stratified splitting with class preservation
- Reproducible with fixed seed
- Fold validation (no overlaps, complete coverage)
- Class distribution tracking per fold
- Support for multiple repetitions

### 3. **ValidationResults** (`results.py`)

Stores and aggregates validation results.

```python
from validation import ValidationResults

# Automatically populated by CrossValidator
results = validator.validate(X_train, y_train, model, 'model', 'dataset')

# Access results
summary = results.get_summary()           # Overall statistics
by_rep = results.get_repetition_summary() # Per-repetition stats
by_fold = results.get_fold_summary()      # Per-fold stats
df = results.to_dataframe()               # All fold results as DataFrame

# Find best/worst folds
best = results.get_best_fold(metric='accuracy')
worst = results.get_worst_fold(metric='f1')

# Log summary
results.log_summary()
```

**Features**:
- Stores fold-level results
- Automatic aggregation
- Summary statistics (mean, std, min, max)
- Per-repetition and per-fold analysis
- DataFrame export for analysis
- Logging integration

### 4. **Metrics** (`metrics/scorers.py`, `metrics/aggregators.py`)

Compute and aggregate classification metrics.

```python
from metrics import compute_classification_metrics, MetricAggregator

# Compute metrics
metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba)

# Aggregate across folds
aggregator = MetricAggregator()
for fold_result in fold_results:
    aggregator.add_fold_result(
        fold_result.metrics,
        repetition_id=fold_result.repetition_id,
        fold_id=fold_result.fold_id
    )

summary = aggregator.aggregate()  # Overall summary
by_rep = aggregator.aggregate_by_repetition()  # Per-repetition
by_fold = aggregator.aggregate_by_fold()  # Per-fold
```

**Metrics Computed**:
- Accuracy: Overall correctness
- Precision: True positives / (TP + FP)
- Recall: True positives / (TP + FN)
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under ROC curve (if probabilities available)
- Matthews CC: Correlation coefficient for binary/multi-class

## Validation Configuration

### Reproducibility Settings
```python
# Fixed configuration for all experiments
n_splits = 5           # 5-fold cross-validation
n_repetitions = 3      # Repeat 3 times
random_state = 42      # Fixed seed for reproducibility
```

### Why These Settings?

1. **5-Fold CV**: Standard in ML research, good balance between variance and bias
2. **3 Repetitions**: Provides statistical robustness, accounts for random variation
3. **Seed = 42**: Ensures reproducibility across runs
4. **Stratification**: Preserves class distribution in each fold (important for imbalanced data)

## Usage Examples

### Example 1: Single Model Validation

```python
from validation import CrossValidator
from sklearn.ensemble import RandomForestClassifier
from preprocessing import prepare_dataset

# Prepare data
X_train, X_test, y_train, y_test, _ = prepare_dataset('iris', test_size=0.3)

# Create and validate model
validator = CrossValidator()
model = RandomForestClassifier(random_state=42)

results = validator.validate(
    X_train, y_train, model,
    model_name='RandomForest',
    dataset_name='iris'
)

# View results
print(results.get_summary())
results.log_summary()
```

### Example 2: Multiple Models, Fair Comparison

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'svm': SVC(random_state=42, probability=True),
}

validator = CrossValidator()

# All models evaluated under identical conditions
all_results = validator.validate_multiple(
    X_train, y_train, models,
    dataset_name='iris'
)

# Compare results
for model_name, results in all_results.items():
    if results:
        summary = results.get_summary()
        print(f"{model_name}: {summary['accuracy_mean']:.4f}")
```

### Example 3: Detailed Analysis

```python
results = validator.validate(X_train, y_train, model, 'rf', 'iris')

# Overall summary
overall = results.get_summary()
print(f"Accuracy: {overall['accuracy_mean']:.4f} ± {overall['accuracy_std']:.4f}")

# Per-repetition performance
rep_summary = results.get_repetition_summary()
print(rep_summary)

# Per-fold consistency
fold_summary = results.get_fold_summary()
print(fold_summary)

# Export to DataFrame for analysis
df = results.to_dataframe()
df.to_csv('results.csv')

# Find outlier folds
best_fold = results.get_best_fold('accuracy')
worst_fold = results.get_worst_fold('accuracy')
print(f"Best fold accuracy: {best_fold.metrics['accuracy']:.4f}")
print(f"Worst fold accuracy: {worst_fold.metrics['accuracy']:.4f}")
```

## Key Design Principles

### ✓ Reproducibility
- Fixed random seed (42)
- Stratified splitting preserves class distribution
- Seeded fold generation
- Deterministic metrics computation

### ✓ Fair Benchmarking
- All models evaluated on identical folds
- Same preprocessing applied
- Identical evaluation conditions
- Prevents comparison bias

### ✓ No Data Leakage
- Folds validated for non-overlap
- Training and test sets independent
- Preprocessing fitted on training folds only
- Statistics computed on test folds

### ✓ Statistical Robustness
- 3 repetitions capture variance
- 5 folds balance bias/variance
- Fold-level metrics tracked
- Standard deviation computed

## Logging

Complete logging at each step:

```
2026-05-08 10:30:45 - validation - INFO - CrossValidator initialized
2026-05-08 10:30:45 - validation - INFO - Starting cross-validation: RandomForest on iris
2026-05-08 10:30:46 - validation - INFO - [1/15] Rep 0, Fold 0: Train 120, Test 30
2026-05-08 10:30:46 - validation - INFO -   Metrics: Accuracy=0.9667, F1=0.9630
...
2026-05-08 10:30:52 - validation - INFO - CROSS-VALIDATION COMPLETE: RandomForest
2026-05-08 10:30:52 - validation - INFO - Total folds evaluated: 15
2026-05-08 10:30:52 - validation - INFO - Accuracy: 0.9533 ± 0.0311
```

## Integration with Data Pipeline

Validation framework integrates seamlessly with data pipeline:

```python
from preprocessing import prepare_dataset
from validation import CrossValidator
from sklearn.ensemble import RandomForestClassifier

# 1. Prepare data
X_train, X_test, y_train, y_test, _ = prepare_dataset(
    'iris',
    test_size=0.3,
    scaling_method='standard',
    encoding_method='onehot'
)

# 2. Validate on training set
validator = CrossValidator()
model = RandomForestClassifier()
results = validator.validate(
    X_train, y_train, model,
    model_name='RandomForest',
    dataset_name='iris'
)

# 3. Final evaluation on held-out test set
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
```

## Output Structure

### Fold Results (Per Fold)
```python
FoldResult:
  - repetition_id: 0
  - fold_id: 0
  - model_name: 'RandomForest'
  - dataset_name: 'iris'
  - metrics: {
      'accuracy': 0.95,
      'precision': 0.94,
      'recall': 0.96,
      'f1': 0.95,
      'roc_auc': 0.98,
      'matthews_cc': 0.92,
      ...
    }
  - train_size: 120
  - test_size: 30
  - train_time: 0.045
  - eval_time: 0.012
```

### Summary Statistics
```python
{
  'accuracy_mean': 0.9533,
  'accuracy_std': 0.0311,
  'accuracy_min': 0.9000,
  'accuracy_max': 1.0000,
  'f1_mean': 0.9530,
  'f1_std': 0.0315,
  ...
  'total_folds': 15,
  'total_train_samples': 1800,
  'total_eval_samples': 600,
}
```

## Next Steps

The validation framework is complete and ready for:
- Model implementation and evaluation
- Benchmark comparisons
- Hyperparameter tuning (within folds)
- Feature selection (with proper CV handling)

## References

- **CrossValidator**: Main validation engine
- **FoldManager**: Fold generation and validation
- **ValidationResults**: Result storage and aggregation
- **Metrics**: Scorers and aggregators
