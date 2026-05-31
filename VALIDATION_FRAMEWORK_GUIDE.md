# Validation Framework Guide

**Purpose**: Comprehensive guide to the reproducible cross-validation framework  
**Target Audience**: ML Engineers implementing benchmarks and experiments  
**Last Updated**: May 8, 2026  
**Total Length**: 120+ KB documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Design Principles](#design-principles)
5. [Implementation Guide](#implementation-guide)
6. [Usage Patterns](#usage-patterns)
7. [Reproducibility Strategy](#reproducibility-strategy)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Overview

The validation framework provides enterprise-grade cross-validation for fair and reproducible machine learning benchmarking.

### Key Features

✓ **Stratified 5-Fold Cross-Validation**: Standard approach with class distribution preservation  
✓ **3 Repetitions**: Statistical robustness across multiple random seeds  
✓ **Fixed Random State (42)**: Ensures reproducibility across runs  
✓ **Fold-Level Logging**: Complete transparency at every step  
✓ **No Data Leakage**: Validated fold separation and training-only preprocessing  
✓ **Multiple Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Matthews CC  
✓ **Fair Benchmarking**: Identical folds across all models  
✓ **Result Aggregation**: Automatic mean/std/min/max computation  
✓ **Classification Only**: Focus on classification tasks  

### When to Use

**Use the validation framework when**:
- Comparing multiple models on the same dataset
- Publishing research results (requires CV metrics)
- Evaluating model stability
- Need reproducible benchmarks
- Want comprehensive fold-level metrics
- Performing hyperparameter tuning

**Don't use when**:
- Simple train/test split is sufficient
- Time is critical (CV is slower than single split)
- Regression tasks (not supported)
- Unsupervised learning (no y labels)

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                  CrossValidator (Main Engine)                │
├─────────────────────────────────────────────────────────────┤
│  - Orchestrates entire CV workflow                           │
│  - Validates fold integrity                                  │
│  - Collects metrics per fold                                 │
│  - Aggregates results                                        │
│  - Logs all operations                                       │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
        ┌──────▼─────┐        ┌──────▼──────┐
        │ FoldManager│        │   Metrics   │
        ├────────────┤        ├─────────────┤
        │- Generate  │        │ - Compute   │
        │  stratified│        │   metrics   │
        │  folds     │        │ - Aggregate │
        │- Validate  │        │ - Statistics│
        │  indices   │        │             │
        └────────────┘        └──────┬──────┘
                                     │
                          ┌──────────▼────────┐
                          │  ValidationResults│
                          ├───────────────────┤
                          │ - Store fold data │
                          │ - Aggregate by    │
                          │   rep/fold        │
                          │ - Export to DF    │
                          │ - Logging         │
                          └───────────────────┘
```

### Data Flow

```
Input Data (X, y)
    │
    ▼
FoldManager.validate_fold_indices()  ◄─ Prevent leakage
    │
    ▼
FoldManager.generate_folds()  ◄─ Stratified splits
    │ (For each fold)
    ├─> Split X, y into train/test
    │
    ├─> model.fit(X_train, y_train)
    │
    ├─> y_pred = model.predict(X_test)
    │
    ├─> compute_classification_metrics()
    │
    ├─> FoldResult(metrics, timing, indices)
    │
    ▼
ValidationResults (accumulate FoldResults)
    │
    ▼
Aggregation & Analysis
    ├─> get_summary() → mean/std per metric
    ├─> to_dataframe() → all folds
    ├─> get_best_fold() → outlier analysis
    │
    ▼
Logging & Export
```

### Module Dependencies

```
validation/
├── cross_validator.py (Main engine)
│   └─ depends on: fold_manager, results, metrics.scorers
├── fold_manager.py (Fold generation)
│   └─ depends on: sklearn.model_selection
├── results.py (Result storage)
│   └─ depends on: pandas, numpy, dataclasses
└── __init__.py (Public API)

metrics/
├── scorers.py (Metric computation)
│   └─ depends on: sklearn.metrics
├── aggregators.py (Aggregation)
│   └─ depends on: pandas, numpy
└── __init__.py (Public API)
```

---

## Core Components

### 1. CrossValidator

**Purpose**: Orchestrate the complete CV workflow

**Class**: `CrossValidator(n_splits=5, n_repetitions=3, random_state=42)`

**Key Methods**:

```python
validate(X, y, model, model_name, dataset_name, predict_proba=True)
    Returns: ValidationResults
    Performs:
    - Fold generation
    - Model training per fold
    - Metric computation
    - Result aggregation
    - Complete logging

validate_multiple(X, y, models_dict, dataset_name)
    Returns: dict[model_name -> ValidationResults]
    Performs:
    - Iterates models dict
    - Calls validate() for each
    - Ensures identical conditions
```

**Example**:

```python
from validation import CrossValidator
from sklearn.ensemble import RandomForestClassifier

validator = CrossValidator(n_splits=5, n_repetitions=3, random_state=42)

model = RandomForestClassifier(random_state=42)

results = validator.validate(
    X_train, y_train, model,
    model_name='RandomForest',
    dataset_name='iris'
)

summary = results.get_summary()
```

### 2. FoldManager

**Purpose**: Generate and validate stratified folds

**Class**: `FoldManager(n_splits=5, n_repetitions=3, random_state=42)`

**Key Methods**:

```python
generate_folds(X, y)
    Yields: (repetition_id, fold_id, train_indices, test_indices)
    Details:
    - Uses StratifiedKFold with shuffle=True
    - Seed varies per repetition (42 + rep_id)
    - Ensures class distribution preservation

validate_fold_indices(X, y)
    Checks:
    - No fold overlaps (train/test independent)
    - Complete coverage (all samples used)
    - Proper stratification

get_all_fold_info(X, y)
    Returns: List of fold_info with class distributions
```

**Example**:

```python
from validation import FoldManager

fold_manager = FoldManager()

for rep_id, fold_id, train_idx, test_idx in fold_manager.generate_folds(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate...

fold_manager.validate_fold_indices(X, y)  # Ensures no leakage
```

### 3. ValidationResults

**Purpose**: Store and aggregate fold-level results

**Class**: `ValidationResults(model_name, dataset_name, random_state)`

**Key Methods**:

```python
add_fold_result(fold_result: FoldResult)
    Accumulates fold results

get_summary()
    Returns: dict with mean/std/min/max per metric
    Example: {
        'accuracy_mean': 0.95,
        'accuracy_std': 0.03,
        'f1_mean': 0.94,
        'f1_std': 0.04,
        'total_folds': 15,
        ...
    }

to_dataframe()
    Returns: pandas DataFrame with all fold metrics

get_repetition_summary()
    Returns: DataFrame with per-repetition statistics

get_fold_summary()
    Returns: DataFrame with per-fold statistics (across reps)

get_best_fold(metric='accuracy')
    Returns: FoldResult with best performance

log_summary()
    Logs all statistics to logger
```

**Example**:

```python
from validation import CrossValidator

validator = CrossValidator()
results = validator.validate(X, y, model, 'rf', 'iris')

summary = results.get_summary()
print(f"Accuracy: {summary['accuracy_mean']:.4f}")

df = results.to_dataframe()
print(df[['fold_id', 'accuracy', 'f1']])

results.log_summary()
```

### 4. FoldResult (Data Class)

**Purpose**: Store single fold result with metadata

**Fields**:

```python
@dataclass
FoldResult:
    repetition_id: int          # Repetition number (0-2)
    fold_id: int               # Fold number (0-4)
    model_name: str            # Model identifier
    dataset_name: str          # Dataset identifier
    metrics: dict              # {metric_name: value}
    train_size: int            # Training samples
    test_size: int             # Test samples
    train_time: float          # Training duration (seconds)
    eval_time: float           # Evaluation duration (seconds)
    timestamp: datetime        # Result timestamp
    y_test: Optional[ndarray]  # Test labels (optional)
    y_pred: Optional[ndarray]  # Predictions (optional)
    y_pred_proba: Optional[ndarray]  # Probabilities (optional)
    test_indices: Optional[list]      # Test indices (optional)
```

---

## Design Principles

### ✓ Stratified Splitting

**Why**: Ensures class distribution is maintained in each fold

```python
# Without stratification (BAD)
fold = KFold(n_splits=5, shuffle=True)

# With stratification (GOOD)
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Example**:
```
Original dataset: 90 Class-0, 10 Class-1 (9:1 ratio)

Stratified folds:
- Fold 0: 72 Class-0, 8 Class-1 (9:1)
- Fold 1: 72 Class-0, 8 Class-1 (9:1)
- ... (ratio preserved)
```

### ✓ Fixed Random State

**Why**: Enables reproducibility

```python
# Fixed seed = 42 for reproducibility
fold_manager = FoldManager(random_state=42)

# Different seed per repetition for diversity
seed_rep_0 = 42 + 0  # 42
seed_rep_1 = 42 + 1  # 43
seed_rep_2 = 42 + 2  # 44

# Results: Different splits but fully reproducible
```

### ✓ No Data Leakage

**How**:
1. Folds validated for non-overlap
2. Preprocessing fitted on training folds only
3. Test data transformed using training statistics

```python
# WRONG - leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fitted on full data
X_train, X_test = split(X_scaled)    # Split preprocessed data

# CORRECT - no leakage
X_train, X_test = split(X)  # Split raw data first
scaler = StandardScaler()
scaler.fit(X_train)          # Fit on training only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)   # Transform with training stats
```

### ✓ Fair Benchmarking

**All models evaluated on identical conditions**:

```python
validator = CrossValidator()

models = {
    'rf': RandomForestClassifier(),
    'svm': SVC(),
    'lr': LogisticRegression(),
}

# All models get same folds, same preprocessing, same metrics
results = validator.validate_multiple(X, y, models, 'iris')

# Valid comparison - no confounding factors
```

---

## Implementation Guide

### Setup: Initialize CrossValidator

```python
from validation import CrossValidator

# Default configuration (recommended)
validator = CrossValidator()

# Custom configuration
validator = CrossValidator(
    n_splits=5,           # 5-fold
    n_repetitions=3,      # Repeat 3 times
    random_state=42       # Fixed seed
)
```

### Step 1: Prepare Data

```python
from preprocessing import prepare_dataset

X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    dataset_name='iris',
    test_size=0.3,
    scaling_method='standard',
    encoding_method='onehot'
)

print(f"Training data: {X_train.shape}")
print(f"Classes: {np.unique(y_train)}")
```

### Step 2: Validate Single Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

results = validator.validate(
    X_train, y_train, model,
    model_name='RandomForest',
    dataset_name='iris',
    predict_proba=True
)
```

### Step 3: Analyze Results

```python
# Overall summary
summary = results.get_summary()
print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")

# Per-repetition consistency
rep_summary = results.get_repetition_summary()
print(rep_summary[['repetition_id', 'accuracy_mean', 'f1_mean']])

# Per-fold stability
fold_summary = results.get_fold_summary()
print(fold_summary[['fold_id', 'accuracy_mean', 'accuracy_std']])

# All fold-level data
df = results.to_dataframe()
print(df)

# Export for external analysis
df.to_csv('results.csv')
```

### Step 4: Compare Multiple Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
}

# Fair comparison - all under identical conditions
all_results = validator.validate_multiple(
    X_train, y_train, models,
    dataset_name='iris'
)

# Compare
for model_name, results in all_results.items():
    summary = results.get_summary()
    print(f"{model_name:20} Accuracy: {summary['accuracy_mean']:.4f}")
```

---

## Usage Patterns

### Pattern 1: Simple Single Model Validation

```python
validator = CrossValidator()
model = RandomForestClassifier(random_state=42)

results = validator.validate(X, y, model, 'rf', 'iris')
print(f"Accuracy: {results.get_summary()['accuracy_mean']:.4f}")
```

### Pattern 2: Hyperparameter Comparison

```python
# Evaluate different hyperparameter settings
configs = {
    'n_estimators_10': RandomForestClassifier(n_estimators=10, random_state=42),
    'n_estimators_50': RandomForestClassifier(n_estimators=50, random_state=42),
    'n_estimators_100': RandomForestClassifier(n_estimators=100, random_state=42),
}

results = validator.validate_multiple(X, y, configs, 'iris')

# Find best
best_config = max(
    results.items(),
    key=lambda x: x[1].get_summary()['accuracy_mean']
)
```

### Pattern 3: Stability Analysis

```python
results = validator.validate(X, y, model, 'rf', 'iris')

# Get fold-level metrics
df = results.to_dataframe()

# Analyze consistency
print(f"Accuracy mean: {df['accuracy'].mean():.4f}")
print(f"Accuracy std:  {df['accuracy'].std():.4f}")
print(f"Coefficient of variation: {df['accuracy'].std() / df['accuracy'].mean():.4f}")

# Identify unstable folds
unstable_folds = df[df['accuracy'] < df['accuracy'].mean() - 2*df['accuracy'].std()]
print(unstable_folds)
```

### Pattern 4: Model Comparison with Statistics

```python
results_rf = validator.validate(X, y, rf_model, 'rf', 'iris')
results_svm = validator.validate(X, y, svm_model, 'svm', 'iris')

# Compare
acc_rf = results_rf.to_dataframe()['accuracy'].values
acc_svm = results_svm.to_dataframe()['accuracy'].values

# Statistical test (paired t-test)
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(acc_rf, acc_svm)
print(f"p-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")
```

---

## Reproducibility Strategy

### Goal: Same Seed = Same Results

```python
# Run 1
validator = CrossValidator(random_state=42)
model = RandomForestClassifier(random_state=42)
results1 = validator.validate(X, y, model, 'rf', 'iris')
acc1 = results1.get_summary()['accuracy_mean']

# Run 2 (even weeks later)
validator = CrossValidator(random_state=42)
model = RandomForestClassifier(random_state=42)
results2 = validator.validate(X, y, model, 'rf', 'iris')
acc2 = results2.get_summary()['accuracy_mean']

assert acc1 == acc2  # ALWAYS TRUE
```

### Factors Affecting Reproducibility

| Factor | Impact | Control |
|--------|--------|---------|
| CrossValidator.random_state | Fold seeding | Always set to 42 |
| Model.random_state | Model initialization | Always set in model |
| FoldManager.random_state | Fold generation | Inherited from CV |
| Preprocessing random_state | Imputation/encoding | Set in pipeline |
| Data order | Fold assignment | Fixed after loading |

### Verification Checklist

```python
# ✓ Check 1: Fixed random state
validator = CrossValidator(random_state=42)

# ✓ Check 2: Model random state
model = RandomForestClassifier(random_state=42)

# ✓ Check 3: Preprocessing random state
pipeline = create_pipeline(random_state=42)

# ✓ Check 4: Data consistency
assert len(X) == 150  # Same dataset
assert y.dtype == np.int64  # Same type

# ✓ Check 5: Results equality
run1 = validator.validate(X, y, model, 'rf', 'iris').get_summary()
run2 = validator.validate(X, y, model, 'rf', 'iris').get_summary()
assert np.isclose(run1['accuracy_mean'], run2['accuracy_mean'])
```

---

## Common Patterns

### Pattern: Export Results to CSV

```python
results = validator.validate(X, y, model, 'rf', 'iris')
df = results.to_dataframe()
df.to_csv('fold_results.csv', index=False)
```

### Pattern: Log Summary to File

```python
import logging

logging.basicConfig(filename='validation.log', level=logging.INFO)
results.log_summary()
```

### Pattern: Get Best Hyperparameters

```python
models = {
    'rf_depth_5': RandomForestClassifier(max_depth=5, random_state=42),
    'rf_depth_10': RandomForestClassifier(max_depth=10, random_state=42),
    'rf_depth_20': RandomForestClassifier(max_depth=20, random_state=42),
}

results = validator.validate_multiple(X, y, models, 'iris')

best_model = max(
    results.items(),
    key=lambda x: x[1].get_summary()['accuracy_mean'] if x[1] else -1
)[0]

print(f"Best hyperparameters: {best_model}")
```

### Pattern: Statistical Comparison

```python
from scipy.stats import ttest_rel, wilcoxon

results_1 = validator.validate(X, y, model_1, 'model_1', 'iris')
results_2 = validator.validate(X, y, model_2, 'model_2', 'iris')

acc_1 = results_1.to_dataframe()['accuracy'].values
acc_2 = results_2.to_dataframe()['accuracy'].values

# Paired t-test
t_stat, p_value = ttest_rel(acc_1, acc_2)
print(f"Paired t-test p-value: {p_value:.4f}")

# Wilcoxon signed-rank test (non-parametric)
stat, p_value = wilcoxon(acc_1, acc_2)
print(f"Wilcoxon p-value: {p_value:.4f}")
```

---

## Troubleshooting

### Issue: Different Results Between Runs

**Symptoms**: Same code produces different metrics

**Causes**:
- Random state not set in model
- Random state not set in validation
- Data shuffled differently
- Preprocessing with randomness

**Solution**:
```python
# ✓ Set all random states
validator = CrossValidator(random_state=42)
model = RandomForestClassifier(random_state=42)
pipeline = create_pipeline(random_state=42)

# ✓ Verify consistency
run1 = validator.validate(X, y, model, 'rf', 'iris')
run2 = validator.validate(X, y, model, 'rf', 'iris')
assert np.isclose(
    run1.get_summary()['accuracy_mean'],
    run2.get_summary()['accuracy_mean']
)
```

### Issue: Low Accuracy in CV

**Symptoms**: CV accuracy much lower than train accuracy

**Causes**:
- Model overfitting
- Improper hyperparameter tuning
- Data quality issues
- Preprocessing issues

**Solution**:
```python
# Analyze per-fold performance
results = validator.validate(X, y, model, 'rf', 'iris')
df = results.to_dataframe()

print(f"Min fold accuracy: {df['accuracy'].min():.4f}")
print(f"Max fold accuracy: {df['accuracy'].max():.4f}")
print(f"Std dev: {df['accuracy'].std():.4f}")

# Check for outlier folds
worst_fold = results.get_worst_fold('accuracy')
print(f"Worst fold: Rep {worst_fold.repetition_id}, Fold {worst_fold.fold_id}")
```

### Issue: Data Leakage Warnings

**Symptoms**: Leakage detected in fold validation

**Causes**:
- Preprocessing fitted on full data
- Features derived from test data
- Scaling/encoding done before splitting

**Solution**:
```python
# ✓ Correct approach
X_train, X_test = split(X, y)

# Fit preprocessing on training data ONLY
scaler = StandardScaler()
scaler.fit(X_train)

# Transform both sets using training statistics
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## FAQ

### Q: Why 5-fold CV?

A: Standard choice balancing:
- **Bias**: Too few folds (e.g., 3) = high bias
- **Variance**: Too many folds (e.g., 10) = high variance, slow computation
- **5-fold**: Proven good balance in practice

### Q: Why 3 repetitions?

A: Captures randomness:
- **1 repetition**: Can be unlucky with particular fold seed
- **3 repetitions**: Sufficient for statistical robustness
- **More**: Diminishing returns vs. computation time

### Q: Why seed 42?

A: Arbitrary choice, but important:
- Same number across all code → reproducibility
- Powers of 2 (32, 64...) work too
- Important: Everyone uses same seed

### Q: Classification only - why?

A: Framework assumptions:
- Stratified splitting assumes discrete classes
- Metrics (accuracy, precision, recall) designed for classification
- ROC-AUC requires probability predictions
- Future: Can extend for regression

### Q: Can I use different preprocessing per fold?

A: Yes, but carefully:

```python
for rep_id, fold_id, train_idx, test_idx in fold_manager.generate_folds(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    
    # Create NEW preprocessor for each fold
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit only on training fold
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model on this fold's data
    model.fit(X_train_scaled, y[train_idx])
```

### Q: How do I save/load results?

A: Export to DataFrame then CSV/database:

```python
results = validator.validate(X, y, model, 'rf', 'iris')

# To CSV
df = results.to_dataframe()
df.to_csv('results.csv', index=False)

# From CSV
df_loaded = pd.read_csv('results.csv')
print(df_loaded)
```

### Q: Can I parallelize across folds?

A: Not yet, but architecture supports it:
- FoldManager can be split across workers
- Each worker processes one fold
- Results aggregated at end
- Future enhancement planned

### Q: Integration with scikit-learn GridSearchCV?

A: GridSearchCV uses internal cross-validation:

```python
# This does its own CV (not using our framework)
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [10, 50, 100]},
    cv=5  # Uses its own 5-fold
)

grid.fit(X, y)

# To use OUR framework, do it manually:
best_params = grid.best_params_
model = RandomForestClassifier(**best_params, random_state=42)
results = validator.validate(X, y, model, 'rf', 'iris')
```

---

## Summary

The validation framework provides:

✓ **Reproducible**: Fixed random states  
✓ **Fair**: Identical conditions for all models  
✓ **Robust**: Multiple repetitions capture variance  
✓ **Transparent**: Complete fold-level logging  
✓ **Valid**: No data leakage, validated folds  
✓ **Comprehensive**: Multiple metrics, aggregation  
✓ **Production-Ready**: Complete error handling, logging  

**Use it for all model evaluation in benchmarks.**
