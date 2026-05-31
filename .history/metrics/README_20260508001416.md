# Comprehensive Metrics Engine

Unified framework for predictive, computational, and stability metrics with fair aggregation and architecture for normalization/CBS.

## Purpose

Provides:
- **Predictive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Computational Metrics**: Training time, inference time, efficiency
- **Stability Metrics**: Standard deviation, coefficient of variation, stability scores
- **Uniform Calculation**: Fair metrics across all models
- **Fold-wise & Aggregated**: Complete metrics at multiple levels
- **Future-Ready**: Architecture for normalization and CBS

## Structure

```
metrics/
├── scorers.py          # Predictive metrics (accuracy, F1, etc.)
├── aggregators.py      # Aggregation across folds/reps
├── computational.py    # Timing and efficiency metrics
├── stability.py        # Stability and variation metrics
├── pipeline.py         # Unified metrics calculation
└── __init__.py         # Public API
```

## Metric Types

### Predictive Metrics

Standard classification metrics:
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True positives / (TP + FP)
- **Recall**: True positives / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (probability-based)

Example:
```python
from metrics import compute_classification_metrics

metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba)
# Returns: {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.96, ...}
```

### Computational Metrics

Efficiency and timing metrics:
- **Train Time**: Time to fit model (seconds)
- **Eval Time**: Time to predict (seconds)
- **Total Time**: Sum of train and eval time
- **Efficiency Score**: 1 / (1 + total_time)

Example:
```python
from metrics import compute_timing_metrics

timing = compute_timing_metrics(
    train_time=0.5,
    eval_time=0.1,
    train_samples=100,
    eval_samples=30
)
# Returns: {'train_time': 0.5, 'eval_time': 0.1, ...}
```

### Stability Metrics

Variation and consistency metrics:
- **Mean**: Average across folds
- **Std**: Standard deviation
- **Coefficient of Variation**: Std / Mean (normalized variance)
- **Stability Score**: 1 / (1 + CV) - higher is more stable

Example:
```python
from metrics import compute_stability_metrics

f1_scores = [0.85, 0.87, 0.84, 0.88, 0.86]
stability = compute_stability_metrics(f1_scores, 'F1')
# Returns: {'mean': 0.86, 'std': 0.015, 'coefficient_of_variation': 0.017, ...}
```

## Quick Start

### 1. Calculate Fold-Level Metrics

```python
from metrics import MetricsCalculator

calculator = MetricsCalculator()

# Calculate metrics for one fold
fold_metrics = calculator.calculate_fold_metrics(
    y_true, y_pred,
    y_pred_proba=y_proba,
    train_time=0.5,
    eval_time=0.1,
    train_samples=100,
    eval_samples=30,
    fold_id=0,
    repetition_id=0
)

# Returns dict with:
# - Predictive: accuracy, precision, recall, f1, roc_auc
# - Computational: train_time, eval_time, total_time
# - Metadata: fold_id, repetition_id
```

### 2. Aggregate Across Folds

```python
fold_metrics_list = [...]  # List of fold results

overall, by_rep, by_fold = calculator.aggregate_fold_metrics(fold_metrics_list)

# overall: Metrics averaged across all folds
# by_rep: Metrics by repetition (if multiple)
# by_fold: Metrics by fold (consistency check)
```

### 3. Calculate Stability Metrics

```python
stability = calculator.calculate_stability_metrics(
    fold_metrics_list,
    primary_metric='f1'
)

# Returns:
# - by_metric: Stability for each metric
# - assessment: Stability level (High/Medium/Low)
```

### 4. Get Comprehensive Summary

```python
summary = calculator.get_comprehensive_summary(
    fold_metrics_list,
    model_name='RandomForest',
    dataset_name='iris'
)

# Contains:
# - overall metrics
# - per-repetition metrics
# - per-fold metrics
# - stability assessment
```

### 5. Export to DataFrame

```python
df = calculator.export_to_dataframe(fold_metrics_list)
df.to_csv('metrics.csv')
```

## Usage Examples

### Example 1: Single Model Evaluation

```python
from metrics import MetricsCalculator
from sklearn.ensemble import RandomForestClassifier
from validation import CrossValidator

# Prepare data
X_train, _, y_train, _, _ = prepare_dataset('iris', test_size=0.3)

# Run validation
validator = CrossValidator()
model = RandomForestClassifier(random_state=42)
results = validator.validate(X_train, y_train, model, 'rf', 'iris')

# Calculate comprehensive metrics
calculator = MetricsCalculator()
fold_metrics = results.to_dataframe()
summary = calculator.get_comprehensive_summary(
    fold_metrics.to_dict('records'),
    'RandomForest',
    'iris'
)

print(f"Accuracy: {summary['overall']['accuracy_mean']:.4f}")
print(f"F1 Std: {summary['overall']['f1_std']:.4f}")
print(f"Stability: {summary['stability']['assessment']['stability_level']}")
```

### Example 2: Fair Model Comparison

```python
models = {
    'LogisticRegression': LogisticRegressionModel(),
    'RandomForest': RandomForestModel(),
    'GradientBoosting': GradientBoostingModel(),
}

validator = CrossValidator()
all_results = validator.validate_multiple(X, y, models, 'iris')

calculator = MetricsCalculator()
comparisons = {}

for name, results in all_results.items():
    metrics = results.to_dataframe()
    summary = calculator.get_comprehensive_summary(
        metrics.to_dict('records'),
        name,
        'iris'
    )
    comparisons[name] = summary

# Compare across models
for model_name, summary in comparisons.items():
    acc = summary['overall']['accuracy_mean']
    acc_std = summary['overall']['accuracy_std']
    f1_std = summary['overall']['f1_std']
    print(f"{model_name}: Acc={acc:.4f}±{acc_std:.4f}, F1 Stability={summary['stability']['assessment']['stability_level']}")
```

### Example 3: Stability Analysis

```python
fold_metrics_list = [
    {'f1': 0.85, 'accuracy': 0.90, 'train_time': 0.5},
    {'f1': 0.87, 'accuracy': 0.92, 'train_time': 0.6},
    {'f1': 0.84, 'accuracy': 0.88, 'train_time': 0.5},
    # ... more folds
]

calculator = MetricsCalculator()
stability = calculator.calculate_stability_metrics(fold_metrics_list, 'f1')

print(f"F1 Mean: {stability['by_metric']['f1']['mean']:.4f}")
print(f"F1 Std: {stability['by_metric']['f1']['std']:.4f}")
print(f"F1 CV: {stability['by_metric']['f1']['coefficient_of_variation']:.4f}")
print(f"Stability Level: {stability['assessment']['stability_level']}")

# Check if model is stable
is_stable = stability['assessment']['is_stable']
print(f"Is Stable: {is_stable}")
```

### Example 4: Computational Efficiency Comparison

```python
from metrics import compare_computational_efficiency

# Aggregate timing for multiple models
timing_results = {
    'LogisticRegression': {'train_time_mean': 0.05, 'eval_time_mean': 0.01},
    'RandomForest': {'train_time_mean': 0.5, 'eval_time_mean': 0.1},
    'SVM': {'train_time_mean': 1.0, 'eval_time_mean': 0.2},
    'GradientBoosting': {'train_time_mean': 2.0, 'eval_time_mean': 0.15},
}

comparison = compare_computational_efficiency(timing_results)

for model_name, efficiency in comparison.items():
    print(f"{model_name}:")
    print(f"  Avg Train Time: {efficiency['avg_train_time']:.4f}s")
    print(f"  Avg Eval Time: {efficiency['avg_eval_time']:.4f}s")
    print(f"  Efficiency Score: {efficiency['efficiency_score']:.4f}")
```

### Example 5: Metrics Validation

```python
from metrics import MetricsValidator

# Validate metrics for consistency
is_valid, errors = MetricsValidator.validate_metrics(fold_metrics_list)

if is_valid:
    print("✓ All metrics are valid")
else:
    print("✗ Metrics validation failed:")
    for error in errors:
        print(f"  - {error}")

# Raise on error
try:
    MetricsValidator.validate_metrics(
        fold_metrics_list,
        raise_on_error=True
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Metric Calculation Workflow

```
1. Per-Fold Calculation
   ├─ Predictive: compute_classification_metrics()
   ├─ Computational: compute_timing_metrics()
   └─ Result: fold_metrics dict

2. Aggregation
   ├─ Overall: Aggregate across all folds
   ├─ By Repetition: Aggregate within each repetition
   └─ By Fold: Show per-fold values

3. Stability Calculation
   ├─ Extract metric values across folds
   ├─ Compute: mean, std, min, max, CV
   └─ Assess: Stability level

4. Summary Generation
   ├─ Combine all aggregates
   ├─ Add stability assessment
   └─ Create comprehensive report
```

## Metric Fairness

### Fair Calculation Practices

1. **Stratified Folds**: Preserve class distribution
2. **Fixed Random States**: Reproducible results
3. **Identical Preprocessing**: Same scaling/encoding per fold
4. **Consistent Metrics**: Same metric definitions
5. **Equal Folds**: Each fold gets equal weight

### Preventing Inconsistencies

- **Validation**: MetricsValidator checks for NaN, Inf, out-of-range
- **Consistency**: All metrics calculated uniformly
- **Documentation**: Clear metric definitions
- **Testing**: Verify metrics across datasets

## Architecture for Normalization and CBS

The metrics engine is designed with extensibility for future enhancements:

### Normalization (Future)

Prepare for metric normalization:
- Metrics stored with raw and normalized values
- Pipeline supports normalization step
- Architecture ready for custom normalizers

### CBS (Conditional Bias Score) (Future)

Architecture prepared for CBS:
- Metrics split by subgroup  - Bias calculation framework ready
- Fairness assessment pipeline

**Note**: CBS not implemented yet, architecture is prepared.

## Design Principles

### ✓ Uniformity

All metrics calculated consistently across models and datasets.

### ✓ Completeness

Predictive, computational, and stability metrics all included.

### ✓ Fairness

Equal treatment of all models, stratified folds, fixed seeds.

### ✓ Clarity

Clear metric definitions, transparent calculation.

### ✓ Extensibility

Architecture ready for normalization and CBS.

## API Reference

### MetricsCalculator

```python
calculator = MetricsCalculator()

# Main methods
calculator.calculate_fold_metrics(...)     # Single fold
calculator.aggregate_fold_metrics(...)     # Across folds
calculator.calculate_stability_metrics(...) # Stability
calculator.get_comprehensive_summary(...)  # Complete report
calculator.export_to_dataframe(...)        # Export
```

### MetricsValidator

```python
is_valid, errors = MetricsValidator.validate_metrics(fold_metrics_list)
```

### Functions

```python
compute_classification_metrics(...)        # Predictive
compute_timing_metrics(...)                # Computational
compute_stability_metrics(...)             # Stability
compute_all_stability_metrics(...)         # All stability
assess_model_stability(...)                # Assessment
compare_computational_efficiency(...)      # Efficiency
```

## Integration

- **With Validation Framework**: Metrics calculated per fold in CrossValidator
- **With Models**: Works with any sklearn-compatible model
- **With Data Pipeline**: Uses preprocessed data for fair evaluation

## Performance Considerations

- Metrics calculation adds minimal overhead
- Aggregation is O(n) where n = number of folds
- Stability calculation is O(m) where m = number of metrics
- Export to DataFrame is fast and memory-efficient

## Common Patterns

### Get Mean and Std

```python
summary = calculator.get_comprehensive_summary(fold_list)
acc_mean = summary['overall']['accuracy_mean']
acc_std = summary['overall']['accuracy_std']
```

### Check Model Stability

```python
stability = calculator.calculate_stability_metrics(fold_list)
level = stability['assessment']['stability_level']  # High/Medium/Low
```

### Compare Models

```python
summaries = {name: calculator.get_comprehensive_summary(...)
             for name, results in all_results.items()}
```

### Export for Analysis

```python
df = calculator.export_to_dataframe(fold_metrics_list)
df.to_csv('metrics.csv')
# Import into Excel or R for further analysis
```

## Next Steps

The metrics engine is complete and ready for:
- Comprehensive model evaluation
- Fair model comparisons
- Stability analysis across datasets
- Architecture ready for future enhancements (normalization, CBS)

## References

- **MetricsCalculator**: Unified metrics computation
- **MetricsValidator**: Metrics validation
- **Predictive Metrics**: scorers.py
- **Computational Metrics**: computational.py
- **Stability Metrics**: stability.py

