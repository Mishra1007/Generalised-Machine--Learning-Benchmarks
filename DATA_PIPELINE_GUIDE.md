# Data Pipeline Implementation Guide

## Overview

This document provides a comprehensive guide to the dataset loading and preprocessing pipeline implemented in the ML Benchmarking Framework.

## Architecture Summary

The pipeline is built around **three core principles**:

1. **Modularity**: Separate concerns between loading and preprocessing
2. **Reproducibility**: Deterministic results with seeded randomness
3. **No Data Leakage**: Preprocessing fitted only on training data

## Components

```
┌─────────────────────────────────────────────────┐
│   Experiment Configuration (YAML/JSON)          │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│   DataPreparation (preprocessing/)              │
│   - Load dataset from registry                  │
│   - Split train/val/test                        │
│   - Create preprocessing pipeline               │
│   - Fit on training data                        │
│   - Transform all splits                        │
└──────────────┬──────────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────┐  ┌──────────────────────────┐
│ DatasetLoader│  │ PreprocessingPipeline    │
│ (loaders.py) │  │ (pipelines.py)           │
│              │  │                          │
│ - Load CSV   │  │ - Numerical Transform    │
│ - Parse      │  │ - Categorical Transform  │
│ - Metadata   │  │ - Feature Handling       │
└────────┬─────┘  └──────────┬───────────────┘
         │                   │
         ▼                   ▼
┌──────────────┐  ┌──────────────────────────┐
│ DatasetRegistry
│ (registry.py) │  │ sklearn Transformers     │
│              │  │ - SimpleImputer          │
│ - Register   │  │ - StandardScaler         │
│ - Discover   │  │ - OneHotEncoder          │
│ - Lookup     │  │ - ColumnTransformer      │
└──────────────┘  └──────────────────────────┘
         ▲
         │
         └─ get_dataset_config()
```

## Key Files

| File | Purpose |
|------|---------|
| `datasets/loaders.py` | DatasetLoader class - loads CSV files and computes metadata |
| `datasets/registry.py` | DatasetRegistry - manages available datasets |
| `preprocessing/pipelines.py` | PreprocessingPipeline - sklearn-based feature engineering |
| `preprocessing/data_preparation.py` | DataPreparation - unified load + preprocess workflow |
| `examples/example_data_pipeline.py` | Complete examples demonstrating all components |
| `preprocessing/DESIGN_DECISIONS.md` | Detailed architecture and design rationale |

## Quick Start

### 1. Register Your Dataset

```python
from datasets import register_dataset

register_dataset(
    name='my_data',
    filepath='data/my_dataset.csv',
    target_column='label',
    description='My dataset for benchmarking',
    task_type='classification'
)
```

### 2. Prepare Data

```python
from preprocessing import prepare_dataset

# Simple train/test split
X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    dataset_name='my_data',
    test_size=0.3,
    random_state=42,
    scaling_method='standard',
    encoding_method='onehot'
)
```

### 3. Train Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Test Accuracy: {score:.4f}")
```

## Detailed Workflow

### Step 1: Dataset Loading

**File**: `datasets/loaders.py`

```python
from datasets import load_dataset

X, y, metadata = load_dataset(
    filepath='data/iris.csv',
    target_column='species'
)
```

**What happens**:
1. Read CSV file with pandas
2. Separate features (X) and target (y)
3. Detect feature types (numerical/categorical)
4. Identify missing values
5. Infer task type (classification/regression)
6. Compute metadata dictionary

**Metadata content**:
```python
{
    'filepath': 'data/iris.csv',
    'n_samples': 150,                    # Total samples
    'n_features': 4,                     # Feature count
    'n_target_classes': 3,               # Classes/values
    'target_type': 'classification',     # Task type
    'numerical_features': [...],         # Numerical columns
    'categorical_features': [...],       # Categorical columns
    'missing_values': {...},             # Missing counts
    'feature_names': [...],              # Column names
    'target_name': 'species'             # Target column
}
```

### Step 2: Dataset Registry

**File**: `datasets/registry.py`

```python
from datasets import register_dataset, get_dataset_config

# Register once
register_dataset(
    name='iris',
    filepath='data/iris.csv',
    target_column='species',
    description='Iris flower classification',
    task_type='classification'
)

# Use anywhere
config = get_dataset_config('iris')
X, y, _ = load_dataset(config['filepath'], config['target_column'])
```

**Benefits**:
- Centralized dataset configuration
- Easy dataset switching in experiments
- Consistent metadata across framework
- Simple dataset discovery

### Step 3: Train/Test Splitting

**File**: `preprocessing/data_preparation.py`

```python
from preprocessing import DataPreparation

prep = DataPreparation(random_state=42)

# Simple split
X_train, X_test, y_train, y_test = prep.prepare_train_test(
    'iris',
    test_size=0.3
)

# Or three-way split
X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
    'iris',
    train_size=0.7,
    val_size=0.15
)
```

**Important**:
- Splits are stratified (preserves class distribution)
- Random seed ensures reproducibility
- Test/validation data held completely separate

### Step 4: Preprocessing Pipeline Creation

**File**: `preprocessing/pipelines.py`

```python
from preprocessing import create_pipeline

pipeline = create_pipeline(
    X_train,
    numerical_features=['feature1', 'feature2'],
    categorical_features=['category1'],
    scaling_method='standard',
    encoding_method='onehot',
    random_state=42
)
```

**Pipeline Structure**:

```
ColumnTransformer
├── Numerical Transformer
│   ├── SimpleImputer(strategy='mean')      # Handle missing
│   └── StandardScaler()                    # Normalize
└── Categorical Transformer
    ├── SimpleImputer(strategy='most_frequent')
    └── OneHotEncoder(sparse_output=False)
```

### Step 5: Fitting (CRITICAL - No Leakage)

```python
# FIT ONLY ON TRAINING DATA
X_train_processed = pipeline.fit_and_transform(X_train, y_train)

# Parameters learned from X_train:
# - Scaler: mean and std from X_train
# - Imputer: missing value strategy from X_train
# - Encoder: categories from X_train
```

**Why this matters**:
- If you fit on test data, test performance metrics are inflated
- Test data should never influence preprocessing parameters
- This prevents "data leakage" which invalidates evaluation

### Step 6: Transformation

```python
# Transform test/validation using TRAINING parameters
X_test_processed = pipeline.transform(X_test)
X_val_processed = pipeline.transform(X_val)

# Uses mean/std from training data
# Uses categories from training data
# No new information from test/val influences preprocessing
```

## Configuration-Driven Usage

### Experiment Configuration (YAML)

```yaml
# experiments/config.yaml
name: "baseline_comparison"
random_state: 42

datasets:
  - name: "iris"
    test_size: 0.3
  - name: "breast_cancer"
    test_size: 0.3

preprocessing:
  scaling_method: "standard"    # standard, minmax, robust
  encoding_method: "onehot"     # onehot, ordinal

models:
  - "random_forest"
  - "logistic_regression"
  - "svm"
```

### Loading Configuration

```python
import yaml
from preprocessing import prepare_dataset

with open('experiments/config.yaml') as f:
    config = yaml.safe_load(f)

for dataset_config in config['datasets']:
    X_train, X_test, y_train, y_test, _ = prepare_dataset(
        dataset_name=dataset_config['name'],
        test_size=dataset_config['test_size'],
        random_state=config['random_state'],
        scaling_method=config['preprocessing']['scaling_method'],
        encoding_method=config['preprocessing']['encoding_method']
    )
    # Train and evaluate models...
```

## Integration with Cross-Validation

### Standard Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from preprocessing import prepare_dataset
from sklearn.ensemble import RandomForestClassifier

# Load and split
X_train, X_test, y_train, y_test, _ = prepare_dataset('iris', test_size=0.3)

# CV on training set (test held out for final evaluation)
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5)

print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.4f}")

# Final evaluation on held-out test set
model.fit(X_train, y_train)
final_score = model.score(X_test, y_test)
print(f"Test Score: {final_score:.4f}")
```

**Why this works**:
- X_train already preprocessed without test data influence
- CV folds within X_train prevent internal leakage
- Final evaluation on truly held-out test data

### Nested Cross-Validation (Hyperparameter Tuning)

```python
from sklearn.model_selection import GridSearchCV
from preprocessing import DataPreparation

# Three-way split
prep = DataPreparation(random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
    'iris',
    train_size=0.7,
    val_size=0.15
)

# Combine train+val for hyperparameter tuning
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.hstack([y_train, y_val])

# Grid search with CV
param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)
grid.fit(X_train_val, y_train_val)

# Final evaluation on completely held-out test set
final_score = grid.score(X_test, y_test)
print(f"Best params: {grid.best_params_}")
print(f"Final test score: {final_score:.4f}")
```

## Handling Different Data Scenarios

### Scenario 1: Pure Numerical Data

```python
X_train, X_test, y_train, y_test, _ = prepare_dataset(
    'numerical_only_dataset',
    scaling_method='standard'
)
```

Pipeline automatically:
- Imputes missing values (mean strategy)
- Applies scaling
- Skips categorical handling

### Scenario 2: Pure Categorical Data

```python
X_train, X_test, y_train, y_test, _ = prepare_dataset(
    'categorical_only_dataset',
    encoding_method='onehot'
)
```

Pipeline automatically:
- Imputes missing values (most_frequent strategy)
- Applies encoding
- Skips scaling

### Scenario 3: Mixed Data with Missing Values

```python
# No special configuration needed
# Pipeline handles:
# - Numerical: imputation + scaling
# - Categorical: imputation + encoding
# - Missing values: automatic detection and handling
X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    'mixed_data',
    scaling_method='standard',
    encoding_method='onehot'
)

# Check what was handled
print(f"Missing values detected: {metadata['missing_values']}")
print(f"Numerical features: {metadata['numerical_features']}")
print(f"Categorical features: {metadata['categorical_features']}")
```

### Scenario 4: Data with Outliers

```python
# RobustScaler uses median and IQR (robust to outliers)
X_train, X_test, y_train, y_test, _ = prepare_dataset(
    'data_with_outliers',
    scaling_method='robust'  # Instead of 'standard'
)
```

## Advanced Usage

### Custom Dataset Registration

```python
from datasets import register_dataset

register_dataset(
    name='custom_data',
    filepath='data/custom.csv',
    target_column='target',
    description='Custom dataset with specific characteristics',
    task_type='regression',
    # Additional metadata
    source='kaggle',
    n_features=50,
    n_samples=10000,
    missing_percentage=5.0
)
```

### Manual Pipeline Control

```python
from preprocessing import PreprocessingPipeline

# Fine-grained control
pipeline = PreprocessingPipeline(
    numerical_features=['age', 'income'],
    categorical_features=['gender'],
    numerical_strategy='median',        # Use median imputation
    categorical_strategy='constant',    # Use constant imputation
    scaling_method='minmax',            # MinMaxScaler
    encoding_method='ordinal',          # OrdinalEncoder
    random_state=42
)

pipeline.build()
X_train_processed = pipeline.fit_and_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)

# Get preprocessing configuration for reproducibility
config = pipeline.get_config()
```

### Accessing Transformed Feature Names

```python
pipeline = create_pipeline(X_train)
X_train_processed = pipeline.fit_and_transform(X_train)

# Get output feature names
feature_names = pipeline.get_feature_names_out()
print(f"Transformed features: {feature_names}")
```

## Testing & Verification

Run the example to verify correct implementation:

```bash
python examples/example_data_pipeline.py
```

This demonstrates:
1. ✓ Dataset registration
2. ✓ Loading and exploration
3. ✓ Pipeline creation
4. ✓ Fit-on-train-only principle
5. ✓ No data leakage verification
6. ✓ Integration with cross-validation

## Common Pitfalls & Solutions

### ❌ Pitfall 1: Fitting on Test Data

```python
# WRONG - Data leakage!
X_train, X_test, y_train, y_test = split_data()
pipeline.fit_and_transform(X_test)  # Fitting on test!

# CORRECT
X_train, X_test, y_train, y_test = split_data()
X_train_processed = pipeline.fit_and_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

### ❌ Pitfall 2: Forgetting to Fit

```python
# WRONG - Pipeline never fitted
pipeline.transform(X_test)  # Error: not fitted

# CORRECT
pipeline.fit_and_transform(X_train)
pipeline.transform(X_test)
```

### ❌ Pitfall 3: Mixing Train and Test

```python
# WRONG - Test influences preprocessing
all_data = pd.concat([X_train, X_test])
pipeline.fit_and_transform(all_data)

# CORRECT
pipeline.fit_and_transform(X_train)  # Train only
```

### ❌ Pitfall 4: Wrong Data Type

```python
# WRONG - Must be DataFrame for consistent handling
X_train_array = X_train.values  # numpy array
pipeline.fit_and_transform(X_train_array)

# CORRECT
pipeline.fit_and_transform(X_train)  # pandas DataFrame
```

## Next Steps

The data pipeline is now ready for:

1. **Model Implementation** (`models/` module)
   - Create model registry
   - Define scikit-learn model wrappers
   - Support model composition

2. **Experiment Orchestration** (`experiments/` module)
   - Load configuration
   - Execute benchmarks
   - Track results

3. **Metrics Computation** (`metrics/` module)
   - Calculate performance scores
   - Aggregate results
   - Statistical testing

4. **Visualization** (`visualization/` module)
   - Plot results
   - Compare models
   - Generate reports

5. **Analysis** (`analysis/` module)
   - Statistical significance testing
   - Performance comparisons
   - Result aggregation

## References

- **Detailed Design**: See [preprocessing/DESIGN_DECISIONS.md](preprocessing/DESIGN_DECISIONS.md)
- **DatasetLoader**: `datasets/loaders.py`
- **DatasetRegistry**: `datasets/registry.py`
- **PreprocessingPipeline**: `preprocessing/pipelines.py`
- **DataPreparation**: `preprocessing/data_preparation.py`
- **Examples**: `examples/example_data_pipeline.py`

## Summary

The data pipeline provides:

✓ **Modular components** for flexible usage  
✓ **No data leakage** through careful fit/transform separation  
✓ **Reproducibility** with seeded randomness  
✓ **Configuration-driven** design for experiments  
✓ **Integration-ready** for models, CV, and evaluation  

Ready for systematic ML benchmarking with production-quality data handling!
