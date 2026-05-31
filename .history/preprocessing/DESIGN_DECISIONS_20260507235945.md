# Data Pipeline Architecture & Design Decisions

## Overview

The data pipeline implements a **reusable, modular, and leakage-free** approach to dataset loading and preprocessing. It's designed for research reproducibility and seamless integration with cross-validation systems.

## Core Components

### 1. **Dataset Loading** (`datasets/loaders.py`)

**Purpose**: Load datasets from various sources with metadata.

**Key Features**:
- CSV loading with pandas
- Automatic feature type detection
- Missing value analysis
- Dataset metadata computation
- Extensible for multiple data sources

**Design Decision**:
```python
# Separation of concerns
loader = DatasetLoader()
X, y, metadata = loader.load_csv(
    filepath='data/iris.csv',
    target_column='species'
)
```

**Metadata Includes**:
- Number of samples and features
- Feature type classification (numerical/categorical)
- Missing value detection
- Target variable information
- Task type inference (classification vs regression)

### 2. **Dataset Registry** (`datasets/registry.py`)

**Purpose**: Centralized catalog of available datasets.

**Key Features**:
- Register datasets with metadata
- Discover datasets by name
- Filter by task type (classification/regression)
- Configuration-based access

**Design Decision**: Registry pattern enables:
- Configuration-driven experiments
- Dataset discovery
- Easy dataset switching in benchmarks

```python
# Register once
register_dataset(
    name='iris',
    filepath='data/iris.csv',
    target_column='species',
    task_type='classification',
)

# Use anywhere
config = get_dataset_config('iris')
X, y, _ = load_dataset(config['filepath'], config['target_column'])
```

### 3. **Preprocessing Pipelines** (`preprocessing/pipelines.py`)

**Purpose**: Feature engineering with reproducibility and no data leakage.

**Architecture**:
```
PreprocessingPipeline
├── Numerical Transformer (per column)
│   ├── SimpleImputer (missing values)
│   └── Scaler (StandardScaler/MinMaxScaler/RobustScaler)
└── Categorical Transformer (per column)
    ├── SimpleImputer (missing values)
    └── Encoder (OneHotEncoder/OrdinalEncoder)
```

**Key Design Decisions**:

#### a) **Separate Feature Type Handling**
- **Numerical**: Missing imputation → Scaling
- **Categorical**: Missing imputation → Encoding
- Uses sklearn's `ColumnTransformer` for clean separation

```python
pipeline = PreprocessingPipeline(
    numerical_features=['age', 'income'],
    categorical_features=['gender', 'education'],
    scaling_method='standard',
    encoding_method='onehot',
)
```

#### b) **Data Leakage Prevention** (CRITICAL)
```python
# ONLY fit on training data
X_train_processed = pipeline.fit_and_transform(X_train, y_train)

# Transform test/val using training parameters
X_test_processed = pipeline.transform(X_test)
```

**Why This Matters**:
- Scaling parameters (mean, std) are learned from training data
- Categorical encodings are based on training values
- Test data never influences these parameters
- Prevents artificially inflated performance metrics

#### c) **Missing Value Strategies**
- **Numerical**: Mean, Median, Constant (configurable)
- **Categorical**: Most frequent, Constant
- Imputation happens before scaling/encoding

#### d) **Encoding Methods**
- **OneHotEncoder**: Default, good for tree-based and linear models
  - `drop='if_binary'`: Prevents multicollinearity for binary features
- **OrdinalEncoder**: Alternative for tree-based models or memory constraints

#### e) **Scaling Methods**
- **StandardScaler**: Default, good for most algorithms
  - Mean=0, Std=1
  - Best for algorithms sensitive to feature magnitude
- **MinMaxScaler**: Bounds features to [0, 1]
  - Good when algorithms assume bounded features
- **RobustScaler**: Uses median and IQR
  - Best when data has outliers

### 4. **Unified Data Preparation** (`preprocessing/data_preparation.py`)

**Purpose**: Complete end-to-end pipeline combining loading and preprocessing.

**Workflow**:
```
1. Load dataset from registry
2. Split into train/validation/test
3. Create preprocessing pipeline from training data
4. Fit pipeline on training data ONLY
5. Transform all splits using fitted parameters
6. Return ready-to-train data
```

**Key Methods**:

#### `prepare()` - Three-way split
```python
X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
    dataset_name='iris',
    train_size=0.7,     # 70% training
    val_size=0.15,      # 15% validation
    scaling_method='standard'
)
```

Useful for:
- Hyperparameter tuning on validation set
- Final evaluation on test set
- Nested cross-validation

#### `prepare_train_test()` - Simple split
```python
X_train, X_test, y_train, y_test = prep.prepare_train_test(
    dataset_name='iris',
    test_size=0.3,
    scaling_method='standard'
)
```

Useful for:
- Cross-validation (splits handled by CV framework)
- Simple train/test evaluation

## Integration Points

### 1. **With Cross-Validation**

Data leakage prevention enables clean CV integration:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# For each CV fold:
# 1. Training fold preprocessed and fitted
# 2. Validation fold transformed using training parameters
# 3. No leakage between folds
```

### 2. **With Model Training**

Preprocessed data integrates directly with sklearn models:

```python
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = prepare_dataset('iris')

# Direct model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. **With Hyperparameter Tuning**

Validation set enables tuning without test leakage:

```python
from sklearn.model_selection import GridSearchCV

X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
    dataset_name='iris',
    train_size=0.7,
    val_size=0.15
)

# Tune on training+validation
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.hstack([y_train, y_val])

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train_val, y_train_val)

# Evaluate on held-out test set
test_score = grid.score(X_test, y_test)
```

## Reproducibility

### Deterministic Output
All randomness is controlled:

```python
prep = DataPreparation(random_state=42)
# Same dataset_name + random_state = identical splits and preprocessing
```

### Configuration Capture
```python
preprocessing_config = prep.get_preprocessing_config()
dataset_metadata = prep.get_dataset_metadata()

# Log or save for experiment tracking
print(preprocessing_config)
# {
#     'numerical_features': [...],
#     'categorical_features': [...],
#     'scaling_method': 'standard',
#     'encoding_method': 'onehot',
#     'random_state': 42
# }
```

## Handling Different Data Scenarios

### Scenario 1: Pure Numerical Data
```python
pipeline = create_pipeline(
    X_train,
    categorical_features=[],  # No categorical features
    scaling_method='standard'
)
```

### Scenario 2: Pure Categorical Data
```python
pipeline = create_pipeline(
    X_train,
    numerical_features=[],  # No numerical features
    encoding_method='onehot'
)
```

### Scenario 3: Mixed with Missing Values
```python
X_train, X_test, y_train, y_test = prepare_dataset(
    'dataset_with_missing_values',
    scaling_method='standard'  # Handles missing automatically
)
```

### Scenario 4: Different Scaling for Outliers
```python
pipeline = create_pipeline(
    X_train,
    scaling_method='robust',  # Robust to outliers
    encoding_method='onehot'
)
```

## Performance Considerations

### 1. **Memory Efficiency**
- OneHotEncoder with `sparse_output=False` for compatibility
- ColumnTransformer avoids duplicate features
- Numpy arrays for processed output

### 2. **Computational Efficiency**
- Imputation and scaling computed once per dataset
- Sklearn pipelines optimized for performance
- No redundant transformations

### 3. **Feature Reduction (Optional)**
Future enhancement for:
```python
from sklearn.feature_selection import SelectKBest

# Can be added to pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(k=10))
])
```

## Common Pitfalls & Solutions

### Pitfall 1: Fitting on test data
```python
# ❌ WRONG: Leakage!
pipeline.fit_and_transform(X_test)

# ✓ CORRECT
pipeline.fit_and_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

### Pitfall 2: Forgetting to fit
```python
# ❌ WRONG: Pipeline not fitted
X_test_processed = pipeline.transform(X_test)

# ✓ CORRECT
X_train_processed = pipeline.fit_and_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

### Pitfall 3: Mixing train and test
```python
# ❌ WRONG: Test features influence preprocessing
scaler = StandardScaler()
all_data = np.vstack([X_train, X_test])
scaler.fit(all_data)

# ✓ CORRECT
scaler.fit(X_train)  # Only training
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Future Extensions

### 1. **Feature Selection**
```python
# Add SelectKBest, RFE, or feature importance
```

### 2. **Custom Transformers**
```python
# Polynomial features, interaction terms
# Domain-specific transformations
```

### 3. **Data Validation**
```python
# Schema validation, outlier detection
# Data quality checks
```

### 4. **Automated Encoding**
```python
# Target encoding for high-cardinality categories
# Binary encoding for memory-constrained scenarios
```

### 5. **Class Imbalance Handling**
```python
# Applied only to training data:
# SMOTE, oversampling, class weights
```

## Example Usage

```python
from preprocessing import prepare_dataset
from datasets import register_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Register dataset once
register_dataset(
    'my_data',
    filepath='data/my_dataset.csv',
    target_column='target',
    task_type='classification'
)

# Prepare data with single function call
X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    'my_data',
    test_size=0.3,
    random_state=42,
    scaling_method='standard',
    encoding_method='onehot'
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

## Testing

Run example:
```bash
python examples/example_data_pipeline.py
```

Verify:
- ✓ Datasets load correctly
- ✓ Preprocessing handles mixed types
- ✓ No data leakage (test parameters independent)
- ✓ Output ready for model training
- ✓ Reproducible with same random_state
