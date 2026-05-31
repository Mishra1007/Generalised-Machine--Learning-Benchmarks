# Preprocessing

This directory contains all data preprocessing and feature engineering logic with **data leakage prevention** and **reproducibility** at its core.

## Purpose

- Feature scaling and normalization
- Feature engineering and transformation
- Handling missing values
- Categorical encoding
- Train-test splitting with stratification
- **No data leakage** (fit on training data only)

## Structure

```
preprocessing/
├── pipelines.py              # PreprocessingPipeline class
├── data_preparation.py       # Unified data loading + preprocessing
├── DESIGN_DECISIONS.md       # Architecture and design rationale
└── __init__.py               # Public API
```

## Core Components

### 1. **PreprocessingPipeline** (`pipelines.py`)

Builds sklearn pipelines with proper feature type handling:

```python
from preprocessing import create_pipeline

# Create pipeline (auto-detects feature types)
pipeline = create_pipeline(
    X_train,
    scaling_method='standard',      # standard, minmax, robust
    encoding_method='onehot',       # onehot, ordinal
    random_state=42
)

# Fit ONLY on training data (prevent leakage)
X_train_processed = pipeline.fit_and_transform(X_train, y_train)

# Transform test data using training parameters
X_test_processed = pipeline.transform(X_test)
```

**Features**:
- Automatic feature type detection
- Separate handling of numerical and categorical features
- SimpleImputer for missing values
- Configurable scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Configurable encoding (OneHotEncoder, OrdinalEncoder)
- Data leakage prevention

### 2. **DataPreparation** (`data_preparation.py`)

Unified pipeline combining dataset loading + preprocessing:

```python
from preprocessing import prepare_dataset

# Complete pipeline: load → split → preprocess
X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    dataset_name='iris',
    test_size=0.3,
    random_state=42,
    scaling_method='standard'
)
```

**Workflow**:
1. Load dataset from registry
2. Split into train/test (with stratification)
3. Create preprocessing pipeline
4. Fit on training data
5. Transform all splits
6. Return ready-to-use data

**Three-way Split** (for CV + tuning):
```python
from preprocessing import DataPreparation

prep = DataPreparation(random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
    dataset_name='iris',
    train_size=0.7,      # 70% training
    val_size=0.15,       # 15% validation
    scaling_method='standard'
)
```

## Key Design Principles

### ✓ Data Leakage Prevention

```python
# CORRECT: Fit ONLY on training data
X_train_processed = pipeline.fit_and_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)  # Uses training parameters

# WRONG: Would inflate test performance
# pipeline.fit_and_transform(X_test)
```

Preprocessing parameters (scaler mean/std, encoder categories) are learned from **training data only**.

### ✓ Reproducibility

```python
prep = DataPreparation(random_state=42)
# Same dataset_name + random_state = identical results
```

All randomness controlled via `random_state` parameter.

### ✓ Modular Design

Separate numerical and categorical pipelines:
- **Numerical**: Imputation → Scaling
- **Categorical**: Imputation → Encoding

## Configuration Options

### Scaling Methods
- `'standard'`: StandardScaler (mean=0, std=1) - **default**
- `'minmax'`: MinMaxScaler (range [0, 1])
- `'robust'`: RobustScaler (robust to outliers)

### Encoding Methods
- `'onehot'`: OneHotEncoder - **default**, good for linear models
- `'ordinal'`: OrdinalEncoder - good for tree-based models

### Imputation Strategies
- **Numerical**: 'mean' (default), 'median', 'constant'
- **Categorical**: 'most_frequent' (default), 'constant'

## Usage Examples

### Example 1: Simple Train/Test Split

```python
from preprocessing import prepare_dataset

X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    dataset_name='iris',
    test_size=0.3,
    random_state=42
)

# Ready for model training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

### Example 2: Manual Pipeline Control

```python
from preprocessing import create_pipeline
from sklearn.model_selection import train_test_split

X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit pipeline
pipeline = create_pipeline(X_train, scaling_method='standard')
X_train_processed = pipeline.fit_and_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)
```

### Example 3: Cross-Validation Integration

```python
from preprocessing import prepare_dataset
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test, _ = prepare_dataset('iris')

# CV on training data (test held out for final evaluation)
model = RandomForestClassifier()
scores = cross_val_score(model, X_train, y_train, cv=5)

# Final evaluation
model.fit(X_train, y_train)
final_score = model.score(X_test, y_test)
```

## Philosophy

- **Use scikit-learn Pipelines** for reproducibility and composition
- **Fit on training data only** to prevent data leakage
- **Separate feature types** for clean, maintainable code
- **Document transformations** for analysis and reproducibility

## Running Examples

```bash
python examples/example_data_pipeline.py
```

This demonstrates:
1. Dataset registration
2. Loading and exploration
3. Pipeline creation and fitting
4. No-leakage train/test processing
5. Integration with cross-validation

## Architecture Deep Dive

See [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) for:
- Complete architecture explanation
- Design rationale for each component
- Data leakage prevention details
- Integration with CV and model training
- Performance considerations
- Common pitfalls and solutions
