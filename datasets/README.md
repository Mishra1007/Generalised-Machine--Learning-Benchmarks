# Datasets

This directory handles all dataset operations including loading, caching, and registry management.

## Purpose

- Load datasets from various sources (CSV, Parquet, sklearn, numpy)
- Cache downloaded datasets locally
- Manage dataset versioning and metadata
- Provide dataset discovery through registry
- Handle data validation and integrity checks

## Structure

```
datasets/
├── loaders.py          # DatasetLoader class for loading from files
├── registry.py         # DatasetRegistry for managing available datasets
└── __init__.py         # Public API exports
```

## Core Components

### 1. **DatasetLoader** (`loaders.py`)

Loads datasets from CSV files with automatic metadata computation:

```python
from datasets import load_dataset

# Load CSV dataset
X, y, metadata = load_dataset(
    filepath='data/iris.csv',
    target_column='species'
)

# Returns:
# - X: Features (pandas DataFrame)
# - y: Target variable (pandas Series)
# - metadata: Dataset information
```

**Metadata Includes**:
```python
{
    'filepath': 'data/iris.csv',
    'n_samples': 150,
    'n_features': 4,
    'n_target_classes': 3,
    'target_type': 'classification',
    'numerical_features': ['sepal_length', 'sepal_width', ...],
    'categorical_features': [],
    'missing_values': {},  # {column: count, ...}
    'feature_names': [...],
    'target_name': 'species'
}
```

### 2. **DatasetRegistry** (`registry.py`)

Centralized registry for discovering and configuring datasets:

```python
from datasets import register_dataset, get_dataset_config, list_datasets

# Register a dataset once
register_dataset(
    name='iris',
    filepath='data/iris.csv',
    target_column='species',
    description='Iris flower classification dataset',
    task_type='classification'
)

# List all datasets
datasets = list_datasets()
# {'iris': 'Iris flower classification dataset', ...}

# Get dataset configuration
config = get_dataset_config('iris')
# {
#     'filepath': 'data/iris.csv',
#     'target_column': 'species',
#     'description': '...',
#     'task_type': 'classification'
# }
```

## Usage Workflow

### Step 1: Register Datasets

```python
from datasets import register_dataset

# Register once in your initialization code
register_dataset(
    name='my_dataset',
    filepath='data/my_dataset.csv',
    target_column='label',
    description='My classification dataset',
    task_type='classification'
)
```

### Step 2: Use in Experiments

```python
from preprocessing import prepare_dataset

# Automatic loading and preprocessing
X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    dataset_name='my_dataset',
    test_size=0.3,
    scaling_method='standard'
)
```

## Supported Data Formats

### CSV Files

```python
from datasets import load_dataset

X, y, metadata = load_dataset(
    filepath='data/dataset.csv',
    target_column='target_column_name',
    sep=',',              # Delimiter
    encoding='utf-8'      # File encoding
)
```

### Numpy Arrays

```python
from datasets import DatasetLoader

loader = DatasetLoader()
X, y, metadata = loader.load_numpy(
    X_filepath='data/features.npy',
    y_filepath='data/targets.npy'
)
```

## Feature Type Detection

Automatically detects feature types:

```python
# Numerical features: int, float
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Categorical features: object, category
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
```

This enables proper handling in preprocessing pipelines.

## Missing Value Detection

Automatically identifies missing values:

```python
metadata['missing_values']
# {
#     'column1': 5,   # 5 missing values
#     'column2': 0,   # No missing values
#     ...
# }
```

Preprocessing pipelines automatically handle these with imputation.

## Task Type Inference

Automatic classification/regression detection:

```python
# Classification: discrete target or < 20 unique values
task_type = 'classification'

# Regression: continuous target with many unique values
task_type = 'regression'
```

## Configuration Example

```python
from datasets import register_dataset

# Register classification dataset
register_dataset(
    name='iris',
    filepath='data/iris.csv',
    target_column='species',
    description='Iris flower classification',
    task_type='classification',
    n_features=4,
    source='sklearn'
)

# Register regression dataset
register_dataset(
    name='housing',
    filepath='data/housing.csv',
    target_column='price',
    description='Housing price prediction',
    task_type='regression',
    source='local'
)
```

## Integration with Preprocessing

Datasets and preprocessing work together seamlessly:

```python
from preprocessing import DataPreparation
from datasets import register_dataset

# Register
register_dataset(
    'my_data',
    filepath='data/dataset.csv',
    target_column='target'
)

# Prepare (load + split + preprocess)
prep = DataPreparation(random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
    dataset_name='my_data',
    train_size=0.7,
    val_size=0.15
)
```

## Best Practices

### 1. Register Datasets Centrally

```python
# config/dataset_config.py
from datasets import register_dataset

def setup_datasets():
    register_dataset('iris', 'data/iris.csv', 'species', ...)
    register_dataset('cancer', 'data/cancer.csv', 'diagnosis', ...)
    register_dataset('housing', 'data/housing.csv', 'price', ...)
```

### 2. Use Registry for Experiment Configuration

```python
# experiments/benchmark.yaml
datasets:
  - iris
  - cancer
  - housing
```

### 3. Consistent Target Column Names

```python
# Good: Explicit target column
register_dataset(..., target_column='target')
register_dataset(..., target_column='label')

# Avoid: Ambiguous names
register_dataset(..., target_column='y')
```

### 4. Descriptive Dataset Names

```python
# Good: Clear dataset identifiers
register_dataset(name='breast_cancer_classification', ...)
register_dataset(name='house_price_regression', ...)

# Avoid: Generic names
register_dataset(name='data1', ...)
register_dataset(name='dataset_csv', ...)
```

## Examples

Run the example to see datasets and preprocessing in action:

```bash
python examples/example_data_pipeline.py
```

This demonstrates:
1. Dataset registration
2. Loading with metadata
3. Feature type detection
4. Integration with preprocessing
5. Train/test splitting

## Future Enhancements

- Remote dataset downloading (URLs)
- Dataset caching and versioning
- Data quality validation
- Schema enforcement
- Automatic encoding detection
- Support for multiple data formats (Parquet, HDF5, etc.)
