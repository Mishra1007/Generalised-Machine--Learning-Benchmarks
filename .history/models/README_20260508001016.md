# Models Module

Unified interface for classification models with registry-based configuration.

## Purpose

Provides:
- Consistent train/predict interface across all models
- Sklearn compatibility with validation engine
- Configuration management and reproducibility
- Model registry for discovery and instantiation
- Baseline models for benchmarking

## Supported Models

| Model | Type | Speed | Best For | Probability |
|-------|------|-------|----------|-------------|
| **LogisticRegression** | Linear | ⚡⚡⚡ Fast | Binary/multi-class, linear boundaries | ✓ Yes |
| **DecisionTree** | Tree | ⚡⚡ Fast | Interpretability, non-linear boundaries | ✓ Yes |
| **RandomForest** | Ensemble | ⚡⚡ Fast | General purpose, robustness | ✓ Yes |
| **SVM** | Kernel | ⚡ Slow | High-dimensional data, non-linear | ✓ Yes |
| **GradientBoosting** | Ensemble | ⚡ Slow | Best performance, tuning needed | ✓ Yes |

## Quick Start

### 1. Create a Model

```python
from models import create_model

# Using convenience function
model = create_model('RandomForest', n_estimators=100, random_state=42)

# Or direct instantiation
from models import RandomForestModel
model = RandomForestModel(n_estimators=100, random_state=42)
```

### 2. Train the Model

```python
# Model is sklearn-compatible
model.fit(X_train, y_train)
```

### 3. Make Predictions

```python
# Standard predictions
y_pred = model.predict(X_test)

# Probability predictions (if supported)
y_proba = model.predict_proba(X_test)
```

### 4. Use with Validation Framework

```python
from validation import CrossValidator

validator = CrossValidator()
results = validator.validate(
    X_train, y_train, model,
    model_name='RandomForest',
    dataset_name='iris'
)

summary = results.get_summary()
print(f"Accuracy: {summary['accuracy_mean']:.4f}")
```

## Architecture

### Class Hierarchy

```
BaseModel (abstract)
├── LogisticRegressionModel
├── DecisionTreeModel
├── RandomForestModel
├── SVMModel
└── GradientBoostingModel
```

### Model Interface

All models provide:

```python
class BaseModel:
    def fit(X_train, y_train) -> self
    def predict(X_test) -> y_pred
    def predict_proba(X_test) -> probabilities  # Optional
    def is_fitted() -> bool
    def get_config() -> dict
```

### Registry System

```python
ModelRegistry
├── register(name, model_class)  # Add new model
├── create(name, **kwargs)       # Instantiate model
├── get(name)                    # Get model class
├── list_models()                # List all models
└── get_model_info(name)         # Get model details
```

## Model Descriptions

### LogisticRegression

Linear classification model using logistic function.

**When to use**:
- Baseline comparison
- Binary classification
- Linear relationships
- Fast training needed

**Configuration**:
```python
model = create_model(
    'LogisticRegression',
    max_iter=1000,          # Maximum iterations
    random_state=42,        # Reproducibility
)
```

**Properties**:
- ⚡⚡⚡ Fastest training and prediction
- ✓ Produces probability estimates
- ✓ Works with any feature scale
- ✗ Only linear boundaries

### DecisionTree

Tree-based classification with interpretable rules.

**When to use**:
- Interpretability important
- Non-linear relationships
- Small datasets
- Feature importance needed

**Configuration**:
```python
model = create_model(
    'DecisionTree',
    max_depth=10,                  # Tree depth limit
    min_samples_split=2,           # Min samples to split
    min_samples_leaf=1,            # Min samples in leaf
    random_state=42,
)
```

**Properties**:
- ⚡⚡ Fast training and prediction
- ✓ Highly interpretable
- ✓ Works with any feature scale
- ✗ Prone to overfitting
- ✗ Unstable with small data changes

### RandomForest

Ensemble of decision trees with bagging.

**When to use**:
- Strong baseline performance
- Feature importance needed
- Imbalanced datasets
- Robust predictions

**Configuration**:
```python
model = create_model(
    'RandomForest',
    n_estimators=100,              # Number of trees
    max_depth=None,                # Unlimited depth
    min_samples_split=2,           # Min samples to split
    min_samples_leaf=1,            # Min samples in leaf
    random_state=42,
    n_jobs=-1,                     # Parallel jobs (-1 = all)
)
```

**Properties**:
- ⚡⚡ Reasonably fast
- ✓ Strong baseline performance
- ✓ Handles non-linear relationships
- ✓ Feature importance
- ✓ Parallelizable
- ✗ Less interpretable than single tree

### SVM

Kernel-based model for complex non-linear boundaries.

**When to use**:
- High-dimensional data
- Non-linear boundaries
- Memory is critical
- Small datasets

**Configuration**:
```python
model = create_model(
    'SVM',
    kernel='rbf',                  # Kernel type
    C=1.0,                         # Regularization
    gamma='scale',                 # Kernel coefficient
    probability=True,              # Enable probabilities
    random_state=42,
)
```

**Kernel Options**:
- `'linear'`: Linear boundaries (fast)
- `'rbf'`: Non-linear (default)
- `'poly'`: Polynomial boundaries
- `'sigmoid'`: Sigmoid kernel

**Properties**:
- ⚡ Slow on large datasets
- ✓ Works in high dimensions
- ✓ Non-linear boundaries
- ✓ Memory efficient
- ✗ Requires feature scaling
- ✗ Slow to train/predict

### GradientBoosting

Sequential ensemble learning model.

**When to use**:
- Maximum accuracy needed
- Hyperparameter tuning acceptable
- Time is not critical
- Both linear and non-linear patterns

**Configuration**:
```python
model = create_model(
    'GradientBoosting',
    n_estimators=100,              # Boosting stages
    learning_rate=0.1,             # Shrinkage rate
    max_depth=3,                   # Tree depth
    min_samples_split=2,           # Min samples to split
    min_samples_leaf=1,            # Min samples in leaf
    subsample=1.0,                 # Fraction for training
    random_state=42,
)
```

**Properties**:
- ⚡ Slow training
- ✓ Often best performance
- ✓ Feature importance
- ✓ Handles non-linear relationships
- ✗ Parameter tuning important
- ✗ Memory intensive

## Registry System

### List Available Models

```python
from models import list_models

models = list_models()
print(models)
# Output: ['DecisionTree', 'GradientBoosting', 'LogisticRegression', 'RandomForest', 'SVM']
```

### Get Model Information

```python
from models import get_model_info

info = get_model_info('RandomForest')
print(info['docstring'])
```

### Register Custom Model

```python
from models import BaseModel, register_model
import numpy as np

class CustomModel(BaseModel):
    def __init__(self, random_state=42, verbose=False):
        super().__init__(
            name='CustomModel',
            random_state=random_state,
            verbose=verbose
        )
    
    def _build_model(self):
        # Return your sklearn-compatible model
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(random_state=self.random_state)

# Register it
register_model('Custom', CustomModel)

# Use it
model = create_model('Custom', random_state=42)
```

## Usage Patterns

### Pattern 1: Quick Model Creation

```python
from models import create_model

model = create_model('RandomForest', n_estimators=50)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Pattern 2: Model Configuration

```python
from models import LogisticRegressionModel

model = LogisticRegressionModel(
    max_iter=2000,
    random_state=42,
    verbose=True
)
```

### Pattern 3: Fair Model Comparison

```python
from models import create_model
from validation import CrossValidator

models = {
    'lr': create_model('LogisticRegression'),
    'rf': create_model('RandomForest'),
    'gb': create_model('GradientBoosting'),
}

validator = CrossValidator()
results = validator.validate_multiple(X, y, models, 'iris')

for name, res in results.items():
    print(f"{name}: {res.get_summary()['accuracy_mean']:.4f}")
```

### Pattern 4: Model Configuration from Dictionary

```python
config = {
    'name': 'RandomForest',
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
}

model = create_model(**config)
```

### Pattern 5: Check Model Status

```python
model = create_model('RandomForest')

print(f"Is fitted: {model.is_fitted()}")
print(f"Config: {model.get_config()}")

model.fit(X_train, y_train)

print(f"Is fitted: {model.is_fitted()}")
print(f"Config: {model.get_config()}")
```

## Performance Characteristics

### Training Time

```
LogisticRegression    ⚡⚡⚡ < 0.1s
DecisionTree          ⚡⚡⚡ < 0.1s
RandomForest          ⚡⚡   0.1-1s
GradientBoosting      ⚡    1-10s
SVM                   ⚡    1-10s (depends on kernel)
```

### Prediction Time (per 10k samples)

```
LogisticRegression    ⚡⚡⚡ < 1ms
DecisionTree          ⚡⚡⚡ < 1ms
RandomForest          ⚡⚡   1-10ms
GradientBoosting      ⚡⚡   1-10ms
SVM                   ⚡    10-100ms
```

### Memory Usage

```
LogisticRegression    ⚡⚡⚡ Minimal (coefficients only)
DecisionTree          ⚡⚡   Small (tree structure)
RandomForest          ⚡    Medium (multiple trees)
GradientBoosting      ⚡    Medium (multiple trees)
SVM                   ⚡⚡⚡ Low (support vectors)
```

## Integration with Framework

### Data Pipeline

```python
from preprocessing import prepare_dataset
from models import create_model

# 1. Prepare data
X_train, X_test, y_train, y_test, _ = prepare_dataset(
    'iris',
    test_size=0.3,
    scaling_method='standard'  # Important for SVM and LogisticRegression
)

# 2. Create model
model = create_model('RandomForest', n_estimators=100)

# 3. Train
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)
```

### Validation Framework

```python
from validation import CrossValidator
from models import create_model

validator = CrossValidator()
model = create_model('RandomForest')

results = validator.validate(X, y, model, 'rf', 'iris')
summary = results.get_summary()
```

## Reproducibility

All models support reproducibility:

```python
# Run 1
model1 = create_model('RandomForest', n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

# Run 2 (same result)
model2 = create_model('RandomForest', n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

assert np.array_equal(y_pred1, y_pred2)  # Always True
```

## Design Principles

### ✓ Sklearn Compatibility

All models follow sklearn's interface:
- `fit(X, y)` → train
- `predict(X)` → predict
- `predict_proba(X)` → probabilities (optional)

### ✓ Reproducibility

- Fixed `random_state` parameter ensures deterministic training
- Same configuration = same results

### ✓ Extensibility

- BaseModel abstract class for custom implementations
- Registry system for registration
- Plugin architecture

### ✓ Simplicity

- Single responsibility per model class
- Minimal wrapper over sklearn
- Clear configuration options

## Common Issues and Solutions

### Issue: SVM Too Slow

**Solution**: Use smaller dataset or linear kernel
```python
model = create_model('SVM', kernel='linear')  # Faster
```

### Issue: DecisionTree Overfitting

**Solution**: Limit tree depth
```python
model = create_model('DecisionTree', max_depth=5)  # Shallower tree
```

### Issue: GradientBoosting Not Converging

**Solution**: Reduce learning rate or increase iterations
```python
model = create_model(
    'GradientBoosting',
    learning_rate=0.05,      # Lower learning rate
    n_estimators=200,        # More iterations
)
```

## Next Steps

The models module is complete and ready for:
- Hyperparameter tuning (within validation folds)
- Benchmark comparisons
- Feature importance analysis
- Integration with experiments module

## References

- **BaseModel**: Abstract base class for all models
- **ModelRegistry**: Model discovery and factory
- **Validation Framework**: For fair model evaluation
- **Data Pipeline**: For preprocessing integration

