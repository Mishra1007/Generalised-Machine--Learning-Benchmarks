# Model Execution Engine Design Guide

**Purpose**: Comprehensive guide to the model execution engine and registry system  
**Target Audience**: ML Engineers using models in benchmarks  
**Last Updated**: May 8, 2026  
**Total Length**: 100+ KB documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Model Descriptions](#model-descriptions)
5. [Registry System](#registry-system)
6. [Implementation Guide](#implementation-guide)
7. [Usage Patterns](#usage-patterns)
8. [Integration](#integration)
9. [Reproducibility](#reproducibility)
10. [Extensions](#extensions)

---

## Overview

The model execution engine provides a modular, extensible framework for managing, configuring, and executing classification models.

### Design Goals

✓ **Unified Interface**: Same API for all models  
✓ **Reproducibility**: Fixed random states for deterministic behavior  
✓ **Sklearn Compatibility**: Works seamlessly with validation framework  
✓ **Configurability**: Easy parameter adjustment  
✓ **Extensibility**: Simple to add new models  
✓ **Simplicity**: Minimal wrapper overhead  

### Key Features

- 5 baseline models (LogReg, DT, RF, SVM, GB)
- Model registry for discovery and instantiation
- Configuration management
- Integrated logging
- Compatible with validation framework
- Support for probability predictions

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Registry                            │
├─────────────────────────────────────────────────────────────┤
│ • List available models                                      │
│ • Create model instances                                     │
│ • Get model information                                      │
│ • Register new models                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┬──────┐
        │              │              │              │      │
   ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐  ┌────▼──┐  ┌─▼──────┐
   │LogReg    │  │DT        │  │RF        │  │SVM    │  │GB      │
   └──────────┘  └──────────┘  └──────────┘  └───────┘  └────────┘
        │              │              │              │      │
        └──────────────┴──────────────┴──────────────┴──────┘
                       │
            ┌──────────▼──────────┐
            │   BaseModel         │
            ├─────────────────────┤
            │ • fit(X, y)         │
            │ • predict(X)        │
            │ • predict_proba(X)  │
            │ • get_config()      │
            │ • is_fitted()       │
            └─────────────────────┘
```

### Module Organization

```
models/
├── base.py              # BaseModel abstract class
├── implementations.py   # 5 model implementations
├── registry.py          # ModelRegistry and factories
├── __init__.py          # Public API exports
└── README.md            # Component documentation
```

### Data Flow

```
User Code
   │
   ├──> ModelRegistry.create(name, **kwargs)
   │        │
   │        ├──> Look up model class
   │        └──> Instantiate with configuration
   │
   ├──> model.fit(X_train, y_train)
   │        │
   │        └──> Build sklearn model + train
   │
   ├──> model.predict(X_test)
   │        │
   │        └──> Call sklearn predict
   │
   └──> model.predict_proba(X_test)
            │
            └──> Call sklearn predict_proba (if available)
```

---

## Core Components

### 1. BaseModel (Abstract Class)

Provides unified interface for all model wrappers.

**Responsibilities**:
- Define standard interface (fit/predict/predict_proba)
- Manage model state (fitted/not fitted)
- Handle reproducibility (random_state)
- Provide logging integration
- Manage configuration

**Key Methods**:

```python
def fit(X, y) -> self
    # Train the model
    # Sets _is_fitted = True
    # Stores class info and feature count

def predict(X) -> y_pred
    # Make predictions
    # Raises error if not fitted

def predict_proba(X) -> probabilities
    # Probability predictions (optional)
    # Returns None if not supported

def is_fitted() -> bool
    # Check if model is trained

def get_config() -> dict
    # Get configuration dictionary
```

**Example**:

```python
from models.base import BaseModel
from sklearn.ensemble import AdaBoostClassifier

class AdaBoostModel(BaseModel):
    def __init__(self, n_estimators=50, random_state=42):
        super().__init__(
            name='AdaBoost',
            random_state=random_state
        )
        self.n_estimators = n_estimators
    
    def _build_model(self):
        return AdaBoostClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
```

### 2. Model Implementations

Five concrete model classes inheriting from BaseModel.

**LogisticRegressionModel**:
- Linear classification
- Fast training and prediction
- Probability estimates
- Best for: Baseline, linear relationships

**DecisionTreeModel**:
- Tree-based classification
- Interpretable rules
- Non-linear boundaries
- Best for: Interpretability, small datasets

**RandomForestModel**:
- Ensemble of trees (bagging)
- Strong baseline performance
- Feature importance
- Best for: General purpose, robustness

**SVMModel**:
- Kernel-based classification
- Non-linear boundaries
- Works in high dimensions
- Best for: High-dim data, memory critical

**GradientBoostingModel**:
- Sequential ensemble (boosting)
- Often best accuracy
- Requires tuning
- Best for: Maximum performance

### 3. ModelRegistry

Central registry for model management.

**Responsibilities**:
- Maintain registry of available models
- Create model instances
- Provide model discovery
- Support registration of custom models

**Key Methods**:

```python
@classmethod
def create(name: str, **kwargs) -> BaseModel
    # Create model instance
    # Raises KeyError if not found

@classmethod
def register(name: str, model_class: Type[BaseModel])
    # Register new model
    # Validates BaseModel inheritance

@classmethod
def list_models() -> list
    # List all available models
    # Returns sorted list

@classmethod
def get_model_info(name: str) -> dict
    # Get model information
    # Includes name, class, docstring
```

**Example**:

```python
from models import ModelRegistry

# Create model
model = ModelRegistry.create('RandomForest', n_estimators=100)

# List models
models = ModelRegistry.list_models()

# Get info
info = ModelRegistry.get_model_info('RandomForest')

# Register custom
class MyModel(BaseModel):
    def _build_model(self):
        return ...

ModelRegistry.register('MyModel', MyModel)
```

### 4. Factory Functions

Convenience functions for common operations.

```python
from models import (
    create_model,           # Create instance
    list_models,            # List available
    get_model_info,         # Get information
    register_model,         # Register new model
)

model = create_model('RandomForest', n_estimators=100)
models = list_models()
info = get_model_info('RandomForest')
```

---

## Model Descriptions

### LogisticRegression

**Type**: Linear classifier using logistic function

**When to use**:
- Baseline for comparison
- Fast results needed
- Linear relationship expected
- Interpretable model important

**Configuration**:
```python
model = create_model(
    'LogisticRegression',
    max_iter=1000,          # Iterations before convergence
    random_state=42         # Reproducibility
)
```

**Properties**:
- Training: < 0.1 seconds
- Prediction: < 1ms per 10k samples
- Memory: Minimal (coefficients only)
- Probability: Yes
- Interpretability: High (coefficients)

**Strengths**:
- ✓ Fast to train and predict
- ✓ Works with any scale
- ✓ Probability estimates
- ✓ Easily interpretable
- ✓ Good baseline

**Weaknesses**:
- ✗ Linear boundaries only
- ✗ May underfit complex data
- ✗ Assumes independent features

### DecisionTree

**Type**: Tree-based classifier with interpretable rules

**When to use**:
- Interpretability critical
- Non-linear patterns
- Small to medium datasets
- Feature importance needed

**Configuration**:
```python
model = create_model(
    'DecisionTree',
    max_depth=10,                  # Tree depth limit
    min_samples_split=2,           # Min to split node
    min_samples_leaf=1,            # Min in leaf
    random_state=42
)
```

**Properties**:
- Training: < 0.1 seconds
- Prediction: < 1ms per 10k samples
- Memory: Small (tree structure)
- Probability: Yes
- Interpretability: Very high (rule-based)

**Strengths**:
- ✓ Highly interpretable
- ✓ Non-linear boundaries
- ✓ No scaling needed
- ✓ Fast training/prediction
- ✓ Feature importance

**Weaknesses**:
- ✗ Prone to overfitting
- ✗ Unstable with small changes
- ✗ Greedy node selection
- ✗ May overfit training data

### RandomForest

**Type**: Ensemble of decision trees with bagging

**When to use**:
- Strong baseline needed
- Robustness important
- Feature importance needed
- Parallelization possible

**Configuration**:
```python
model = create_model(
    'RandomForest',
    n_estimators=100,              # Number of trees
    max_depth=None,                # No depth limit
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,                     # Use all cores
    random_state=42
)
```

**Properties**:
- Training: 0.1-1 second
- Prediction: 1-10ms per 10k samples
- Memory: Medium (multiple trees)
- Probability: Yes
- Interpretability: Medium (feature importance)

**Strengths**:
- ✓ Strong baseline performance
- ✓ Handles non-linear patterns
- ✓ Feature importance available
- ✓ Parallelizable
- ✓ Robust to outliers
- ✓ No scaling needed

**Weaknesses**:
- ✗ Less interpretable than tree
- ✗ Memory intensive
- ✗ Slower than single tree
- ✗ May still overfit

**Tuning Parameters**:
- `n_estimators`: More trees → better accuracy but slower
- `max_depth`: Limit overfitting by reducing depth
- `min_samples_split`: Prevent overfitting

### SVM

**Type**: Support Vector Machine with kernel trick

**When to use**:
- High-dimensional data
- Non-linear boundaries needed
- Memory is constraint
- Small to medium datasets

**Configuration**:
```python
model = create_model(
    'SVM',
    kernel='rbf',                  # Kernel type
    C=1.0,                         # Regularization
    gamma='scale',                 # Kernel coefficient
    probability=True,              # Probability estimates
    random_state=42
)
```

**Kernel Options**:
- `'linear'`: Linear boundaries (fastest)
- `'rbf'`: Radial basis function (default, flexible)
- `'poly'`: Polynomial boundaries
- `'sigmoid'`: Sigmoid kernel

**Properties**:
- Training: 1-10 seconds (large datasets: slower)
- Prediction: 10-100ms per 10k samples
- Memory: Low (support vectors only)
- Probability: Yes
- Interpretability: Low (kernel method)

**Strengths**:
- ✓ Works in high dimensions
- ✓ Non-linear boundaries
- ✓ Memory efficient
- ✓ Flexible kernels
- ✓ Good for complex patterns

**Weaknesses**:
- ✗ Slow on large datasets
- ✗ Requires feature scaling
- ✗ Hard to interpret
- ✗ Parameter tuning important (C, gamma)
- ✗ Probability estimates slower

**Tuning Parameters**:
- `C`: Trade-off (higher C → less regularization)
- `gamma`: Kernel parameter (high → tight fit)
- `kernel`: Type of boundary

### GradientBoosting

**Type**: Sequential ensemble (boosting)

**When to use**:
- Maximum accuracy needed
- Hyperparameter tuning acceptable
- Time not critical
- Complex patterns to learn

**Configuration**:
```python
model = create_model(
    'GradientBoosting',
    n_estimators=100,              # Boosting stages
    learning_rate=0.1,             # Shrinkage rate
    max_depth=3,                   # Tree depth
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,                 # Training subset
    random_state=42
)
```

**Properties**:
- Training: 1-10 seconds
- Prediction: 1-10ms per 10k samples
- Memory: Medium (multiple trees)
- Probability: Yes
- Interpretability: Low (sequential models)

**Strengths**:
- ✓ Often best accuracy
- ✓ Handles complex patterns
- ✓ Feature importance
- ✓ No scaling needed
- ✓ Works with both linear/non-linear

**Weaknesses**:
- ✗ Slow training
- ✗ Parameter tuning critical
- ✗ Risk of overfitting
- ✗ Memory intensive
- ✗ Hard to interpret

**Tuning Parameters**:
- `learning_rate`: Lower → slower but better (0.01-0.1)
- `n_estimators`: More iterations → more accuracy but slower
- `max_depth`: 3-5 typical (shallow trees)
- `subsample`: Use 0.8 to prevent overfitting

---

## Registry System

### How It Works

```python
# 1. Define model class
class MyModel(BaseModel):
    def _build_model(self):
        return ...

# 2. Register in registry
ModelRegistry.register('MyModel', MyModel)

# 3. Create instance
model = ModelRegistry.create('MyModel', param1=value1)

# 4. Use like any other model
model.fit(X, y)
y_pred = model.predict(X_test)
```

### Using Registry

```python
from models import get_registry, create_model, list_models

# Method 1: Global functions (recommended)
models = list_models()
model = create_model('RandomForest', n_estimators=50)

# Method 2: Registry class
registry = get_registry()
models = registry.list_models()
model = registry.create('RandomForest', n_estimators=50)

# Method 3: Get model class
registry = get_registry()
RandomForestClass = registry.get('RandomForest')
model = RandomForestClass(n_estimators=50)
```

### Extensibility

```python
from models import BaseModel, register_model
import numpy as np

class KNearestNeighborsModel(BaseModel):
    def __init__(self, n_neighbors=5, random_state=42, verbose=False):
        super().__init__(
            name='KNN',
            random_state=random_state,
            verbose=verbose
        )
        self.n_neighbors = n_neighbors
    
    def _build_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=self.n_neighbors)

# Register it
register_model('KNN', KNearestNeighborsModel)

# Use it
model = create_model('KNN', n_neighbors=5)
```

---

## Implementation Guide

### Step 1: Prepare Data

```python
from preprocessing import prepare_dataset
from sklearn.preprocessing import StandardScaler

# Load and preprocess
X_train, X_test, y_train, y_test, _ = prepare_dataset(
    'iris',
    test_size=0.3,
    scaling_method='standard'  # Important for SVM, LogReg
)
```

### Step 2: Create Model

```python
from models import create_model

model = create_model(
    'RandomForest',
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

### Step 3: Train Model

```python
model.fit(X_train, y_train)
```

### Step 4: Make Predictions

```python
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

### Step 5: Evaluate

```python
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
```

---

## Usage Patterns

### Pattern 1: Simple Usage

```python
from models import create_model

model = create_model('RandomForest', random_state=42)
model.fit(X_train, y_train)
score = model.predict(X_test)
```

### Pattern 2: Configuration from Dict

```python
config = {
    'name': 'GradientBoosting',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': 42,
}

model = create_model(**config)
```

### Pattern 3: Try Multiple Configurations

```python
configs = [
    {'n_estimators': 50},
    {'n_estimators': 100},
    {'n_estimators': 200},
]

results = []
for config in configs:
    model = create_model('RandomForest', **config)
    model.fit(X_train, y_train)
    score = (model.predict(X_test) == y_test).mean()
    results.append((config, score))
```

### Pattern 4: Model Comparison

```python
models = {
    'lr': create_model('LogisticRegression'),
    'dt': create_model('DecisionTree'),
    'rf': create_model('RandomForest'),
    'svm': create_model('SVM'),
    'gb': create_model('GradientBoosting'),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = (model.predict(X_test) == y_test).mean()
    print(f"{name}: {score:.4f}")
```

### Pattern 5: Check Model State

```python
model = create_model('RandomForest')

print(f"Fitted: {model.is_fitted()}")           # False
print(f"Config: {model.get_config()}")

model.fit(X_train, y_train)

print(f"Fitted: {model.is_fitted()}")           # True
print(f"Config: {model.get_config()}")          # Has feature count
```

---

## Integration

### With Data Pipeline

```python
from preprocessing import prepare_dataset
from models import create_model

# 1. Load and preprocess
X_train, X_test, y_train, y_test, metadata = prepare_dataset(
    'iris',
    test_size=0.3,
    scaling_method='standard'
)

print(f"Features: {metadata['feature_names']}")
print(f"Classes: {metadata['n_target_classes']}")

# 2. Train model
model = create_model('RandomForest', n_estimators=100)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
score = (y_pred == y_test).mean()
```

### With Validation Framework

```python
from validation import CrossValidator
from models import create_model

# 1. Load data
X_train, _, y_train, _, _ = prepare_dataset('iris', test_size=0.3)

# 2. Create validator
validator = CrossValidator(n_splits=5, n_repetitions=3)

# 3. Validate single model
model = create_model('RandomForest', n_estimators=100)
results = validator.validate(
    X_train, y_train, model,
    model_name='RandomForest',
    dataset_name='iris'
)

# 4. Get summary
summary = results.get_summary()
print(f"Accuracy: {summary['accuracy_mean']:.4f}")
```

### Fair Model Comparison

```python
from validation import CrossValidator
from models import create_model

models = {
    'LogisticRegression': create_model('LogisticRegression'),
    'RandomForest': create_model('RandomForest'),
    'GradientBoosting': create_model('GradientBoosting'),
}

validator = CrossValidator()
results = validator.validate_multiple(X, y, models, 'iris')

for name, res in results.items():
    summary = res.get_summary()
    print(f"{name}: {summary['accuracy_mean']:.4f}")
```

---

## Reproducibility

### Achieving Reproducibility

All randomness in models is controlled by `random_state`:

```python
# Same configuration = same results
model1 = create_model('RandomForest', n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

model2 = create_model('RandomForest', n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

assert np.array_equal(y_pred1, y_pred2)  # Always True
```

### Factors Affecting Reproducibility

| Factor | Control | Impact |
|--------|---------|--------|
| Model random_state | Set to 42 | Initialization, splits |
| Training data order | Fixed after load | Feature computation |
| sklearn version | Pin version | Some algorithms |
| Data preprocessing | Fit on train only | Feature values |
| Model parameters | Fixed in config | Model structure |

---

## Extensions

### Adding a New Model

```python
from models.base import BaseModel
from models import register_model

class NaiveBayesModel(BaseModel):
    def __init__(self, random_state=42, verbose=False):
        super().__init__(
            name='NaiveBayes',
            random_state=random_state,
            verbose=verbose
        )
    
    def _build_model(self):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

# Register
register_model('NaiveBayes', NaiveBayesModel)

# Use
model = create_model('NaiveBayes')
```

### Creating a Wrapper for Custom Model

```python
class CustomEnsembleModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__(
            name='CustomEnsemble',
            random_state=random_state
        )
    
    def _build_model(self):
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        lr = LogisticRegression(random_state=self.random_state)
        rf = RandomForestClassifier(random_state=self.random_state)
        
        return VotingClassifier(
            estimators=[('lr', lr), ('rf', rf)],
            voting='soft'
        )

register_model('CustomEnsemble', CustomEnsembleModel)
```

---

## Design Principles

### ✓ Sklearn Compatibility

All models follow sklearn's interface for seamless integration:
- `fit(X, y)` for training
- `predict(X)` for predictions
- `predict_proba(X)` for probabilities (optional)

### ✓ Reproducibility

Fixed `random_state` ensures deterministic results across runs.

### ✓ Simplicity

Minimal wrapper overhead, keeping models close to sklearn.

### ✓ Extensibility

BaseModel abstract class enables easy addition of new models.

### ✓ Configuration

All model parameters configurable at instantiation.

---

## Summary

The model execution engine provides:

✓ **5 Baseline Models**: Covering range of approaches  
✓ **Unified Interface**: Same API for all models  
✓ **Registry System**: Discovery and instantiation  
✓ **Reproducibility**: Fixed random states  
✓ **Extensibility**: Easy to add new models  
✓ **Integration**: Works with validation framework  

Ready for benchmarking, comparisons, and experiments.
