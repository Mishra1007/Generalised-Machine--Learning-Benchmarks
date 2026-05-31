# Experiments

This directory orchestrates benchmarking experiments and workflows.

## Purpose

- Define benchmark scenarios
- Execute model training and evaluation
- Manage experimental state and results
- Support cross-validation and nested evaluation

## Structure

```
experiments/
├── benchmark.py        # Benchmark orchestration
├── workflows.py        # Experimental workflows
├── runner.py           # Experiment execution engine
└── tracking.py         # Experiment state and metadata
```

## Experiment Workflow

1. Load configuration
2. Prepare datasets and preprocessing
3. Initialize models
4. Execute training/evaluation (with CV if specified)
5. Collect metrics
6. Log results

## Usage

```python
from experiments.benchmark import Benchmark

benchmark = Benchmark("configs/experiment.yaml")
results = benchmark.run()
```

## Result Format

```python
{
    'model': 'model_name',
    'dataset': 'dataset_name',
    'metrics': {'accuracy': 0.95, 'f1': 0.92, ...},
    'folds': [...]  # if cross-validation used
}
```
