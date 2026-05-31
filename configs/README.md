# Configurations

This directory contains experiment configuration files in YAML or JSON format.

## Purpose

Centralize all experiment parameters and settings:
- Dataset specifications
- Model hyperparameters
- Preprocessing pipelines
- Evaluation metrics
- Random seeds for reproducibility

## File Structure

Each configuration file should include:
```yaml
experiment:
  name: "experiment_name"
  description: "Detailed description"
  random_state: 42

datasets:
  - name: "dataset_name"
    source: "local|remote"
    test_size: 0.3

models:
  - name: "model_name"
    hyperparameters:
      param1: value1
      param2: value2

metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1"

preprocessing:
  scaler: "standard|minmax|robust"
  feature_selection: true
```

## Example Configuration

See `example_config.yaml` for a template.
