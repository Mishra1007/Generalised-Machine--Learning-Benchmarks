# Results

This directory stores all experiment outputs, metrics, and generated reports.

## Purpose

- Store experiment results and scores
- Archive visualizations and plots
- Maintain result metadata and provenance
- Enable result retrieval and comparison

## Structure

```
results/
├── metrics/            # Raw metric outputs (JSON/CSV)
├── plots/              # Generated visualizations
├── reports/            # Summary reports (HTML/PDF)
├── archives/           # Historical results
└── manifest.json       # Results index and metadata
```

## File Naming Convention

```
{timestamp}_{experiment_name}_{model}_{dataset}.json
{timestamp}_{visualization_type}_{experiment_name}.png
```

## Metadata

Each result file should include:
- Experiment name and configuration
- Dataset name and version
- Model and hyperparameters
- Execution timestamp
- Random seed
- Metrics and scores
- Data split information
