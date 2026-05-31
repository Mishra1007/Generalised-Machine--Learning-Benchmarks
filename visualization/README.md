# Visualization

This directory contains plotting and visual analysis tools.

## Purpose

- Generate performance comparison charts
- Create visual summaries of results
- Support publication-quality figures
- Enable exploratory visual analysis

## Structure

```
visualization/
├── plotters.py         # Main plotting functions
├── styles.py           # Matplotlib styles and themes
├── comparisons.py      # Comparative visualizations
└── reports.py          # Report generation utilities
```

## Supported Visualizations

- Model comparison bar charts
- Performance by dataset heatmaps
- Metric evolution plots
- ROC/PR curves
- Confusion matrices
- Box plots for cross-validation results

## Usage

```python
from visualization.plotters import plot_model_comparison

plot_model_comparison(results_df, save_path='results/comparison.png')
```

## Output

All visualizations saved to `results/` directory with:
- PNG format for publication
- High DPI for clarity
- Legend and annotations for clarity
