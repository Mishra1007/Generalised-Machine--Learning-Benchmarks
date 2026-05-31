# Analysis

This directory performs statistical analysis and comparative reporting.

## Purpose

- Conduct statistical significance testing
- Generate comparative performance reports
- Compute summary statistics
- Identify performance trends and patterns

## Structure

```
analysis/
├── stats.py            # Statistical tests (t-test, ANOVA, etc.)
├── reporters.py        # Report generation
├── comparisons.py      # Pairwise model comparisons
└── utils.py            # Analysis utilities
```

## Analysis Types

- Cross-dataset performance aggregation
- Statistical significance testing (t-tests, Mann-Whitney U)
- Confidence interval computation
- Ranking and leaderboards
- Performance aggregation and summaries

## Usage

```python
from analysis.stats import compare_models

report = compare_models(results_df)
print(report)
```

## Outputs

- Summary tables (CSV, LaTeX)
- Statistical test results
- Ranking tables
- Performance summaries
