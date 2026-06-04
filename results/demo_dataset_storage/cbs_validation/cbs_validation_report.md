# CBS Validation Report

Dataset artifact: `results/demo_dataset_storage`

## Methodology

CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.

## Baseline Ranking

- Rank 1.0: RandomForest CBS=0.8538
- Rank 2.0: SVM CBS=0.2952
- Rank 3.0: LogisticRegression CBS=0.2000

## Sensitivity And Robustness

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Metric Dominance

Highest CBS variance contribution: `f1` (48.32%).

| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| f1 | 0.3000 | 0.1429 | 48.32 | 1.0000 | 0.0685 |
| roc_auc | 0.2000 | 0.0800 | 23.89 | 1.0000 | 0.0368 |
| accuracy | 0.1000 | 0.0400 | 5.97 | 1.0000 | 0.0163 |
| precision | 0.1000 | 0.0444 | 5.53 | 1.0000 | 0.0142 |
| recall | 0.1000 | 0.0444 | 5.53 | 1.0000 | 0.0142 |
| stability_score | 0.1000 | 0.0467 | 5.40 | -1.0000 | 0.0651 |
| time_score | 0.1000 | 0.0513 | 5.34 | -0.5000 | 0.0522 |

## Normalization Audit

Normalization artifacts flagged:
- cbs_proxy: raw proxy rank differs from normalized CBS rank

## Monte Carlo Weight Analysis

Most frequent top model under sampled valid weights: `RandomForest` (99.84%).

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| RandomForest | 1.0 | 1.0016 | 0.0400 | 99.84% | 0.16% |
| SVM | 2.0 | 2.1776 | 0.3822 | 0.00% | 17.76% |
| LogisticRegression | 3.0 | 2.8208 | 0.3877 | 0.16% | 17.76% |

## Recommendations

- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.
- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.
- Report robustness artifacts alongside CBS rankings in publication material.
- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.

## Visualizations

- sensitivity_plot: `results\demo_dataset_storage\cbs_validation\plots\tornado_plot.png`
- weight_impact_plot: `results\demo_dataset_storage\cbs_validation\plots\weight_sensitivity_plot.png`
- ranking_stability_plot: `results\demo_dataset_storage\cbs_validation\plots\ranking_stability_plot.png`
- metric_contribution_plot: `results\demo_dataset_storage\cbs_validation\plots\metric_contribution_plot.png`
- correlation_heatmap: `results\demo_dataset_storage\cbs_validation\plots\correlation_heatmap.png`
- stability_distribution: `results\demo_dataset_storage\cbs_validation\plots\stability_distribution.png`
