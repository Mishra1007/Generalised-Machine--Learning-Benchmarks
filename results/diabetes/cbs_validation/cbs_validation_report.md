# CBS Validation Report

Dataset artifact: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\diabetes`

## Methodology

CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.

## Baseline Ranking

- Rank 1.0: LogisticRegression CBS=0.9989
- Rank 2.0: SVM CBS=0.8100
- Rank 3.0: RandomForest CBS=0.7589
- Rank 4.0: GradientBoosting CBS=0.6807
- Rank 5.0: DecisionTree CBS=0.1000

## Sensitivity And Robustness

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Metric Dominance

Highest CBS variance contribution: `f1` (46.07%).

| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| f1 | 0.3000 | 0.2045 | 46.07 | 0.9000 | 0.0223 |
| roc_auc | 0.2000 | 0.1530 | 24.70 | 0.9000 | 0.0339 |
| stability_score | 0.1000 | 0.0439 | 7.19 | 0.9000 | 0.0307 |
| time_score | 0.1000 | 0.0650 | 6.76 | -0.1000 | 0.0463 |
| accuracy | 0.1000 | 0.0681 | 5.12 | 1.0000 | 0.0057 |
| recall | 0.1000 | 0.0681 | 5.12 | 1.0000 | 0.0057 |
| precision | 0.1000 | 0.0671 | 5.03 | 1.0000 | 0.0046 |

## Normalization Audit

No boundedness failures, constant-metric compression flags, or rank distortions were detected in the audited artifact.

## Monte Carlo Weight Analysis

Most frequent top model under sampled valid weights: `LogisticRegression` (100.00%).

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 1.0 | 1.0000 | 0.0000 | 100.00% | 0.00% |
| SVM | 2.0 | 2.2220 | 0.4157 | 0.00% | 22.20% |
| RandomForest | 3.0 | 2.8505 | 0.5218 | 0.00% | 29.45% |
| GradientBoosting | 4.0 | 3.9275 | 0.2594 | 0.00% | 7.25% |
| DecisionTree | 5.0 | 5.0000 | 0.0000 | 0.00% | 0.00% |

## Recommendations

- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.
- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.
- Report robustness artifacts alongside CBS rankings in publication material.
- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.

## Visualizations

- sensitivity_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\diabetes\cbs_validation\plots\tornado_plot.png`
- weight_impact_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\diabetes\cbs_validation\plots\weight_sensitivity_plot.png`
- ranking_stability_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\diabetes\cbs_validation\plots\ranking_stability_plot.png`
- metric_contribution_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\diabetes\cbs_validation\plots\metric_contribution_plot.png`
- correlation_heatmap: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\diabetes\cbs_validation\plots\correlation_heatmap.png`
- stability_distribution: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\diabetes\cbs_validation\plots\stability_distribution.png`
