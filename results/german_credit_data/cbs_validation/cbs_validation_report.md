# CBS Validation Report

Dataset artifact: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\german_credit_data`

## Methodology

CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.

## Baseline Ranking

- Rank 1.0: LogisticRegression CBS=0.9612
- Rank 2.0: GradientBoosting CBS=0.8494
- Rank 3.0: RandomForest CBS=0.6758
- Rank 4.0: SVM CBS=0.6154
- Rank 5.0: DecisionTree CBS=0.1444

## Sensitivity And Robustness

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Metric Dominance

Highest CBS variance contribution: `f1` (49.00%).

| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| f1 | 0.3000 | 0.1770 | 49.00 | 0.9000 | 0.0538 |
| roc_auc | 0.2000 | 0.1525 | 23.57 | 0.7000 | 0.0428 |
| time_score | 0.1000 | 0.0628 | 6.05 | -0.3000 | 0.0493 |
| recall | 0.1000 | 0.0746 | 5.75 | 0.9000 | 0.0172 |
| accuracy | 0.1000 | 0.0746 | 5.75 | 0.9000 | 0.0172 |
| precision | 0.1000 | 0.0679 | 5.39 | 0.9000 | 0.0103 |
| stability_score | 0.1000 | 0.0398 | 4.50 | 0.1000 | 0.0429 |

## Normalization Audit

No boundedness failures, constant-metric compression flags, or rank distortions were detected in the audited artifact.

## Monte Carlo Weight Analysis

Most frequent top model under sampled valid weights: `LogisticRegression` (98.30%).

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 1.0 | 1.0170 | 0.1293 | 98.30% | 1.70% |
| GradientBoosting | 2.0 | 1.9845 | 0.1352 | 1.70% | 1.85% |
| RandomForest | 3.0 | 3.2105 | 0.4126 | 0.00% | 20.85% |
| SVM | 4.0 | 3.7900 | 0.4111 | 0.00% | 20.85% |
| DecisionTree | 5.0 | 4.9980 | 0.0447 | 0.00% | 0.20% |

## Recommendations

- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.
- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.
- Report robustness artifacts alongside CBS rankings in publication material.
- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.

## Visualizations

- sensitivity_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\german_credit_data\cbs_validation\plots\tornado_plot.png`
- weight_impact_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\german_credit_data\cbs_validation\plots\weight_sensitivity_plot.png`
- ranking_stability_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\german_credit_data\cbs_validation\plots\ranking_stability_plot.png`
- metric_contribution_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\german_credit_data\cbs_validation\plots\metric_contribution_plot.png`
- correlation_heatmap: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\german_credit_data\cbs_validation\plots\correlation_heatmap.png`
- stability_distribution: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\german_credit_data\cbs_validation\plots\stability_distribution.png`
