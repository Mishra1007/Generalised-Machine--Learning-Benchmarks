# CBS Validation Report

Dataset artifact: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial`

## Methodology

CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.

## Baseline Ranking

- Rank 1.0: RandomForest CBS=0.9352
- Rank 2.0: GradientBoosting CBS=0.7769
- Rank 3.0: DecisionTree CBS=0.4225
- Rank 4.0: SVM CBS=0.3486
- Rank 5.0: LogisticRegression CBS=0.0614

## Sensitivity And Robustness

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Metric Dominance

Highest CBS variance contribution: `f1` (46.38%).

| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| f1 | 0.3000 | 0.1702 | 46.38 | 1.0000 | 0.0510 |
| roc_auc | 0.2000 | 0.1031 | 22.92 | 0.7000 | 0.0439 |
| accuracy | 0.1000 | 0.0453 | 6.46 | 0.9000 | 0.0150 |
| recall | 0.1000 | 0.0453 | 6.46 | 0.9000 | 0.0150 |
| time_score | 0.1000 | 0.0400 | 6.36 | 0.3000 | 0.0437 |
| precision | 0.1000 | 0.0587 | 6.32 | 1.0000 | 0.0192 |
| stability_score | 0.1000 | 0.0463 | 5.09 | 0.1000 | 0.0351 |

## Normalization Audit

No boundedness failures, constant-metric compression flags, or rank distortions were detected in the audited artifact.

## Monte Carlo Weight Analysis

Most frequent top model under sampled valid weights: `RandomForest` (100.00%).

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| RandomForest | 1.0 | 1.0000 | 0.0000 | 100.00% | 0.00% |
| GradientBoosting | 2.0 | 2.0085 | 0.0918 | 0.00% | 0.85% |
| DecisionTree | 3.0 | 3.2700 | 0.4628 | 0.00% | 28.70% |
| SVM | 4.0 | 3.7215 | 0.4484 | 0.00% | 27.85% |
| LogisticRegression | 5.0 | 5.0000 | 0.0000 | 0.00% | 0.00% |

## Recommendations

- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.
- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.
- Report robustness artifacts alongside CBS rankings in publication material.
- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.

## Visualizations

- sensitivity_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\cbs_validation\plots\tornado_plot.png`
- weight_impact_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\cbs_validation\plots\weight_sensitivity_plot.png`
- ranking_stability_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\cbs_validation\plots\ranking_stability_plot.png`
- metric_contribution_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\cbs_validation\plots\metric_contribution_plot.png`
- correlation_heatmap: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\cbs_validation\plots\correlation_heatmap.png`
- stability_distribution: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Financial\cbs_validation\plots\stability_distribution.png`
