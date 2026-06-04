# CBS Validation Report

Dataset artifact: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart`

## Methodology

CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.

## Baseline Ranking

- Rank 1.0: RandomForest CBS=0.9000
- Rank 2.0: GradientBoosting CBS=0.8386
- Rank 3.0: DecisionTree CBS=0.7509
- Rank 4.0: SVM CBS=0.4423
- Rank 5.0: LogisticRegression CBS=0.1027

## Sensitivity And Robustness

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Metric Dominance

Highest CBS variance contribution: `f1` (50.69%).

| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| f1 | 0.3000 | 0.1901 | 50.69 | 0.9000 | 0.0361 |
| roc_auc | 0.2000 | 0.1252 | 19.34 | 0.9000 | 0.0325 |
| stability_score | 0.1000 | 0.0379 | 7.54 | 0.9000 | 0.0298 |
| precision | 0.1000 | 0.0629 | 5.65 | 0.9000 | 0.0095 |
| recall | 0.1000 | 0.0633 | 5.64 | 0.9000 | 0.0094 |
| accuracy | 0.1000 | 0.0633 | 5.64 | 0.9000 | 0.0094 |
| time_score | 0.1000 | 0.0641 | 5.50 | -0.7000 | 0.0599 |

## Normalization Audit

No boundedness failures, constant-metric compression flags, or rank distortions were detected in the audited artifact.

## Monte Carlo Weight Analysis

Most frequent top model under sampled valid weights: `RandomForest` (89.85%).

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| RandomForest | 1.0 | 1.1630 | 0.5095 | 89.85% | 10.15% |
| GradientBoosting | 2.0 | 2.0120 | 0.2737 | 3.15% | 7.50% |
| DecisionTree | 3.0 | 2.8250 | 0.5334 | 7.00% | 10.50% |
| SVM | 4.0 | 4.0000 | 0.0000 | 0.00% | 0.00% |
| LogisticRegression | 5.0 | 5.0000 | 0.0000 | 0.00% | 0.00% |

## Recommendations

- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.
- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.
- Report robustness artifacts alongside CBS rankings in publication material.
- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.

## Visualizations

- sensitivity_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\cbs_validation\plots\tornado_plot.png`
- weight_impact_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\cbs_validation\plots\weight_sensitivity_plot.png`
- ranking_stability_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\cbs_validation\plots\ranking_stability_plot.png`
- metric_contribution_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\cbs_validation\plots\metric_contribution_plot.png`
- correlation_heatmap: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\cbs_validation\plots\correlation_heatmap.png`
- stability_distribution: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\heart\cbs_validation\plots\stability_distribution.png`
