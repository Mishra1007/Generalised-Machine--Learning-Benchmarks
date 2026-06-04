# CBS Validation Report

Dataset artifact: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-xApi`

## Methodology

CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.

## Baseline Ranking

- Rank 1.0: RandomForest CBS=0.9497
- Rank 2.0: GradientBoosting CBS=0.8451
- Rank 3.0: LogisticRegression CBS=0.7901
- Rank 4.0: SVM CBS=0.6870
- Rank 5.0: DecisionTree CBS=0.1000

## Sensitivity And Robustness

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Metric Dominance

Highest CBS variance contribution: `f1` (50.61%).

| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| f1 | 0.3000 | 0.2017 | 50.61 | 1.0000 | 0.0353 |
| roc_auc | 0.2000 | 0.1455 | 22.11 | 0.9000 | 0.0233 |
| time_score | 0.1000 | 0.0681 | 6.14 | -0.9000 | 0.0583 |
| accuracy | 0.1000 | 0.0671 | 5.63 | 0.9000 | 0.0094 |
| recall | 0.1000 | 0.0671 | 5.63 | 0.9000 | 0.0094 |
| precision | 0.1000 | 0.0658 | 5.34 | 1.0000 | 0.0088 |
| stability_score | 0.1000 | 0.0589 | 4.54 | 0.9000 | 0.0117 |

## Normalization Audit

No boundedness failures, constant-metric compression flags, or rank distortions were detected in the audited artifact.

## Monte Carlo Weight Analysis

Most frequent top model under sampled valid weights: `RandomForest` (99.35%).

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| RandomForest | 1.0 | 1.0065 | 0.0804 | 99.35% | 0.65% |
| GradientBoosting | 2.0 | 2.2780 | 0.5735 | 0.00% | 21.40% |
| LogisticRegression | 3.0 | 2.7805 | 0.4318 | 0.65% | 21.50% |
| SVM | 4.0 | 3.9350 | 0.2466 | 0.00% | 6.50% |
| DecisionTree | 5.0 | 5.0000 | 0.0000 | 0.00% | 0.00% |

## Recommendations

- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.
- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.
- Report robustness artifacts alongside CBS rankings in publication material.
- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.

## Visualizations

- sensitivity_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-xApi\cbs_validation\plots\tornado_plot.png`
- weight_impact_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-xApi\cbs_validation\plots\weight_sensitivity_plot.png`
- ranking_stability_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-xApi\cbs_validation\plots\ranking_stability_plot.png`
- metric_contribution_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-xApi\cbs_validation\plots\metric_contribution_plot.png`
- correlation_heatmap: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-xApi\cbs_validation\plots\correlation_heatmap.png`
- stability_distribution: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-xApi\cbs_validation\plots\stability_distribution.png`
