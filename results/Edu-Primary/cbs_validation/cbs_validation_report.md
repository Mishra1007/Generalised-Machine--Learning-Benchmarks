# CBS Validation Report

Dataset artifact: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary`

## Methodology

CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.

## Baseline Ranking

- Rank 1.0: DecisionTree CBS=0.8937
- Rank 2.0: RandomForest CBS=0.8207
- Rank 3.0: GradientBoosting CBS=0.8000
- Rank 4.0: SVM CBS=0.4154
- Rank 5.0: LogisticRegression CBS=0.4096

## Sensitivity And Robustness

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 0.9857 | 0.9714 | 0.1429 | 1.0000 |
| 20% | 0.9786 | 0.9571 | 0.2143 | 1.0000 |

## Metric Dominance

Highest CBS variance contribution: `f1` (63.00%).

| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| f1 | 0.3000 | 0.1702 | 63.00 | 0.6000 | 0.0773 |
| stability_score | 0.1000 | 0.0600 | 8.64 | 0.5000 | 0.0385 |
| accuracy | 0.1000 | 0.0555 | 7.65 | 0.6000 | 0.0240 |
| recall | 0.1000 | 0.0555 | 7.65 | 0.6000 | 0.0240 |
| precision | 0.1000 | 0.0594 | 6.82 | 0.6000 | 0.0171 |
| time_score | 0.1000 | 0.0672 | 6.23 | 0.2000 | 0.0456 |
| roc_auc | 0.2000 | 0.2000 | 0.00 | nan | 0.0830 |

## Normalization Audit

No boundedness failures, constant-metric compression flags, or rank distortions were detected in the audited artifact.

## Monte Carlo Weight Analysis

Most frequent top model under sampled valid weights: `DecisionTree` (81.60%).

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| DecisionTree | 1.0 | 1.1965 | 0.4278 | 81.60% | 18.40% |
| RandomForest | 2.0 | 2.4105 | 0.4981 | 0.30% | 41.65% |
| GradientBoosting | 3.0 | 2.4165 | 0.8120 | 18.10% | 44.60% |
| SVM | 4.0 | 4.4825 | 0.5374 | 0.00% | 52.15% |
| LogisticRegression | 5.0 | 4.4940 | 0.5011 | 0.00% | 50.55% |

## Recommendations

- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.
- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.
- Report robustness artifacts alongside CBS rankings in publication material.
- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.

## Visualizations

- sensitivity_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\cbs_validation\plots\tornado_plot.png`
- weight_impact_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\cbs_validation\plots\weight_sensitivity_plot.png`
- ranking_stability_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\cbs_validation\plots\ranking_stability_plot.png`
- metric_contribution_plot: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\cbs_validation\plots\metric_contribution_plot.png`
- correlation_heatmap: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\cbs_validation\plots\correlation_heatmap.png`
- stability_distribution: `C:\Users\mshar\Projects\Generalised Machine learning Benchmarks\results\Edu-Primary\cbs_validation\plots\stability_distribution.png`
