# CBS Validation Report

Phase 9 formal scientific validation of the Composite Benchmark Score (CBS).

## Scope

CBS was analyzed without changing:

- `metrics/cbs.py`
- CBS weights
- Normalization implementation
- Benchmark execution logic

Validation was run against the available artifact:

```text
results/demo_dataset_storage
```

The generated artifact bundle is stored at:

```text
results/demo_dataset_storage/cbs_validation/
```

## CBS Definition Audited

```text
CBS =
0.30 * F1
+ 0.20 * ROC-AUC
+ 0.10 * Accuracy
+ 0.10 * Precision
+ 0.10 * Recall
+ 0.10 * Time Score
+ 0.10 * Stability Score
```

All components are normalized to `[0, 1]` with higher values better before composition.

## Methodology

The validation layer recomputed CBS from normalized component metrics and performed:

- Sensitivity analysis under weight perturbations of +/-5%, +/-10%, and +/-20%.
- Weight robustness analysis using Spearman rank correlation, Kendall rank correlation, rank shifts, and pairwise ranking reversal counts.
- Metric dominance analysis using contribution shares, contribution variance, leave-one-out impact, and metric-CBS correlations.
- Normalization audit for boundedness, score compression, and rank distortion.
- Monte Carlo weight analysis using 5,000 valid random non-negative weight vectors that sum to 1.

## Baseline Ranking

| Rank | Model | CBS |
|---:|---|---:|
| 1 | RandomForest | 0.8538 |
| 2 | SVM | 0.2952 |
| 3 | LogisticRegression | 0.2000 |

## Sensitivity Analysis

Weight perturbations at all tested levels preserved the baseline ranking.

| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |
|---:|---:|---:|---:|---:|
| 5% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20% | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

Finding: the observed ranking is stable under local CBS weight perturbations for this artifact.

## Weight Robustness

Across +/-5%, +/-10%, and +/-20% single-metric weight perturbations:

- No rank reversals occurred.
- Spearman rank correlation remained 1.0.
- Kendall rank correlation remained 1.0.
- Maximum absolute rank shift was 0.

This indicates strong local robustness for the available three-model artifact. Broader publication claims should be rerun on each final benchmark dataset.

## Metric Dominance

The largest CBS variance contribution came from F1.

| Metric | Weight | Mean Contribution | Variance Contribution | Spearman With CBS | Leave-One-Out Impact |
|---|---:|---:|---:|---:|---:|
| F1 | 0.30 | 0.1429 | 48.32% | 1.0000 | 0.0685 |
| ROC-AUC | 0.20 | 0.0800 | 23.89% | 1.0000 | 0.0368 |
| Accuracy | 0.10 | 0.0400 | 5.97% | 1.0000 | 0.0163 |
| Precision | 0.10 | 0.0444 | 5.53% | 1.0000 | 0.0142 |
| Recall | 0.10 | 0.0444 | 5.53% | 1.0000 | 0.0142 |
| Stability Score | 0.10 | 0.0467 | 5.40% | -1.0000 | 0.0651 |
| Time Score | 0.10 | 0.0513 | 5.34% | -0.5000 | 0.0522 |

Findings:

- F1 is the dominant contributor to CBS variance in this artifact.
- ROC-AUC is the second-largest contributor.
- Time Score and Stability Score have meaningful leave-one-out impact despite lower variance contribution.
- Stability and Time correlations are negative in this artifact because the highest-performing model trades off against those components.

## Normalization Audit

Checks performed:

- Normalized component boundedness in `[0, 1]`.
- Unique-value preservation.
- Rank correlation between raw values and normalized values.
- Raw proxy CBS rank compared with normalized CBS rank.

Finding:

- Component boundedness passed.
- A `cbs_proxy` rank-distortion warning was detected: the raw proxy rank differs from normalized CBS rank.

Interpretation: per-dataset min-max normalization is behaving as designed, but it can change aggregate ranking relative to raw unnormalized weighted scores. This is expected for a relative score, but it should be disclosed in publication methodology.

## Monte Carlo Weight Analysis

5,000 valid random weight vectors were sampled from a Dirichlet distribution centered on the canonical CBS weights.

| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |
|---|---:|---:|---:|---:|---:|
| RandomForest | 1 | 1.0016 | 0.0400 | 99.84% | 0.16% |
| SVM | 2 | 2.1776 | 0.3822 | 0.00% | 17.76% |
| LogisticRegression | 3 | 2.8208 | 0.3877 | 0.16% | 17.76% |

Findings:

- RandomForest remains the top model under nearly all sampled valid weight vectors.
- The main instability is between SVM and LogisticRegression, not the top model.
- Mean Spearman correlation vs baseline across Monte Carlo samples was 0.9096.
- Mean Kendall correlation vs baseline across Monte Carlo samples was 0.8805.
- Mean pairwise reversals per sample were 0.1792.

## Visualizations

Generated plots:

- `results/demo_dataset_storage/cbs_validation/plots/tornado_plot.png`
- `results/demo_dataset_storage/cbs_validation/plots/weight_sensitivity_plot.png`
- `results/demo_dataset_storage/cbs_validation/plots/ranking_stability_plot.png`
- `results/demo_dataset_storage/cbs_validation/plots/metric_contribution_plot.png`
- `results/demo_dataset_storage/cbs_validation/plots/correlation_heatmap.png`
- `results/demo_dataset_storage/cbs_validation/plots/stability_distribution.png`

Generated CSV artifacts:

- `results/demo_dataset_storage/cbs_validation/sensitivity_analysis.csv`
- `results/demo_dataset_storage/cbs_validation/weight_robustness.csv`
- `results/demo_dataset_storage/cbs_validation/metric_influence.csv`
- `results/demo_dataset_storage/cbs_validation/ranking_stability.csv`
- `results/demo_dataset_storage/cbs_validation/monte_carlo_weight_samples.csv`
- `results/demo_dataset_storage/cbs_validation/normalization_validation.csv`
- `results/demo_dataset_storage/cbs_validation/correlation_analysis.csv`

## Recommendations

- Do not change CBS weights based on this analysis alone.
- Report CBS together with robustness diagnostics in publication materials.
- Treat CBS as dataset-relative because min-max normalization can alter rankings relative to raw weighted scores.
- Re-run `run_cbs_validation` on every final benchmark dataset before publication.
- Flag any dataset where F1 or ROC-AUC contributes an overwhelming share of CBS variance, or where Monte Carlo top-model frequency is diffuse.

## Validation Command

```powershell
.\.venv311\Scripts\python.exe -c "from analysis.cbs_validation import run_cbs_validation; run_cbs_validation('results/demo_dataset_storage', mc_iterations=5000, random_state=42)"
```

Result: CBS validation artifacts and plots generated successfully.
