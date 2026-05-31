# CBS Validation Report

## Summary
- CBS was evaluated without changing the score definition.
- Analysis covers perturbation sensitivity, weight robustness, metric influence, correlation structure, normalization behavior, and Monte Carlo ranking stability.

## Sensitivity Analysis
The most sensitive CBS component is f1 under fold_variability perturbations, with mean absolute CBS change 0.0040.

## Weight Robustness
Rank stability across perturbations: 1.0000 average Spearman correlation.

## Metric Dominance
- Highest mean leave-one-out impact: f1 (0.0685).

## Correlation Analysis
- Pearson and Spearman correlations were computed between CBS and each component metric.
- Potentially redundant pairs (|Pearson r| >= 0.90): 20

## Normalization Validation
- Normalization was checked for score compression and rank distortions.

## Monte Carlo Stability
- Mean CBS variance across models: 0.0079; top-1 retention: 1.0000.

## Acceptance Criteria
- Robustness: assessed via perturbation and weight sensitivity.
- Interpretability: assessed via contributions and dominance analysis.
- Stability: assessed via Monte Carlo ranking consistency and confidence bounds.
- Methodological soundness: assessed via correlations, normalization checks, and ranking distortion estimates.
