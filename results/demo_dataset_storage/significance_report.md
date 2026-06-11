# Significance Report

## Test Assumptions
- Corrected Resampled t-test (Nadeau-Bengio): protects against inflated Type I error due to overlapping training sets in repeated k-fold cross-validation.
- Holm-Bonferroni: controls Family-Wise Error Rate (FWER) across multiple pairwise comparisons.
- Note: Friedman and Nemenyi tests are strictly omitted for single-dataset benchmarks because cross-validation folds violate the independence assumptions required by these tests.

## Conclusions
Use the comparison_table.csv for pairwise significance and ranking_table.csv for publication-ready ordering.

## Confidence Intervals
- LogisticRegression: mean=0.856, 95% CI=[0.841843, 0.870157]
- RandomForest: mean=0.914, 95% CI=[0.899843, 0.928157]
- SVM: mean=0.874, 95% CI=[0.859843, 0.888157]

## Pairwise Corrected t-test Summary
- LogisticRegression vs RandomForest: raw p=0.00435913, adj p=0.00871826, significant=True
- LogisticRegression vs SVM: raw p=0.145557, adj p=0.145557, significant=False
- RandomForest vs SVM: raw p=0, adj p=0, significant=True

## Ranking Table
- RandomForest: rank=1
- SVM: rank=2
- LogisticRegression: rank=3
