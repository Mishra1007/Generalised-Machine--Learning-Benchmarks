# Significance Report

## Test Assumptions
- Friedman: paired comparisons across the same datasets/folds.
- Wilcoxon Signed-Rank: paired sample differences per model pair.
- Nemenyi: post-hoc multiple comparison after Friedman.

## Statistical Outputs
Friedman statistic: None
Friedman p-value: None
Friedman interpretation: None
Nemenyi critical difference: None

## Conclusions
Use the comparison_table.csv for pairwise significance and ranking_table.csv for publication-ready ordering.

## Confidence Intervals
- LogisticRegression: mean=0.856, 95% CI=[0.841843, 0.870157]
- RandomForest: mean=0.914, 95% CI=[0.899843, 0.928157]
- SVM: mean=0.874, 95% CI=[0.859843, 0.888157]

## Pairwise Wilcoxon Summary
- LogisticRegression vs RandomForest: raw p=0.00435913, adj p=0.00871826, significant=True
- LogisticRegression vs SVM: raw p=0.145557, adj p=0.145557, significant=False
- RandomForest vs SVM: raw p=0, adj p=0, significant=True

## Ranking Table
- RandomForest: rank=1
- SVM: rank=2
- LogisticRegression: rank=3
