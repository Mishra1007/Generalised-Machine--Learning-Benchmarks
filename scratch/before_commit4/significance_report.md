# Significance Report

## Test Assumptions
- Friedman: paired comparisons across the same datasets/folds.
- Wilcoxon Signed-Rank: paired sample differences per model pair.
- Nemenyi: post-hoc multiple comparison after Friedman.

## Statistical Outputs
Friedman statistic: 8.4
Friedman p-value: 0.0149956
Friedman interpretation: significant differences detected
Nemenyi critical difference: 1.48229

## Conclusions
Use the comparison_table.csv for pairwise significance and ranking_table.csv for publication-ready ordering.

## Confidence Intervals
- LogisticRegression: mean=0.856, 95% CI=[0.841843, 0.870157]
- RandomForest: mean=0.914, 95% CI=[0.899843, 0.928157]
- SVM: mean=0.874, 95% CI=[0.859843, 0.888157]

## Pairwise Wilcoxon Summary
- LogisticRegression vs RandomForest: raw p=0.0625, adj p=0.1875, significant=False
- LogisticRegression vs SVM: raw p=0.125, adj p=0.1875, significant=False
- RandomForest vs SVM: raw p=0.0625, adj p=0.1875, significant=False

## Ranking Table
- RandomForest: rank=1
- SVM: rank=2
- LogisticRegression: rank=3
