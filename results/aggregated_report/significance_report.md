# Significance Report

## Test Assumptions
- Friedman: paired comparisons across independent datasets.
- Wilcoxon Signed-Rank: paired sample differences per model pair across independent datasets.
- Nemenyi: post-hoc multiple comparison after Friedman.

## Statistical Outputs
Friedman statistic: 3.42857
Friedman p-value: 0.180092
Friedman interpretation: no significant differences detected
Nemenyi critical difference: 1.25276

## Conclusions
Use the comparison_table.csv for pairwise significance and ranking_table.csv for publication-ready ordering.

## Pairwise Wilcoxon Summary
- LogisticRegression vs RandomForest: raw p=0.109375, adj p=0.21875, significant=False
- LogisticRegression vs SVM: raw p=0.375, adj p=0.375, significant=False
- RandomForest vs SVM: raw p=0.03125, adj p=0.09375, significant=False

## Ranking Table
- RandomForest: rank=1
- SVM: rank=2
- LogisticRegression: rank=3
