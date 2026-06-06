# Financial Dataset Runtime Profile

This document details the runtime profiling results for the Financial dataset benchmark. Measurements are obtained from existing benchmark execution logs and metrics files (`results/Financial/metrics.json` and system execution measurements).

## 1. Executive Summary

* **Total Benchmark Runtime**: **16,546.80 seconds** (~275.78 minutes / **4.60 hours**)
* **Component Breakdown**:
  * **Preprocessing Stage**: 6.17 seconds (0.04% of total)
  * **Model Validation Stage (5-fold CV × 3 Repetitions)**: 16,528.55 seconds (99.89% of total)
  * **CBS Validation Stage**: 12.08 seconds (0.07% of total)

---

## 2. Detailed Component Profile

### Preprocessing Stage (Upfront)
* **Total Runtime**: **6.1694 seconds**
* **Breakdown**:
  * **Raw CSV Loading**: 0.2852 seconds (4.6%)
  * **Dataset Sanitization / Remediation**: 1.4600 seconds (23.7%)
  * **Train/Test Split & Preprocessing Pipeline Fit/Transform**: 4.4243 seconds (71.7%)

### Model Validation Stage (Cross-Validation)
* **Total Runtime**: **16,528.55 seconds** (15 folds evaluated)
* **Runtime per Model**:
  * **SVM**: 12,503.91 seconds (75.57% of total benchmark runtime)
  * **GradientBoosting**: 3,734.51 seconds (22.57% of total benchmark runtime)
  * **LogisticRegression**: 216.99 seconds (1.31% of total benchmark runtime)
  * **RandomForest**: 42.43 seconds (0.26% of total benchmark runtime)
  * **DecisionTree**: 30.71 seconds (0.19% of total benchmark runtime)

### Runtime per Fold (Mean across Repetitions)
The table below shows the mean runtime (training + evaluation in seconds) per fold for each model:

| Fold | DecisionTree | GradientBoosting | LogisticRegression | RandomForest | SVM | Fold Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 2.15s | 257.62s | 14.49s | 3.07s | 939.27s | **1216.60s** |
| **Fold 1** | 2.04s | 269.50s | 14.85s | 2.93s | 721.65s | **1010.97s** |
| **Fold 2** | 2.03s | 247.78s | 14.44s | 3.11s | 778.40s | **1045.76s** |
| **Fold 3** | 1.90s | 232.43s | 14.41s | 2.46s | 844.09s | **1095.29s** |
| **Fold 4** | 2.11s | 237.50s | 14.14s | 2.57s | 884.56s | **1140.88s** |

### CBS Validation Stage
* **Total Runtime**: **12.0849 seconds**
* **Breakdown**: Recomputation of CBS scores, weight sensitivity analysis (Spearman/Kendall rank correlations), metric dominance auditing, and 2,000-iteration Monte Carlo weight simulations.
