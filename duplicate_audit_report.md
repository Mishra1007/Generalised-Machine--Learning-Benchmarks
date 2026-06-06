# Heart Dataset Duplication Audit Report

This report presents an independent verification of record duplication and data leakage within the Heart dataset (`datasets/heart.csv`).

---

## 1. Quantitative Duplication Evidence

* **Dataset Path**: `datasets/heart.csv`
* **Total Rows**: `1,025`
* **Exact Duplicate Rows**: `723`
* **Unique Rows**: `302` (Original Cleveland dataset has 303 unique samples)
* **Percentage of Dataset Duplicated**: **70.54%**

---

## 2. Impact on Train/Test Splits and Cross-Validation Folds

To prove whether duplicate records leak between splits, we simulated a standard 70/30 train/test split and a 5-fold cross-validation setup:

### A. Standard 70/30 Train/Test Split
* **Training Set Size**: `717` samples
* **Test Set Size**: `308` samples
* **Overlapping Test Rows**: **290 out of 308 samples (94.16%)** in the test set exist as exact duplicates in the training set.
* **Unique Overlapping Rows**: 208 unique patient records are simultaneously present in both train and test sets.

### B. 5-Fold Cross-Validation Fold Contamination
Standard `KFold` cross-validation (with shuffle=True, random_state=42) was run on the dataset:

* **Fold 1 Leakage**: **97.07%** of test samples are duplicates of training samples.
* **Fold 2 Leakage**: **98.05%** of test samples are duplicates of training samples.
* **Fold 3 Leakage**: **98.54%** of test samples are duplicates of training samples.
* **Fold 4 Leakage**: **95.61%** of test samples are duplicates of training samples.
* **Fold 5 Leakage**: **98.54%** of test samples are duplicates of training samples.
* **Mean Fold Contamination / Leakage**: **97.56%**

### Scientific Conclusion
Because 97.56% of the validation samples are already seen by the model during training, the cross-validation error rate is heavily biased toward training error. The benchmark results on the `heart` dataset do not represent generalization capability but rather near-perfect training set memorization, violating the core assumptions of machine learning evaluation.

---

## 3. Proposed Remediation

We propose the following remediation:
1. **Deduplication**: Drop all duplicate rows during dataset loading in `DatasetLoader.load_csv`, keeping only the first occurrence of each unique row.
2. **Metadata Update**: Update the dataset size and feature count metadata dynamically to reflect the 302 unique samples.
3. **Execution**: Deduplicate before splits are generated in `prepare_dataset` or `ExperimentExecutor.run`.
