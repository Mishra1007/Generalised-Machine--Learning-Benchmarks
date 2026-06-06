# Financial Dataset Optimization Report

This document evaluates the five proposed optimization candidates for the Financial dataset revalidation benchmark and provides a recommended execution strategy.

---

## 1. Optimization Candidates Evaluation

### Candidate A: Reduce CV folds from 5 → 3
* **Runtime Reduction**: Saves **6,611.42 seconds** (~110 minutes, **40.0% reduction** in validation time).
* **Scientific Impact**: High. Training set size is reduced from 80% to 66.7% of the dataset, which increases the variance of validation metrics and may degrade absolute model performance scores.
* **Risk Level**: Medium.
* **Comparability Preserved**: **No**. Comparing 3-fold CV metrics with 5-fold CV benchmarks from other datasets is statistically invalid.

### Candidate B: Enable/verify `n_jobs=-1` where supported
* **Runtime Reduction**: **0 seconds** (already enabled by default in RandomForest wrapper; other slow models do not support it natively).
* **Scientific Impact**: None. Model predictions and performance metrics remain identical.
* **Risk Level**: Low.
* **Comparability Preserved**: **Yes**.

### Candidate C: Replace kernel SVM with LinearSVC
* **Runtime Reduction**: Saves **~12,450 seconds** (~207.5 minutes, **75.2% reduction** in total runtime).
* **Scientific Impact**: High. Substitutes a non-linear RBF kernel model with a linear decision boundary classifier. Model capacity will drop significantly on complex relationships, skewing CBS rankings.
* **Risk Level**: High.
* **Comparability Preserved**: **No**. Substitutes model architecture.

### Candidate D: Replace kernel SVM with SGDClassifier(loss="hinge")
* **Runtime Reduction**: Saves **~12,500 seconds** (~208.3 minutes, **75.5% reduction** in total runtime).
* **Scientific Impact**: Critical. Fits a linear SVM using SGD. Does not natively support probability estimates (`predict_proba`), which breaks metric computations such as ROC-AUC.
* **Risk Level**: Very High.
* **Comparability Preserved**: **No**. Substitutes model architecture and breaks standard evaluation metrics.

### Candidate E: Run CBS validation after benchmark completion only
* **Runtime Reduction**: **0 seconds** (already runs post-hoc; consumes only 12.08 seconds or 0.07% of runtime).
* **Scientific Impact**: None.
* **Risk Level**: None.
* **Comparability Preserved**: **Yes**.

---

## 2. Summary Comparison Matrix

| Candidate | Est. Time Saved | Sci. Validity | Risk | Comparability? | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A: 3-Fold CV** | ~110m (40%) | Compromised | Medium | No | Reject |
| **B: `n_jobs=-1`**| 0m (0%) | Preserved | Low | Yes | Accept (Verify) |
| **C: LinearSVC** | ~207.5m (75.2%)| Compromised | High | No | Reject |
| **D: SGDClassifier**| ~208.3m (75.5%)| Broken (No Prob)| V. High | No | Reject |
| **E: Post-CBS** | 0m (0%) | Preserved | None | Yes | Accept (Already Active) |

---

## 3. Recommendation

**Proceed with Staged Revalidation**

### Rationale:
1. **Comparability and Validity**: Scientific rigor requires keeping the 5-fold CV × 3 repetitions protocol and the RBF kernel SVM model. Altering these (Candidates A, C, and D) would prevent direct comparison with baseline benchmarks and jeopardize scientific validity.
2. **Speed & Verification Flow**: To prevent long-running hanging behavior and verify pipeline sanity:
   - **Stage 1 (Fast verification)**: Run only the fast models (`DecisionTree`, `RandomForest`, `LogisticRegression`) first. These take **under 5 minutes** total to run.
   - **Stage 2 (Heavy compute)**: Execute `GradientBoosting` and `SVM` models sequentially or in the background once pipeline verification is complete.
   - **Execution Ordering**: Follow the execution manifest recommendation, placing the Financial dataset last.
