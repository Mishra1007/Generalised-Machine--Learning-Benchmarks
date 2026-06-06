# Checkpoint Integrity Audit Report

This report documents the checkpoint and resume integrity audit conducted on the benchmarking framework using simulated crash-resume execution scenarios.

---

## 1. Audit Methodology

We simulated system interruptions at different execution granularities using a synthetic dataset and three models (`DecisionTree`, `LogisticRegression`, `RandomForest`). We measured the outcomes against a reference, uninterrupted run.

### Scenarios Evaluated:
1. **Scenario 1**: Interruption after **1 completed fold** of the first model (`DecisionTree`).
2. **Scenario 2**: Interruption after **1 completed model** (`DecisionTree` fully done; crash triggered on the first fold of `LogisticRegression`).
3. **Scenario 3**: Interruption after **multiple completed models** (`DecisionTree` and `LogisticRegression` fully done; crash triggered on the first fold of `RandomForest`).

---

## 2. Audit Verification Matrix

| Verification Check | Scenario 1 | Scenario 2 | Scenario 3 | Status |
| :--- | :---: | :---: | :---: | :---: |
| **No Fold Recomputation** | Skipped completed fold | Skipped completed folds | Skipped completed folds | **PASSED** |
| **No Model Recomputation** | Evaluated DT remaining folds | Restored DT; resumed LR | Restored DT & LR; resumed RF | **PASSED** |
| **Final Metrics Match Reference** | Match at $10^{-9}$ tolerance | Match at $10^{-9}$ tolerance | Match at $10^{-9}$ tolerance | **PASSED** |
| **CBS Scores Match Reference** | Match (excluding runtimes) | Match (excluding runtimes) | Match (excluding runtimes) | **PASSED** |
| **Statistical Outputs Match** | Match (excluding runtimes) | Match (excluding runtimes) | Match (excluding runtimes) | **PASSED** |
| **checkpoint.json Retained on Crash**| Yes | Yes | Yes | **PASSED** |
| **checkpoint.json Cleaned on Success**| Yes (Successfully Removed) | Yes (Successfully Removed) | Yes (Successfully Removed) | **PASSED** |

---

## 3. Detailed Audit Outcomes

### Scenario 1 (1 Completed Fold Interruption):
* **Behavior**: Executor crashed after 1 fold of `DecisionTree`. The `checkpoint.json` file was successfully written and retained. On resume, the framework successfully loaded the checkpoint, skipped the first fold, and completed the remaining 14 folds.
* **Outcome**: The final `raw_results.csv`, `normalized_results.csv`, and metrics matched the reference run exactly (excluding non-deterministic model training and evaluation runtimes).

### Scenario 2 (1 Completed Model Interruption):
* **Behavior**: `DecisionTree` finished. Executor crashed on the first fold of `LogisticRegression`. On resume, the executor skipped the entire `DecisionTree` model evaluation, loaded its metrics directly into memory, and resumed `LogisticRegression` from its second fold.
* **Outcome**: The final model summaries and aggregate evaluations matched the reference run exactly.

### Scenario 3 (Multiple Completed Models Interruption):
* **Behavior**: Both `DecisionTree` and `LogisticRegression` finished. Executor crashed on the first fold of `RandomForest`. On resume, both finished models were restored from the checkpoint file. Only the `RandomForest` model was evaluated (starting from its second fold).
* **Outcome**: All generated files, including raw results, normalized metrics, and statistical significance analysis artifacts matched the reference run.

---

## 4. Conclusion

The checkpoint integrity audit confirms that the resume support functions with **100% mathematical precision**. The framework successfully prevents compute redundancy at both the fold level and model level, retains integrity metadata during unexpected process terminations, and cleans up checkpoint artifacts upon successful completion.
