# Group Leakage Validation Report — Stage 3

This report provides an evidence-driven assessment of whether any dataset 
requires grouped cross-validation (GroupKFold / StratifiedGroupKFold) 
instead of the current StratifiedKFold strategy.

> [!IMPORTANT]
> All conclusions are based on **data analysis**, not column name heuristics alone. 
Prior claims from Stage 1 are re-evaluated with quantitative evidence.

## Executive Summary

| Dataset | Recommendation | Confidence | Prior Results Invalid? |
|---|---|---|---|
| Edu-Primary | Safe for current StratifiedKFold | HIGH | No |
| Edu-xApi | Safe for current StratifiedKFold | HIGH | No |
| Financial | Safe for current StratifiedKFold | HIGH | No |
| german_credit_data | Safe for current StratifiedKFold | HIGH | No |
| diabetes | Safe for current StratifiedKFold | HIGH | No |
| heart | Safe for current StratifiedKFold | HIGH | No |


## Dataset: Edu-Primary

- **Samples**: 649
- **Features**: 33
- **Target**: `G3`
- **Overall Recommendation**: **Safe for current StratifiedKFold**
- **Prior Results Invalid**: No

### Column: `paid`

> [!NOTE]
> **FALSE POSITIVE** — This column was flagged by name heuristic but is actually a categorical feature.

Column 'paid' has only 2 unique values across 649 rows (unique ratio: 0.0031). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.

**Recommendation**: Safe for current StratifiedKFold (Confidence: HIGH)

#### Group Statistics

| Metric | Value |
|---|---|
| Total unique groups | 2 |
| Total observations | 649 |
| Mean obs per group | 324.50 |
| Median obs per group | 324.5 |
| Max obs per group | 610 |
| % repeated groups | 100.0% |
| % samples in repeated groups | 100.0% |

#### Cross-Validation Fold Overlap

| Fold | % Test Groups Overlapping | % Test Samples Leaked |
|---|---|---|
| 1 | 100.0% | 100.0% |
| 2 | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% |
| 4 | 100.0% | 100.0% |
| 5 | 100.0% | 100.0% |
| **Mean** | — | **100.0%** |

#### Impact Assessment

- **Expected impact**: None. Current benchmark results remain valid.
- **Prior results invalid**: No

---

## Dataset: Edu-xApi

- **Samples**: 480
- **Features**: 17
- **Target**: `Class`
- **Overall Recommendation**: **Safe for current StratifiedKFold**
- **Prior Results Invalid**: No

### Column: `StageID`

> [!NOTE]
> **FALSE POSITIVE** — This column was flagged by name heuristic but is actually a categorical feature.

Column 'StageID' has only 3 unique values across 480 rows (unique ratio: 0.0063). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.

**Recommendation**: Safe for current StratifiedKFold (Confidence: HIGH)

#### Group Statistics

| Metric | Value |
|---|---|
| Total unique groups | 3 |
| Total observations | 480 |
| Mean obs per group | 160.00 |
| Median obs per group | 199.0 |
| Max obs per group | 248 |
| % repeated groups | 100.0% |
| % samples in repeated groups | 100.0% |

#### Train/Test Split Overlap

| Metric | Value |
|---|---|
| Overlapping groups | 3 |
| % test groups overlapping | 100.0% |
| Test samples with group in train | 144 |
| % test samples leaked | 100.0% |

#### Cross-Validation Fold Overlap

| Fold | % Test Groups Overlapping | % Test Samples Leaked |
|---|---|---|
| 1 | 100.0% | 100.0% |
| 2 | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% |
| 4 | 100.0% | 100.0% |
| 5 | 100.0% | 100.0% |
| **Mean** | — | **100.0%** |

#### Impact Assessment

- **Expected impact**: None. Current benchmark results remain valid.
- **Prior results invalid**: No

### Column: `GradeID`

> [!NOTE]
> **FALSE POSITIVE** — This column was flagged by name heuristic but is actually a categorical feature.

Column 'GradeID' has only 10 unique values across 480 rows (unique ratio: 0.0208). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.

**Recommendation**: Safe for current StratifiedKFold (Confidence: HIGH)

#### Group Statistics

| Metric | Value |
|---|---|
| Total unique groups | 10 |
| Total observations | 480 |
| Mean obs per group | 48.00 |
| Median obs per group | 22.5 |
| Max obs per group | 147 |
| % repeated groups | 100.0% |
| % samples in repeated groups | 100.0% |

#### Train/Test Split Overlap

| Metric | Value |
|---|---|
| Overlapping groups | 9 |
| % test groups overlapping | 100.0% |
| Test samples with group in train | 144 |
| % test samples leaked | 100.0% |

#### Cross-Validation Fold Overlap

| Fold | % Test Groups Overlapping | % Test Samples Leaked |
|---|---|---|
| 1 | 100.0% | 100.0% |
| 2 | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% |
| 4 | 100.0% | 100.0% |
| 5 | 100.0% | 100.0% |
| **Mean** | — | **100.0%** |

#### Impact Assessment

- **Expected impact**: None. Current benchmark results remain valid.
- **Prior results invalid**: No

### Column: `SectionID`

> [!NOTE]
> **FALSE POSITIVE** — This column was flagged by name heuristic but is actually a categorical feature.

Column 'SectionID' has only 3 unique values across 480 rows (unique ratio: 0.0063). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.

**Recommendation**: Safe for current StratifiedKFold (Confidence: HIGH)

#### Group Statistics

| Metric | Value |
|---|---|
| Total unique groups | 3 |
| Total observations | 480 |
| Mean obs per group | 160.00 |
| Median obs per group | 167.0 |
| Max obs per group | 283 |
| % repeated groups | 100.0% |
| % samples in repeated groups | 100.0% |

#### Train/Test Split Overlap

| Metric | Value |
|---|---|
| Overlapping groups | 3 |
| % test groups overlapping | 100.0% |
| Test samples with group in train | 144 |
| % test samples leaked | 100.0% |

#### Cross-Validation Fold Overlap

| Fold | % Test Groups Overlapping | % Test Samples Leaked |
|---|---|---|
| 1 | 100.0% | 100.0% |
| 2 | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% |
| 4 | 100.0% | 100.0% |
| 5 | 100.0% | 100.0% |
| **Mean** | — | **100.0%** |

#### Impact Assessment

- **Expected impact**: None. Current benchmark results remain valid.
- **Prior results invalid**: No

---

## Dataset: Financial

- **Samples**: 50,000
- **Features**: 27
- **Target**: `Payment_Behaviour`
- **Overall Recommendation**: **Safe for current StratifiedKFold**
- **Prior Results Invalid**: No

### Column: `Customer_ID`

**Recommendation**: StratifiedGroupKFold recommended (Confidence: HIGH)

#### Group Statistics

| Metric | Value |
|---|---|
| Total unique groups | 12,500 |
| Total observations | 50,000 |
| Mean obs per group | 4.00 |
| Median obs per group | 4.0 |
| Max obs per group | 4 |
| % repeated groups | 100.0% |
| % samples in repeated groups | 100.0% |

#### Train/Test Split Overlap

| Metric | Value |
|---|---|
| Overlapping groups | 9383 |
| % test groups overlapping | 98.8% |
| Test samples with group in train | 14536 |
| % test samples leaked | 96.9% |

#### Cross-Validation Fold Overlap

| Fold | % Test Groups Overlapping | % Test Samples Leaked |
|---|---|---|
| 1 | 99.7% | 99.1% |
| 2 | 99.7% | 99.2% |
| 3 | 99.8% | 99.4% |
| 4 | 99.6% | 98.7% |
| 5 | 99.7% | 99.2% |
| **Mean** | — | **99.1%** |

#### Impact Assessment

- **Expected impact**: Cross-validation scores may be inflated because 99.1% of test samples have their entity group present in training. Models may memorize group-level patterns instead of learning generalizable features.
- **Prior results invalid**: No

> [!TIP]
> **Already Remediated**: Customer_ID is already dropped during Stage 2 remediation (DatasetSanitizer.remediate_financial). This column never enters the training pipeline, so there is no group leakage in current benchmarks.

---

## Dataset: german_credit_data

- **Samples**: 1,000
- **Features**: 21
- **Target**: `kredit`
- **Overall Recommendation**: **Safe for current StratifiedKFold**
- **Prior Results Invalid**: No

*No candidate grouping columns found. Dataset is safe for StratifiedKFold.*

---

## Dataset: diabetes

- **Samples**: 768
- **Features**: 9
- **Target**: `Outcome`
- **Overall Recommendation**: **Safe for current StratifiedKFold**
- **Prior Results Invalid**: No

*No candidate grouping columns found. Dataset is safe for StratifiedKFold.*

---

## Dataset: heart

- **Samples**: 1,025
- **Features**: 14
- **Target**: `target`
- **Overall Recommendation**: **Safe for current StratifiedKFold**
- **Prior Results Invalid**: No

*No candidate grouping columns found. Dataset is safe for StratifiedKFold.*

---

## Re-evaluation of Prior Claims

### Edu-Primary

- **Column `paid`**: **FALSE POSITIVE** — Safe for current StratifiedKFold
  - Column 'paid' has only 2 unique values across 649 rows (unique ratio: 0.0031). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.
- **Conclusion**: Safe for current StratifiedKFold

### Edu-xApi

- **Column `StageID`**: **FALSE POSITIVE** — Safe for current StratifiedKFold
  - Column 'StageID' has only 3 unique values across 480 rows (unique ratio: 0.0063). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.
- **Column `GradeID`**: **FALSE POSITIVE** — Safe for current StratifiedKFold
  - Column 'GradeID' has only 10 unique values across 480 rows (unique ratio: 0.0208). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.
- **Column `SectionID`**: **FALSE POSITIVE** — Safe for current StratifiedKFold
  - Column 'SectionID' has only 3 unique values across 480 rows (unique ratio: 0.0063). This is a categorical feature, not an entity identifier. High group overlap in CV is expected and harmless — it simply means the same category value appears in both train and test, which is normal for categorical features.
- **Conclusion**: Safe for current StratifiedKFold
