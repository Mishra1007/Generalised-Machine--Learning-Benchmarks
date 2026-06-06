# Financial Dataset Remediation Report

This report documents the remediations applied to the Financial dataset (`datasets/Financial.csv`) 
via `DatasetSanitizer.remediate_financial()`.

## Before / After Summary

| Metric | Before | After |
|---|---|---|
| Total rows | 50,000 | 46,200 |
| Total columns | 27 | 23 |
| Target classes | 7 | 6 |

## Remediations Applied

### 1. Corrupted Target Label Removal

- **Corrupted label**: `!@9#%8`
- **Rows dropped**: 3,800
- **Remaining target classes**: 6

### 2. Identifier Columns Dropped

The following identifier columns were removed from the feature set to prevent target leakage:

- `ID`
- `Customer_ID`
- `Name`
- `SSN`

### 3. Placeholder Normalization

The following dirty placeholder strings were converted to `NaN`:

- `_______`, `---`, `#F%$D@*&8`, `_`, `nan`, `NAN`, `NaN`
- Regex patterns: `__<value>__`, `**<value>**`

### 4. Numeric Column Coercion

The following columns were coerced from string/object dtype to numeric after cleaning trailing underscores:

- `Age`
- `Annual_Income`
- `Num_of_Loan`
- `Num_of_Delayed_Payment`
- `Changed_Credit_Limit`
- `Outstanding_Debt`
- `Amount_invested_monthly`
- `Monthly_Balance`

## Post-Remediation Target Distribution

| Class | Count |
|---|---|
| Low_spent_Small_value_payments | 12,694 |
| High_spent_Medium_value_payments | 8,922 |
| High_spent_Large_value_payments | 6,844 |
| Low_spent_Medium_value_payments | 6,837 |
| High_spent_Small_value_payments | 5,651 |
| Low_spent_Large_value_payments | 5,252 |

## Integration

These remediations are automatically applied when loading the Financial dataset via:
```python
X, y, metadata = load_dataset("datasets/Financial.csv", target_column="Payment_Behaviour")
```
The logic is implemented in `DatasetSanitizer.remediate_financial()` and invoked by `DatasetLoader.load_csv()`.
