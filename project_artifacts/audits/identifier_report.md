# Identifier and Feature Leakage Report

This report lists all identifier columns and target leakage columns detected across the audited datasets.

## Summary Table

| Dataset | Total Features | Identified IDs | Leakage Columns |
|---|---|---|---|
| Edu-Primary | 33 | 1 | 0 |
| Edu-xApi | 17 | 4 | 0 |
| Financial | 27 | 4 | 1 |
| german_credit_data | 21 | 0 | 0 |
| diabetes | 9 | 0 | 0 |
| heart | 14 | 0 | 0 |


## Dataset: Edu-Primary

### Detected Identifier Columns
- **`paid`**: Name heuristic (uniqueness ratio: 0.3%)

### Target Leakage Columns
- *No target leakage columns detected.*

---

## Dataset: Edu-xApi

### Detected Identifier Columns
- **`StageID`**: Name heuristic (uniqueness ratio: 0.6%)
- **`GradeID`**: Name heuristic (uniqueness ratio: 2.1%)
- **`SectionID`**: Name heuristic (uniqueness ratio: 0.6%)
- **`StudentAbsenceDays`**: Name heuristic (uniqueness ratio: 0.4%)

### Target Leakage Columns
- *No target leakage columns detected.*

---

## Dataset: Financial

### Detected Identifier Columns
- **`ID`**: Name heuristic and fully unique values (uniqueness ratio: 100.0%)
- **`Customer_ID`**: Name heuristic (uniqueness ratio: 25.0%)
- **`Name`**: Name heuristic (uniqueness ratio: 20.3%)
- **`SSN`**: Name heuristic (uniqueness ratio: 25.0%)

### Target Leakage Columns
- **`ID`**: Feature perfectly separates target categories (Perfect separation: 1.0)

---

## Dataset: german_credit_data

### Detected Identifier Columns
- *No identifier columns detected.*

### Target Leakage Columns
- *No target leakage columns detected.*

---

## Dataset: diabetes

### Detected Identifier Columns
- *No identifier columns detected.*

### Target Leakage Columns
- *No target leakage columns detected.*

---

## Dataset: heart

### Detected Identifier Columns
- *No identifier columns detected.*

### Target Leakage Columns
- *No target leakage columns detected.*

---
