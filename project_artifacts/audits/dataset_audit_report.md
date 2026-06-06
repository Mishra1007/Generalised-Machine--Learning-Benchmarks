# Dataset Integrity Audit Report

Generated automatically on all registered datasets.

## Executive Overview

| Dataset | Samples | Features | Duplicates | Near-Dupes | Identifiers | Groups | Imbalance Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Edu-Primary | 649 | 33 | 0 | 0 | 1 | 1 | 104.00 |
| Edu-xApi | 480 | 17 | 2 | 0 | 4 | 3 | 1.66 |
| Financial | 50000 | 27 | 0 | 0 | 4 | 1 | 3.34 |
| german_credit_data | 1000 | 21 | 0 | 0 | 0 | 0 | 2.33 |
| diabetes | 768 | 9 | 0 | 0 | 0 | 0 | 1.87 |
| heart | 1025 | 14 | 723 | 0 | 0 | 0 | 1.05 |


## Dataset: Edu-Primary

- **Path**: `datasets/Edu-Primary.csv` (Size: 93,220 bytes)
- **Target Column**: `G3`
- **Dimensions**: 649 rows × 33 features

### Grouping & Leakage Risks
- **Grouped Entities Detected**:
  - Column `paid` represents `2` entities (max repetitions: 610, avg: 324.5). This requires a grouped validation strategy (e.g. GroupKFold).
- **Identifier Columns Detected (Target/Feature Leakage)**:
  - `paid` (Name heuristic, uniqueness: 0.3%)
- *No target leakage columns detected.*


### Data Quality & Encoding Issues
- **Duplicate Rows**: 0
- **Near-Duplicate Rows (excluding IDs)**: 0
- *No dirty missing value placeholders detected.*
- *No corrupted target labels detected.*
- *No high-cardinality features detected.*

---

## Dataset: Edu-xApi

- **Path**: `datasets/Edu-xApi.csv` (Size: 38,026 bytes)
- **Target Column**: `Class`
- **Dimensions**: 480 rows × 17 features

### Grouping & Leakage Risks
- **Grouped Entities Detected**:
  - Column `StageID` represents `3` entities (max repetitions: 248, avg: 160.0). This requires a grouped validation strategy (e.g. GroupKFold).
  - Column `GradeID` represents `10` entities (max repetitions: 147, avg: 48.0). This requires a grouped validation strategy (e.g. GroupKFold).
  - Column `SectionID` represents `3` entities (max repetitions: 283, avg: 160.0). This requires a grouped validation strategy (e.g. GroupKFold).
- **Identifier Columns Detected (Target/Feature Leakage)**:
  - `StageID` (Name heuristic, uniqueness: 0.6%)
  - `GradeID` (Name heuristic, uniqueness: 2.1%)
  - `SectionID` (Name heuristic, uniqueness: 0.6%)
  - `StudentAbsenceDays` (Name heuristic, uniqueness: 0.4%)
- *No target leakage columns detected.*


### Data Quality & Encoding Issues
- **Duplicate Rows**: 2
- **Near-Duplicate Rows (excluding IDs)**: 0
- *No dirty missing value placeholders detected.*
- *No corrupted target labels detected.*
- *No high-cardinality features detected.*

---

## Dataset: Financial

- **Path**: `datasets/Financial.csv` (Size: 15,366,486 bytes)
- **Target Column**: `Payment_Behaviour`
- **Dimensions**: 50000 rows × 27 features

### Grouping & Leakage Risks
- **Grouped Entities Detected**:
  - Column `Customer_ID` represents `12500` entities (max repetitions: 4, avg: 4.0). This requires a grouped validation strategy (e.g. GroupKFold).
- **Identifier Columns Detected (Target/Feature Leakage)**:
  - `ID` (Name heuristic and fully unique values, uniqueness: 100.0%)
  - `Customer_ID` (Name heuristic, uniqueness: 25.0%)
  - `Name` (Name heuristic, uniqueness: 20.3%)
  - `SSN` (Name heuristic, uniqueness: 25.0%)
- **Target Leakage Columns Detected**:
  - `ID`: Feature perfectly separates target categories (value: 1.0)


### Data Quality & Encoding Issues
- **Duplicate Rows**: 0
- **Near-Duplicate Rows (excluding IDs)**: 0
- **Dirty Missing Value Placeholders Detected**:
  - Column `Name` contains: 'nan': 5015
  - Column `SSN` contains: '#F%$D@*&8': 2828
  - Column `Occupation` contains: '_______': 3438
  - Column `Monthly_Inhand_Salary` contains: 'nan': 7498
  - Column `Num_Bank_Accounts` contains: '999': 1
  - Column `Type_of_Loan` contains: 'nan': 5704
  - Column `Num_of_Delayed_Payment` contains: 'nan': 3498
  - Column `Changed_Credit_Limit` contains: '_': 1059
  - Column `Num_Credit_Inquiries` contains: 'nan': 1035
  - Column `Credit_Mix` contains: '_': 9805
  - Column `Credit_History_Age` contains: 'nan': 4470
  - Column `Amount_invested_monthly` contains: 'nan': 2271, 'regex(^__\d+__$ e.g. __10000__)': 2175
  - Column `Monthly_Balance` contains: 'nan': 562
- **Corrupted Target Labels**:
  - Target class value `Low_spent_Small_value_payments` occurs 12694 times.
  - Target class value `High_spent_Medium_value_payments` occurs 8922 times.
  - Target class value `High_spent_Large_value_payments` occurs 6844 times.
  - Target class value `Low_spent_Medium_value_payments` occurs 6837 times.
  - Target class value `High_spent_Small_value_payments` occurs 5651 times.
  - Target class value `Low_spent_Large_value_payments` occurs 5252 times.
  - Target class value `!@9#%8` occurs 3800 times.
- **High Cardinality Categorical Features**:
  - Column `ID` has 50000 unique values (ratio: 100.0%).
  - Column `Customer_ID` has 12500 unique values (ratio: 25.0%).
  - Column `Name` has 10139 unique values (ratio: 20.3%).
  - Column `Age` has 976 unique values (ratio: 2.0%).
  - Column `SSN` has 12501 unique values (ratio: 25.0%).
  - Column `Annual_Income` has 16121 unique values (ratio: 32.2%).
  - Column `Num_of_Loan` has 263 unique values (ratio: 0.5%).
  - Column `Type_of_Loan` has 6260 unique values (ratio: 12.5%).
  - Column `Num_of_Delayed_Payment` has 443 unique values (ratio: 0.9%).
  - Column `Changed_Credit_Limit` has 3927 unique values (ratio: 7.9%).
  - Column `Outstanding_Debt` has 12685 unique values (ratio: 25.4%).
  - Column `Credit_History_Age` has 399 unique values (ratio: 0.8%).
  - Column `Amount_invested_monthly` has 45450 unique values (ratio: 90.9%).
  - Column `Monthly_Balance` has 49433 unique values (ratio: 98.9%).

---

## Dataset: german_credit_data

- **Path**: `datasets/german_credit_data.csv` (Size: 47,940 bytes)
- **Target Column**: `kredit`
- **Dimensions**: 1000 rows × 21 features

### Grouping & Leakage Risks
- *No grouped entities detected.*
- *No identifier columns detected.*
- *No target leakage columns detected.*


### Data Quality & Encoding Issues
- **Duplicate Rows**: 0
- **Near-Duplicate Rows (excluding IDs)**: 0
- **Dirty Missing Value Placeholders Detected**:
  - Column `hoehe` contains: '999': 1
- *No corrupted target labels detected.*
- *No high-cardinality features detected.*

---

## Dataset: diabetes

- **Path**: `datasets/diabetes.csv` (Size: 23,873 bytes)
- **Target Column**: `Outcome`
- **Dimensions**: 768 rows × 9 features

### Grouping & Leakage Risks
- *No grouped entities detected.*
- *No identifier columns detected.*
- *No target leakage columns detected.*


### Data Quality & Encoding Issues
- **Duplicate Rows**: 0
- **Near-Duplicate Rows (excluding IDs)**: 0
- *No dirty missing value placeholders detected.*
- *No corrupted target labels detected.*
- *No high-cardinality features detected.*

---

## Dataset: heart

- **Path**: `datasets/heart.csv` (Size: 38,114 bytes)
- **Target Column**: `target`
- **Dimensions**: 1025 rows × 14 features

### Grouping & Leakage Risks
- *No grouped entities detected.*
- *No identifier columns detected.*
- *No target leakage columns detected.*


### Data Quality & Encoding Issues
- **Duplicate Rows**: 723
- **Near-Duplicate Rows (excluding IDs)**: 0
- *No dirty missing value placeholders detected.*
- *No corrupted target labels detected.*
- *No high-cardinality features detected.*

---
