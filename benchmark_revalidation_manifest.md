# Benchmark Revalidation Manifest

## 1. Execution Order Recommendation (Safest to Riskiest)

We recommend the following execution order to minimize risk and avoid blocking on slow datasets:

1. **diabetes** (Safe to rerun now — < 1 min)
2. **Edu-xApi** (Safe to rerun now — < 1 min)
3. **Edu-Primary** (Safe to rerun now — < 1 min)
4. **heart** (Safe to rerun now — < 1 min)
5. **german_credit_data** (High compute cost — ~ 1 mins)
6. **Financial** (Very high compute cost — ~ 3895 mins)

## 2. Dataset Overview & Estimates

| Dataset            |   Rows |   Features |   Classes |   CV Folds |   Model Fits | Expected Runtime   |
|:-------------------|-------:|-----------:|----------:|-----------:|-------------:|:-------------------|
| diabetes           |    768 |          8 |         2 |          5 |           25 | < 1 min            |
| Edu-xApi           |    480 |         16 |         3 |          5 |           25 | < 1 min            |
| Edu-Primary        |    649 |         32 |        17 |          2 |           10 | < 1 min            |
| heart              |   1025 |         13 |         2 |          5 |           25 | < 1 min            |
| german_credit_data |   1000 |         20 |         2 |          5 |           25 | ~ 1 mins           |
| Financial          |  50000 |         26 |         7 |          5 |           25 | ~ 3895 mins        |

## 3. Configuration & Models

- **DecisionTree**
- **GradientBoosting**
- **LogisticRegression**
- **RandomForest**
- **SVM**

## 4. Risk Analysis & Classifications

| Dataset            | Cost Classification    | Bottlenecks                                                                    | Failure Points                                                                           |
|:-------------------|:-----------------------|:-------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|
| diabetes           | Safe to rerun now      | None                                                                           | Low risk                                                                                 |
| Edu-xApi           | Safe to rerun now      | None                                                                           | Low risk                                                                                 |
| Edu-Primary        | Safe to rerun now      | None                                                                           | Low risk                                                                                 |
| heart              | Safe to rerun now      | None                                                                           | Low risk                                                                                 |
| german_credit_data | High compute cost      | None                                                                           | Low risk                                                                                 |
| Financial          | Very high compute cost | SVM bottlenecks (O(N^2) to O(N^3) scaling); Memory bottlenecks (Dataset scale) | SVM could time out or appear hung; OOM during Parallel CV or SVM dense matrix allocation |
