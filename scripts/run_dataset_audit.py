"""Dataset Integrity Audit Tool.

Scans all registered datasets and reports duplicate rows, identifier leakage,
missing value placeholders, corrupted class labels, and group leakage.
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

DATASETS = {
    'Edu-Primary': DATASETS_DIR / 'Edu-Primary.csv',
    'Edu-xApi': DATASETS_DIR / 'Edu-xApi.csv',
    'Financial': DATASETS_DIR / 'Financial.csv',
    'german_credit_data': DATASETS_DIR / 'german_credit_data.csv',
    'diabetes': DATASETS_DIR / 'diabetes.csv',
    'heart': DATASETS_DIR / 'heart.csv',
}

TARGET_COLUMNS = {
    'Edu-Primary': 'G3',
    'Edu-xApi': 'Class',
    'Financial': 'Payment_Behaviour',
    'german_credit_data': 'kredit',
    'diabetes': 'Outcome',
    'heart': 'target',
}

def detect_missing_value_placeholders(df: pd.DataFrame) -> dict:
    placeholders = ["_______", "#F%$D@*&8", "?", "NULL", "None", "NA", "N/A", "NaN", "", "-", "nan", "_", "--"]
    results = {}
    for col in df.columns:
        col_series = df[col].astype(str).str.strip()
        found = {}
        for p in placeholders:
            count = (col_series == p).sum()
            if count > 0:
                found[p] = int(count)
        # Check for numeric weird placeholders (e.g. -999, -9999, 9999, __10000__)
        if df[col].dtype in [np.number]:
            for p in [-999, -9999, 999, 9999]:
                count = (df[col] == p).sum()
                if count > 0:
                    found[str(p)] = int(count)
        # Check strings like __10000__
        counts_und = col_series.str.match(r"^__\d+__$").sum()
        if counts_und > 0:
            sample_val = col_series[col_series.str.match(r"^__\d+__$")].iloc[0]
            found[f"regex(^__\\d+__$ e.g. {sample_val})"] = int(counts_und)

        if found:
            results[col] = found
    return results

def detect_identifiers(df: pd.DataFrame) -> list:
    ids = []
    n_rows = len(df)
    for col in df.columns:
        col_lower = col.lower()
        # Name heuristic
        is_id_name = any(kw in col_lower for kw in ["id", "ssn", "name", "key", "subject", "student", "customer", "user", "member", "identity"])
        # Uniqueness heuristic
        n_unique = df[col].nunique()
        is_fully_unique = (n_unique == n_rows) and (df[col].dtype == object or df[col].dtype == int)
        
        if is_id_name or is_fully_unique:
            ids.append({
                "column": col,
                "reason": "Name heuristic" if is_id_name else "Fully unique values",
                "unique_ratio": float(n_unique / n_rows)
            })
    return ids

def detect_high_cardinality(df: pd.DataFrame, target_col: str) -> list:
    hc = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == object or df[col].dtype.name == 'category':
            n_unique = df[col].nunique()
            if n_unique > 50 or (n_unique / len(df) > 0.05 and n_unique > 10):
                hc.append({
                    "column": col,
                    "unique_values": int(n_unique),
                    "ratio": float(n_unique / len(df))
                })
    return hc

def detect_grouped_entities(df: pd.DataFrame) -> list:
    groups = []
    for col in df.columns:
        col_lower = col.lower()
        # Group identifiers usually have ID in the name, but repeating records
        if any(kw in col_lower for kw in ["customer_id", "student_id", "user_id", "subject_id", "member_id", "group_id", "id"]):
            n_unique = df[col].nunique()
            if 1 < n_unique < len(df): # not unique, repeating groups
                max_rep = int(df[col].value_counts().max())
                groups.append({
                    "column": col,
                    "n_groups": int(n_unique),
                    "max_repetitions": max_rep,
                    "avg_repetitions": float(len(df) / n_unique)
                })
    return groups

def detect_corrupted_labels(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        return {}
    target_series = df[target_col]
    counts = target_series.value_counts(dropna=False).to_dict()
    corrupted = {}
    for val, count in counts.items():
        val_str = str(val)
        # gibberish check: special characters, non-alphanumeric (allowing spaces or standard classes)
        # e.g., '!@9#%8'
        import re
        if re.search(r"[!@#$%^&*()_+={}\[\]|\\:;\"'<>,.?/~`]", val_str):
            corrupted[val_str] = int(count)
    return {
        "corrupted_classes": corrupted,
        "class_counts": {str(k): int(v) for k, v in counts.items()}
    }

def detect_target_leakage(df: pd.DataFrame, target_col: str) -> list:
    leakage = []
    if target_col not in df.columns:
        return leakage
    
    # Simple correlation check for numeric features
    target_encoded = df[target_col]
    if target_encoded.dtype == object or target_encoded.dtype.name == 'category':
        target_encoded = pd.factorize(target_encoded)[0]
    
    for col in df.columns:
        if col == target_col:
            continue
        col_series = df[col]
        if col_series.dtype in [np.number]:
            corr = pd.Series(target_encoded).corr(col_series)
            if abs(corr) > 0.95:
                leakage.append({
                    "column": col,
                    "metric": "Pearson Correlation",
                    "value": float(corr),
                    "reason": "Extremely high correlation with target (>0.95)"
                })
        else:
            # Overlap check for categoricals
            # If a categorical feature perfectly separates the target
            ct = pd.crosstab(df[col], df[target_col])
            # If for each category, only one target label is active
            row_entropy = ct.apply(lambda r: (r > 0).sum(), axis=1)
            if (row_entropy <= 1).all() and df[col].nunique() > 1:
                leakage.append({
                    "column": col,
                    "metric": "Perfect separation",
                    "value": 1.0,
                    "reason": "Feature perfectly separates target categories"
                })
    return leakage

def run_audit():
    audit_results = {}
    
    for name, path in DATASETS.items():
        path = Path(path)
        if not path.exists():
            logger.warning(f"Dataset {name} does not exist at {path}")
            continue
        
        logger.info(f"Auditing dataset {name}...")
        df = pd.read_csv(path)
        target = TARGET_COLUMNS.get(name)
        
        # Duplicate checks
        duplicates = int(df.duplicated().sum())
        
        # Near-duplicate check: drop identifier-like columns and see if duplicates increase
        ids_detected = [info["column"] for info in detect_identifiers(df)]
        if ids_detected:
            near_duplicates = int(df.drop(columns=ids_detected).duplicated().sum()) - duplicates
        else:
            near_duplicates = 0
            
        # Constant / near-constant features
        constant_features = []
        near_constant_features = []
        for col in df.columns:
            if col == target:
                continue
            n_unique = df[col].nunique()
            if n_unique == 1:
                constant_features.append(col)
            elif n_unique > 1:
                top_freq = df[col].value_counts(normalize=True).iloc[0]
                if top_freq > 0.99:
                    near_constant_features.append({
                        "column": col,
                        "frequency": float(top_freq)
                    })
                    
        # Date/time columns
        datetime_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["date", "time", "timestamp", "year", "month"]):
                datetime_cols.append(col)
                
        # Class imbalance ratio
        imbalance_ratio = 1.0
        if target and target in df.columns:
            counts = df[target].value_counts()
            if len(counts) > 1:
                imbalance_ratio = float(counts.max() / counts.min())

        # Compile report for this dataset
        audit_results[name] = {
            "file_size_bytes": int(path.stat().st_size),
            "n_samples": int(df.shape[0]),
            "n_features": int(df.shape[1]),
            "target_column": target,
            "duplicate_rows": duplicates,
            "near_duplicate_rows": near_duplicates,
            "identifier_columns": detect_identifiers(df),
            "high_cardinality_columns": detect_high_cardinality(df, target),
            "grouped_entities": detect_grouped_entities(df),
            "corrupted_labels": detect_corrupted_labels(df, target),
            "target_leakage_columns": detect_target_leakage(df, target),
            "constant_features": constant_features,
            "near_constant_features": near_constant_features,
            "missing_value_placeholders": detect_missing_value_placeholders(df),
            "datetime_columns": datetime_cols,
            "class_imbalance_ratio": imbalance_ratio
        }

    # Save dataset_audit.json
    with open(PROJECT_ROOT / "project_artifacts/audits/dataset_audit.json", "w", encoding="utf8") as f:
        json.dump(audit_results, f, indent=2)
    logger.info("Saved dataset_audit.json")

    # Generate dataset_audit_report.md
    md_content = []
    md_content.append("# Dataset Integrity Audit Report\n")
    md_content.append("Generated automatically by `scripts/run_dataset_audit.py` on all registered datasets.\n")
    
    # Summary Table
    md_content.append("## Executive Overview\n")
    headers = ["Dataset", "Samples", "Features", "Duplicates", "Near-Dupes", "Identifiers", "Groups", "Imbalance Ratio"]
    md_content.append("| " + " | ".join(headers) + " |")
    md_content.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for name, res in audit_results.items():
        row = [
            name,
            str(res["n_samples"]),
            str(res["n_features"]),
            str(res["duplicate_rows"]),
            str(res["near_duplicate_rows"]),
            str(len(res["identifier_columns"])),
            str(len(res["grouped_entities"])),
            f"{res['class_imbalance_ratio']:.2f}"
        ]
        md_content.append("| " + " | ".join(row) + " |")
    md_content.append("\n")

    # Per-Dataset Findings
    for name, res in audit_results.items():
        md_content.append(f"## Dataset: {name}\n")
        md_content.append(f"- **Path**: `datasets/{name}.csv` (Size: {res['file_size_bytes']:,} bytes)")
        md_content.append(f"- **Target Column**: `{res['target_column']}`")
        md_content.append(f"- **Dimensions**: {res['n_samples']} rows × {res['n_features']} features\n")
        
        # 1. Leakage & Grouping
        md_content.append("### Grouping & Leakage Risks")
        if res["grouped_entities"]:
            md_content.append("- **Grouped Entities Detected**:")
            for g in res["grouped_entities"]:
                md_content.append(f"  - Column `{g['column']}` represents `{g['n_groups']}` entities (max repetitions: {g['max_repetitions']}, avg: {g['avg_repetitions']:.1f}). This requires a grouped validation strategy (e.g. GroupKFold).")
        else:
            md_content.append("- *No grouped entities detected.*")
            
        if res["identifier_columns"]:
            md_content.append("- **Identifier Columns Detected (Target/Feature Leakage)**:")
            for idx in res["identifier_columns"]:
                md_content.append(f"  - `{idx['column']}` ({idx['reason']}, uniqueness: {idx['unique_ratio']:.1%})")
        else:
            md_content.append("- *No identifier columns detected.*")
            
        if res["target_leakage_columns"]:
            md_content.append("- **Target Leakage Columns Detected**:")
            for l in res["target_leakage_columns"]:
                md_content.append(f"  - `{l['column']}`: {l['reason']} (value: {l['value']})")
        else:
            md_content.append("- *No target leakage columns detected.*")
        md_content.append("\n")

        # 2. Preprocessing & Quality
        md_content.append("### Data Quality & Encoding Issues")
        md_content.append(f"- **Duplicate Rows**: {res['duplicate_rows']}")
        md_content.append(f"- **Near-Duplicate Rows (excluding IDs)**: {res['near_duplicate_rows']}")
        
        if res["missing_value_placeholders"]:
            md_content.append("- **Dirty Missing Value Placeholders Detected**:")
            for col, placeholders in res["missing_value_placeholders"].items():
                place_str = ", ".join([f"'{k}': {v}" for k, v in placeholders.items()])
                md_content.append(f"  - Column `{col}` contains: {place_str}")
        else:
            md_content.append("- *No dirty missing value placeholders detected.*")
            
        if res["corrupted_labels"] and res["corrupted_labels"]["corrupted_classes"]:
            md_content.append("- **Corrupted Target Labels**:")
            for k, v in res["corrupted_labels"]["corrupted_classes"].items():
                md_content.append(f"  - Target class value `{k}` occurs {v} times.")
        else:
            md_content.append("- *No corrupted target labels detected.*")
            
        if res["high_cardinality_columns"]:
            md_content.append("- **High Cardinality Categorical Features**:")
            for hc in res["high_cardinality_columns"]:
                md_content.append(f"  - Column `{hc['column']}` has {hc['unique_values']} unique values (ratio: {hc['ratio']:.1%}).")
        else:
            md_content.append("- *No high-cardinality features detected.*")
            
        if res["constant_features"]:
            md_content.append(f"- **Constant Features (to be dropped)**: `{', '.join(res['constant_features'])}`")
        if res["near_constant_features"]:
            md_content.append("- **Near-Constant Features**:")
            for nc in res["near_constant_features"]:
                md_content.append(f"  - Column `{nc['column']}` has a single value representing {nc['frequency']:.1%} of samples.")
                
        md_content.append("\n" + "---" + "\n")

    with open(PROJECT_ROOT / "project_artifacts/audits/dataset_audit_report.md", "w", encoding="utf8") as f:
        f.write("\n".join(md_content))
    logger.info("Saved dataset_audit_report.md")

if __name__ == "__main__":
    run_audit()
