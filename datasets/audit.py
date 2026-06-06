"""
Dataset Audit Module.

Provides comprehensive validation, data quality checks, and leakage detection
for datasets registered in the benchmarking framework.
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from datasets.registry import get_registry
from datasets.identifier import IdentifierDetection

logger = logging.getLogger(__name__)


class DatasetAudit:
    """
    Audits datasets for duplicate rows, near-duplicates, grouped entities,
    corrupted labels, target leakage, high cardinality, and missing value placeholders.
    """

    def __init__(self, identifier_detector: Optional[IdentifierDetection] = None):
        self.detector = identifier_detector or IdentifierDetection()

    def detect_missing_value_placeholders(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Identify common dirty placeholders used for missing values."""
        placeholders = ["_______", "#F%$D@*&8", "?", "NULL", "None", "NA", "N/A", "NaN", "", "-", "nan", "_", "--"]
        results = {}
        for col in df.columns:
            col_series = df[col].astype(str).str.strip()
            found = {}
            for p in placeholders:
                count = (col_series == p).sum()
                if count > 0:
                    found[p] = int(count)
            # Check for numeric weird placeholders (e.g. -999, -9999, 999, 9999)
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                for p in [-999, -9999, 999, 9999]:
                    count = (df[col] == p).sum()
                    if count > 0:
                        found[str(p)] = int(count)
            
            # Check strings like __10000__ or **10000**
            counts_und = col_series.str.match(r"^__\d+__$").sum()
            if counts_und > 0:
                sample_val = col_series[col_series.str.match(r"^__\d+__$")].iloc[0]
                found[f"regex(^__\\d+__$ e.g. {sample_val})"] = int(counts_und)

            counts_ast = col_series.str.match(r"^\*\*.*\*\*$").sum()
            if counts_ast > 0:
                sample_val = col_series[col_series.str.match(r"^\*\*.*\*\*$")].iloc[0]
                found[f"regex(^\\*\\*.*\\*\\*$ e.g. {sample_val})"] = int(counts_ast)

            if found:
                results[col] = found
        return results

    def detect_high_cardinality(self, df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
        """Detect object/categorical columns with high cardinality."""
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

    def detect_grouped_entities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify columns that define groupings or entities (e.g. repeated user/customer IDs)."""
        groups = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ["customer_id", "student_id", "user_id", "subject_id", "member_id", "group_id", "id"]):
                n_unique = df[col].nunique()
                if 1 < n_unique < len(df):  # not unique, repeating groups
                    max_rep = int(df[col].value_counts().max())
                    groups.append({
                        "column": col,
                        "n_groups": int(n_unique),
                        "max_repetitions": max_rep,
                        "avg_repetitions": float(len(df) / n_unique)
                    })
        return groups

    def detect_corrupted_labels(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Check for corrupted/gibberish class labels in the target column."""
        if target_col not in df.columns:
            return {}
        target_series = df[target_col]
        counts = target_series.value_counts(dropna=False).to_dict()
        corrupted = {}
        for val, count in counts.items():
            val_str = str(val)
            # Check for non-standard target value naming patterns
            if re.search(r"[!@#$%^&*()_+={}\[\]|\\:;\"'<>,.?/~`]", val_str):
                corrupted[val_str] = int(count)
        return {
            "corrupted_classes": corrupted,
            "class_counts": {str(k): int(v) for k, v in counts.items()}
        }

    def detect_target_leakage(self, df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
        """Scan columns for high correlation or perfect separation targeting leakage."""
        leakage = []
        if target_col not in df.columns:
            return leakage

        target_encoded = df[target_col]
        if target_encoded.dtype == object or target_encoded.dtype.name == 'category':
            target_encoded = pd.factorize(target_encoded)[0]

        for col in df.columns:
            if col == target_col:
                continue
            col_series = df[col]
            if pd.api.types.is_numeric_dtype(col_series.dtype):
                try:
                    corr = pd.Series(target_encoded).corr(col_series)
                    if abs(corr) > 0.95:
                        leakage.append({
                            "column": col,
                            "metric": "Pearson Correlation",
                            "value": float(corr),
                            "reason": "Extremely high correlation with target (>0.95)"
                        })
                except Exception:
                    pass
            else:
                try:
                    ct = pd.crosstab(df[col], df[target_col])
                    row_entropy = ct.apply(lambda r: (r > 0).sum(), axis=1)
                    if (row_entropy <= 1).all() and df[col].nunique() > 1:
                        leakage.append({
                            "column": col,
                            "metric": "Perfect separation",
                            "value": 1.0,
                            "reason": "Feature perfectly separates target categories"
                        })
                except Exception:
                    pass
        return leakage

    def audit_dataframe(self, df: pd.DataFrame, target_col: str, filepath: Optional[Path] = None) -> Dict[str, Any]:
        """Perform a full quality and integrity audit on a DataFrame."""
        n_samples = len(df)
        n_features = len(df.columns)
        
        # Check duplicate rows
        duplicates = int(df.duplicated().sum())

        # Detect identifiers
        ids_detected = self.detector.detect_identifiers(df)
        id_cols = [x["column"] for x in ids_detected]

        # Near-duplicate check: drop identifier columns and see if duplicates increase
        if id_cols:
            near_duplicates = int(df.drop(columns=id_cols).duplicated().sum()) - duplicates
        else:
            near_duplicates = 0

        # Constant / near-constant features
        constant_features = []
        near_constant_features = []
        for col in df.columns:
            if col == target_col:
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
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ["date", "time", "timestamp", "year", "month"]):
                datetime_cols.append(col)

        # Class imbalance ratio
        imbalance_ratio = 1.0
        if target_col and target_col in df.columns:
            counts = df[target_col].value_counts()
            if len(counts) > 1:
                imbalance_ratio = float(counts.max() / counts.min())

        file_size = int(filepath.stat().st_size) if filepath and filepath.exists() else 0

        return {
            "file_size_bytes": file_size,
            "n_samples": n_samples,
            "n_features": n_features,
            "target_column": target_col,
            "duplicate_rows": duplicates,
            "near_duplicate_rows": near_duplicates,
            "identifier_columns": ids_detected,
            "high_cardinality_columns": self.detect_high_cardinality(df, target_col),
            "grouped_entities": self.detect_grouped_entities(df),
            "corrupted_labels": self.detect_corrupted_labels(df, target_col),
            "target_leakage_columns": self.detect_target_leakage(df, target_col),
            "constant_features": constant_features,
            "near_constant_features": near_constant_features,
            "missing_value_placeholders": self.detect_missing_value_placeholders(df),
            "datetime_columns": datetime_cols,
            "class_imbalance_ratio": imbalance_ratio
        }

    def audit_all_registered(self) -> Dict[str, Dict[str, Any]]:
        """Audit all datasets in the global registry."""
        registry = get_registry()
        results = {}
        for name in registry.list_datasets():
            cfg = registry.get(name)
            filepath = Path(cfg["filepath"])
            target_col = cfg["target_column"]
            
            if not filepath.exists():
                logger.warning(f"Dataset '{name}' filepath '{filepath}' does not exist.")
                continue

            df = pd.read_csv(filepath)
            results[name] = self.audit_dataframe(df, target_col, filepath)
        return results

    def generate_markdown_report(self, audit_results: Dict[str, Any]) -> str:
        """Format the audit results into a detailed Markdown report."""
        md = []
        md.append("# Dataset Integrity Audit Report\n")
        md.append("Generated automatically on all registered datasets.\n")
        
        md.append("## Executive Overview\n")
        headers = ["Dataset", "Samples", "Features", "Duplicates", "Near-Dupes", "Identifiers", "Groups", "Imbalance Ratio"]
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")
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
            md.append("| " + " | ".join(row) + " |")
        md.append("\n")

        for name, res in audit_results.items():
            md.append(f"## Dataset: {name}\n")
            md.append(f"- **Path**: `{res.get('filepath', f'datasets/{name}.csv')}` (Size: {res['file_size_bytes']:,} bytes)")
            md.append(f"- **Target Column**: `{res['target_column']}`")
            md.append(f"- **Dimensions**: {res['n_samples']} rows × {res['n_features']} features\n")
            
            md.append("### Grouping & Leakage Risks")
            if res["grouped_entities"]:
                md.append("- **Grouped Entities Detected**:")
                for g in res["grouped_entities"]:
                    md.append(f"  - Column `{g['column']}` represents `{g['n_groups']}` entities (max repetitions: {g['max_repetitions']}, avg: {g['avg_repetitions']:.1f}). This requires a grouped validation strategy (e.g. GroupKFold).")
            else:
                md.append("- *No grouped entities detected.*")
                
            if res["identifier_columns"]:
                md.append("- **Identifier Columns Detected (Target/Feature Leakage)**:")
                for idx in res["identifier_columns"]:
                    md.append(f"  - `{idx['column']}` ({idx['reason']}, uniqueness: {idx['unique_ratio']:.1%})")
            else:
                md.append("- *No identifier columns detected.*")
                
            if res["target_leakage_columns"]:
                md.append("- **Target Leakage Columns Detected**:")
                for l in res["target_leakage_columns"]:
                    md.append(f"  - `{l['column']}`: {l['reason']} (value: {l['value']})")
            else:
                md.append("- *No target leakage columns detected.*")
            md.append("\n")

            md.append("### Data Quality & Encoding Issues")
            md.append(f"- **Duplicate Rows**: {res['duplicate_rows']}")
            md.append(f"- **Near-Duplicate Rows (excluding IDs)**: {res['near_duplicate_rows']}")
            
            if res["missing_value_placeholders"]:
                md.append("- **Dirty Missing Value Placeholders Detected**:")
                for col, placeholders in res["missing_value_placeholders"].items():
                    place_str = ", ".join([f"'{k}': {v}" for k, v in placeholders.items()])
                    md.append(f"  - Column `{col}` contains: {place_str}")
            else:
                md.append("- *No dirty missing value placeholders detected.*")
                
            if res["corrupted_labels"] and res["corrupted_labels"].get("corrupted_classes"):
                md.append("- **Corrupted Target Labels**:")
                for k, v in res["corrupted_labels"]["corrupted_classes"].items():
                    md.append(f"  - Target class value `{k}` occurs {v} times.")
            else:
                md.append("- *No corrupted target labels detected.*")
                
            if res["high_cardinality_columns"]:
                md.append("- **High Cardinality Categorical Features**:")
                for hc in res["high_cardinality_columns"]:
                    md.append(f"  - Column `{hc['column']}` has {hc['unique_values']} unique values (ratio: {hc['ratio']:.1%}).")
            else:
                md.append("- *No high-cardinality features detected.*")
                
            if res["constant_features"]:
                md.append(f"- **Constant Features (to be dropped)**: `{', '.join(res['constant_features'])}`")
            if res["near_constant_features"]:
                md.append("- **Near-Constant Features**:")
                for nc in res["near_constant_features"]:
                    md.append(f"  - Column `{nc['column']}` has a single value representing {nc['frequency']:.1%} of samples.")
                    
            md.append("\n" + "---" + "\n")

        return "\n".join(md)
