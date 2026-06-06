"""
Identifier Detection module.

Provides logic to detect unique identifiers and potential ID-like columns
within a dataset using heuristics.
"""

import logging
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class IdentifierDetection:
    """
    Detects potential identifier columns (e.g. IDs, SSNs, names, keys)
    in a pandas DataFrame based on naming heuristics and value uniqueness.
    """

    def __init__(self, keywords: List[str] = None):
        """
        Initialize IdentifierDetection with name-based keyword patterns.
        """
        self.keywords = keywords or [
            "id", "ssn", "name", "key", "subject", "student",
            "customer", "user", "member", "identity"
        ]

    def detect_identifiers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scan features in a DataFrame to detect potential identifier columns.

        Args:
            df: The input pandas DataFrame.

        Returns:
            A list of dicts containing column name, reason, and uniqueness ratio.
        """
        identifiers = []
        n_rows = len(df)
        if n_rows == 0:
            return identifiers

        for col in df.columns:
            col_lower = str(col).lower()
            
            # 1. Name heuristic check
            is_id_name = any(kw in col_lower for kw in self.keywords)
            
            # 2. Uniqueness check
            n_unique = df[col].nunique()
            is_fully_unique = (n_unique == n_rows) and (df[col].dtype == object or pd.api.types.is_integer_dtype(df[col].dtype))

            if is_id_name or is_fully_unique:
                reason = "Name heuristic" if is_id_name else "Fully unique values"
                # If both are true, reason is both
                if is_id_name and is_fully_unique:
                    reason = "Name heuristic and fully unique values"
                
                identifiers.append({
                    "column": col,
                    "reason": reason,
                    "unique_ratio": float(n_unique / n_rows)
                })

        return identifiers

    def generate_identifier_report(self, audit_results: Dict[str, Any]) -> str:
        """
        Generate a markdown report of identifier detection results across datasets.

        Args:
            audit_results: The aggregated audit JSON structure.

        Returns:
            A markdown formatted report string.
        """
        md = []
        md.append("# Identifier and Feature Leakage Report\n")
        md.append("This report lists all identifier columns and target leakage columns detected across the audited datasets.\n")

        md.append("## Summary Table\n")
        md.append("| Dataset | Total Features | Identified IDs | Leakage Columns |")
        md.append("|---|---|---|---|")

        for name, res in audit_results.items():
            ids_count = len(res.get("identifier_columns", []))
            leak_count = len(res.get("target_leakage_columns", []))
            md.append(f"| {name} | {res['n_features']} | {ids_count} | {leak_count} |")
        md.append("\n")

        for name, res in audit_results.items():
            md.append(f"## Dataset: {name}\n")
            
            # Identifiers
            md.append("### Detected Identifier Columns")
            ids = res.get("identifier_columns", [])
            if ids:
                for item in ids:
                    md.append(f"- **`{item['column']}`**: {item['reason']} (uniqueness ratio: {item['unique_ratio']:.1%})")
            else:
                md.append("- *No identifier columns detected.*")
            md.append("")

            # Target Leakage
            md.append("### Target Leakage Columns")
            leakage = res.get("target_leakage_columns", [])
            if leakage:
                for item in leakage:
                    md.append(f"- **`{item['column']}`**: {item['reason']} ({item.get('metric', 'Metric')}: {item.get('value', 1.0)})")
            else:
                md.append("- *No target leakage columns detected.*")
            
            md.append("\n---\n")

        return "\n".join(md)
