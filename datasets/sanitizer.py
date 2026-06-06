"""
Dataset Sanitizer Module.

Provides in-memory sanitization operations to clean datasets before training.
Note: This module does NOT modify files on disk.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from datasets.identifier import IdentifierDetection

logger = logging.getLogger(__name__)


class DatasetSanitizer:
    """
    Sanitizes DataFrames in-memory.

    Provides routines to drop duplicate rows, remove identifier columns,
    coerces noisy placeholders to NaNs, and filters out corrupted labels.
    """

    def __init__(self, identifier_detector: Optional[IdentifierDetection] = None):
        self.detector = identifier_detector or IdentifierDetection()

    def sanitize(
        self,
        df: pd.DataFrame,
        target_col: str,
        drop_ids: bool = True,
        drop_duplicates: bool = True,
        clean_placeholders: bool = True,
        clean_corrupted_targets: bool = True
    ) -> pd.DataFrame:
        """
        Applies a suite of sanitization steps to a DataFrame in-memory.

        Args:
            df: Input DataFrame.
            target_col: Name of the target column.
            drop_ids: If True, detects and drops identifier features.
            drop_duplicates: If True, drops duplicate rows.
            clean_placeholders: If True, normalizes common dirty placeholders to NaN.
            clean_corrupted_targets: If True, cleans corrupted labels from the target column.

        Returns:
            A sanitized pandas DataFrame.
        """
        # Create a copy to prevent in-place modification of caller's DataFrame
        clean_df = df.copy()

        # 1. Clean corrupted target values (e.g. drop rows with '!@9#%8' or NaNs in target)
        if clean_corrupted_targets and target_col in clean_df.columns:
            # Clean corrupted target label "!@9#%8"
            clean_df = clean_df[clean_df[target_col].astype(str) != "!@9#%8"]
            # Drop any actual NaNs in target
            clean_df = clean_df.dropna(subset=[target_col])

        # 2. Drop identifiers
        if drop_ids:
            # Detect identifiers on the full DataFrame, but exclude target_col from dropping
            features_df = clean_df.drop(columns=[target_col], errors='ignore')
            detected = self.detector.detect_identifiers(features_df)
            id_cols = [item["column"] for item in detected]
            if id_cols:
                logger.info(f"Sanitizer: dropping identifier columns: {id_cols}")
                clean_df = clean_df.drop(columns=id_cols, errors='ignore')

        # 3. Clean placeholders
        if clean_placeholders:
            placeholders = ["_______", "---", "#F%$D@*&8", "_", "nan", "NAN", "NaN"]
            clean_df = clean_df.replace(placeholders, np.nan)
            
            # Replace regex patterns for double underscores/asterisks placeholders
            clean_df = clean_df.replace(to_replace=r'^__.*__$', value=np.nan, regex=True)
            clean_df = clean_df.replace(to_replace=r'^\*\*.*__$', value=np.nan, regex=True)
            clean_df = clean_df.replace(to_replace=r'^\*\*.*\*\*', value=np.nan, regex=True)

            # Try to coerce numerical columns that might be read as object/string due to placeholders
            for col in clean_df.columns:
                if col != target_col and clean_df[col].dtype == object:
                    try:
                        # Clean trailing underscores (e.g., "24_" -> "24")
                        cleaned = clean_df[col].astype(str).str.rstrip('_')
                        cleaned = cleaned.replace(['nan', 'None', 'NaN'], np.nan)
                        clean_df[col] = pd.to_numeric(cleaned, errors='raise')
                    except Exception:
                        pass

        # 4. Drop duplicate rows
        if drop_duplicates:
            original_len = len(clean_df)
            clean_df = clean_df.drop_duplicates()
            dropped = original_len - len(clean_df)
            if dropped > 0:
                logger.info(f"Sanitizer: dropped {dropped} duplicate rows.")

        return clean_df

    # ------------------------------------------------------------------ #
    # Dataset-specific remediation methods
    # ------------------------------------------------------------------ #

    def remediate_financial(self, df: pd.DataFrame, target_col: str = "Payment_Behaviour") -> Dict[str, Any]:
        """
        Apply Financial dataset-specific remediations.

        Steps:
        1. Drop rows with corrupted target label '!@9#%8'.
        2. Drop identifier columns (ID, Customer_ID, Name, SSN).
        3. Convert known placeholders to NaN.
        4. Coerce noisy numeric columns back to numeric dtype.

        Args:
            df: Raw Financial DataFrame loaded from CSV.
            target_col: Name of the target column.

        Returns:
            Dict with keys:
            - 'df': Sanitized DataFrame.
            - 'stats': Before/after statistics dictionary.
        """
        before_rows = len(df)
        before_cols = len(df.columns)
        before_target_classes = df[target_col].nunique()

        clean_df = df.copy()

        # 1. Drop corrupted target label
        corrupted_count = int((clean_df[target_col].astype(str) == "!@9#%8").sum())
        clean_df = clean_df[clean_df[target_col].astype(str) != "!@9#%8"]

        # 2. Drop identifiers explicitly
        identifiers = ["ID", "Customer_ID", "Name", "SSN"]
        id_cols_present = [c for c in identifiers if c in clean_df.columns]
        clean_df = clean_df.drop(columns=id_cols_present, errors='ignore')

        # 3. Replace known placeholders with NaN
        placeholders = ["_______", "---", "#F%$D@*&8", "_", "nan", "NAN", "NaN"]
        clean_df = clean_df.replace(placeholders, np.nan)

        # Replace regex pattern placeholders (e.g. __10000__, **10000**)
        clean_df = clean_df.replace(to_replace=r'^__.*__$', value=np.nan, regex=True)
        clean_df = clean_df.replace(to_replace=r'^\*\*.*__$', value=np.nan, regex=True)
        clean_df = clean_df.replace(to_replace=r'^\*\*.*\*\*', value=np.nan, regex=True)

        # 4. Coerce noisy numeric columns
        coerced_cols = []
        for col in clean_df.columns:
            if col != target_col and clean_df[col].dtype == object:
                try:
                    cleaned = clean_df[col].astype(str).str.rstrip('_')
                    cleaned = cleaned.replace(['nan', 'None', 'NaN'], np.nan)
                    clean_df[col] = pd.to_numeric(cleaned, errors='raise')
                    coerced_cols.append(col)
                except Exception:
                    pass

        after_rows = len(clean_df)
        after_cols = len(clean_df.columns)
        after_target_classes = clean_df[target_col].nunique()

        stats = {
            "before_rows": before_rows,
            "before_cols": before_cols,
            "before_target_classes": before_target_classes,
            "after_rows": after_rows,
            "after_cols": after_cols,
            "after_target_classes": after_target_classes,
            "corrupted_target_rows_dropped": corrupted_count,
            "identifier_columns_dropped": id_cols_present,
            "numeric_columns_coerced": coerced_cols,
        }

        return {"df": clean_df, "stats": stats}

    def remediate_heart(self, df: pd.DataFrame, target_col: str = "target") -> Dict[str, Any]:
        """
        Apply Heart dataset-specific remediations.

        Steps:
        1. Deduplicate rows (keep first occurrence).

        Args:
            df: Raw Heart DataFrame loaded from CSV.
            target_col: Name of the target column.

        Returns:
            Dict with keys:
            - 'df': Sanitized DataFrame.
            - 'stats': Before/after statistics dictionary.
        """
        before_rows = len(df)
        before_unique = len(df.drop_duplicates())
        duplicate_count = before_rows - before_unique

        # Class distribution before deduplication
        before_class_dist = df[target_col].value_counts().to_dict()

        # Deduplicate
        clean_df = df.drop_duplicates().reset_index(drop=True)

        after_rows = len(clean_df)
        after_class_dist = clean_df[target_col].value_counts().to_dict()

        stats = {
            "before_rows": before_rows,
            "before_unique_rows": before_unique,
            "duplicate_rows_dropped": duplicate_count,
            "duplicate_percentage": float(duplicate_count / before_rows * 100) if before_rows > 0 else 0.0,
            "after_rows": after_rows,
            "before_class_distribution": {str(k): int(v) for k, v in before_class_dist.items()},
            "after_class_distribution": {str(k): int(v) for k, v in after_class_dist.items()},
        }

        return {"df": clean_df, "stats": stats}
