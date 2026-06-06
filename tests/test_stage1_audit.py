import pytest
import pandas as pd
import numpy as np
from datasets import DatasetAudit, DatasetSanitizer, IdentifierDetection


def test_identifier_detection():
    detector = IdentifierDetection()

    df = pd.DataFrame({
        "id_col": [1, 2, 3, 4],
        "SSN_col": ["a", "b", "c", "d"],
        "normal_col": [10, 20, 30, 10],  # Not fully unique
        "unique_str": ["x", "y", "z", "w"],  # Fully unique, should be detected
        "duplicate_str": ["x", "x", "z", "z"],  # Not unique, not an ID name
    })

    detected = detector.detect_identifiers(df)
    cols_detected = [item["column"] for item in detected]

    assert "id_col" in cols_detected
    assert "SSN_col" in cols_detected
    assert "unique_str" in cols_detected
    assert "normal_col" not in cols_detected
    assert "duplicate_str" not in cols_detected


def test_dataset_audit():
    audit = DatasetAudit()

    # Create 200 rows to satisfy near-constant threshold (>0.99 frequency)
    near_constant_values = [1] * 199 + [2]
    df = pd.DataFrame({
        "customer_id": [1, 2, 1, 2, 3] * 40,  # Grouped entities
        "target": ["ok", "ok", "fail", "fail", "!@9#%8"] * 40,  # Corrupted target value
        "constant_feat": [5] * 200,  # Constant
        "near_constant": near_constant_values,  # Near constant
        "normal": list(range(200)),
        "leaked": list(range(200)),  # Perfect correlation with normal
        "dirty_val": ["_______", "10", "---", "15", "NAN"] * 40,  # Placeholders
    })

    # Test audit_dataframe with normal as target
    res = audit.audit_dataframe(df, target_col="target")

    assert res["n_samples"] == 200
    assert res["n_features"] == 7
    assert "constant_feat" in res["constant_features"]
    assert len(res["near_constant_features"]) == 1
    assert res["near_constant_features"][0]["column"] == "near_constant"

    # Grouped entities
    grouped_cols = [g["column"] for g in res["grouped_entities"]]
    assert "customer_id" in grouped_cols

    # Corrupted labels
    assert "!@9#%8" in res["corrupted_labels"]["corrupted_classes"]

    # Placeholders
    assert "dirty_val" in res["missing_value_placeholders"]
    assert "_______" in res["missing_value_placeholders"]["dirty_val"]

    # Leakage
    res_leak = audit.audit_dataframe(df, target_col="normal")
    leak_cols = [l["column"] for l in res_leak["target_leakage_columns"]]
    assert "leaked" in leak_cols


def test_dataset_sanitizer():
    sanitizer = DatasetSanitizer()

    # Create dummy dataframe with duplicates, identifiers, corrupted targets, and placeholders
    df = pd.DataFrame({
        "id": [1, 2, 3, 3, 4, 5],  # Identifier column
        "feature": ["_______", "10_", "---", "---", "25", "10_"],
        "target": ["ok", "ok", "!@9#%8", "!@9#%8", "fail", "different_target"],
    })

    sanitized = sanitizer.sanitize(df, target_col="target", drop_ids=True, drop_duplicates=True)

    # Corrupted target rows (with "!@9#%8") should be removed
    assert "!@9#%8" not in sanitized["target"].unique()
    assert "id" not in sanitized.columns
    # We should have rows 0, 1, 4, 5 left (4 rows)
    assert len(sanitized) == 4

    # Check placeholder sanitization
    # Row 0: "_______" -> NaN
    # Row 1: "10_" -> 10.0
    # Row 4: "25" -> 25.0
    # Row 5: "10_" -> 10.0
    assert pd.isna(sanitized["feature"].iloc[0])
    assert sanitized["feature"].iloc[1] == 10.0
