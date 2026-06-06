import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datasets.loaders import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_financial_remediations():
    # Load Financial dataset
    filepath = PROJECT_ROOT / "datasets" / "Financial.csv"
    if not filepath.exists():
        pytest.skip("Financial.csv not found")
        
    X, y, metadata = load_dataset(filepath, target_column="Payment_Behaviour")
    
    # Assert 1: Identifiers never enter model training
    identifiers = ["ID", "Customer_ID", "Name", "SSN"]
    for id_col in identifiers:
        assert id_col not in X.columns, f"Identifier {id_col} was not excluded from features"
        
    # Assert 2: Placeholders become NaN
    # We check that the original dirty strings are no longer in the loaded features
    for col in X.columns:
        # Check string presence, ignoring actual NaNs
        non_null_series = X[col].dropna()
        if non_null_series.dtype == object:
            assert not non_null_series.str.contains("_______").any(), f"Placeholder '_______' found in column {col}"
            assert not non_null_series.str.contains(r"#F%\$D@\*&8").any(), f"Placeholder '#F%$D@*&8' found in column {col}"
            assert not (non_null_series == "_").any(), f"Placeholder '_' found in column {col}"
            assert not (non_null_series == "__10000__").any(), f"Placeholder '__10000__' found in column {col}"
            assert not (non_null_series == "**10000**").any(), f"Placeholder '**10000**' found in column {col}"
            assert not (non_null_series == "nan").any(), f"Placeholder string 'nan' found in column {col}"
        
    # Verify that numerical features containing noise are now correctly parsed as numeric types
    numeric_expected = ["Age", "Changed_Credit_Limit", "Amount_invested_monthly", "Monthly_Balance"]
    for num_col in numeric_expected:
        if num_col in X.columns:
            assert np.issubdtype(X[num_col].dtype, np.number), f"Column {num_col} was not coerced to numeric type (dtype: {X[num_col].dtype})"

    # Assert 3: Corrupted target rows are handled correctly
    assert "!@9#%8" not in y.unique(), "Corrupted label '!@9#%8' was not removed from targets"
    assert not y.isna().any(), "Target contains NaN values"
    
    # Verify metadata is correct
    assert "ID" not in metadata["feature_names"]
    assert "Customer_ID" not in metadata["feature_names"]
    assert "Name" not in metadata["feature_names"]
    assert "SSN" not in metadata["feature_names"]
    assert "!@9#%8" not in metadata["class_distribution"]
    assert len(metadata["class_distribution"]) == 6
