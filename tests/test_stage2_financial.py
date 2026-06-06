"""
Stage 2 Regression Tests: Financial Dataset Remediation.

Validates that the DatasetSanitizer.remediate_financial() method and the
DatasetLoader integration correctly clean the Financial dataset.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from datasets.loaders import load_dataset
from datasets.sanitizer import DatasetSanitizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Unit tests on DatasetSanitizer.remediate_financial (synthetic data)
# --------------------------------------------------------------------------- #

class TestFinancialSanitizerUnit:
    """Tests remediate_financial on small synthetic DataFrames."""

    def _make_financial_df(self):
        """Create a miniature Financial-like DataFrame."""
        return pd.DataFrame({
            "ID": ["CUS_0001", "CUS_0002", "CUS_0003", "CUS_0004", "CUS_0005"],
            "Customer_ID": ["C100", "C101", "C102", "C103", "C104"],
            "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "SSN": ["111-22-3333", "444-55-6666", "777-88-9999", "000-11-2222", "333-44-5555"],
            "Age": ["25_", "30", "_______", "40_", "50"],
            "Monthly_Inhand_Salary": ["5000", "nan", "__10000__", "7000", "**8000**"],
            "Occupation": ["Engineer", "_______", "Doctor", "_", "Teacher"],
            "Credit_Mix": ["Good", "_", "Standard", "Good", "_"],
            "Payment_Behaviour": [
                "Low_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "!@9#%8",
                "Low_spent_Large_value_payments",
                "!@9#%8",
            ],
        })

    def test_corrupted_target_removed(self):
        df = self._make_financial_df()
        result = DatasetSanitizer().remediate_financial(df, target_col="Payment_Behaviour")
        clean = result["df"]
        assert "!@9#%8" not in clean["Payment_Behaviour"].values

    def test_identifiers_dropped(self):
        df = self._make_financial_df()
        result = DatasetSanitizer().remediate_financial(df, target_col="Payment_Behaviour")
        clean = result["df"]
        for col in ["ID", "Customer_ID", "Name", "SSN"]:
            assert col not in clean.columns, f"{col} should be dropped"

    def test_placeholders_replaced_with_nan(self):
        df = self._make_financial_df()
        result = DatasetSanitizer().remediate_financial(df, target_col="Payment_Behaviour")
        clean = result["df"]
        for col in clean.columns:
            non_null = clean[col].dropna()
            if non_null.dtype == object:
                assert not (non_null == "_______").any(), f"'_______' still in {col}"
                assert not (non_null == "_").any(), f"'_' still in {col}"
                assert not (non_null == "nan").any(), f"'nan' still in {col}"

    def test_numeric_coercion(self):
        df = self._make_financial_df()
        result = DatasetSanitizer().remediate_financial(df, target_col="Payment_Behaviour")
        clean = result["df"]
        if "Age" in clean.columns:
            assert np.issubdtype(clean["Age"].dtype, np.number), \
                f"Age dtype should be numeric, got {clean['Age'].dtype}"

    def test_stats_populated(self):
        df = self._make_financial_df()
        result = DatasetSanitizer().remediate_financial(df, target_col="Payment_Behaviour")
        stats = result["stats"]
        assert stats["before_rows"] == 5
        assert stats["corrupted_target_rows_dropped"] == 2
        assert stats["after_rows"] == 3
        assert "ID" in stats["identifier_columns_dropped"]
        assert "Customer_ID" in stats["identifier_columns_dropped"]
        assert "Name" in stats["identifier_columns_dropped"]
        assert "SSN" in stats["identifier_columns_dropped"]


# --------------------------------------------------------------------------- #
# Integration test on the real Financial.csv via DatasetLoader
# --------------------------------------------------------------------------- #

class TestFinancialLoaderIntegration:
    """Tests that DatasetLoader.load_csv delegates to DatasetSanitizer correctly."""

    @pytest.fixture(autouse=True)
    def _check_file(self):
        filepath = PROJECT_ROOT / "datasets" / "Financial.csv"
        if not filepath.exists():
            pytest.skip("Financial.csv not found")

    def test_identifiers_excluded(self):
        X, y, metadata = load_dataset(
            PROJECT_ROOT / "datasets" / "Financial.csv",
            target_column="Payment_Behaviour"
        )
        for col in ["ID", "Customer_ID", "Name", "SSN"]:
            assert col not in X.columns, f"Identifier {col} leaked into features"
            assert col not in metadata["feature_names"]

    def test_corrupted_target_absent(self):
        X, y, metadata = load_dataset(
            PROJECT_ROOT / "datasets" / "Financial.csv",
            target_column="Payment_Behaviour"
        )
        assert "!@9#%8" not in y.unique()
        assert not y.isna().any()
        assert "!@9#%8" not in metadata["class_distribution"]
        assert len(metadata["class_distribution"]) == 6

    def test_placeholders_cleaned(self):
        X, y, _ = load_dataset(
            PROJECT_ROOT / "datasets" / "Financial.csv",
            target_column="Payment_Behaviour"
        )
        for col in X.columns:
            non_null = X[col].dropna()
            if non_null.dtype == object:
                assert not (non_null == "_______").any(), f"'_______' in {col}"
                assert not (non_null == "_").any(), f"'_' in {col}"
                assert not (non_null == "__10000__").any(), f"'__10000__' in {col}"
                assert not (non_null == "**10000**").any(), f"'**10000**' in {col}"
                assert not (non_null == "nan").any(), f"'nan' in {col}"

    def test_numeric_columns_coerced(self):
        X, _, _ = load_dataset(
            PROJECT_ROOT / "datasets" / "Financial.csv",
            target_column="Payment_Behaviour"
        )
        numeric_expected = ["Age", "Changed_Credit_Limit", "Amount_invested_monthly", "Monthly_Balance"]
        for col in numeric_expected:
            if col in X.columns:
                assert np.issubdtype(X[col].dtype, np.number), \
                    f"{col} should be numeric (got {X[col].dtype})"
