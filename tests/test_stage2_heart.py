"""
Stage 2 Regression Tests: Heart Dataset Remediation.

Validates that the DatasetSanitizer.remediate_heart() method and the
DatasetLoader integration correctly deduplicate the Heart dataset.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from datasets.loaders import load_dataset
from datasets.sanitizer import DatasetSanitizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Unit tests on DatasetSanitizer.remediate_heart (synthetic data)
# --------------------------------------------------------------------------- #

class TestHeartSanitizerUnit:
    """Tests remediate_heart on small synthetic DataFrames."""

    def _make_heart_df(self):
        """Create a miniature Heart-like DataFrame with known duplicates."""
        rows = [
            {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "target": 1},
            {"age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "target": 0},
            {"age": 41, "sex": 0, "cp": 1, "trestbps": 130, "chol": 204, "target": 0},
            {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "target": 1},  # dup of row 0
            {"age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "target": 0},  # dup of row 1
            {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "target": 1},  # dup of row 0
            {"age": 56, "sex": 1, "cp": 1, "trestbps": 120, "chol": 236, "target": 0},
        ]
        return pd.DataFrame(rows)

    def test_duplicates_removed(self):
        df = self._make_heart_df()
        result = DatasetSanitizer().remediate_heart(df, target_col="target")
        clean = result["df"]
        assert len(clean) == 4  # 4 unique rows
        assert clean.duplicated().sum() == 0

    def test_stats_correct(self):
        df = self._make_heart_df()
        result = DatasetSanitizer().remediate_heart(df, target_col="target")
        stats = result["stats"]
        assert stats["before_rows"] == 7
        assert stats["before_unique_rows"] == 4
        assert stats["duplicate_rows_dropped"] == 3
        assert stats["after_rows"] == 4
        assert abs(stats["duplicate_percentage"] - (3 / 7 * 100)) < 0.01

    def test_class_distribution_preserved(self):
        df = self._make_heart_df()
        result = DatasetSanitizer().remediate_heart(df, target_col="target")
        stats = result["stats"]
        # Before: target 1 appears 3 times (dupes), target 0 appears 4 times (dupes)
        # After:  target 1 appears 1 time, target 0 appears 3 times
        assert stats["after_class_distribution"]["1"] == 1
        assert stats["after_class_distribution"]["0"] == 3

    def test_no_data_loss_for_unique_rows(self):
        df = self._make_heart_df()
        unique_original = df.drop_duplicates()
        result = DatasetSanitizer().remediate_heart(df, target_col="target")
        clean = result["df"]
        # Every unique row from the original should be present
        for _, row in unique_original.iterrows():
            matches = clean[(clean == row).all(axis=1)]
            assert len(matches) == 1, f"Unique row missing after dedup: {row.to_dict()}"


# --------------------------------------------------------------------------- #
# Integration test on the real heart.csv via DatasetLoader
# --------------------------------------------------------------------------- #

class TestHeartLoaderIntegration:
    """Tests that DatasetLoader.load_csv deduplicates Heart data correctly."""

    @pytest.fixture(autouse=True)
    def _check_file(self):
        filepath = PROJECT_ROOT / "datasets" / "heart.csv"
        if not filepath.exists():
            pytest.skip("heart.csv not found")

    def test_no_duplicates_after_load(self):
        X, y, metadata = load_dataset(
            PROJECT_ROOT / "datasets" / "heart.csv",
            target_column="target"
        )
        combined = X.copy()
        combined["target"] = y
        assert combined.duplicated().sum() == 0, "Duplicate rows remain after loading"

    def test_unique_row_count(self):
        """The Cleveland heart dataset has 302 unique rows (from 1025 total)."""
        X, y, metadata = load_dataset(
            PROJECT_ROOT / "datasets" / "heart.csv",
            target_column="target"
        )
        assert metadata["n_samples"] == 302, \
            f"Expected 302 unique samples, got {metadata['n_samples']}"

    def test_features_intact(self):
        X, y, _ = load_dataset(
            PROJECT_ROOT / "datasets" / "heart.csv",
            target_column="target"
        )
        expected_features = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                             "restecg", "thalach", "exang", "oldpeak", "slope",
                             "ca", "thal"]
        for feat in expected_features:
            assert feat in X.columns, f"Expected feature '{feat}' missing"

    def test_target_classes_preserved(self):
        X, y, metadata = load_dataset(
            PROJECT_ROOT / "datasets" / "heart.csv",
            target_column="target"
        )
        assert set(y.unique()) == {0, 1}, "Target should have binary classes {0, 1}"
