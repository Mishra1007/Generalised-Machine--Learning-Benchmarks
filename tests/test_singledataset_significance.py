"""Tests for pairwise_corrected_t_tests and analyze_singledataset_significance.

These tests verify the single-dataset significance testing pathway that uses
the Nadeau-Bengio corrected resampled t-test instead of Friedman/Nemenyi.

Statistical properties protected:
- DataFrame schema matches pairwise_wilcoxon (drop-in compatibility)
- Holm-Bonferroni correction is applied correctly
- p-values are bounded [0, 1]
- Adjusted p-values >= raw p-values (monotonicity)
- Identical scores produce no false positives
- Ranking respects higher_is_better / lower_is_better
- No Friedman/Nemenyi keys appear in single-dataset output
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.significance import (
    analyze_singledataset_significance,
    holm_bonferroni,
    pairwise_corrected_t_tests,
    pairwise_wilcoxon,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _separable_matrix():
    """3 models, 10 observations: model A >> B >> C with clear separation."""
    rng = np.random.default_rng(42)
    n = 10
    return np.column_stack([
        0.90 + rng.normal(0, 0.01, n),  # model A - best
        0.80 + rng.normal(0, 0.01, n),  # model B - middle
        0.70 + rng.normal(0, 0.01, n),  # model C - worst
    ])


def _identical_matrix():
    """3 models with identical constant performance."""
    return np.full((8, 3), 0.85, dtype=float)


def _two_model_matrix():
    """Minimal 2-model case."""
    rng = np.random.default_rng(7)
    n = 10
    return np.column_stack([
        0.85 + rng.normal(0, 0.02, n),
        0.80 + rng.normal(0, 0.02, n),
    ])


# ── pairwise_corrected_t_tests ───────────────────────────────────────────


class TestPairwiseCorrectedTTests:
    """Tests for pairwise_corrected_t_tests()."""

    def test_returns_dataframe_with_expected_columns(self):
        """Schema must include all columns downstream consumers expect."""
        df = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        expected = {
            'model_a', 'model_b', 'statistic', 'p_value',
            'p_value_raw', 'p_value_adj', 'significant', 'decision',
        }
        assert set(df.columns) == expected

    def test_number_of_pairs_equals_c_n_2(self):
        """C(3,2) = 3 pairs for 3 models."""
        df = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        assert len(df) == 3

    def test_two_models_produces_single_row(self):
        df = pairwise_corrected_t_tests(
            _two_model_matrix(), model_names=['X', 'Y'], n_splits=5,
        )
        assert len(df) == 1
        assert df.iloc[0]['model_a'] == 'X'
        assert df.iloc[0]['model_b'] == 'Y'

    def test_p_values_in_valid_range(self):
        """All p-values must be in [0, 1]."""
        df = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        for col in ('p_value', 'p_value_raw', 'p_value_adj'):
            assert (df[col] >= 0.0).all(), f'{col} has negative values'
            assert (df[col] <= 1.0).all(), f'{col} has values > 1'

    def test_adjusted_p_values_ge_raw(self):
        """Holm-Bonferroni adjustment can only increase p-values."""
        df = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        assert (df['p_value_adj'] >= df['p_value_raw'] - 1e-15).all()

    def test_holm_bonferroni_applied_correctly(self):
        """Cross-check that Holm correction matches standalone function."""
        df = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        expected_adj, expected_reject = holm_bonferroni(
            df['p_value_raw'].to_numpy(dtype=float), alpha=0.05,
        )
        assert np.allclose(
            df['p_value_adj'].to_numpy(dtype=float),
            expected_adj, rtol=1e-12, atol=1e-12,
        )
        assert np.array_equal(
            df['significant'].to_numpy(dtype=bool), expected_reject,
        )

    def test_decision_matches_significance(self):
        """decision column must be consistent with significant column."""
        df = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        for _, row in df.iterrows():
            if row['significant']:
                assert row['decision'] == 'reject null'
            else:
                assert row['decision'] == 'fail to reject null'

    def test_identical_scores_not_significant(self):
        """Models with identical scores must never be significant."""
        df = pairwise_corrected_t_tests(
            _identical_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        assert not df['significant'].any()

    def test_default_model_names(self):
        """Auto-generated names follow model_0, model_1, ... pattern."""
        df = pairwise_corrected_t_tests(_separable_matrix(), n_splits=5)
        assert df.iloc[0]['model_a'] == 'model_0'
        assert df.iloc[0]['model_b'] == 'model_1'

    def test_schema_matches_pairwise_wilcoxon(self):
        """Column set must be identical to pairwise_wilcoxon for compatibility."""
        matrix = _separable_matrix()
        wilcoxon_df = pairwise_wilcoxon(matrix, model_names=['A', 'B', 'C'])
        corrected_df = pairwise_corrected_t_tests(
            matrix, model_names=['A', 'B', 'C'], n_splits=5,
        )
        assert set(wilcoxon_df.columns) == set(corrected_df.columns)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            pairwise_corrected_t_tests(
                np.array([1, 2, 3]), model_names=['A'], n_splits=5,
            )

    def test_custom_alpha(self):
        """Stricter alpha should produce fewer or equal rejections."""
        df_strict = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'],
            n_splits=5, alpha=0.001,
        )
        df_lenient = pairwise_corrected_t_tests(
            _separable_matrix(), model_names=['A', 'B', 'C'],
            n_splits=5, alpha=0.10,
        )
        assert df_lenient['significant'].sum() >= df_strict['significant'].sum()


# ── analyze_singledataset_significance ────────────────────────────────────


class TestAnalyzeSingledatasetSignificance:
    """Tests for analyze_singledataset_significance()."""

    def test_returns_expected_keys(self):
        result = analyze_singledataset_significance(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        assert 'pairwise_corrected_t' in result
        assert 'ranking_table' in result

    def test_does_not_return_friedman_or_nemenyi(self):
        """Single-dataset pathway must not contain multi-dataset test keys."""
        result = analyze_singledataset_significance(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        assert 'friedman' not in result
        assert 'nemenyi' not in result
        assert 'pairwise_wilcoxon' not in result

    def test_pairwise_dataframe_schema(self):
        result = analyze_singledataset_significance(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        expected = {
            'model_a', 'model_b', 'statistic', 'p_value',
            'p_value_raw', 'p_value_adj', 'significant', 'decision',
        }
        assert set(result['pairwise_corrected_t'].columns) == expected

    def test_ranking_table_shape(self):
        result = analyze_singledataset_significance(
            _separable_matrix(), model_names=['A', 'B', 'C'], n_splits=5,
        )
        assert result['ranking_table'].shape[0] == 3
        assert 'model' in result['ranking_table'].columns
        assert 'rank' in result['ranking_table'].columns

    def test_ranking_order_higher_is_better(self):
        """Model A has highest scores; it should rank first."""
        result = analyze_singledataset_significance(
            _separable_matrix(), model_names=['A', 'B', 'C'],
            n_splits=5, higher_is_better=True,
        )
        assert result['ranking_table'].iloc[0]['model'] == 'A'

    def test_ranking_order_lower_is_better(self):
        """With lower_is_better, model C (lowest scores) should rank first."""
        result = analyze_singledataset_significance(
            _separable_matrix(), model_names=['A', 'B', 'C'],
            n_splits=5, higher_is_better=False,
        )
        assert result['ranking_table'].iloc[0]['model'] == 'C'

    def test_rejects_single_observation(self):
        """Need at least 2 observations for meaningful comparison."""
        with pytest.raises(ValueError):
            analyze_singledataset_significance(
                np.array([[0.9, 0.8]]), model_names=['A', 'B'], n_splits=5,
            )
