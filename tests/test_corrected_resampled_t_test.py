"""Tests for corrected_resampled_t_test (Nadeau-Bengio 2003).

These tests verify statistical correctness, input validation, and edge-case
behaviour of the corrected resampled t-test used for single-dataset model
comparison under repeated k-fold cross-validation.

Each test category protects a specific statistical or engineering property:

1. INPUT VALIDATION — ensures the function rejects malformed inputs before
   any computation occurs, preventing silent wrong results.

2. ZERO VARIANCE — verifies the guard against division-by-zero when all
   paired differences are identical (zero sample variance).

3. STATISTICAL SANITY — confirms the output obeys basic statistical
   axioms: p-values bounded in [0,1], monotonicity with effect size,
   and that the Nadeau-Bengio correction is more conservative than a
   naive paired t-test (wider confidence = larger p-values).

4. REGRESSION PROTECTION — locks the exact formula so that refactors
   cannot silently change the correction factor or variance estimator.
"""

import math

import numpy as np
import pytest
from scipy import stats

from analysis.statistics import corrected_resampled_t_test


# ===================================================================
# 1. INPUT VALIDATION
#    Protect: function rejects invalid inputs with clear errors.
# ===================================================================

class TestInputValidation:
    """The function must reject mismatched lengths, empty inputs,
    and n_splits <= 1 before performing any computation."""

    def test_mismatched_lengths_raises(self):
        """Paired test requires equal-length vectors."""
        with pytest.raises(ValueError, match='equal length'):
            corrected_resampled_t_test([1, 2, 3], [1, 2], n_splits=5)

    def test_empty_inputs_raises(self):
        """Zero observations cannot produce a test statistic."""
        with pytest.raises(ValueError, match='at least one'):
            corrected_resampled_t_test([], [], n_splits=5)

    def test_n_splits_zero_raises(self):
        """n_splits=0 would produce division by zero in 1/(K-1)."""
        with pytest.raises(ValueError, match='n_splits > 1'):
            corrected_resampled_t_test([1, 2], [3, 4], n_splits=0)

    def test_n_splits_one_raises(self):
        """n_splits=1 makes the correction factor 1/(1-1) = 1/0."""
        with pytest.raises(ValueError, match='n_splits > 1'):
            corrected_resampled_t_test([1, 2], [3, 4], n_splits=1)

    def test_n_splits_negative_raises(self):
        """Negative n_splits is nonsensical."""
        with pytest.raises(ValueError, match='n_splits > 1'):
            corrected_resampled_t_test([1, 2], [3, 4], n_splits=-3)


# ===================================================================
# 2. ZERO VARIANCE CASES
#    Protect: no division-by-zero; sensible outputs when var(diff)=0.
# ===================================================================

class TestZeroVariance:
    """When all paired differences are identical, sample variance is 0.
    The function must not crash and must return interpretable results."""

    def test_identical_vectors_returns_p_one(self):
        """If x == y everywhere, mean_diff=0 and var_diff=0.
        There is no evidence of a difference, so p_value should be 1.0."""
        x = [0.8, 0.8, 0.8, 0.8, 0.8]
        result = corrected_resampled_t_test(x, x, n_splits=5)
        assert result['statistic'] == 0.0
        assert result['p_value'] == 1.0
        assert result['significant'] is False

    def test_constant_positive_difference_returns_plus_inf(self):
        """If every diff is identical and positive, var_diff=0 but
        mean_diff > 0. The function should return t=+inf and p=0.0."""
        x = [0.9, 0.9, 0.9, 0.9, 0.9]
        y = [0.8, 0.8, 0.8, 0.8, 0.8]
        result = corrected_resampled_t_test(x, y, n_splits=5)
        assert result['statistic'] == float('inf')
        assert result['p_value'] == 0.0
        assert result['significant'] is True

    def test_constant_negative_difference_returns_minus_inf(self):
        """If every diff is identical and negative, var_diff=0 but
        mean_diff < 0. The function should return t=-inf and p=0.0."""
        x = [0.8, 0.8, 0.8, 0.8, 0.8]
        y = [0.9, 0.9, 0.9, 0.9, 0.9]
        result = corrected_resampled_t_test(x, y, n_splits=5)
        assert result['statistic'] == float('-inf')
        assert result['p_value'] == 0.0
        assert result['significant'] is True


# ===================================================================
# 3. STATISTICAL SANITY CHECKS
#    Protect: basic axioms that any valid statistical test must satisfy.
# ===================================================================

class TestStatisticalSanity:
    """These tests verify properties that must hold for ANY correct
    implementation of a two-tailed t-test, regardless of correction."""

    def test_p_value_bounded_zero_one(self):
        """p-values must lie in [0, 1] by definition."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.normal(0.8, 0.05, size=10)
            y = rng.normal(0.78, 0.05, size=10)
            result = corrected_resampled_t_test(x, y, n_splits=5)
            assert 0.0 <= result['p_value'] <= 1.0

    def test_larger_difference_produces_smaller_p_value(self):
        """Increasing the effect size while keeping similar variance
        must yield smaller p-values (stronger evidence).

        We use data with natural variance in the paired differences
        to avoid the zero-variance guard path."""
        # Small effect: mean diff ≈ 0.02, with variance
        x_small = [0.80, 0.82, 0.81, 0.83, 0.79, 0.84, 0.80, 0.82, 0.81, 0.83]
        y_small = [0.78, 0.81, 0.79, 0.82, 0.77, 0.83, 0.79, 0.80, 0.80, 0.81]

        # Large effect: mean diff ≈ 0.05, with similar variance
        y_large = [0.75, 0.77, 0.76, 0.78, 0.74, 0.79, 0.75, 0.77, 0.76, 0.78]

        p_small = corrected_resampled_t_test(x_small, y_small, n_splits=5)['p_value']
        p_large = corrected_resampled_t_test(x_small, y_large, n_splits=5)['p_value']

        assert p_large < p_small, (
            f'Larger effect should produce smaller p-value: '
            f'p_large={p_large:.6f} vs p_small={p_small:.6f}'
        )

    def test_corrected_test_more_conservative_than_uncorrected(self):
        """The Nadeau-Bengio correction inflates the variance estimate,
        which should produce LARGER p-values (more conservative) than
        a standard paired t-test on the same observations.

        This is the core property the correction exists to enforce:
        preventing false positives from non-independent CV folds."""
        x = np.array([0.85, 0.87, 0.83, 0.86, 0.84,
                       0.88, 0.82, 0.85, 0.87, 0.84])
        y = np.array([0.80, 0.82, 0.79, 0.81, 0.80,
                       0.83, 0.78, 0.80, 0.82, 0.79])

        corrected = corrected_resampled_t_test(x, y, n_splits=5)
        # Standard paired t-test (uncorrected)
        _, uncorrected_p = stats.ttest_rel(x, y)

        assert corrected['p_value'] >= uncorrected_p, (
            f'Corrected p-value ({corrected["p_value"]:.6f}) should be >= '
            f'uncorrected ({uncorrected_p:.6f}). '
            f'The correction inflates variance to reduce Type I error.'
        )

    def test_significant_flag_matches_threshold(self):
        """The 'significant' flag must exactly reflect p_value < 0.05.

        Uses data with natural variance so the normal code path runs
        (not the zero-variance guard)."""
        x = [0.90, 0.88, 0.92, 0.85, 0.91, 0.87, 0.93, 0.86, 0.89, 0.90]
        y = [0.80, 0.82, 0.79, 0.83, 0.81, 0.78, 0.84, 0.80, 0.82, 0.79]
        result = corrected_resampled_t_test(x, y, n_splits=5)
        # Verify the flag is consistent with the threshold
        assert result['significant'] == (result['p_value'] < 0.05)

    def test_return_keys_complete(self):
        """Output dict must contain exactly the documented keys."""
        result = corrected_resampled_t_test([1, 2, 3], [4, 5, 6], n_splits=3)
        assert set(result.keys()) == {'statistic', 'p_value', 'significant'}


# ===================================================================
# 4. REGRESSION PROTECTION
#    Protect: exact formula so refactors cannot silently change it.
# ===================================================================

class TestRegressionProtection:
    """These tests lock the Nadeau-Bengio formula to known manual
    computations, preventing accidental changes to the correction
    factor or variance estimator."""

    def test_correction_factor_uses_k_minus_one(self):
        """Verify the correction factor is (1/n) + (1/(K-1)).

        For n=10 observations with K=5 folds:
          correction = 1/10 + 1/4 = 0.35

        This is NOT 1/(n_test) as in the original Nadeau-Bengio paper's
        train/test formulation — we use 1/(K-1) for repeated K-fold CV
        as recommended by the literature review."""
        x = np.array([0.80, 0.82, 0.81, 0.83, 0.79,
                       0.84, 0.80, 0.82, 0.81, 0.83])
        y = np.array([0.78, 0.79, 0.80, 0.81, 0.77,
                       0.82, 0.78, 0.80, 0.79, 0.81])

        n = 10
        k = 5
        diff = x - y
        mean_diff = diff.mean()
        var_diff = np.var(diff, ddof=1)
        expected_correction = (1.0 / n) + (1.0 / (k - 1))
        corrected_var = var_diff * expected_correction
        expected_t = mean_diff / math.sqrt(corrected_var)
        expected_p = 2.0 * stats.t.cdf(-abs(expected_t), df=n - 1)

        result = corrected_resampled_t_test(x, y, n_splits=k)

        assert math.isclose(result['statistic'], expected_t, rel_tol=1e-12), \
            f"t-stat mismatch: {result['statistic']} vs {expected_t}"
        assert math.isclose(result['p_value'], expected_p, rel_tol=1e-12), \
            f"p-value mismatch: {result['p_value']} vs {expected_p}"

    def test_ddof_one_is_default(self):
        """The default ddof=1 produces unbiased sample variance.
        Verify that explicitly passing ddof=1 gives the same result
        as the default."""
        x = [0.9, 0.85, 0.88, 0.92, 0.87]
        y = [0.8, 0.78, 0.82, 0.81, 0.79]

        default_result = corrected_resampled_t_test(x, y, n_splits=5)
        explicit_result = corrected_resampled_t_test(x, y, n_splits=5, ddof=1)

        assert default_result == explicit_result

    def test_ddof_zero_gives_different_result(self):
        """Using ddof=0 (biased/population variance) should produce a
        different statistic than ddof=1, confirming ddof is actually
        used in the computation."""
        x = [0.9, 0.85, 0.88, 0.92, 0.87]
        y = [0.8, 0.78, 0.82, 0.81, 0.79]

        result_ddof1 = corrected_resampled_t_test(x, y, n_splits=5, ddof=1)
        result_ddof0 = corrected_resampled_t_test(x, y, n_splits=5, ddof=0)

        assert not math.isclose(
            result_ddof1['statistic'], result_ddof0['statistic'], rel_tol=1e-6
        ), 'ddof should affect the variance estimate and thus the t-statistic'

    def test_varying_n_splits_changes_correction(self):
        """Different n_splits must produce different correction factors
        and therefore different t-statistics and p-values.

        K=2: correction = 1/10 + 1/1 = 1.1   (very conservative)
        K=5: correction = 1/10 + 1/4 = 0.35
        K=10: correction = 1/10 + 1/9 ≈ 0.211 (least conservative)

        Larger K -> smaller correction -> larger |t| -> smaller p."""
        x = np.array([0.85, 0.87, 0.83, 0.86, 0.84,
                       0.88, 0.82, 0.85, 0.87, 0.84])
        y = np.array([0.80, 0.82, 0.79, 0.81, 0.80,
                       0.83, 0.78, 0.80, 0.82, 0.79])

        p_k2 = corrected_resampled_t_test(x, y, n_splits=2)['p_value']
        p_k5 = corrected_resampled_t_test(x, y, n_splits=5)['p_value']
        p_k10 = corrected_resampled_t_test(x, y, n_splits=10)['p_value']

        assert p_k2 > p_k5 > p_k10, (
            f'Larger K should produce smaller correction and smaller p: '
            f'p(K=2)={p_k2:.6f}, p(K=5)={p_k5:.6f}, p(K=10)={p_k10:.6f}'
        )
