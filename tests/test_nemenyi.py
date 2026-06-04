import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.significance import nemenyi_critical_difference, nemenyi_q_alpha, nemenyi_test


def _reference_ranks(matrix, higher_is_better=True):
    values = np.asarray(matrix, dtype=float)
    return np.array(
        [stats.rankdata(-row if higher_is_better else row, method='average') for row in values],
        dtype=float,
    )


def _reference_p_values(matrix, higher_is_better=True):
    ranks = _reference_ranks(matrix, higher_is_better=higher_is_better)
    avg_ranks = ranks.mean(axis=0)
    n, k = ranks.shape
    standard_error = math.sqrt(k * (k + 1) / (6.0 * n))
    p_values = np.ones((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            statistic = abs(avg_ranks[i] - avg_ranks[j]) / standard_error * math.sqrt(2.0)
            p_values[i, j] = p_values[j, i] = stats.studentized_range.sf(statistic, k, np.inf)
    return avg_ranks, p_values


def test_nemenyi_q_values_match_demsar_table():
    # Demsar 2006, Table 5(a): two-tailed Nemenyi q values.
    expected_005 = {3: 2.343, 5: 2.728, 10: 3.164}
    expected_010 = {3: 2.052, 5: 2.459, 10: 2.920}

    for k, expected in expected_005.items():
        assert math.isclose(nemenyi_q_alpha(k, alpha=0.05), expected, abs_tol=1e-3)
    for k, expected in expected_010.items():
        assert math.isclose(nemenyi_q_alpha(k, alpha=0.10), expected, abs_tol=1e-3)


def test_nemenyi_critical_difference_matches_demsar_example():
    # Demsar 2006, Table 6 example: k=4, N=14, alpha=.05 has CD about 1.25.
    cd = nemenyi_critical_difference(4, 14, alpha=0.05)
    expected = 2.569 * math.sqrt(4 * 5 / (6 * 14))

    assert math.isclose(cd, expected, abs_tol=5e-4)
    assert math.isclose(cd, 1.25, abs_tol=0.01)


def test_rank_generation_matches_reference_with_ties():
    matrix = np.array(
        [
            [0.90, 0.90, 0.70],
            [0.80, 0.75, 0.75],
            [0.60, 0.60, 0.60],
            [0.95, 0.85, 0.85],
        ],
        dtype=float,
    )
    result = nemenyi_test(matrix, model_names=['A', 'B', 'C'])
    expected_ranks = _reference_ranks(matrix)
    expected_avg = expected_ranks.mean(axis=0)

    assert np.allclose(
        [result['average_ranks']['A'], result['average_ranks']['B'], result['average_ranks']['C']],
        expected_avg,
        rtol=1e-12,
        atol=1e-12,
    )


def test_significance_decisions_match_studentized_range_reference_for_3_models():
    matrix = np.array(
        [
            [0.90, 0.80, 0.70],
            [0.85, 0.82, 0.72],
            [0.88, 0.81, 0.73],
            [0.87, 0.83, 0.71],
            [0.91, 0.79, 0.74],
            [0.89, 0.84, 0.70],
        ],
        dtype=float,
    )

    result = nemenyi_test(matrix, model_names=['A', 'B', 'C'], alpha=0.05)
    _, expected_p = _reference_p_values(matrix)
    expected_significant = expected_p <= 0.05
    np.fill_diagonal(expected_significant, False)

    assert np.allclose(result['p_value_matrix'].to_numpy(dtype=float), expected_p, rtol=1e-12, atol=1e-12)
    assert np.array_equal(result['significance_matrix'].to_numpy(dtype=bool), expected_significant)


def test_significance_decisions_match_studentized_range_reference_for_5_models():
    matrix = np.array(
        [
            [0.95, 0.90, 0.84, 0.78, 0.70],
            [0.96, 0.89, 0.85, 0.79, 0.69],
            [0.94, 0.91, 0.83, 0.77, 0.71],
            [0.97, 0.88, 0.86, 0.80, 0.68],
            [0.93, 0.92, 0.82, 0.76, 0.72],
            [0.98, 0.87, 0.87, 0.81, 0.67],
            [0.92, 0.93, 0.81, 0.75, 0.73],
            [0.99, 0.86, 0.88, 0.82, 0.66],
        ],
        dtype=float,
    )

    result = nemenyi_test(matrix, model_names=list('ABCDE'), alpha=0.05)
    _, expected_p = _reference_p_values(matrix)
    expected_significant = expected_p <= 0.05
    np.fill_diagonal(expected_significant, False)

    assert np.allclose(result['p_value_matrix'].to_numpy(dtype=float), expected_p, rtol=1e-12, atol=1e-12)
    assert np.array_equal(result['significance_matrix'].to_numpy(dtype=bool), expected_significant)


def test_significance_decisions_match_studentized_range_reference_for_10_models_small_sample():
    matrix = np.array(
        [
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [9, 10, 8, 7, 6, 5, 4, 3, 2, 1],
            [10, 8, 9, 7, 6, 5, 4, 3, 2, 1],
            [9, 8, 10, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=float,
    )
    names = [f'M{i}' for i in range(10)]

    result = nemenyi_test(matrix, model_names=names, alpha=0.10)
    _, expected_p = _reference_p_values(matrix)
    expected_significant = expected_p <= 0.10
    np.fill_diagonal(expected_significant, False)

    assert result['significance_matrix'].shape == (10, 10)
    assert np.allclose(result['p_value_matrix'].to_numpy(dtype=float), expected_p, rtol=1e-12, atol=1e-12)
    assert np.array_equal(result['significance_matrix'].to_numpy(dtype=bool), expected_significant)


def test_equal_performance_produces_equal_ranks_and_no_significance():
    matrix = np.full((5, 5), 0.75, dtype=float)
    result = nemenyi_test(matrix, model_names=list('ABCDE'), alpha=0.05)

    assert all(math.isclose(rank, 3.0, abs_tol=1e-12) for rank in result['average_ranks'].values())
    assert np.allclose(result['p_value_matrix'].to_numpy(dtype=float), np.ones((5, 5)), rtol=1e-12, atol=1e-12)
    assert not result['significance_matrix'].to_numpy(dtype=bool).any()
