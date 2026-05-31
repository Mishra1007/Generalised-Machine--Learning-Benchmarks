import math

import numpy as np
from scipy import stats

from analysis.effect_size import cliffs_delta, cohens_d, effect_size_summary
from analysis.significance import global_significance_analysis, nemenyi_critical_difference, nemenyi_test
from analysis.statistics import bootstrap_confidence_interval, friedman_test, mean_confidence_interval, wilcoxon_signed_rank


def test_friedman_matches_scipy_reference():
    matrix = np.array(
        [
            [0.80, 0.78, 0.76],
            [0.82, 0.79, 0.77],
            [0.81, 0.80, 0.75],
            [0.83, 0.81, 0.78],
        ],
        dtype=float,
    )

    expected = stats.friedmanchisquare(matrix[:, 0], matrix[:, 1], matrix[:, 2])
    actual = friedman_test(matrix)

    assert math.isclose(actual['statistic'], expected.statistic, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(actual['p_value'], expected.pvalue, rel_tol=1e-12, abs_tol=1e-12)


def test_wilcoxon_matches_scipy_reference():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.9, 2.1, 2.8, 4.1, 4.7])

    expected = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', mode='auto')
    actual = wilcoxon_signed_rank(x, y)

    assert math.isclose(actual['statistic'], expected.statistic, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(actual['p_value'], expected.pvalue, rel_tol=1e-12, abs_tol=1e-12)


def test_confidence_interval_helpers_behave_sensibly():
    values = np.array([0.8, 0.85, 0.82, 0.83, 0.81], dtype=float)

    mean_ci = mean_confidence_interval(values, confidence=0.95)
    boot_ci = bootstrap_confidence_interval(values, confidence=0.95, n_bootstrap=2000, random_state=7)

    assert math.isclose(mean_ci['mean'], values.mean(), rel_tol=1e-12, abs_tol=1e-12)
    assert mean_ci['lower'] <= mean_ci['mean'] <= mean_ci['upper']
    assert boot_ci['lower'] <= boot_ci['mean'] <= boot_ci['upper']


def test_effect_size_helpers_match_manual_values():
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    d = cohens_d(x, y)
    delta = cliffs_delta(x, y)
    summary = effect_size_summary(x, y)

    pooled_var = (((len(x) - 1) * x.var(ddof=1)) + ((len(y) - 1) * y.var(ddof=1))) / (len(x) + len(y) - 2)
    expected_d = (x.mean() - y.mean()) / math.sqrt(pooled_var)
    expected_delta = 0.4375

    assert math.isclose(d, expected_d, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(delta, expected_delta, rel_tol=1e-12, abs_tol=1e-12)
    assert summary['cohens_d_interpretation'] in {'small', 'medium', 'large', 'negligible'}
    assert summary['cliffs_delta_interpretation'] in {'small', 'medium', 'large', 'negligible'}


def test_posthoc_analysis_shapes_are_consistent():
    matrix = np.array(
        [
            [0.80, 0.78, 0.76],
            [0.82, 0.79, 0.77],
            [0.81, 0.80, 0.75],
            [0.83, 0.81, 0.78],
        ],
        dtype=float,
    )

    analysis = global_significance_analysis(matrix, model_names=['A', 'B', 'C'])
    nemenyi = analysis['nemenyi']

    assert nemenyi['significance_matrix'].shape == (3, 3)
    assert nemenyi['ranking_table'].shape[0] == 3
    assert nemenyi_critical_difference(3, 4) > 0
    assert not nemenyi_test(matrix, model_names=['A', 'B', 'C'])['ranking_table'].empty
