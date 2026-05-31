import math
import numpy as np

from analysis.effect_size import cohens_d, cliffs_delta, effect_size_summary


def test_independent_cohens_d_matches_manual():
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    pooled_var = (((len(x) - 1) * x.var(ddof=1)) + ((len(y) - 1) * y.var(ddof=1))) / (len(x) + len(y) - 2)
    expected_d = (x.mean() - y.mean()) / math.sqrt(pooled_var)

    d = cohens_d(x, y, paired=False)
    assert math.isclose(d, expected_d, rel_tol=1e-12, abs_tol=1e-12)


def test_paired_cohens_d_simple():
    # differences: [1.0, 1.0, 1.0, 1.0] -> sd = 0 -> NaN expected
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = cohens_d(x, y, paired=True)
    assert math.isnan(d)

    # non-constant differences
    x2 = np.array([2.0, 3.0, 4.0, 6.0])
    y2 = np.array([1.0, 2.0, 3.0, 4.0])
    diffs = x2 - y2
    expected = diffs.mean() / diffs.std(ddof=1)
    d2 = cohens_d(x2, y2, paired=True)
    assert math.isclose(d2, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_effect_size_summary_paired_flag_propagates():
    x = np.array([2.0, 3.0, 4.0, 6.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    summ = effect_size_summary(x, y, paired=True)
    assert 'cohens_d' in summ and 'cliffs_delta' in summ
    assert math.isfinite(summ['cohens_d'])


if __name__ == '__main__':
    test_independent_cohens_d_matches_manual()
    test_paired_cohens_d_simple()
    test_effect_size_summary_paired_flag_propagates()
    print('Effect size tests passed')
