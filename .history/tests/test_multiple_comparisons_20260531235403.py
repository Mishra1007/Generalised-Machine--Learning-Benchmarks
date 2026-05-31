import math
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path for test execution
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.significance import holm_bonferroni, pairwise_wilcoxon


def test_holm_bonferroni_expected_values():
    pvals = np.array([0.01, 0.04, 0.03, 0.002], dtype=float)
    adjusted, reject = holm_bonferroni(pvals, alpha=0.05)

    expected_adj = np.array([0.03, 0.06, 0.06, 0.008], dtype=float)
    expected_reject = np.array([True, False, False, True])

    assert np.allclose(adjusted, expected_adj, rtol=1e-12, atol=1e-12)
    assert np.array_equal(reject, expected_reject)

    # Cross-check with statsmodels if available
    try:
        from statsmodels.stats.multitest import multipletests
        sm_reject, sm_adj, _, _ = multipletests(pvals, alpha=0.05, method='holm')
        assert np.allclose(adjusted, sm_adj, rtol=1e-12, atol=1e-12)
        assert np.array_equal(reject, sm_reject)
    except Exception:
        # Statsmodels is optional in this environment; expected values above are from reference output.
        pass


def test_pairwise_wilcoxon_includes_adjusted_p():
    # 6 observations, 3 models
    matrix = np.array(
        [
            [0.80, 0.78, 0.76],
            [0.82, 0.79, 0.77],
            [0.81, 0.80, 0.75],
            [0.83, 0.81, 0.78],
            [0.79, 0.77, 0.74],
            [0.84, 0.82, 0.79],
        ],
        dtype=float,
    )

    df = pairwise_wilcoxon(matrix, model_names=['A', 'B', 'C'])
    assert 'p_value' in df.columns
    assert 'p_value_raw' in df.columns
    assert 'p_value_adj' in df.columns
    assert 'significant' in df.columns
    assert all(df['p_value_raw'] == df['p_value'])

    adjusted, reject = holm_bonferroni(df['p_value'].to_numpy(dtype=float), alpha=0.05)
    assert np.allclose(df['p_value_adj'].to_numpy(dtype=float), adjusted, rtol=1e-12, atol=1e-12)
    assert np.array_equal(df['significant'].to_numpy(dtype=bool), reject)


if __name__ == '__main__':
    test_holm_bonferroni_expected_values()
    test_pairwise_wilcoxon_includes_adjusted_p()
    print('Multiple comparison tests passed')
