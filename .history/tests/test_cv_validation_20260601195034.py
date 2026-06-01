import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.fold_manager import FoldManager


def test_cv_validation_raises_for_small_samples():
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1])
    fm = FoldManager(n_splits=5, n_repetitions=1, random_state=42)
    try:
        fm.validate_fold_indices(X, y)
        assert False, 'Expected validation to fail for n_splits > n_samples'
    except ValueError as exc:
        assert 'n_splits=5' in str(exc)


def test_cv_validation_raises_for_small_class():
    X = np.zeros((6, 2))
    y = np.array([0, 0, 0, 0, 1, 1])
    fm = FoldManager(n_splits=3, n_repetitions=1, random_state=42)
    try:
        fm.validate_fold_indices(X, y)
        assert False, 'Expected validation to fail for n_splits > min class count'
    except ValueError as exc:
        assert 'smallest class count' in str(exc)


if __name__ == '__main__':
    test_cv_validation_raises_for_small_samples()
    test_cv_validation_raises_for_small_class()
    print('CV validation tests passed')
