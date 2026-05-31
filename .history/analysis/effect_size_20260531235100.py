"""Effect size utilities for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Tuple

import numpy as np


def cohens_d(x: np.ndarray, y: np.ndarray, paired: bool = False) -> float:
    """Compute Cohen's d.

    Args:
        x, y: numeric arrays
        paired: if True compute paired (within-subject) Cohen's d using the
            mean of differences divided by the standard deviation of differences.
            If False (default) compute the pooled (independent-samples) Cohen's d
            to preserve legacy behavior.

    Returns:
        Cohen's d (float) or NaN when undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if paired:
        # Paired (within-subject) Cohen's d: mean(diff) / sd(diff)
        if len(x) != len(y) or len(x) < 2:
            return float('nan')
        diffs = x - y
        mean_diff = float(diffs.mean())
        sd_diff = float(diffs.std(ddof=1))
        if sd_diff == 0 or sd_diff != sd_diff:  # handle zero or nan
            return float('nan')
        return float(mean_diff / sd_diff)

    # Legacy: independent-samples (pooled) Cohen's d
    if len(x) < 2 or len(y) < 2:
        return float('nan')

    mean_diff = x.mean() - y.mean()
    pooled_var = ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)
    if pooled_var <= 0 or pooled_var != pooled_var:
        return 0.0
    return float(mean_diff / sqrt(pooled_var))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float('nan')

    greater = 0
    lower = 0
    for value_x in x:
        greater += int(np.sum(value_x > y))
        lower += int(np.sum(value_x < y))
    return float((greater - lower) / (len(x) * len(y)))


def interpret_cohens_d(value: float) -> str:
    magnitude = abs(value)
    if np.isnan(magnitude):
        return 'undefined'
    if magnitude < 0.2:
        return 'negligible'
    if magnitude < 0.5:
        return 'small'
    if magnitude < 0.8:
        return 'medium'
    return 'large'


def interpret_cliffs_delta(value: float) -> str:
    magnitude = abs(value)
    if np.isnan(magnitude):
        return 'undefined'
    if magnitude < 0.147:
        return 'negligible'
    if magnitude < 0.33:
        return 'small'
    if magnitude < 0.474:
        return 'medium'
    return 'large'


def effect_size_summary(x: np.ndarray, y: np.ndarray, paired: bool = False) -> Dict[str, Any]:
    d = cohens_d(x, y, paired=paired)
    delta = cliffs_delta(x, y)
    return {
        'cohens_d': d,
        'cohens_d_interpretation': interpret_cohens_d(d),
        'cliffs_delta': delta,
        'cliffs_delta_interpretation': interpret_cliffs_delta(delta),
    }
