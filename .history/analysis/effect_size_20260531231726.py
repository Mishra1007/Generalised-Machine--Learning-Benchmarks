"""Effect size utilities for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Tuple

import numpy as np


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float('nan')

    mean_diff = x.mean() - y.mean()
    pooled_var = ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)
    if pooled_var <= 0:
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


def effect_size_summary(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    d = cohens_d(x, y)
    delta = cliffs_delta(x, y)
    return {
        'cohens_d': d,
        'cohens_d_interpretation': interpret_cohens_d(d),
        'cliffs_delta': delta,
        'cliffs_delta_interpretation': interpret_cliffs_delta(delta),
    }
