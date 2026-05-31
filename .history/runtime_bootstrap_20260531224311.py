"""Early-binding helpers for reproducible benchmark launches.

This module is intentionally stdlib-only so it can be imported before any
framework modules without triggering project imports.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


EARLY_BINDING_VARS = (
    'PYTHONHASHSEED',
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
)


def build_early_binding_env(seed: Optional[int] = None, thread_count: int = 1) -> Dict[str, str]:
    """Build environment values that must exist before framework imports."""
    hash_seed = os.environ.get('PYTHONHASHSEED')
    if seed is not None:
        hash_seed = str(seed)
    elif not hash_seed:
        hash_seed = '0'

    thread_value = str(thread_count)
    return {
        'PYTHONHASHSEED': hash_seed,
        'OMP_NUM_THREADS': thread_value,
        'OPENBLAS_NUM_THREADS': thread_value,
        'MKL_NUM_THREADS': thread_value,
    }


def apply_early_binding(seed: Optional[int] = None, thread_count: int = 1) -> Dict[str, str]:
    """Apply early-binding environment variables in the current process."""
    values = build_early_binding_env(seed=seed, thread_count=thread_count)
    for key, value in values.items():
        os.environ[key] = value
    return values
