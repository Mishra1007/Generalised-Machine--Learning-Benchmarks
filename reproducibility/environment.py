"""Environment capture helpers for reproducible benchmark runs."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PACKAGE_NAMES = ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib')


def _version_for_package(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        try:
            module = importlib.import_module(name)
            return getattr(module, '__version__', 'unknown')
        except Exception:
            return 'not-installed'


def capture_package_versions(package_names: Iterable[str] = PACKAGE_NAMES) -> Dict[str, str]:
    return {name: _version_for_package(name) for name in package_names}


def capture_git_info(project_root: Optional[str] = None) -> Dict[str, Any]:
    info = {'git_commit': None, 'git_dirty': None}
    if not project_root:
        return info

    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        info['git_commit'] = commit.decode().strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=project_root)
        info['git_dirty'] = bool(status.strip())
    except Exception:
        pass
    return info


def capture_environment() -> Dict[str, Any]:
    uname = platform.uname()
    installed_packages = []
    try:
        for dist in importlib_metadata.distributions():
            installed_packages.append({
                'name': dist.metadata['Name'] if 'Name' in dist.metadata else dist.name,
                'version': dist.version,
            })
    except Exception:
        installed_packages = []

    pip_freeze = []
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'],
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()
    except Exception:
        pip_freeze = []

    package_versions = capture_package_versions()

    return {
        'python_version': sys.version.replace('\n', ' '),
        'python_executable': sys.executable,
        'python_implementation': platform.python_implementation(),
        'os': {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'cpu': {
            'logical_count': os.cpu_count(),
            'uname': uname._asdict(),
        },
        'packages': package_versions,
        'library_versions': package_versions,
        'installed_packages': installed_packages,
        'pip_freeze': sorted(pip_freeze),
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'thread_environment': {
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
            'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        },
        'launcher_environment': {
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
            'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        },
    }
