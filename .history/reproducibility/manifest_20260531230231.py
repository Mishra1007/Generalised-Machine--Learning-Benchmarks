"""Manifest schema and serialization helpers."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentManifest:
    experiment_id: str
    run_id: str
    timestamp: str
    framework_version: str
    git_commit: Optional[str]
    dataset: Dict[str, Any]
    reproducibility: Dict[str, Any]
    model: Dict[str, Any]
    environment: Dict[str, Any]
    artifacts: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_manifest(manifest: Dict[str, Any]) -> None:
    required_keys = [
        'experiment_id',
        'run_id',
        'timestamp',
        'framework_version',
        'dataset',
        'reproducibility',
        'model',
        'environment',
        'artifacts',
    ]
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        raise ValueError(f'Missing required manifest fields: {missing}')


def write_json(path: str | Path, payload: Dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=str(target.parent))
    try:
        with open(fd, 'w', encoding='utf8', closefd=True) as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            import os

            os.fsync(fh.fileno())
        import os

        os.replace(tmp_name, target)
    except Exception:
        try:
            import os

            os.unlink(tmp_name)
        except Exception:
            pass
        raise
    return str(target)


def load_manifest(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf8') as fh:
        return json.load(fh)
