from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import hashlib
import json
import sys
import importlib
import importlib.metadata as importlib_metadata
from pathlib import Path


def compute_dataset_fingerprint(filepath: str, target_column: Optional[str] = None) -> str:
    """Compute a content-based deterministic hash for a dataset CSV file.

    The function reads the CSV in a canonical way and returns a hex sha256.
    It is independent of the file path (based on content).
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(filepath)

    # Read as text and normalize line endings to LF
    data = p.read_text(encoding='utf8')
    # Optionally remove leading/trailing whitespace lines
    data = '\n'.join([line.rstrip() for line in data.splitlines()])

    # If target_column provided, we can try to remove the column header order differences,
    # but for simplicity we hash the full content; registries should ensure canonical CSVs.

    h = hashlib.sha256()
    h.update(data.encode('utf8'))
    return h.hexdigest()


def capture_library_versions(packages: List[str]) -> Dict[str, str]:
    out = {}
    for pkg in packages:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except Exception:
            try:
                mod = importlib.import_module(pkg)
                out[pkg] = getattr(mod, '__version__', 'unknown')
            except Exception:
                out[pkg] = 'not-installed'
    return out


@dataclass
class ExperimentMetadata:
    experiment_id: str
    timestamp: str
    framework_version: str
    dataset_name: str
    dataset_hash: str
    dataset_size: int
    target_column: Optional[str]
    preprocessing_pipeline: Dict[str, Any]
    model_names: List[str]
    model_hyperparameters: Dict[str, Any]
    validation_strategy: Dict[str, Any]
    folds: List[Dict[str, Any]]
    repetitions: int
    random_seed: int
    config_snapshot: Dict[str, Any]
    python_version: str
    library_versions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
