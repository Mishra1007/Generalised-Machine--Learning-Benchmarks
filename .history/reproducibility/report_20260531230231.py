"""Human-readable reproducibility report rendering."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def render_reproducibility_report(manifest: Dict[str, Any]) -> str:
    dataset = manifest.get('dataset', {})
    model = manifest.get('model', {})
    reproducibility = manifest.get('reproducibility', {})
    environment = manifest.get('environment', {})
    artifacts = manifest.get('artifacts', {})

    lines = [
        '# Reproducibility Report',
        '',
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Experiment ID: {manifest.get('experiment_id')}",
        f"Run ID: {manifest.get('run_id')}",
        f"Framework Version: {manifest.get('framework_version')}",
        f"Git Commit: {manifest.get('git_commit') or 'unavailable'}",
        '',
        '## Environment Summary',
        f"Python: {environment.get('python_version')}",
        f"OS: {environment.get('os', {}).get('platform')}",
        f"CPU: {environment.get('cpu', {}).get('logical_count')} logical cores",
        f"NumPy: {environment.get('packages', {}).get('numpy')}",
        f"Pandas: {environment.get('packages', {}).get('pandas')}",
        f"Scikit-learn: {environment.get('packages', {}).get('scikit-learn')}",
        f"SciPy: {environment.get('packages', {}).get('scipy')}",
        '',
        '## Dataset Summary',
        f"Dataset: {dataset.get('name')}",
        f"Source: {dataset.get('source')}",
        f"Size: {dataset.get('size')}",
        f"Feature Count: {dataset.get('feature_count')}",
        f"Target: {dataset.get('target_variable')}",
        f"Fingerprint: {dataset.get('fingerprint')}",
        f"Class Distribution: {dataset.get('class_distribution')}",
        '',
        '## Model Summary',
        f"Models: {model.get('model_names')}",
        f"Hyperparameters: {model.get('hyperparameters')}",
        f"Search Space: {model.get('search_space')}",
        f"Final Configuration: {model.get('final_selected_configuration')}",
        '',
        '## Validation Protocol',
        f"Validation Strategy: {reproducibility.get('validation_strategy')}",
        f"CV Fold Count: {reproducibility.get('cv_fold_count')}",
        f"Repetition Count: {reproducibility.get('repetition_count')}",
        f"Random Seed: {reproducibility.get('random_seed')}",
        f"Train/Test Split: {reproducibility.get('train_test_split')}",
        f"Deterministic: {reproducibility.get('deterministic')}",
        '',
        '## Reproduction Instructions',
        '1. Use the stored experiment_manifest.json as the single source of truth.',
        '2. Restore the dataset from the recorded source and verify the fingerprint.',
        '3. Reuse the recorded random seed, validation strategy, and model configuration.',
        '4. Recreate outputs using the stored benchmark artifacts and fold metadata.',
        '',
        '## Stored Artifacts',
    ]
    for name, value in artifacts.items():
        lines.append(f"- {name}: {value}")
    lines.append('')
    return '\n'.join(lines)


def write_report(path: str | Path, manifest: Dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_reproducibility_report(manifest), encoding='utf8')
    return str(target)


def load_report(path: str | Path) -> str:
    return Path(path).read_text(encoding='utf8')
