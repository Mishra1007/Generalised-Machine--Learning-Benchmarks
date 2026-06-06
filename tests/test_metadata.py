import tempfile
from pathlib import Path
import json

from datasets.registry import register_dataset, get_registry
from experiments.metadata import compute_dataset_fingerprint
from experiments.executor import ExperimentExecutor


def test_dataset_fingerprint_consistency():
    tmpdir = Path(tempfile.mkdtemp())
    csv = tmpdir / 'data.csv'
    csv.write_text('a,b,target\n1,2,0\n3,4,1\n')

    h1 = compute_dataset_fingerprint(str(csv))
    h2 = compute_dataset_fingerprint(str(csv))
    assert h1 == h2


def test_executor_writes_metadata_and_predictions(tmp_path):
    # Create small dataset
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    csv = data_dir / 'sm.csv'
    rows = ['feat1,feat2,target']
    for i in range(10):
        rows.append('0.1,1,0')
        rows.append('0.2,2,1')
    csv.write_text('\n'.join(rows) + '\n')

    # Register dataset
    register_dataset('meta_demo', str(csv), target_column='target')

    # Build config
    cfg = {
        'dataset': {'registered_name': 'meta_demo', 'test_size': 0.5},
        'models': [{'name': 'lr', 'type': 'LogisticRegression', 'params': {'random_state': 42}}],
        'random_state': 42,
        'persist_cv_splits': True,
        'save_predictions': True,
    }

    out = tmp_path / 'results'
    exe = ExperimentExecutor(seed=42)
    res = exe.run(cfg, out_root=str(out))

    base = out / 'meta_demo'
    assert (base / 'metadata.json').exists()
    assert (base / 'metrics.json').exists()
    # Predictions file (per model)
    assert any(p.exists() for p in base.glob('predictions_*.csv'))
