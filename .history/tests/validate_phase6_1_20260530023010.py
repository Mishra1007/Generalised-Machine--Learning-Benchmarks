import tempfile
import json
import shutil
from pathlib import Path
import filecmp
import os

from datasets.registry import register_dataset
from experiments.executor import ExperimentExecutor
from validation.fold_manager import FoldManager
import metrics.storage as storage


def write_csv(path: Path, rows):
    path.write_text('\n'.join(rows) + '\n')


def test_metadata_validation_rejects():
    tmp = Path(tempfile.mkdtemp(prefix='meta_val_'))
    try:
        try:
            storage.save_experiment_results('ds', {}, out_root=str(tmp), config={}, seed=42, metadata={'foo': 'bar'})
            return False, 'expected exception not raised'
        except Exception:
            return True, 'raised as expected'
    finally:
        shutil.rmtree(tmp)


def test_config_snapshot_immutable():
    tmp = Path(tempfile.mkdtemp(prefix='cfgsnap_'))
    data = tmp / 'data'
    data.mkdir()
    csv = data / 'd.csv'
    rows = ['a,b,target'] + [f"{i},{i%2},{i%2}" for i in range(50)]
    write_csv(csv, rows)
    register_dataset('phase6_ds', str(csv), target_column='target')

    cfg = {'dataset': {'registered_name': 'phase6_ds', 'test_size': 0.5}, 'models': [{'name': 'lr', 'type': 'LogisticRegression'}], 'random_state': 123, 'deterministic': True}

    exe = ExperimentExecutor(seed=123)
    out = tmp / 'results1'
    res = exe.run(cfg, out_root=str(out))

    meta_path = out / 'phase6_ds' / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    meta_snapshot = meta.get('config_snapshot')
    ok = meta_snapshot is not None
    return ok, 'config snapshot present' if ok else 'missing config snapshot'


def test_environment_snapshot_complete():
    tmp = Path(tempfile.mkdtemp(prefix='env_'))
    data = tmp / 'data'
    data.mkdir()
    csv = data / 'd.csv'
    rows = ['a,b,target'] + [f"{i},{i%2},{i%2}" for i in range(20)]
    write_csv(csv, rows)
    register_dataset('env_ds', str(csv), target_column='target')

    cfg = {'dataset': {'registered_name': 'env_ds'}, 'models': ['lr'], 'random_state': 1}
    exe = ExperimentExecutor(seed=1)
    out = tmp / 'results'
    res = exe.run(cfg, out_root=str(out))
    meta = json.loads((out / 'env_ds' / 'metadata.json').read_text())
    env = meta.get('environment') or meta.get('env') or {}
    required = ['python_version', 'os', 'cpu_count', 'pip_freeze', 'library_versions']
    missing = [k for k in required if k not in env]
    return (len(missing) == 0), f"missing: {missing}" if missing else 'env snapshot complete'


def test_deterministic_mode_identical_runs(n=10):
    tmp = Path(tempfile.mkdtemp(prefix='det_'))
    data = tmp / 'data'
    data.mkdir()
    csv = data / 'd.csv'
    rows = ['feat1,feat2,target'] + [f"{i*0.1},{i%3},{i%2}" for i in range(200)]
    write_csv(csv, rows)
    register_dataset('det_ds', str(csv), target_column='target')

    cfg = {'dataset': {'registered_name': 'det_ds'}, 'models': ['lr'], 'random_state': 42, 'deterministic': True, 'persist_cv_splits': True, 'save_predictions': True}

    outputs = []
    for i in range(n):
        out = tmp / f'res_{i}'
        exe = ExperimentExecutor(seed=42)
        exe.run(cfg, out_root=str(out))
        base = out / 'det_ds'
        outputs.append({
            'metadata': (base / 'metadata.json').read_text(),
            'metrics': (base / 'metrics.json').read_text() if (base / 'metrics.json').exists() else '',
            'preds': sorted([p.read_text() for p in base.glob('predictions_*.csv')])
        })

    first = outputs[0]
    for o in outputs[1:]:
        if o['metadata'] != first['metadata'] or o['metrics'] != first['metrics'] or o['preds'] != first['preds']:
            return False, 'differences found between runs'
    return True, 'identical across runs'


def test_fold_persistence_reconstructs():
    tmp = Path(tempfile.mkdtemp(prefix='fold_'))
    data = tmp / 'data'
    data.mkdir()
    csv = data / 'd.csv'
    rows = ['a,b,target'] + [f"{i},{i%3},{i%2}" for i in range(120)]
    write_csv(csv, rows)
    register_dataset('fold_ds', str(csv), target_column='target')

    cfg = {'dataset': {'registered_name': 'fold_ds'}, 'models': ['lr'], 'random_state': 7, 'persist_cv_splits': True}
    exe = ExperimentExecutor(seed=7)
    out = tmp / 'results'
    exe.run(cfg, out_root=str(out))
    base = out / 'fold_ds'
    meta = json.loads((base / 'metadata.json').read_text())
    folds = meta.get('folds', [])

    from datasets.loaders import DatasetLoader
    loader = DatasetLoader()
    X, y, _ = loader.load_csv(str(csv), 'target')
    fm = FoldManager(n_splits=exe.validator.n_splits, n_repetitions=exe.validator.n_repetitions, random_state=exe.validator.random_state)
    all_info = fm.get_all_fold_info(X.values, y.values)
    if len(folds) != len(all_info):
        return False, f"fold length mismatch {len(folds)} vs {len(all_info)}"
    for i, f in enumerate(folds):
        if list(f['test_indices']) != list(all_info[i]['test_indices']):
            return False, f"mismatch fold {i}"
    return True, 'folds reconstructed exactly'


def main():
    results = {}
    results['metadata_validation'] = test_metadata_validation_rejects()
    results['config_snapshot'] = test_config_snapshot_immutable()
    results['environment_snapshot'] = test_environment_snapshot_complete()
    results['deterministic_runs'] = test_deterministic_mode_identical_runs(10)
    results['fold_persistence'] = test_fold_persistence_reconstructs()

    print('PHASE 6.1 VALIDATION RESULTS')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
