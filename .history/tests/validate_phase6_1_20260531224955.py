import tempfile
import json
import shutil
from pathlib import Path
import filecmp
import os

from datasets.registry import register_dataset
from preprocessing.data_preparation import prepare_dataset
from experiments.executor import ExperimentExecutor
from experiments.runner import ExperimentRunner
from validation.fold_manager import FoldManager
import metrics.storage as storage
from runtime_bootstrap import build_early_binding_env


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
    try:
        cfg_path = tmp / 'config.yaml'
        cfg_path.write_text(
            'dataset:\n  registered_name: demo_ds\nmodels:\n  - name: lr\n    type: LogisticRegression\nrandom_state: 123\n',
            encoding='utf8',
        )
        runner = ExperimentRunner()
        loaded = runner._load_config(str(cfg_path))
        snapshot_before = json.dumps(loaded['_config_snapshot'], sort_keys=True, default=str)
        loaded['dataset']['registered_name'] = 'mutated_ds'
        snapshot_after = json.dumps(loaded['_config_snapshot'], sort_keys=True, default=str)
        ok = snapshot_before == snapshot_after and loaded['_config_snapshot']['dataset']['registered_name'] == 'demo_ds'
        return ok, 'deepcopy snapshot preserved' if ok else 'snapshot drifted'
    finally:
        shutil.rmtree(tmp)


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
    required = ['python_version', 'python_executable', 'os', 'cpu', 'pip_freeze', 'packages', 'thread_environment', 'launcher_environment']
    missing = [k for k in required if k not in env]
    package_names = ['numpy', 'pandas', 'scikit-learn', 'joblib']
    missing_packages = [name for name in package_names if name not in env.get('packages', {})]
    thread_names = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']
    missing_threads = [name for name in thread_names if name not in env.get('thread_environment', {})]
    ok = len(missing) == 0 and len(missing_packages) == 0 and len(missing_threads) == 0
    details = {'missing': missing, 'missing_packages': missing_packages, 'missing_threads': missing_threads}
    return ok, json.dumps(details) if not ok else 'env snapshot complete'


def test_early_binding_launcher_values():
    values = build_early_binding_env(seed=99)
    required = {
        'PYTHONHASHSEED': '99',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
    }
    ok = all(values.get(key) == expected for key, expected in required.items())
    return ok, json.dumps(values, sort_keys=True)


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
        meta = json.loads((base / 'metadata.json').read_text())
        stable_meta = {k: v for k, v in meta.items() if k not in {'experiment_id', 'timestamp'}}
        outputs.append({
            'metadata': json.dumps(stable_meta, sort_keys=True, default=str),
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

    X_train, X_test, y_train, y_test, _ = prepare_dataset('fold_ds', test_size=0.3, random_state=7, scaling_method='standard', encoding_method='onehot')
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    fm = FoldManager(n_splits=exe.validator.n_splits, n_repetitions=exe.validator.n_repetitions, random_state=exe.validator.random_state)
    all_info = fm.get_all_fold_info(X_train, y_train)
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
    results['early_binding'] = test_early_binding_launcher_values()
    results['environment_snapshot'] = test_environment_snapshot_complete()
    results['deterministic_runs'] = test_deterministic_mode_identical_runs(10)
    results['fold_persistence'] = test_fold_persistence_reconstructs()

    print('PHASE 6.1 VALIDATION RESULTS')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
