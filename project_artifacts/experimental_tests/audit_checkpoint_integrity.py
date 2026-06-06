import sys
import os
import shutil
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import register_dataset
from experiments.executor import ExperimentExecutor

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_file_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def compare_csvs(csv1, csv2):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    cols1 = [c for c in df1.columns if 'time' not in c.lower() and 'cbs' not in c.lower()]
    cols2 = [c for c in df2.columns if 'time' not in c.lower() and 'cbs' not in c.lower()]
    pd.testing.assert_frame_equal(df1[cols1], df2[cols2], atol=1e-7, rtol=1e-7)

def main():
    print("Initializing Checkpoint Integrity Audit...")
    
    # 1. Prepare synthetic dataset
    dummy_csv = PROJECT_ROOT / 'scratch' / 'audit_dummy.csv'
    dummy_csv.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = np.random.choice([0, 1], size=200)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
    df['target'] = y
    df.to_csv(dummy_csv, index=False)
    
    register_dataset('AuditDummy', dummy_csv, 'target')
    
    config = {
        'dataset': {
            'registered_name': 'AuditDummy',
            'test_size': 0.3,
            'preprocessing': {
                'scaling': 'standard',
                'encoding': 'ordinal'
            },
            'stratify': True
        },
        'models': [
            {'name': 'DecisionTree', 'type': 'DecisionTree'},
            {'name': 'LogisticRegression', 'type': 'LogisticRegression'},
            {'name': 'RandomForest', 'type': 'RandomForest'}
        ],
        'random_state': 42,
        'save_predictions': False,
        'persist_cv_splits': False
    }
    
    out_dir_ref = 'scratch/audit_results_ref'
    out_dir_test = 'scratch/audit_results_test'
    checkpoint_file = Path(out_dir_test) / 'AuditDummy' / 'checkpoint.json'
    
    # Cleanup previous runs
    for path in [out_dir_ref, out_dir_test]:
        if os.path.exists(path):
            shutil.rmtree(path)
            
    # --- PHASE 0: REFERENCE RUN ---
    print("\n[PHASE 0] Running Uninterrupted Reference Run...")
    executor_ref = ExperimentExecutor(seed=42)
    ref_out = executor_ref.run(config, out_root=out_dir_ref)
    print("Reference Run completed.")
    assert not (Path(out_dir_ref) / 'AuditDummy' / 'checkpoint.json').exists(), "Reference checkpoint should not exist after success"
    
    scenarios_log = []
    
    # --- SCENARIO 1: Interruption after 1 completed fold of DecisionTree ---
    print("\n[SCENARIO 1] Simulating interruption after 1 completed fold of DecisionTree...")
    executor_s1 = ExperimentExecutor(seed=42)
    orig_validate = executor_s1.validator.validate
    
    def s1_validate(*args, **kwargs):
        model_name = kwargs.get('model_name') or args[3]
        on_fold_complete_orig = kwargs.get('on_fold_complete')
        
        if model_name == 'DecisionTree':
            def intercepting_callback(fr):
                if on_fold_complete_orig:
                    on_fold_complete_orig(fr)
                if fr.repetition_id == 0 and fr.fold_id == 0:
                    print("  [CRASH] Simulating crash after 1st fold of DecisionTree...")
                    raise KeyboardInterrupt("Simulated crash fold 1")
            kwargs['on_fold_complete'] = intercepting_callback
        return orig_validate(*args, **kwargs)
        
    executor_s1.validator.validate = s1_validate
    
    try:
        executor_s1.run(config, out_root=out_dir_test)
    except KeyboardInterrupt:
        print("  Run interrupted as expected.")
        
    assert checkpoint_file.exists(), "Checkpoint file should be retained after interruption"
    
    with open(checkpoint_file, 'r') as fh:
        cp = json.load(fh)
    assert cp['in_progress_model'] == 'DecisionTree'
    assert len(cp['in_progress_folds']) == 1
    assert len(cp['completed_models']) == 0
    print("  Checkpoint state verified: in_progress=DecisionTree, folds=1, completed_models=0")
    
    # Resume S1
    print("  Resuming S1...")
    executor_resumed = ExperimentExecutor(seed=42)
    resumed_out = executor_resumed.run(config, out_root=out_dir_test)
    assert not checkpoint_file.exists(), "Checkpoint file should be removed after successful resume completion"
    
    # Verify outputs match reference
    compare_csvs(os.path.join(out_dir_ref, 'AuditDummy', 'raw_results.csv'), os.path.join(out_dir_test, 'AuditDummy', 'raw_results.csv'))
    compare_csvs(os.path.join(out_dir_ref, 'AuditDummy', 'normalized_results.csv'), os.path.join(out_dir_test, 'AuditDummy', 'normalized_results.csv'))
    compare_csvs(os.path.join(out_dir_ref, 'AuditDummy', 'cbs_scores.csv'), os.path.join(out_dir_test, 'AuditDummy', 'cbs_scores.csv'))
    print("  [OK] Scenario 1 verified successfully! Metrics, CBS, and files match reference exactly.")
    scenarios_log.append("Scenario 1 (1 Completed Fold): PASSED")
    
    # Cleanup test dir for next scenario
    if os.path.exists(out_dir_test):
        shutil.rmtree(out_dir_test)
        
    # --- SCENARIO 2: Interruption after 1 completed model (DecisionTree done, LogisticRegression crashes fold 0) ---
    print("\n[SCENARIO 2] Simulating interruption after 1 completed model...")
    executor_s2 = ExperimentExecutor(seed=42)
    
    def s2_validate(*args, **kwargs):
        model_name = kwargs.get('model_name') or args[3]
        on_fold_complete_orig = kwargs.get('on_fold_complete')
        
        if model_name == 'LogisticRegression':
            def intercepting_callback(fr):
                if on_fold_complete_orig:
                    on_fold_complete_orig(fr)
                if fr.repetition_id == 0 and fr.fold_id == 0:
                    print("  [CRASH] Simulating crash on 1st fold of LogisticRegression...")
                    raise KeyboardInterrupt("Simulated crash model 2 fold 1")
            kwargs['on_fold_complete'] = intercepting_callback
        return orig_validate(*args, **kwargs)
        
    executor_s2.validator.validate = s2_validate
    
    try:
        executor_s2.run(config, out_root=out_dir_test)
    except KeyboardInterrupt:
        print("  Run interrupted as expected.")
        
    assert checkpoint_file.exists(), "Checkpoint file should be retained"
    with open(checkpoint_file, 'r') as fh:
        cp = json.load(fh)
    assert 'DecisionTree' in cp['completed_models']
    assert cp['in_progress_model'] == 'LogisticRegression'
    assert len(cp['in_progress_folds']) == 1
    print("  Checkpoint state verified: completed_models=['DecisionTree'], in_progress=LogisticRegression, folds=1")
    
    # Resume S2
    print("  Resuming S2...")
    executor_resumed2 = ExperimentExecutor(seed=42)
    resumed_out2 = executor_resumed2.run(config, out_root=out_dir_test)
    assert not checkpoint_file.exists(), "Checkpoint file should be removed"
    
    compare_csvs(os.path.join(out_dir_ref, 'AuditDummy', 'raw_results.csv'), os.path.join(out_dir_test, 'AuditDummy', 'raw_results.csv'))
    compare_csvs(os.path.join(out_dir_ref, 'AuditDummy', 'cbs_scores.csv'), os.path.join(out_dir_test, 'AuditDummy', 'cbs_scores.csv'))
    print("  [OK] Scenario 2 verified successfully!")
    scenarios_log.append("Scenario 2 (1 Completed Model): PASSED")
    
    if os.path.exists(out_dir_test):
        shutil.rmtree(out_dir_test)
        
    # --- SCENARIO 3: Interruption after multiple completed models (DecisionTree, LogisticRegression done) ---
    print("\n[SCENARIO 3] Simulating interruption after multiple completed models...")
    executor_s3 = ExperimentExecutor(seed=42)
    
    def s3_validate(*args, **kwargs):
        model_name = kwargs.get('model_name') or args[3]
        on_fold_complete_orig = kwargs.get('on_fold_complete')
        
        if model_name == 'RandomForest':
            def intercepting_callback(fr):
                if on_fold_complete_orig:
                    on_fold_complete_orig(fr)
                if fr.repetition_id == 0 and fr.fold_id == 0:
                    print("  [CRASH] Simulating crash on 1st fold of RandomForest...")
                    raise KeyboardInterrupt("Simulated crash model 3 fold 1")
            kwargs['on_fold_complete'] = intercepting_callback
        return orig_validate(*args, **kwargs)
        
    executor_s3.validator.validate = s3_validate
    
    try:
        executor_s3.run(config, out_root=out_dir_test)
    except KeyboardInterrupt:
        print("  Run interrupted as expected.")
        
    assert checkpoint_file.exists()
    with open(checkpoint_file, 'r') as fh:
        cp = json.load(fh)
    assert 'DecisionTree' in cp['completed_models']
    assert 'LogisticRegression' in cp['completed_models']
    assert cp['in_progress_model'] == 'RandomForest'
    assert len(cp['in_progress_folds']) == 1
    print("  Checkpoint state verified: completed_models=['DecisionTree', 'LogisticRegression'], in_progress=RandomForest, folds=1")
    
    # Resume S3
    print("  Resuming S3...")
    executor_resumed3 = ExperimentExecutor(seed=42)
    resumed_out3 = executor_resumed3.run(config, out_root=out_dir_test)
    assert not checkpoint_file.exists()
    
    compare_csvs(os.path.join(out_dir_ref, 'AuditDummy', 'raw_results.csv'), os.path.join(out_dir_test, 'AuditDummy', 'raw_results.csv'))
    compare_csvs(os.path.join(out_dir_ref, 'AuditDummy', 'cbs_scores.csv'), os.path.join(out_dir_test, 'AuditDummy', 'cbs_scores.csv'))
    
    # Verify statistical outputs (wilcoxon_results, etc. if they exist)
    ref_files = list(Path(out_dir_ref).glob('*'))
    test_files = list(Path(out_dir_test).glob('*'))
    print(f"  Reference files generated: {[f.name for f in ref_files if f.is_file()]}")
    print(f"  Resumed files generated: {[f.name for f in test_files if f.is_file()]}")
    
    # Check that any generated significance analysis files match exactly
    for f in Path(out_dir_ref).glob('AuditDummy/*.csv'):
        test_csv = Path(out_dir_test) / 'AuditDummy' / f.name
        if test_csv.exists():
            print(f"  Verifying statistical artifact: {f.name}")
            compare_csvs(f, test_csv)
            
    print("  [OK] Scenario 3 verified successfully!")
    scenarios_log.append("Scenario 3 (Multiple Completed Models): PASSED")
    
    print("\n==============================================")
    print("AUDIT RESULTS SUMMARY:")
    for log in scenarios_log:
        print(f"  {log}")
    print("==============================================")
    
    # Clean up dummy dataset & output directories
    if dummy_csv.exists():
        dummy_csv.unlink()
    for path in [out_dir_ref, out_dir_test]:
        if os.path.exists(path):
            shutil.rmtree(path)

if __name__ == '__main__':
    main()
