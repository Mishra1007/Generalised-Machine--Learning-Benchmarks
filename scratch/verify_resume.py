import sys
import os
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import register_dataset
from experiments.executor import ExperimentExecutor
from validation.results import FoldResult

def main():
    print("Preparing verification dataset...")
    # Create a small dummy CSV dataset for verification
    dummy_path = PROJECT_ROOT / 'scratch' / 'dummy.csv'
    dummy_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.random.choice([0, 1], size=100)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
    df['target'] = y
    df.to_csv(dummy_path, index=False)
    
    # Register dataset
    register_dataset('Dummy', dummy_path, 'target')
    
    config = {
        'dataset': {
            'registered_name': 'Dummy',
            'test_size': 0.3,
            'preprocessing': {
                'scaling': 'standard',
                'encoding': 'ordinal'
            },
            'stratify': True
        },
        'models': [
            {'name': 'DecisionTree', 'type': 'DecisionTree'},
            {'name': 'LogisticRegression', 'type': 'LogisticRegression'}
        ],
        'random_state': 42,
        'experiment_id': 'exp-verify-resume-1234',
        'run_id': 'run-verify-resume-1234',
        'save_predictions': False,
        'persist_cv_splits': False
    }
    
    out_root = 'scratch/verify_resume_results'
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
        
    # Phase 1: Reference non-interrupted run
    print("\n--- Running reference (non-interrupted) run ---")
    executor_ref = ExperimentExecutor(seed=42)
    ref_out_root = out_root + "_ref"
    if os.path.exists(ref_out_root):
        shutil.rmtree(ref_out_root)
    ref_results = executor_ref.run(config, out_root=ref_out_root)
    print("Reference run complete.")
    
    # Phase 2: Run with interruption
    print("\n--- Running interrupted run ---")
    # We will subclass CrossValidator or mock validate to raise an exception on a specific model / fold
    executor = ExperimentExecutor(seed=42)
    
    # Let's mock validator's validate to raise exception on the 2nd fold of LogisticRegression
    orig_validate = executor.validator.validate
    
    def interrupted_validate(*args, **kwargs):
        model_name = kwargs.get('model_name') or args[3]
        on_fold_complete_orig = kwargs.get('on_fold_complete')
        
        if model_name == 'LogisticRegression':
            def intercepting_callback(fr):
                if on_fold_complete_orig:
                    on_fold_complete_orig(fr)
                if fr.repetition_id == 0 and fr.fold_id == 1:
                    print("  [INTERRUPTION TRIGGERED] Raising exception to simulate crash...")
                    raise KeyboardInterrupt("Simulated crash mid-run")
            kwargs['on_fold_complete'] = intercepting_callback
            
        return orig_validate(*args, **kwargs)
        
    executor.validator.validate = interrupted_validate
    
    checkpoint_file = Path(out_root) / 'Dummy' / 'checkpoint.json'
    
    try:
        executor.run(config, out_root=out_root)
    except KeyboardInterrupt as e:
        print(f"Run crashed as expected: {e}")
        
    # Assert checkpoint exists and has correct info
    assert checkpoint_file.exists(), "Checkpoint file was not created!"
    import json
    with open(checkpoint_file, 'r') as fh:
        cp_data = json.load(fh)
        
    print("Checkpoint contents after crash:")
    print(f"  Completed models: {list(cp_data['completed_models'].keys())}")
    print(f"  In progress model: {cp_data['in_progress_model']}")
    print(f"  In progress folds count: {len(cp_data['in_progress_folds'])}")
    
    assert 'DecisionTree' in cp_data['completed_models'], "DecisionTree should have been saved in checkpoint!"
    assert cp_data['in_progress_model'] == 'LogisticRegression', "LogisticRegression should be in progress!"
    assert len(cp_data['in_progress_folds']) == 2, "Should have checkpointed 2 folds of LogisticRegression!"
    
    # Phase 3: Resume run
    print("\n--- Resuming interrupted run ---")
    executor_resumed = ExperimentExecutor(seed=42)
    # This time it should load the checkpoint, skip DecisionTree, skip first 2 folds of LogisticRegression, and complete.
    resumed_results = executor_resumed.run(config, out_root=out_root)
    print("Resumed run complete.")
    
    assert not checkpoint_file.exists(), "Checkpoint file should have been cleaned up after success!"
    
    # Phase 4: Compare results
    print("\n--- Comparing results ---")
    ref_metrics = ref_results['model_summaries']
    resumed_metrics = resumed_results['model_summaries']
    
    for model_name in ['DecisionTree', 'LogisticRegression']:
        ref_f1 = ref_metrics[model_name]['overall']['f1_mean']
        resumed_f1 = resumed_metrics[model_name]['overall']['f1_mean']
        print(f"  {model_name} F1-score: Ref={ref_f1:.6f}, Resumed={resumed_f1:.6f}")
        assert abs(ref_f1 - resumed_f1) < 1e-9, f"Metrics mismatch for {model_name}!"
        
    print("\n[OK] SUCCESS: Verification test passed! Resume/checkpoint works perfectly.")
    
    # Cleanup dummy files
    if dummy_path.exists():
        dummy_path.unlink()
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    if os.path.exists(ref_out_root):
        shutil.rmtree(ref_out_root)

if __name__ == '__main__':
    main()
