#!/usr/bin/env python
"""Quick validation test for metrics modules."""

import sys
import numpy as np

print("Testing metrics modules...")
print("=" * 70)

try:
    from metrics.pipeline import MetricsCalculator, MetricsValidator
    print("✓ MetricsCalculator and MetricsValidator imported")
except Exception as e:
    print(f"✗ Failed to import MetricsCalculator: {e}")
    sys.exit(1)

try:
    from metrics import (
        compute_stability_metrics,
        compute_timing_metrics,
        compute_all_stability_metrics,
        assess_model_stability,
        compare_computational_efficiency,
    )
    print("✓ Metrics functions imported")
except Exception as e:
    print(f"✗ Failed to import metrics functions: {e}")
    sys.exit(1)

# Test 1: Timing metrics
print("\n[Test 1] compute_timing_metrics")
try:
    timing = compute_timing_metrics(0.5, 0.1, 100, 30)
    assert 'train_time' in timing
    assert 'eval_time' in timing
    assert 'total_time' in timing
    assert timing['train_time'] == 0.5
    assert timing['eval_time'] == 0.1
    assert timing['total_time'] == 0.6
    print(f"  ✓ Timing metrics computed: {list(timing.keys())}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Stability metrics
print("\n[Test 2] compute_stability_metrics")
try:
    f1_scores = [0.85, 0.87, 0.84, 0.88, 0.86]
    stability = compute_stability_metrics(f1_scores, 'F1')
    assert 'mean' in stability
    assert 'std' in stability
    assert 'coefficient_of_variation' in stability
    assert 'stability_score' in stability
    assert 0 < stability['mean'] < 1
    assert stability['std'] >= 0
    print(f"  ✓ Stability metrics computed:")
    print(f"    Mean: {stability['mean']:.4f}")
    print(f"    Std: {stability['std']:.4f}")
    print(f"    CV: {stability['coefficient_of_variation']:.4f}")
    print(f"    Stability Score: {stability['stability_score']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: MetricsCalculator
print("\n[Test 3] MetricsCalculator.calculate_fold_metrics")
try:
    calc = MetricsCalculator()
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1])
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.85, 0.15],
                        [0.1, 0.9], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7],
                        [0.8, 0.2], [0.1, 0.9]])
    
    fold_metrics = calc.calculate_fold_metrics(
        y_true, y_pred,
        y_pred_proba=y_proba,
        train_time=0.5,
        eval_time=0.1,
        train_samples=100,
        eval_samples=30,
        fold_id=0,
        repetition_id=0
    )
    
    assert 'accuracy' in fold_metrics
    assert 'f1' in fold_metrics
    assert 'train_time' in fold_metrics
    print(f"  ✓ Fold metrics calculated:")
    print(f"    Accuracy: {fold_metrics['accuracy']:.4f}")
    print(f"    F1: {fold_metrics['f1']:.4f}")
    print(f"    Train Time: {fold_metrics['train_time']:.4f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Aggregation
print("\n[Test 4] MetricsCalculator.aggregate_fold_metrics")
try:
    fold_metrics_list = []
    for fold_id in range(5):
        np.random.seed(42 + fold_id)
        fold_metrics_list.append({
            'fold_id': fold_id,
            'repetition_id': 0,
            'accuracy': 0.90 + np.random.normal(0, 0.02),
            'f1': 0.88 + np.random.normal(0, 0.02),
            'train_time': 0.5 + np.random.normal(0, 0.05),
        })
    
    overall, by_rep, by_fold = calc.aggregate_fold_metrics(fold_metrics_list)
    
    assert 'accuracy_mean' in overall
    assert 'f1_std' in overall
    print(f"  ✓ Aggregation successful:")
    print(f"    Accuracy: {overall['accuracy_mean']:.4f} ± {overall['accuracy_std']:.4f}")
    print(f"    F1: {overall['f1_mean']:.4f} ± {overall['f1_std']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Stability calculation
print("\n[Test 5] MetricsCalculator.calculate_stability_metrics")
try:
    stability_result = calc.calculate_stability_metrics(fold_metrics_list, 'f1')
    assert 'by_metric' in stability_result
    assert 'assessment' in stability_result
    assessment = stability_result['assessment']
    assert 'primary_metric_cv' in assessment
    assert 'stability_level' in assessment
    print(f"  ✓ Stability calculation successful:")
    print(f"    CV: {assessment['primary_metric_cv']:.4f}")
    print(f"    Level: {assessment['stability_level']}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Comprehensive summary
print("\n[Test 6] MetricsCalculator.get_comprehensive_summary")
try:
    summary = calc.get_comprehensive_summary(
        fold_metrics_list,
        model_name='TestModel',
        dataset_name='test_dataset'
    )
    assert 'model_name' in summary
    assert 'overall' in summary
    assert 'stability' in summary
    print(f"  ✓ Comprehensive summary generated:")
    print(f"    Model: {summary['model_name']}")
    print(f"    Folds: {summary['n_folds']}")
    print(f"    Dataset: {summary['dataset_name']}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Validation
print("\n[Test 7] MetricsValidator.validate_metrics")
try:
    is_valid, errors = MetricsValidator.validate_metrics(fold_metrics_list)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
    print(f"  ✓ Validation successful:")
    print(f"    Valid: {is_valid}")
    print(f"    Errors: {len(errors)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All tests passed!")
print("=" * 70)
