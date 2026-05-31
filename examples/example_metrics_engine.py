"""
Examples of the comprehensive metrics engine.

Demonstrates:
1. Single model evaluation
2. Fair model comparison
3. Stability analysis
4. Computational efficiency tracking
5. Metrics validation and export
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List

from metrics.pipeline import MetricsCalculator, MetricsValidator
from metrics import (
    compute_classification_metrics,
    compute_timing_metrics,
    compute_stability_metrics,
    compare_computational_efficiency,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Single Model Evaluation
# ============================================================================

def example_single_model_evaluation():
    """
    Evaluate a single model across all folds with comprehensive metrics.
    
    Shows how to:
    - Calculate fold-level metrics
    - Aggregate across folds
    - Generate comprehensive summary
    - Export results
    """
    logger.info("=" * 70)
    logger.info("Example 1: Single Model Evaluation")
    logger.info("=" * 70)
    
    # Simulated fold results (would come from CrossValidator in practice)
    fold_metrics_list = []
    
    # Generate 5 folds × 3 repetitions = 15 fold results
    for rep_id in range(3):
        for fold_id in range(5):
            # Simulate metrics with slight variation
            np.random.seed(42 + rep_id * 100 + fold_id)
            
            metrics = {
                'repetition_id': rep_id,
                'fold_id': fold_id,
                'accuracy': 0.90 + np.random.normal(0, 0.02),
                'precision': 0.89 + np.random.normal(0, 0.02),
                'recall': 0.91 + np.random.normal(0, 0.02),
                'f1': 0.90 + np.random.normal(0, 0.02),
                'roc_auc': 0.93 + np.random.normal(0, 0.02),
                'train_time': 0.5 + np.random.normal(0, 0.05),
                'eval_time': 0.1 + np.random.normal(0, 0.01),
                'total_time': 0.6 + np.random.normal(0, 0.05),
            }
            fold_metrics_list.append(metrics)
    
    # Calculate comprehensive summary
    calculator = MetricsCalculator()
    summary = calculator.get_comprehensive_summary(
        fold_metrics_list,
        model_name='RandomForest',
        dataset_name='iris'
    )
    
    logger.info(f"\nModel: {summary['model_name']}")
    logger.info(f"Dataset: {summary['dataset_name']}")
    logger.info(f"Folds: {summary['n_folds']}")
    logger.info(f"Repetitions: {summary['n_repetitions']}")
    
    logger.info("\nOverall Performance Metrics:")
    overall = summary['overall']
    logger.info(f"  Accuracy: {overall['accuracy_mean']:.4f} ± {overall['accuracy_std']:.4f}")
    logger.info(f"  Precision: {overall['precision_mean']:.4f} ± {overall['precision_std']:.4f}")
    logger.info(f"  Recall: {overall['recall_mean']:.4f} ± {overall['recall_std']:.4f}")
    logger.info(f"  F1-Score: {overall['f1_mean']:.4f} ± {overall['f1_std']:.4f}")
    logger.info(f"  ROC-AUC: {overall['roc_auc_mean']:.4f} ± {overall['roc_auc_std']:.4f}")
    
    logger.info("\nComputational Metrics:")
    logger.info(f"  Train Time: {overall['train_time_mean']:.4f}s ± {overall['train_time_std']:.4f}s")
    logger.info(f"  Eval Time: {overall['eval_time_mean']:.4f}s ± {overall['eval_time_std']:.4f}s")
    logger.info(f"  Total Time: {overall['total_time_mean']:.4f}s ± {overall['total_time_std']:.4f}s")
    
    logger.info("\nStability Assessment:")
    assessment = summary['stability']['assessment']
    logger.info(f"  Primary Metric: {assessment['primary_metric']}")
    logger.info(f"  Mean: {assessment['primary_metric_mean']:.4f}")
    logger.info(f"  Std: {assessment['primary_metric_std']:.4f}")
    logger.info(f"  Coefficient of Variation: {assessment['primary_metric_cv']:.4f}")
    logger.info(f"  Stability Level: {assessment['stability_level']}")
    logger.info(f"  Is Stable: {assessment['is_stable']}")
    
    # Export to DataFrame
    df = calculator.export_to_dataframe(fold_metrics_list)
    logger.info(f"\nExported {len(df)} fold results to DataFrame")
    logger.info("\nDataFrame Summary:")
    logger.info(df.describe())
    
    return summary, df


# ============================================================================
# Example 2: Fair Model Comparison
# ============================================================================

def example_fair_model_comparison():
    """
    Compare multiple models fairly using uniform metrics.
    
    Shows how to:
    - Calculate metrics for multiple models
    - Compare across models
    - Identify best performer
    - Show statistical summary
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Fair Model Comparison")
    logger.info("=" * 70)
    
    models = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'SVM', 'GradientBoosting']
    
    # Simulate metrics for each model (would come from CrossValidator)
    all_results = {}
    model_performance = {
        'LogisticRegression': (0.85, 0.02),  # (mean, std)
        'DecisionTree': (0.82, 0.04),
        'RandomForest': (0.92, 0.015),
        'SVM': (0.88, 0.03),
        'GradientBoosting': (0.93, 0.012),
    }
    
    calculator = MetricsCalculator()
    
    for model_name in models:
        mean, std = model_performance[model_name]
        
        fold_metrics_list = []
        for rep_id in range(3):
            for fold_id in range(5):
                np.random.seed(42 + hash(model_name) % 100 + rep_id * 100 + fold_id)
                
                metrics = {
                    'repetition_id': rep_id,
                    'fold_id': fold_id,
                    'accuracy': np.clip(mean + np.random.normal(0, std), 0, 1),
                    'precision': np.clip(mean + np.random.normal(0, std), 0, 1),
                    'recall': np.clip(mean + np.random.normal(0, std), 0, 1),
                    'f1': np.clip(mean + np.random.normal(0, std), 0, 1),
                    'roc_auc': np.clip(mean + np.random.normal(0, std), 0, 1),
                    'train_time': np.abs(np.random.normal(1, 0.3)),
                    'eval_time': np.abs(np.random.normal(0.15, 0.05)),
                    'total_time': np.abs(np.random.normal(1.15, 0.3)),
                }
                fold_metrics_list.append(metrics)
        
        summary = calculator.get_comprehensive_summary(
            fold_metrics_list,
            model_name=model_name,
            dataset_name='wine'
        )
        all_results[model_name] = summary
    
    # Display comparison
    logger.info("\nModel Performance Comparison:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<20} {'Accuracy':<15} {'F1':<15} {'Train Time':<15}")
    logger.info("-" * 80)
    
    for model_name, summary in all_results.items():
        overall = summary['overall']
        acc = overall['accuracy_mean']
        f1 = overall['f1_mean']
        train_time = overall['train_time_mean']
        
        logger.info(f"{model_name:<20} {acc:.4f}±{overall['accuracy_std']:.3f} {f1:.4f}±{overall['f1_std']:.3f} {train_time:.4f}s")
    
    # Find best overall
    best_model = max(all_results.items(),
                     key=lambda x: x[1]['overall']['f1_mean'])
    logger.info("-" * 80)
    logger.info(f"\n✓ Best Model: {best_model[0]} (F1={best_model[1]['overall']['f1_mean']:.4f})")
    
    # Stability comparison
    logger.info("\nStability Comparison (F1 Score):")
    logger.info("-" * 60)
    logger.info(f"{'Model':<20} {'Stability Level':<20} {'CV':<15}")
    logger.info("-" * 60)
    
    for model_name, summary in all_results.items():
        assessment = summary['stability']['assessment']
        cv = assessment['primary_metric_cv']
        level = assessment['stability_level']
        logger.info(f"{model_name:<20} {level:<20} {cv:.4f}")
    
    return all_results


# ============================================================================
# Example 3: Stability Deep Dive
# ============================================================================

def example_stability_analysis():
    """
    Analyze model stability across folds and repetitions.
    
    Shows how to:
    - Track metrics across folds
    - Assess stability level
    - Identify unstable models
    - Recommend investigation
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Stability Deep Dive")
    logger.info("=" * 70)
    
    calculator = MetricsCalculator()
    
    # Stable model scenario
    logger.info("\nScenario 1: Stable Model")
    stable_f1_scores = [0.88, 0.87, 0.89, 0.88, 0.87]
    stable_metrics = [
        {'f1': score, 'accuracy': score + 0.02}
        for score in stable_f1_scores
    ]
    
    stability_stable = calculator.calculate_stability_metrics(
        stable_metrics,
        primary_metric='f1'
    )
    
    logger.info(f"  F1 Scores: {[f'{s:.3f}' for s in stable_f1_scores]}")
    logger.info(f"  Mean: {stability_stable['assessment']['primary_metric_mean']:.4f}")
    logger.info(f"  Std: {stability_stable['assessment']['primary_metric_std']:.4f}")
    logger.info(f"  CV: {stability_stable['assessment']['primary_metric_cv']:.4f}")
    logger.info(f"  Status: {stability_stable['assessment']['stability_level']}")
    
    # Unstable model scenario
    logger.info("\nScenario 2: Unstable Model")
    unstable_f1_scores = [0.75, 0.92, 0.78, 0.95, 0.73]
    unstable_metrics = [
        {'f1': score, 'accuracy': score + 0.02}
        for score in unstable_f1_scores
    ]
    
    stability_unstable = calculator.calculate_stability_metrics(
        unstable_metrics,
        primary_metric='f1'
    )
    
    logger.info(f"  F1 Scores: {[f'{s:.3f}' for s in unstable_f1_scores]}")
    logger.info(f"  Mean: {stability_unstable['assessment']['primary_metric_mean']:.4f}")
    logger.info(f"  Std: {stability_unstable['assessment']['primary_metric_std']:.4f}")
    logger.info(f"  CV: {stability_unstable['assessment']['primary_metric_cv']:.4f}")
    logger.info(f"  Status: {stability_unstable['assessment']['stability_level']}")
    logger.info(f"  ⚠ Recommendation: Investigate fold differences or model hyperparameters")
    
    # Per-metric stability
    logger.info("\nPer-Metric Stability (Stable Model):")
    by_metric = stability_stable['by_metric']
    for metric_name, stability_metrics in by_metric.items():
        logger.info(f"  {metric_name}:")
        logger.info(f"    Mean: {stability_metrics['mean']:.4f}")
        logger.info(f"    Std: {stability_metrics['std']:.4f}")
        logger.info(f"    Stability Score: {stability_metrics['stability_score']:.4f}")


# ============================================================================
# Example 4: Computational Efficiency Tracking
# ============================================================================

def example_computational_efficiency():
    """
    Track and compare computational efficiency across models.
    
    Shows how to:
    - Calculate timing metrics
    - Compare efficiency
    - Identify fast vs slow models
    - Trade-off analysis
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Computational Efficiency Tracking")
    logger.info("=" * 70)
    
    # Simulated timing data for models
    timing_results = {
        'LogisticRegression': {
            'train_time_mean': 0.05,
            'eval_time_mean': 0.01,
            'n_folds': 5,
        },
        'DecisionTree': {
            'train_time_mean': 0.08,
            'eval_time_mean': 0.02,
            'n_folds': 5,
        },
        'RandomForest': {
            'train_time_mean': 0.5,
            'eval_time_mean': 0.1,
            'n_folds': 5,
        },
        'SVM': {
            'train_time_mean': 2.0,
            'eval_time_mean': 0.3,
            'n_folds': 5,
        },
        'GradientBoosting': {
            'train_time_mean': 3.0,
            'eval_time_mean': 0.2,
            'n_folds': 5,
        },
    }
    
    comparison = compare_computational_efficiency(timing_results)
    
    logger.info("\nComputational Efficiency Comparison:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<20} {'Avg Train':<15} {'Avg Eval':<15} {'Efficiency':<15}")
    logger.info("-" * 80)
    
    for model_name, metrics in comparison.items():
        logger.info(
            f"{model_name:<20} "
            f"{metrics['avg_train_time']:.4f}s {' ':<10} "
            f"{metrics['avg_eval_time']:.4f}s {' ':<10} "
            f"{metrics['efficiency_score']:.4f}"
        )
    
    logger.info("-" * 80)
    
    # Find most efficient
    most_efficient = max(comparison.items(),
                        key=lambda x: x[1]['efficiency_score'])
    logger.info(f"\n✓ Most Efficient: {most_efficient[0]}")
    
    # Find fastest training
    fastest_training = min(comparison.items(),
                          key=lambda x: x[1]['avg_train_time'])
    logger.info(f"✓ Fastest Training: {fastest_training[0]}")


# ============================================================================
# Example 5: Metrics Validation and Export
# ============================================================================

def example_metrics_validation_and_export():
    """
    Validate metrics consistency and export results.
    
    Shows how to:
    - Validate fold metrics
    - Check for anomalies
    - Export to DataFrame
    - Generate reports
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Metrics Validation and Export")
    logger.info("=" * 70)
    
    # Valid metrics
    valid_metrics = [
        {'fold_id': 0, 'accuracy': 0.90, 'f1': 0.88},
        {'fold_id': 1, 'accuracy': 0.92, 'f1': 0.90},
        {'fold_id': 2, 'accuracy': 0.91, 'f1': 0.89},
    ]
    
    is_valid, errors = MetricsValidator.validate_metrics(valid_metrics)
    logger.info(f"\nValid Metrics: is_valid={is_valid}, errors={errors}")
    
    # Invalid metrics (with errors)
    invalid_metrics = [
        {'fold_id': 0, 'accuracy': 0.90, 'f1': 0.88},
        {'fold_id': 1, 'accuracy': 1.5, 'f1': 0.90},  # Out of range
        {'fold_id': 2, 'accuracy': np.nan, 'f1': 0.89},  # NaN
    ]
    
    is_valid, errors = MetricsValidator.validate_metrics(invalid_metrics)
    logger.info(f"\nInvalid Metrics: is_valid={is_valid}")
    if errors:
        logger.info("Errors found:")
        for error in errors:
            logger.info(f"  - {error}")
    
    # Export to DataFrame
    logger.info("\nExporting to DataFrame:")
    df = pd.DataFrame(valid_metrics)
    logger.info(f"Shape: {df.shape}")
    logger.info("\nDataFrame:")
    logger.info(df)
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(df.describe())


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    logger.info("Running Comprehensive Metrics Engine Examples\n")
    
    # Run all examples
    example_single_model_evaluation()
    example_fair_model_comparison()
    example_stability_analysis()
    example_computational_efficiency()
    example_metrics_validation_and_export()
    
    logger.info("\n" + "=" * 70)
    logger.info("All examples completed!")
    logger.info("=" * 70)
