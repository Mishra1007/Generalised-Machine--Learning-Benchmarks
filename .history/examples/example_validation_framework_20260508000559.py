"""
Example: Reproducible Validation Framework

Demonstrates:
1. Cross-validation setup (5-fold × 3 repetitions)
2. Single model validation
3. Multiple model comparison (fair benchmarking)
4. Result analysis and aggregation
5. Logging and reporting

Run with: python examples/example_validation_framework.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_breast_cancer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'validation_example.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from validation import CrossValidator, FoldManager
from preprocessing import prepare_dataset
from datasets import register_dataset


def example_1_fold_manager():
    """Example 1: Understand fold generation."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Fold Manager - Understanding the Stratified Folds")
    logger.info("=" * 70)
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create fold manager
    fold_manager = FoldManager(n_splits=5, n_repetitions=2, random_state=42)
    
    logger.info(f"\nDataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    logger.info(f"Fold Configuration: 5-fold CV × 2 repetitions\n")
    
    fold_count = 0
    for rep_id, fold_id, train_idx, test_idx in fold_manager.generate_folds(X, y):
        fold_count += 1
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Calculate class distributions
        train_dist = {cls: np.sum(y_train == cls) for cls in np.unique(y)}
        test_dist = {cls: np.sum(y_test == cls) for cls in np.unique(y)}
        
        logger.info(
            f"[{fold_count}] Rep {rep_id}, Fold {fold_id}: "
            f"Train {len(train_idx)}, Test {len(test_idx)} | "
            f"Train dist: {train_dist}, Test dist: {test_dist}"
        )
    
    # Validate folds
    logger.info(f"\n✓ Validating fold integrity...")
    fold_manager.validate_fold_indices(X, y)
    logger.info(f"✓ Fold validation passed: no overlaps, complete coverage\n")
    
    print()


def example_2_single_model():
    """Example 2: Validate a single model."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 2: Single Model Validation")
    logger.info("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create validator
    validator = CrossValidator(n_splits=5, n_repetitions=3, random_state=42)
    
    # Create and validate model
    logger.info("\nValidating RandomForestClassifier (5-fold × 3 reps)...\n")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    results = validator.validate(
        X, y, model,
        model_name='RandomForest',
        dataset_name='iris',
        predict_proba=True
    )
    
    # Get summary
    summary = results.get_summary()
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total folds: {summary['total_folds']}")
    logger.info(f"Repetitions: {summary['n_repetitions']}")
    logger.info(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    logger.info(f"Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    logger.info(f"Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    logger.info(f"F1-Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    
    # Log detailed results
    results.log_summary()
    
    print()
    return results


def example_3_multiple_models():
    """Example 3: Compare multiple models fairly."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 3: Multiple Models - Fair Benchmarking")
    logger.info("=" * 70)
    
    # Load data
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    logger.info(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Classes: {np.unique(y)} | Class dist: {np.bincount(y)}\n")
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    }
    
    logger.info(f"Evaluating {len(models)} models under identical conditions:\n")
    
    validator = CrossValidator(n_splits=5, n_repetitions=2, random_state=42)
    
    all_results = validator.validate_multiple(
        X, y, models, dataset_name='breast_cancer'
    )
    
    # Compare results
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    
    comparison_data = []
    for model_name, results in all_results.items():
        if results:
            summary = results.get_summary()
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}",
                'F1-Score': f"{summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
                'ROC-AUC': f"{summary.get('roc_auc_mean', 0):.4f} ± {summary.get('roc_auc_std', 0):.4f}",
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    logger.info("\n" + comparison_df.to_string())
    
    print()


def example_4_result_analysis():
    """Example 4: Detailed result analysis."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 4: Detailed Result Analysis")
    logger.info("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    validator = CrossValidator()
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    results = validator.validate(
        X, y, model,
        model_name='RandomForest',
        dataset_name='iris'
    )
    
    # Get results as DataFrame
    df = results.to_dataframe()
    
    logger.info("\nAll fold-level results (first 5 folds):\n")
    logger.info(df[['repetition_id', 'fold_id', 'accuracy', 'f1', 'train_time', 'eval_time']].head().to_string())
    
    # Per-repetition summary
    logger.info("\n\nPer-repetition summary:\n")
    rep_summary = results.get_repetition_summary()
    logger.info(rep_summary[['repetition_id', 'accuracy_mean', 'accuracy_std', 'f1_mean']].to_string())
    
    # Per-fold summary
    logger.info("\n\nPer-fold summary (consistency across repetitions):\n")
    fold_summary = results.get_fold_summary()
    logger.info(fold_summary[['fold_id', 'accuracy_mean', 'accuracy_std', 'f1_mean']].to_string())
    
    # Find outliers
    logger.info("\n\nOutlier folds:")
    best = results.get_best_fold('accuracy')
    worst = results.get_worst_fold('accuracy')
    
    logger.info(
        f"Best fold:  Rep {best.repetition_id}, Fold {best.fold_id} | "
        f"Accuracy: {best.metrics['accuracy']:.4f}"
    )
    logger.info(
        f"Worst fold: Rep {worst.repetition_id}, Fold {worst.fold_id} | "
        f"Accuracy: {worst.metrics['accuracy']:.4f}"
    )
    
    # Statistics
    logger.info(f"\n\nAccuracy statistics:")
    accuracies = df['accuracy'].values
    logger.info(f"  Mean: {np.mean(accuracies):.4f}")
    logger.info(f"  Std:  {np.std(accuracies):.4f}")
    logger.info(f"  Min:  {np.min(accuracies):.4f}")
    logger.info(f"  Max:  {np.max(accuracies):.4f}")
    logger.info(f"  CoV:  {np.std(accuracies) / np.mean(accuracies):.4f}")  # Coefficient of variation
    
    print()


def example_5_reproducibility():
    """Example 5: Demonstrate reproducibility."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 5: Reproducibility Verification")
    logger.info("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    logger.info("\nRunning validation twice with same configuration...\n")
    
    # Run 1
    validator1 = CrossValidator(n_splits=5, n_repetitions=2, random_state=42)
    model1 = RandomForestClassifier(n_estimators=50, random_state=42)
    results1 = validator1.validate(X, y, model1, 'rf', 'iris')
    summary1 = results1.get_summary()
    
    # Run 2
    validator2 = CrossValidator(n_splits=5, n_repetitions=2, random_state=42)
    model2 = RandomForestClassifier(n_estimators=50, random_state=42)
    results2 = validator2.validate(X, y, model2, 'rf', 'iris')
    summary2 = results2.get_summary()
    
    logger.info("\n" + "=" * 70)
    logger.info("REPRODUCIBILITY CHECK")
    logger.info("=" * 70)
    
    # Compare results
    accuracy_match = np.isclose(
        summary1['accuracy_mean'],
        summary2['accuracy_mean'],
        atol=1e-6
    )
    f1_match = np.isclose(
        summary1['f1_mean'],
        summary2['f1_mean'],
        atol=1e-6
    )
    
    logger.info(f"\nRun 1 Accuracy: {summary1['accuracy_mean']:.6f}")
    logger.info(f"Run 2 Accuracy: {summary2['accuracy_mean']:.6f}")
    logger.info(f"Match: {'✓ YES' if accuracy_match else '✗ NO'}")
    
    logger.info(f"\nRun 1 F1-Score: {summary1['f1_mean']:.6f}")
    logger.info(f"Run 2 F1-Score: {summary2['f1_mean']:.6f}")
    logger.info(f"Match: {'✓ YES' if f1_match else '✗ NO'}")
    
    if accuracy_match and f1_match:
        logger.info("\n✓ REPRODUCIBILITY VERIFIED: Identical results across runs")
    else:
        logger.warning("\n✗ REPRODUCIBILITY FAILED: Different results")
    
    print()


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 70)
    logger.info("ML BENCHMARKING FRAMEWORK - VALIDATION EXAMPLES")
    logger.info("=" * 70 + "\n")
    
    try:
        example_1_fold_manager()
        example_2_single_model()
        example_3_multiple_models()
        example_4_result_analysis()
        example_5_reproducibility()
        
        logger.info("=" * 70)
        logger.info("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        logger.info("\nKey Takeaways:")
        logger.info("  1. ✓ Stratified 5-fold CV × 3 repetitions")
        logger.info("  2. ✓ Prevents data leakage (folds validated)")
        logger.info("  3. ✓ Fair model comparison (identical conditions)")
        logger.info("  4. ✓ Complete logging of all operations")
        logger.info("  5. ✓ Reproducible results (seed=42)")
        logger.info("  6. ✓ Comprehensive metrics per fold")
        logger.info("  7. ✓ Statistical summaries and aggregation")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
