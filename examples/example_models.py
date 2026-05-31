"""
Example: Model Execution Engine

Demonstrates:
1. Creating individual models
2. Model registry and discovery
3. Training and prediction
4. Fair model comparison with validation framework
5. Model configuration and reproducibility
6. Integration with data pipeline

Run with: python examples/example_models.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'models_example.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from models import (
    create_model,
    list_models,
    get_model_info,
    LogisticRegressionModel,
    DecisionTreeModel,
    RandomForestModel,
    SVMModel,
    GradientBoostingModel,
)
from validation import CrossValidator
from preprocessing import prepare_dataset


def example_1_model_registry():
    """Example 1: Model registry and discovery."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Model Registry and Discovery")
    logger.info("=" * 70)
    
    # List all available models
    models = list_models()
    logger.info(f"\nAvailable models ({len(models)}):")
    for model_name in models:
        logger.info(f"  - {model_name}")
    
    # Get model information
    logger.info("\nModel information:")
    for model_name in models[:2]:  # Show first 2
        info = get_model_info(model_name)
        logger.info(f"\n{model_name}:")
        logger.info(f"  {info['docstring'].split(chr(10))[0]}")
    
    print()


def example_2_create_models():
    """Example 2: Create models using registry."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 2: Creating Models")
    logger.info("=" * 70)
    
    logger.info("\nMethod 1: Using convenience function")
    model1 = create_model('RandomForest', n_estimators=100, random_state=42)
    logger.info(f"Created: {model1}")
    
    logger.info("\nMethod 2: Direct class instantiation")
    model2 = LogisticRegressionModel(max_iter=1000, random_state=42)
    logger.info(f"Created: {model2}")
    
    logger.info("\nMethod 3: From configuration dictionary")
    config = {
        'name': 'DecisionTree',
        'max_depth': 5,
        'random_state': 42,
    }
    model3 = create_model(**config)
    logger.info(f"Created: {model3}")
    
    print()


def example_3_single_model_training():
    """Example 3: Train and use a single model."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 3: Single Model Training")
    logger.info("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"\nDataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    logger.info(f"Train/Test split: {len(X_train)} / {len(X_test)}")
    
    # Create and train model
    logger.info("\nCreating and training RandomForestModel...")
    model = create_model('RandomForest', n_estimators=50, random_state=42)
    
    logger.info(f"Before training: {model}")
    
    model.fit(X_train, y_train)
    
    logger.info(f"After training: {model}")
    logger.info(f"Config: {model.get_config()}")
    
    # Make predictions
    logger.info("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    logger.info(f"Predictions shape: {y_pred.shape}")
    logger.info(f"Probabilities shape: {y_proba.shape if y_proba is not None else None}")
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    print()


def example_4_model_comparison():
    """Example 4: Compare all 5 models."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 4: Model Comparison")
    logger.info("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features for LogisticRegression and SVM
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"\nComparing {len(list_models())} models on Iris dataset")
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    results = []
    
    # LogisticRegression
    logger.info("1. Training LogisticRegression...")
    model_lr = create_model('LogisticRegression', max_iter=1000, random_state=42)
    model_lr.fit(X_train_scaled, y_train)
    acc_lr = model_lr.predict(X_test_scaled)
    score_lr = (acc_lr == y_test).mean()
    logger.info(f"   Accuracy: {score_lr:.4f}")
    results.append(('LogisticRegression', score_lr))
    
    # DecisionTree
    logger.info("2. Training DecisionTree...")
    model_dt = create_model('DecisionTree', max_depth=10, random_state=42)
    model_dt.fit(X_train, y_train)
    acc_dt = model_dt.predict(X_test)
    score_dt = (acc_dt == y_test).mean()
    logger.info(f"   Accuracy: {score_dt:.4f}")
    results.append(('DecisionTree', score_dt))
    
    # RandomForest
    logger.info("3. Training RandomForest...")
    model_rf = create_model('RandomForest', n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    acc_rf = model_rf.predict(X_test)
    score_rf = (acc_rf == y_test).mean()
    logger.info(f"   Accuracy: {score_rf:.4f}")
    results.append(('RandomForest', score_rf))
    
    # SVM
    logger.info("4. Training SVM...")
    model_svm = create_model('SVM', kernel='rbf', probability=True, random_state=42)
    model_svm.fit(X_train_scaled, y_train)
    acc_svm = model_svm.predict(X_test_scaled)
    score_svm = (acc_svm == y_test).mean()
    logger.info(f"   Accuracy: {score_svm:.4f}")
    results.append(('SVM', score_svm))
    
    # GradientBoosting
    logger.info("5. Training GradientBoosting...")
    model_gb = create_model('GradientBoosting', n_estimators=100, random_state=42)
    model_gb.fit(X_train, y_train)
    acc_gb = model_gb.predict(X_test)
    score_gb = (acc_gb == y_test).mean()
    logger.info(f"   Accuracy: {score_gb:.4f}")
    results.append(('GradientBoosting', score_gb))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    logger.info("\n" + results_df.to_string(index=False))
    logger.info(f"\nBest model: {results_df.iloc[0]['Model']}")
    
    print()


def example_5_validation_framework():
    """Example 5: Use with validation framework."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 5: Integration with Validation Framework")
    logger.info("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    logger.info(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    
    # Create models
    models = {
        'LogisticRegression': create_model('LogisticRegression', max_iter=1000, random_state=42),
        'RandomForest': create_model('RandomForest', n_estimators=50, random_state=42),
        'GradientBoosting': create_model('GradientBoosting', n_estimators=50, random_state=42),
    }
    
    logger.info(f"\nValidating {len(models)} models (5-fold CV × 3 reps)...")
    
    # Use validation framework
    validator = CrossValidator(n_splits=5, n_repetitions=2, random_state=42)  # Reduced for demo
    
    all_results = validator.validate_multiple(
        X, y, models,
        dataset_name='iris'
    )
    
    # Compare results
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    
    comparison = []
    for model_name, results in all_results.items():
        if results:
            summary = results.get_summary()
            comparison.append({
                'Model': model_name,
                'Accuracy': f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}",
                'F1': f"{summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
            })
    
    comparison_df = pd.DataFrame(comparison)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    print()


def example_6_reproducibility():
    """Example 6: Model reproducibility."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 6: Reproducibility Verification")
    logger.info("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info("\nTraining same model configuration twice...")
    
    # Run 1
    logger.info("Run 1:")
    model1 = create_model('RandomForest', n_estimators=100, random_state=42)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    score1 = (y_pred1 == y_test).mean()
    logger.info(f"  Accuracy: {score1:.6f}")
    
    # Run 2
    logger.info("Run 2:")
    model2 = create_model('RandomForest', n_estimators=100, random_state=42)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    score2 = (y_pred2 == y_test).mean()
    logger.info(f"  Accuracy: {score2:.6f}")
    
    # Check reproducibility
    logger.info("\n" + "=" * 70)
    if np.array_equal(y_pred1, y_pred2) and score1 == score2:
        logger.info("✓ REPRODUCIBILITY VERIFIED: Identical results")
    else:
        logger.warning("✗ REPRODUCIBILITY FAILED: Different results")
    logger.info("=" * 70)
    
    print()


def example_7_model_configuration():
    """Example 7: Advanced model configuration."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 7: Advanced Model Configuration")
    logger.info("=" * 70)
    
    # Example configurations
    logger.info("\nLogisticRegression configurations:")
    configs_lr = [
        {'max_iter': 100},
        {'max_iter': 1000},
        {'max_iter': 5000},
    ]
    
    for config in configs_lr:
        model = create_model('LogisticRegression', **config)
        logger.info(f"  {config} → {model}")
    
    logger.info("\nRandomForest configurations:")
    configs_rf = [
        {'n_estimators': 10},
        {'n_estimators': 100},
        {'n_estimators': 500},
    ]
    
    for config in configs_rf:
        model = create_model('RandomForest', **config)
        logger.info(f"  {config} → {model}")
    
    logger.info("\nGradientBoosting configurations:")
    configs_gb = [
        {'learning_rate': 0.01, 'n_estimators': 100},
        {'learning_rate': 0.1, 'n_estimators': 100},
        {'learning_rate': 0.1, 'n_estimators': 500},
    ]
    
    for config in configs_gb:
        model = create_model('GradientBoosting', **config)
        logger.info(f"  {config} → {model}")
    
    print()


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 70)
    logger.info("ML BENCHMARKING FRAMEWORK - MODELS EXAMPLES")
    logger.info("=" * 70 + "\n")
    
    try:
        example_1_model_registry()
        example_2_create_models()
        example_3_single_model_training()
        example_4_model_comparison()
        example_5_validation_framework()
        example_6_reproducibility()
        example_7_model_configuration()
        
        logger.info("=" * 70)
        logger.info("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        logger.info("\nKey Takeaways:")
        logger.info("  1. ✓ 5 models available (LogReg, DT, RF, SVM, GB)")
        logger.info("  2. ✓ Model registry for discovery and creation")
        logger.info("  3. ✓ Sklearn-compatible interface (fit/predict)")
        logger.info("  4. ✓ Reproducible training with random_state")
        logger.info("  5. ✓ Fair comparison with validation framework")
        logger.info("  6. ✓ Extensible design for custom models")
        logger.info("  7. ✓ Configuration management")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
