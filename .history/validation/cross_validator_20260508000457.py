"""
Cross-validation engine for reproducible benchmarking.

Orchestrates stratified 5-fold CV repeated 3 times with complete logging.
"""

import logging
import time
from typing import Tuple, Optional, Callable
import numpy as np

from validation.fold_manager import FoldManager
from validation.results import ValidationResults, FoldResult
from metrics.scorers import compute_fold_metrics

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Reproducible cross-validation engine.
    
    Performs stratified 5-fold CV repeated 3 times with full logging
    and metric collection.
    
    Architecture:
    - Fixed random seed (42) for reproducibility
    - Stratified splits preserve class distribution
    - Complete fold-level metric logging
    - Fair benchmarking across all models
    
    Usage:
        >>> validator = CrossValidator(n_splits=5, n_repetitions=3, random_state=42)
        >>> results = validator.validate(
        ...     X_train, y_train, model, 'model_name', 'dataset_name'
        ... )
        >>> summary = results.get_summary()
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_repetitions: int = 3,
        random_state: int = 42
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds (default 5 for stratified 5-fold CV)
            n_repetitions: Number of repetitions (default 3)
            random_state: Random seed for reproducibility (default 42)
        """
        self.n_splits = n_splits
        self.n_repetitions = n_repetitions
        self.random_state = random_state
        
        self.fold_manager = FoldManager(
            n_splits=n_splits,
            n_repetitions=n_repetitions,
            random_state=random_state
        )
        
        logger.info(
            f"CrossValidator initialized: {n_splits}-fold CV × {n_repetitions} reps, "
            f"random_state={random_state}"
        )
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        model_name: str,
        dataset_name: str,
        predict_proba: bool = True,
        return_predictions: bool = False,
    ) -> ValidationResults:
        """
        Perform cross-validation on a model.
        
        Executes stratified 5-fold CV repeated 3 times with complete logging.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            model: sklearn-compatible model with fit() and predict() methods
            model_name: Name of model for logging
            dataset_name: Name of dataset for logging
            predict_proba: Whether to collect prediction probabilities
            return_predictions: Whether to store predictions in results
        
        Returns:
            ValidationResults object with all fold metrics
        
        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> X_train, y_train = load_preprocessed_data()
            >>> model = RandomForestClassifier(random_state=42)
            >>> validator = CrossValidator()
            >>> results = validator.validate(
            ...     X_train, y_train, model,
            ...     model_name='RandomForest',
            ...     dataset_name='iris'
            ... )
            >>> summary = results.get_summary()
            >>> print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        """
        logger.info(
            f"\n{'=' * 60}\n"
            f"Starting cross-validation: {model_name} on {dataset_name}\n"
            f"Configuration: {self.n_splits}-fold CV × {self.n_repetitions} reps\n"
            f"{'=' * 60}\n"
        )
        
        results = ValidationResults(
            model_name=model_name,
            dataset_name=dataset_name,
            random_state=self.random_state
        )
        
        # Track overall timing
        cv_start_time = time.time()
        
        # Validate fold indices to ensure no leakage
        try:
            self.fold_manager.validate_fold_indices(X, y)
        except Exception as e:
            logger.error(f"Fold validation failed: {e}")
            raise
        
        # Iterate through all folds
        fold_count = 0
        total_folds = self.n_splits * self.n_repetitions
        
        for rep_id, fold_id, train_idx, test_idx in self.fold_manager.generate_folds(X, y):
            fold_count += 1
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            logger.info(
                f"\n[{fold_count}/{total_folds}] Rep {rep_id}, Fold {fold_id}: "
                f"Train {len(train_idx)}, Test {len(test_idx)}"
            )
            
            # Train model
            train_start = time.time()
            try:
                model.fit(X_train, y_train)
                train_time = time.time() - train_start
                logger.debug(f"  Training completed in {train_time:.4f}s")
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                raise
            
            # Make predictions
            eval_start = time.time()
            try:
                y_pred = model.predict(X_test)
                
                # Try to get prediction probabilities
                y_pred_proba = None
                if predict_proba:
                    try:
                        y_pred_proba = model.predict_proba(X_test)
                    except (AttributeError, NotImplementedError):
                        logger.debug(f"  Model doesn't support predict_proba")
                
                eval_time = time.time() - eval_start
                logger.debug(f"  Prediction completed in {eval_time:.4f}s")
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                raise
            
            # Compute metrics
            try:
                metrics = compute_fold_metrics(
                    y_test, y_pred, y_pred_proba,
                    fold_id=fold_id,
                    repetition_id=rep_id
                )
                logger.info(
                    f"  Metrics: Accuracy={metrics.get('accuracy', 0):.4f}, "
                    f"F1={metrics.get('f1', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"Metric computation failed: {e}")
                raise
            
            # Store result
            fold_result = FoldResult(
                repetition_id=rep_id,
                fold_id=fold_id,
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_time=train_time,
                eval_time=eval_time,
                y_test=y_test if return_predictions else None,
                y_pred=y_pred if return_predictions else None,
                y_pred_proba=y_pred_proba if return_predictions else None,
                test_indices=test_idx.tolist() if return_predictions else None,
            )
            
            results.add_fold_result(fold_result)
        
        cv_total_time = time.time() - cv_start_time
        
        # Log summary
        summary = results.get_summary()
        
        logger.info(
            f"\n{'=' * 60}\n"
            f"CROSS-VALIDATION COMPLETE: {model_name}\n"
            f"{'=' * 60}"
        )
        logger.info(f"Total folds evaluated: {results.get_summary()['total_folds']}")
        logger.info(f"Total time: {cv_total_time:.4f}s")
        logger.info(
            f"Accuracy: {summary.get('accuracy_mean', 0):.4f} ± "
            f"{summary.get('accuracy_std', 0):.4f}"
        )
        logger.info(
            f"F1-Score: {summary.get('f1_mean', 0):.4f} ± "
            f"{summary.get('f1_std', 0):.4f}"
        )
        logger.info("=" * 60 + "\n")
        
        return results
    
    def validate_multiple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: dict,
        dataset_name: str,
    ) -> dict:
        """
        Validate multiple models on the same dataset.
        
        Ensures all models are evaluated under identical conditions.
        
        Args:
            X: Training features
            y: Training labels
            models: Dictionary mapping model_name to model instance
            dataset_name: Name of dataset
        
        Returns:
            Dictionary mapping model_name to ValidationResults
        
        Example:
            >>> models = {
            ...     'rf': RandomForestClassifier(random_state=42),
            ...     'svm': SVC(random_state=42),
            ...     'lr': LogisticRegression(random_state=42),
            ... }
            >>> results = validator.validate_multiple(
            ...     X_train, y_train, models, 'iris'
            ... )
            >>> for name, res in results.items():
            ...     print(f"{name}: {res.get_summary()['accuracy_mean']:.4f}")
        """
        all_results = {}
        
        logger.info(
            f"\nValidating {len(models)} models on {dataset_name}\n"
            f"All models evaluated under identical conditions\n"
        )
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating: {model_name}")
            
            try:
                results = self.validate(
                    X, y, model,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    predict_proba=True,
                    return_predictions=False,
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Validation failed for {model_name}: {e}")
                all_results[model_name] = None
        
        logger.info(
            f"\nValidation of all models complete\n"
            f"Results available for {len([r for r in all_results.values() if r])} models"
        )
        
        return all_results


def validate_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    fold_id: int = 0,
    repetition_id: int = 0,
) -> FoldResult:
    """
    Validate a single fold (utility function).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model: sklearn-compatible model
        fold_id: Fold identifier
        repetition_id: Repetition identifier
    
    Returns:
        FoldResult with metrics
    """
    # Train
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # Predict
    eval_start = time.time()
    y_pred = model.predict(X_test)
    
    y_pred_proba = None
    try:
        y_pred_proba = model.predict_proba(X_test)
    except (AttributeError, NotImplementedError):
        pass
    
    eval_time = time.time() - eval_start
    
    # Compute metrics
    metrics = compute_fold_metrics(
        y_test, y_pred, y_pred_proba,
        fold_id=fold_id,
        repetition_id=repetition_id
    )
    
    return FoldResult(
        repetition_id=repetition_id,
        fold_id=fold_id,
        model_name='model',
        dataset_name='dataset',
        metrics=metrics,
        train_size=len(X_train),
        test_size=len(X_test),
        train_time=train_time,
        eval_time=eval_time,
    )
