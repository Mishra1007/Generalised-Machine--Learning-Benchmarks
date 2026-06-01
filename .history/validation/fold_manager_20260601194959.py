"""
Fold management for stratified k-fold cross-validation.

Handles fold generation, tracking, and reproducibility.
"""

import logging
from typing import List, Tuple, Generator, Dict, Any
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class FoldManager:
    """
    Manage stratified k-fold cross-validation splits.
    
    Features:
    - Generates stratified folds respecting class distribution
    - Reproducible splits with fixed random seed
    - Tracks fold information
    - Supports multiple repetitions
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_repetitions: int = 3,
        random_state: int = 42
    ):
        """
        Initialize fold manager.
        
        Args:
            n_splits: Number of folds (default 5)
            n_repetitions: Number of repetitions (default 3)
            random_state: Random seed for reproducibility (default 42)
        
        Example:
            >>> fold_manager = FoldManager(n_splits=5, n_repetitions=3, random_state=42)
        """
        self.n_splits = n_splits
        self.n_repetitions = n_repetitions
        self.random_state = random_state
        
        self.folds_cache = {}
        
        logger.info(
            f"FoldManager initialized: {n_splits}-fold CV, "
            f"{n_repetitions} repetitions, seed={random_state}"
        )
    
    def generate_folds(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Generator[Tuple[int, int, np.ndarray, np.ndarray], None, None]:
        """
        Generate stratified folds.
        
        Yields folds for multiple repetitions:
        - Repetition 0, Fold 0: train/test split
        - Repetition 0, Fold 1: train/test split
        - ...
        - Repetition 1, Fold 0: train/test split (different split pattern)
        - ...
        
        Args:
            X: Features (n_samples, n_features)
            y: Target labels (n_samples,)
        
        Yields:
            Tuple of (repetition_id, fold_id, train_indices, test_indices)
        
        Example:
            >>> for rep_id, fold_id, train_idx, test_idx in fold_manager.generate_folds(X, y):
            ...     X_train = X[train_idx]
            ...     X_test = X[test_idx]
            ...     # Train and evaluate...
        """
        for rep_id in range(self.n_repetitions):
            # Use different seed for each repetition
            # This creates different splits while maintaining reproducibility
            seed = self.random_state + rep_id
            
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=seed
            )
            
            fold_id = 0
            for train_idx, test_idx in skf.split(X, y):
                logger.debug(
                    f"Generated fold - Repetition: {rep_id}, Fold: {fold_id}, "
                    f"Train: {len(train_idx)}, Test: {len(test_idx)}"
                )
                
                yield rep_id, fold_id, train_idx, test_idx
                fold_id += 1
    
    def get_fold_info(
        self,
        X: np.ndarray,
        y: np.ndarray,
        repetition_id: int,
        fold_id: int
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific fold.
        
        Args:
            X: Features
            y: Target labels
            repetition_id: Repetition identifier (0 to n_repetitions-1)
            fold_id: Fold identifier (0 to n_splits-1)
        
        Returns:
            Dictionary with fold information
        """
        seed = self.random_state + repetition_id
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=seed
        )
        
        fold_idx = 0
        for train_idx, test_idx in skf.split(X, y):
            if fold_idx == fold_id:
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                return {
                    'repetition_id': repetition_id,
                    'fold_id': fold_id,
                    'train_indices': train_idx,
                    'test_indices': test_idx,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_class_distribution': _class_distribution(y_train),
                    'test_class_distribution': _class_distribution(y_test),
                }
            fold_idx += 1
        
        raise ValueError(
            f"Fold {fold_id} not found for repetition {repetition_id}"
        )
    
    def get_all_fold_info(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """
        Get information for all folds and repetitions.
        
        Args:
            X: Features
            y: Target labels
        
        Returns:
            List of dictionaries, one per fold
        """
        all_folds = []
        
        for rep_id in range(self.n_repetitions):
            for fold_id in range(self.n_splits):
                fold_info = self.get_fold_info(X, y, rep_id, fold_id)
                all_folds.append(fold_info)
        
        logger.info(f"Generated fold info for {len(all_folds)} folds total")
        
        return all_folds
    
    def validate_fold_indices(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate that fold indices are non-overlapping and complete.
        
        Args:
            X: Features
            y: Target labels
        
        Returns:
            True if validation passes, raises exception otherwise
        """
        n_samples = len(y)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Fold validation failed: n_splits={self.n_splits} greater than n_samples={n_samples}"
            )

        classes, counts = np.unique(y, return_counts=True)
        if len(counts) == 0:
            raise ValueError('Fold validation failed: empty target labels')
        min_class = int(counts.min())
        if self.n_splits > min_class:
            raise ValueError(
                f"Fold validation failed: n_splits={self.n_splits} greater than smallest class count={min_class}"
            )

        all_test_idx = []
        
        for rep_id, fold_id, train_idx, test_idx in self.generate_folds(X, y):
            # Check no overlap between train and test
            overlap = set(train_idx) & set(test_idx)
            if overlap:
                raise ValueError(
                    f"Overlap in fold indices (Rep {rep_id}, Fold {fold_id})"
                )
            
            # Collect test indices
            all_test_idx.extend(test_idx)
        
        # Check all indices used
        all_test_idx = sorted(all_test_idx)
        expected_idx = sorted(list(range(len(X))) * self.n_repetitions)
        
        if all_test_idx != expected_idx:
            raise ValueError("Fold indices don't cover all samples correctly")
        
        logger.info("✓ Fold indices validation passed")
        return True


def _class_distribution(y: np.ndarray) -> Dict[int, float]:
    """
    Get class distribution as percentages.
    
    Args:
        y: Target labels
    
    Returns:
        Dictionary mapping class to percentage
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    return {
        int(cls): float(count / total * 100)
        for cls, count in zip(classes, counts)
    }


def get_fold_statistics(fold_info_list: List[Dict]) -> Dict[str, Any]:
    """
    Compute statistics about folds.
    
    Args:
        fold_info_list: List of fold information dictionaries
    
    Returns:
        Dictionary with fold statistics
    """
    train_sizes = [f['train_size'] for f in fold_info_list]
    test_sizes = [f['test_size'] for f in fold_info_list]
    
    return {
        'total_folds': len(fold_info_list),
        'train_size_mean': np.mean(train_sizes),
        'train_size_std': np.std(train_sizes),
        'test_size_mean': np.mean(test_sizes),
        'test_size_std': np.std(test_sizes),
        'train_sizes': train_sizes,
        'test_sizes': test_sizes,
    }
