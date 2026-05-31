"""
Unified data preparation module.

Combines dataset loading and preprocessing into a single workflow.
Handles the complete data pipeline with reproducibility and leakage prevention.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from datasets.loaders import DatasetLoader
from datasets.registry import get_dataset_config
from preprocessing.pipelines import create_pipeline, PreprocessingPipeline

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Complete data preparation pipeline combining loading and preprocessing.
    
    Workflow:
    1. Load dataset
    2. Split into train/validation/test
    3. Fit preprocessing on training data
    4. Transform all splits
    5. Ready for model training
    
    Data Leakage Prevention:
    - Preprocessing is fitted ONLY on training data
    - Test/validation data never influences preprocessing parameters
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize data preparation.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.loader = DatasetLoader()
        self.preprocessing: Optional[PreprocessingPipeline] = None
        self.metadata: Dict[str, Any] = {}
        
    def prepare(
        self,
        dataset_name: str,
        train_size: float = 0.7,
        val_size: float = 0.15,
        scaling_method: str = 'standard',
        encoding_method: str = 'onehot',
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], pd.Series, pd.Series, Optional[pd.Series]]:
        """
        Complete data preparation pipeline.
        
        Loads dataset from registry, splits, and preprocesses.
        
        Args:
            dataset_name: Name registered in dataset registry
            train_size: Proportion for training (0.0-1.0)
            val_size: Proportion for validation (0.0-1.0)
            scaling_method: Feature scaling ('standard', 'minmax', 'robust')
            encoding_method: Categorical encoding ('onehot', 'ordinal')
            stratify: Whether to stratify splits (for classification)
        
        Returns:
            Tuple of:
            - X_train_processed: Processed training features
            - X_val_processed: Processed validation features
            - X_test_processed: Processed test features (or None if no test set)
            - y_train: Training target
            - y_val: Validation target
            - y_test: Test target (or None)
        
        Example:
            >>> prep = DataPreparation(random_state=42)
            >>> X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
            ...     dataset_name='iris',
            ...     train_size=0.7,
            ...     val_size=0.15,
            ...     scaling_method='standard'
            ... )
        """
        # Load dataset
        config = get_dataset_config(dataset_name)
        X, y, dataset_metadata = self.loader.load_csv(
            config['filepath'],
            config['target_column']
        )
        
        self.metadata = dataset_metadata
        logger.info(f"Loaded dataset: {dataset_name}")
        
        # Split into train/val/test
        X_train, X_remaining, y_train, y_remaining = self._first_split(
            X, y, train_size, stratify
        )
        
        # Calculate val/test split
        val_proportion = val_size / (1 - train_size)
        
        X_val, X_test, y_val, y_test = self._second_split(
            X_remaining, y_remaining, val_proportion, stratify
        )
        
        logger.info(
            f"Train/Val/Test split: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}"
        )
        
        # Create preprocessing pipeline
        self.preprocessing = create_pipeline(
            X_train,
            scaling_method=scaling_method,
            encoding_method=encoding_method,
            random_state=self.random_state,
        )
        
        # Fit on training data only - CRITICAL for preventing leakage
        logger.info("Fitting preprocessing on training data")
        X_train_processed = self.preprocessing.fit_and_transform(X_train, y_train)
        
        # Transform other splits using fitted parameters
        X_val_processed = self.preprocessing.transform(X_val)
        X_test_processed = self.preprocessing.transform(X_test)
        
        logger.info(
            f"Preprocessing complete. Output shape: {X_train_processed.shape}"
        )
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
    
    def prepare_train_test(
        self,
        dataset_name: str,
        test_size: float = 0.3,
        scaling_method: str = 'standard',
        encoding_method: str = 'onehot',
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Simpler preparation with just train/test split.
        
        Args:
            dataset_name: Name registered in dataset registry
            test_size: Proportion for testing (0.0-1.0)
            scaling_method: Feature scaling
            encoding_method: Categorical encoding
            stratify: Whether to stratify split
        
        Returns:
            Tuple of (X_train_processed, X_test_processed, y_train, y_test)
        
        Example:
            >>> prep = DataPreparation(random_state=42)
            >>> X_train, X_test, y_train, y_test = prep.prepare_train_test(
            ...     'iris',
            ...     test_size=0.3
            ... )
        """
        # Load dataset
        config = get_dataset_config(dataset_name)
        X, y, self.metadata = self.loader.load_csv(
            config['filepath'],
            config['target_column']
        )
        
        # Split
        X_train, X_test, y_train, y_test = self._first_split(
            X, y, 1 - test_size, stratify
        )
        
        # Preprocessing
        self.preprocessing = create_pipeline(
            X_train,
            scaling_method=scaling_method,
            encoding_method=encoding_method,
            random_state=self.random_state,
        )
        
        X_train_processed = self.preprocessing.fit_and_transform(X_train, y_train)
        X_test_processed = self.preprocessing.transform(X_test)
        
        logger.info(
            f"Preparation complete. "
            f"Train: {X_train_processed.shape}, Test: {X_test_processed.shape}"
        )
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def _first_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_size: float,
        stratify: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform first train/test split."""
        from sklearn.model_selection import train_test_split
        
        stratify_col = y if stratify else None
        
        return train_test_split(
            X, y,
            train_size=split_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
    
    def _second_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_proportion: float,
        stratify: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform second split of remaining data."""
        from sklearn.model_selection import train_test_split
        
        stratify_col = y if stratify else None
        
        return train_test_split(
            X, y,
            train_size=split_proportion,
            random_state=self.random_state,
            stratify=stratify_col
        )
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration for reproducibility."""
        if self.preprocessing is None:
            raise RuntimeError("No preprocessing configured yet")
        return self.preprocessing.get_config()
    
    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Get loaded dataset metadata."""
        return self.metadata


# Convenience function
def prepare_dataset(
    dataset_name: str,
    test_size: float = 0.3,
    random_state: int = 42,
    scaling_method: str = 'standard',
    encoding_method: str = 'onehot',
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Quick data preparation.
    
    Args:
        dataset_name: Registered dataset name
        test_size: Test proportion
        random_state: Random seed
        scaling_method: Feature scaling
        encoding_method: Categorical encoding
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    
    Example:
        >>> X_train, X_test, y_train, y_test, meta = prepare_dataset('iris', test_size=0.3)
    """
    prep = DataPreparation(random_state=random_state)
    X_train, X_test, y_train, y_test = prep.prepare_train_test(
        dataset_name,
        test_size=test_size,
        scaling_method=scaling_method,
        encoding_method=encoding_method,
    )
    return X_train, X_test, y_train, y_test, prep.get_dataset_metadata()
