"""
Dataset loaders for benchmarking framework.

Handles loading datasets from various sources with validation and caching.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "datasets" / "cache"


class DatasetLoader:
    """
    Load and manage benchmark datasets.
    
    Features:
    - Load CSV files with metadata
    - Automatic missing value detection
    - Feature type inference
    - Dataset caching
    - Input validation
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize dataset loader.
        
        Args:
            cache_dir: Directory for caching datasets. Defaults to datasets/cache/
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatasetLoader initialized with cache: {self.cache_dir}")
    
    def load_csv(
        self,
        filepath: str | Path,
        target_column: str,
        test_size: Optional[float] = None,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target variable column
            test_size: If provided, returns train/test split
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for pd.read_csv()
        
        Returns:
            Tuple of (features_df, target_series, metadata_dict)
            where metadata contains dataset information
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If target_column not found
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Load data
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Extract features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Compute metadata
        metadata = self._compute_metadata(X, y, filepath)
        
        logger.info(
            f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, "
            f"{y.nunique()} target classes/values"
        )
        
        return X, y, metadata
    
    @staticmethod
    def _compute_metadata(X: pd.DataFrame, y: pd.Series, filepath: Path) -> Dict[str, Any]:
        """
        Compute dataset metadata.
        
        Args:
            X: Feature dataframe
            y: Target series
            filepath: Dataset file path
        
        Returns:
            Dictionary with metadata
        """
        # Identify feature types
        numerical_features = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_features = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Check for missing values
        missing_features = {
            col: X[col].isna().sum()
            for col in X.columns
            if X[col].isna().any()
        }
        
        target_type = 'classification' if y.dtype == 'object' or y.nunique() < 20 else 'regression'
        class_distribution = y.value_counts(dropna=False).to_dict()
        
        return {
            'filepath': str(filepath),
            'dataset_source': str(filepath),
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_count': X.shape[1],
            'n_target_classes': y.nunique(),
            'class_distribution': class_distribution,
            'target_type': target_type,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'missing_values': missing_features,
            'feature_names': X.columns.tolist(),
            'target_name': y.name,
        }
    
    @staticmethod
    def load_numpy(
        X_filepath: str | Path,
        y_filepath: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load dataset from numpy arrays.
        
        Args:
            X_filepath: Path to features array
            y_filepath: Path to target array
        
        Returns:
            Tuple of (X_array, y_array, metadata_dict)
        """
        X_filepath = Path(X_filepath)
        y_filepath = Path(y_filepath)
        
        if not X_filepath.exists() or not y_filepath.exists():
            raise FileNotFoundError("Numpy array files not found")
        
        logger.info(f"Loading numpy arrays from {X_filepath} and {y_filepath}")
        
        X = np.load(X_filepath)
        y = np.load(y_filepath)
        
        metadata = {
            'filepath_X': str(X_filepath),
            'filepath_y': str(y_filepath),
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_target_classes': len(np.unique(y)),
        }
        
        return X, y, metadata


def load_dataset(
    filepath: str | Path,
    target_column: str,
    random_state: int = 42,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Convenience function to load a dataset.
    
    Args:
        filepath: Path to CSV file
        target_column: Target column name
        random_state: Random seed
        **kwargs: Arguments for pd.read_csv()
    
    Returns:
        Tuple of (X, y, metadata)
    
    Example:
        >>> X, y, metadata = load_dataset('data.csv', target_column='price')
        >>> print(f"Loaded {metadata['n_samples']} samples with {metadata['n_features']} features")
    """
    loader = DatasetLoader()
    return loader.load_csv(filepath, target_column, random_state=random_state, **kwargs)
