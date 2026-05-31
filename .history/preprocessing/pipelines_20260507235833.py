"""
Preprocessing pipelines for feature engineering.

Uses scikit-learn pipelines to ensure reproducibility and prevent data leakage.
Pipelines are fitted only on training data.
"""

import logging
from typing import Optional, List, Literal, Tuple
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Build and manage preprocessing pipelines.
    
    Features:
    - Separate handling of numerical and categorical features
    - Configurable scaling and encoding strategies
    - Missing value imputation
    - Data leakage prevention (fit on train only)
    - Sklearn Pipeline compatibility
    
    Workflow:
    1. Create pipeline with fit_and_transform(X_train, y_train)
    2. Transform test/validation data with transform(X_test)
    """
    
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        numerical_strategy: str = 'mean',
        categorical_strategy: str = 'most_frequent',
        scaling_method: Literal['standard', 'minmax', 'robust'] = 'standard',
        encoding_method: Literal['onehot', 'ordinal'] = 'onehot',
        random_state: int = 42,
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            numerical_strategy: Missing value strategy for numerical features
                ('mean', 'median', 'constant')
            categorical_strategy: Missing value strategy for categorical features
                ('most_frequent', 'constant')
            scaling_method: Feature scaling ('standard', 'minmax', 'robust')
            encoding_method: Categorical encoding ('onehot', 'ordinal')
            random_state: Random seed for reproducibility
        
        Example:
            >>> pipeline = PreprocessingPipeline(
            ...     numerical_features=['age', 'income'],
            ...     categorical_features=['gender', 'city'],
            ...     scaling_method='standard'
            ... )
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.random_state = random_state
        
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        
        logger.info(
            f"PreprocessingPipeline created: "
            f"{len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical features"
        )
    
    def build(self) -> Pipeline:
        """
        Build the sklearn pipeline.
        
        Constructs:
        - Numerical transformer: Imputation → Scaling
        - Categorical transformer: Imputation → Encoding
        - ColumnTransformer to apply both
        
        Returns:
            Fitted Pipeline object
        """
        # Numerical pipeline: Handle missing → Scale
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.numerical_strategy)),
            ('scaler', self._get_scaler()),
        ])
        
        # Categorical pipeline: Handle missing → Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.categorical_strategy, fill_value='missing')),
            ('encoder', self._get_encoder()),
        ])
        
        # Combine using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features),
            ],
            remainder='drop',  # Drop unknown columns
            verbose_feature_names_out=False,
        )
        
        # Wrap in pipeline for easy fit/transform
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
        ])
        
        logger.debug("Preprocessing pipeline built")
        return self.pipeline
    
    def fit_and_transform(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit pipeline on training data and transform it.
        
        CRITICAL: Only fit on training data to prevent data leakage.
        
        Args:
            X_train: Training features (fit will be applied here)
            y_train: Training target (not used in preprocessing, but included for API consistency)
        
        Returns:
            Transformed training data as numpy array
        
        Example:
            >>> X_train_processed = pipeline.fit_and_transform(X_train, y_train)
        """
        if self.pipeline is None:
            self.build()
        
        logger.info(f"Fitting pipeline on {X_train.shape[0]} samples")
        X_processed = self.pipeline.fit_transform(X_train)
        self.is_fitted = True
        
        logger.info(f"Training data transformed: {X_processed.shape}")
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted pipeline to new data.
        
        Uses parameters learned from training data.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed data as numpy array
        
        Raises:
            RuntimeError: If pipeline not fitted yet
        
        Example:
            >>> X_test_processed = pipeline.transform(X_test)
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError(
                "Pipeline not fitted. Call fit_and_transform() on training data first."
            )
        
        logger.debug(f"Transforming {X.shape[0]} samples")
        X_processed = self.pipeline.transform(X)
        
        return X_processed
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit pipeline on data and transform it (for compatibility).
        
        Note: Use fit_and_transform() with separate train/test splits.
        
        Args:
            X: Features
            y: Target (optional)
        
        Returns:
            Transformed data
        """
        return self.fit_and_transform(X, y)
    
    def get_feature_names_out(self) -> np.ndarray:
        """
        Get names of output features after transformation.
        
        Returns:
            Array of feature names
        
        Raises:
            RuntimeError: If pipeline not fitted
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Pipeline must be fitted first")
        
        return self.pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    def _get_scaler(self):
        """Get scaler based on configuration."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        return scalers[self.scaling_method]
    
    def _get_encoder(self):
        """Get encoder based on configuration."""
        encoders = {
            'onehot': OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='if_binary'
            ),
            'ordinal': OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ),
        }
        return encoders[self.encoding_method]
    
    def get_config(self) -> dict:
        """
        Get pipeline configuration for reproducibility.
        
        Returns:
            Dictionary with all pipeline settings
        """
        return {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'numerical_strategy': self.numerical_strategy,
            'categorical_strategy': self.categorical_strategy,
            'scaling_method': self.scaling_method,
            'encoding_method': self.encoding_method,
            'random_state': self.random_state,
        }


def create_pipeline(
    X_train: pd.DataFrame,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scaling_method: str = 'standard',
    encoding_method: str = 'onehot',
    random_state: int = 42,
) -> PreprocessingPipeline:
    """
    Create and build a preprocessing pipeline.
    
    Auto-infers feature types if not provided.
    
    Args:
        X_train: Training data for type inference
        numerical_features: List of numerical columns (auto-inferred if None)
        categorical_features: List of categorical columns (auto-inferred if None)
        scaling_method: Scaling method ('standard', 'minmax', 'robust')
        encoding_method: Encoding method ('onehot', 'ordinal')
        random_state: Random seed
    
    Returns:
        PreprocessingPipeline instance (not yet fitted)
    
    Example:
        >>> pipeline = create_pipeline(
        ...     X_train,
        ...     scaling_method='standard',
        ...     encoding_method='onehot'
        ... )
        >>> X_train_processed = pipeline.fit_and_transform(X_train)
        >>> X_test_processed = pipeline.transform(X_test)
    """
    # Auto-infer types if not provided
    if numerical_features is None:
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if categorical_features is None:
        categorical_features = X_train.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
    
    logger.info(
        f"Creating pipeline with {len(numerical_features)} numerical, "
        f"{len(categorical_features)} categorical features"
    )
    
    pipeline = PreprocessingPipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        scaling_method=scaling_method,
        encoding_method=encoding_method,
        random_state=random_state,
    )
    
    pipeline.build()
    return pipeline
