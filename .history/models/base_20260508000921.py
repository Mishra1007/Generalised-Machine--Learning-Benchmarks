"""Base model wrapper providing unified interface."""

from abc import ABC, abstractmethod
import logging
from typing import Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all model wrappers.
    
    Provides unified interface for training, prediction, and model management.
    All models inherit from this class to ensure consistency.
    
    Architecture:
    - Wraps sklearn models
    - Provides logging and configuration
    - Ensures reproducibility with random_state
    - Maintains sklearn interface for validation engine compatibility
    
    Example:
        >>> from models import LogisticRegressionModel
        >>> model = LogisticRegressionModel(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        name: str,
        task_type: str = 'classification',
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize base model.
        
        Args:
            name: Model identifier (e.g., 'LogisticRegression')
            task_type: 'classification' or 'regression' (classification only for now)
            random_state: Random seed for reproducibility
            verbose: Enable verbose logging
        """
        self.name = name
        self.task_type = task_type
        self.random_state = random_state
        self.verbose = verbose
        
        self._model = None  # Actual sklearn model, set by subclass
        self._is_fitted = False
        self._classes = None
        self._n_features = None
        
        logger.debug(
            f"Initialized {self.__class__.__name__} "
            f"(name={name}, random_state={random_state})"
        )
    
    @abstractmethod
    def _build_model(self) -> Any:
        """
        Build and return the underlying sklearn model.
        
        Must be implemented by subclasses.
        
        Returns:
            sklearn model instance
        """
        pass
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self (for method chaining)
        """
        if not self._model:
            self._model = self._build_model()
        
        if self.verbose:
            logger.info(
                f"Training {self.name}: {X.shape[0]} samples, {X.shape[1]} features"
            )
        
        try:
            self._model.fit(X, y)
            self._is_fitted = True
            self._classes = np.unique(y)
            self._n_features = X.shape[1]
            
            logger.debug(
                f"{self.name} fitted successfully. "
                f"Classes: {len(self._classes)}, Features: {self._n_features}"
            )
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before prediction")
        
        if self.verbose:
            logger.info(f"Predicting {self.name} on {X.shape[0]} samples")
        
        try:
            return self._model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (if supported).
        
        Args:
            X: Features to predict on
        
        Returns:
            Probability predictions (n_samples, n_classes) or None
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before prediction")
        
        if not hasattr(self._model, 'predict_proba'):
            logger.debug(f"{self.name} doesn't support predict_proba")
            return None
        
        try:
            return self._model.predict_proba(X)
        except Exception as e:
            logger.debug(f"predict_proba failed for {self.name}: {e}")
            return None
    
    def get_config(self) -> dict:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model name, type, and parameters
        """
        return {
            'name': self.name,
            'task_type': self.task_type,
            'random_state': self.random_state,
            'is_fitted': self._is_fitted,
            'n_features': self._n_features,
            'n_classes': len(self._classes) if self._classes is not None else None,
        }
    
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted
    
    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self.name}, {status})"
