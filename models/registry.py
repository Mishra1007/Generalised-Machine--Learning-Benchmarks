"""Model registry and factory functions."""

import logging
from typing import Dict, Type, Any, Optional

from models.base import BaseModel
from models.implementations import (
    LogisticRegressionModel,
    DecisionTreeModel,
    RandomForestModel,
    SVMModel,
    GradientBoostingModel,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing available models.
    
    Provides:
    - Centralized model catalog
    - Model creation factory
    - Configuration management
    - Model discovery
    
    Architecture:
    - Static registry of model classes
    - Global instance for application-wide use
    - Factory methods for instantiation
    
    Example:
        >>> registry = ModelRegistry()
        >>> model = registry.create('RandomForest', n_estimators=100)
        >>> available = registry.list_models()
    """
    
    # Registry of available models
    _models: Dict[str, Type[BaseModel]] = {
        'LogisticRegression': LogisticRegressionModel,
        'DecisionTree': DecisionTreeModel,
        'RandomForest': RandomForestModel,
        'SVM': SVMModel,
        'GradientBoosting': GradientBoostingModel,
    }
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model class.
        
        Args:
            name: Model identifier (e.g., 'MyModel')
            model_class: Model class (must inherit from BaseModel)
        
        Example:
            >>> class CustomModel(BaseModel):
            ...     def _build_model(self):
            ...         return ...
            >>> ModelRegistry.register('Custom', CustomModel)
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must inherit from BaseModel")
        
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create(
        cls,
        name: str,
        **kwargs,
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            name: Model identifier
            **kwargs: Model-specific configuration
        
        Returns:
            Model instance
        
        Raises:
            KeyError: If model not registered
        
        Example:
            >>> model = ModelRegistry.create('RandomForest', n_estimators=50)
            >>> model = ModelRegistry.create('LogisticRegression', max_iter=2000)
        """
        # Accept simple aliases for backward compatibility (e.g., 'lr' -> 'LogisticRegression')
        aliases = {
            'lr': 'LogisticRegression',
            'rf': 'RandomForest',
            'dt': 'DecisionTree',
            'svm': 'SVM',
            'gb': 'GradientBoosting',
        }
        if name not in cls._models:
            lname = name.lower() if isinstance(name, str) else name
            if isinstance(lname, str) and lname in aliases:
                name = aliases[lname]

        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available: {available}"
            )
        
        model_class = cls._models[name]
        
        try:
            model = model_class(**kwargs)
            logger.debug(f"Created model: {name} with config {kwargs}")
            return model
        except TypeError as e:
            logger.error(f"Invalid configuration for {name}: {e}")
            raise
    
    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """
        Get model class (not instance).
        
        Args:
            name: Model identifier
        
        Returns:
            Model class
        
        Raises:
            KeyError: If model not registered
        """
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not registered")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list:
        """
        List all registered models.
        
        Returns:
            List of model names
        
        Example:
            >>> models = ModelRegistry.list_models()
            >>> print(models)
            ['LogisticRegression', 'DecisionTree', 'RandomForest', 'SVM', 'GradientBoosting']
        """
        return sorted(list(cls._models.keys()))
    
    @classmethod
    def get_model_info(cls, name: str) -> dict:
        """
        Get information about a model.
        
        Args:
            name: Model identifier
        
        Returns:
            Dictionary with model info
        
        Example:
            >>> info = ModelRegistry.get_model_info('RandomForest')
        """
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not registered")
        
        model_class = cls._models[name]
        
        return {
            'name': name,
            'class': model_class.__name__,
            'docstring': model_class.__doc__,
        }


# Global registry instance
_global_registry = None


def get_registry() -> ModelRegistry:
    """
    Get global model registry instance.
    
    Returns:
        ModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def create_model(name: str, **kwargs) -> BaseModel:
    """
    Convenience function to create a model.
    
    Args:
        name: Model identifier
        **kwargs: Model-specific configuration
    
    Returns:
        Model instance
    
    Example:
        >>> model = create_model('RandomForest', n_estimators=50, max_depth=10)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """
    return get_registry().create(name, **kwargs)


def list_models() -> list:
    """
    Convenience function to list available models.
    
    Returns:
        List of model names
    
    Example:
        >>> models = list_models()
        >>> print(models)
    """
    return get_registry().list_models()


def get_model_info(name: str) -> dict:
    """
    Convenience function to get model information.
    
    Args:
        name: Model identifier
    
    Returns:
        Model info dictionary
    """
    return get_registry().get_model_info(name)


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Convenience function to register a model.
    
    Args:
        name: Model identifier
        model_class: Model class
    
    Example:
        >>> class MyModel(BaseModel):
        ...     def _build_model(self):
        ...         return ...
        >>> register_model('MyModel', MyModel)
    """
    get_registry().register(name, model_class)
