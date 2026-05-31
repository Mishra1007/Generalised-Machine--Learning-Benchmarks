"""
Dataset registry for managing available datasets.

Maintains metadata about datasets and enables discovery and configuration.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class DatasetRegistry:
    """
    Registry of available datasets for benchmarking.
    
    Maintains dataset metadata and configurations for easy lookup
    and experiment configuration.
    """
    
    def __init__(self):
        """Initialize dataset registry."""
        self._datasets: Dict[str, Dict[str, Any]] = {}
        logger.info("DatasetRegistry initialized")
    
    def register(
        self,
        name: str,
        filepath: str | Path,
        target_column: str,
        description: str = "",
        task_type: str = "classification",
        **metadata
    ) -> None:
        """
        Register a dataset.
        
        Args:
            name: Unique dataset identifier
            filepath: Path to dataset file
            target_column: Name of target column
            description: Dataset description
            task_type: 'classification' or 'regression'
            **metadata: Additional metadata (features, source, etc.)
        
        Example:
            >>> registry = DatasetRegistry()
            >>> registry.register(
            ...     name='iris',
            ...     filepath='data/iris.csv',
            ...     target_column='species',
            ...     description='Iris flower classification',
            ...     task_type='classification'
            ... )
        """
        filepath = Path(filepath)
        
        self._datasets[name] = {
            'filepath': str(filepath),
            'target_column': target_column,
            'description': description,
            'task_type': task_type,
            **metadata
        }
        
        logger.info(f"Registered dataset: {name} ({task_type})")
    
    def get(self, name: str) -> Dict[str, Any]:
        """
        Get dataset configuration by name.
        
        Args:
            name: Dataset identifier
        
        Returns:
            Dataset configuration dictionary
        
        Raises:
            KeyError: If dataset not found
        """
        if name not in self._datasets:
            available = list(self._datasets.keys())
            raise KeyError(
                f"Dataset '{name}' not found. "
                f"Available datasets: {available}"
            )
        
        return self._datasets[name]
    
    def list_datasets(self) -> Dict[str, str]:
        """
        List all registered datasets.
        
        Returns:
            Dictionary mapping dataset names to descriptions
        """
        return {
            name: config.get('description', '')
            for name, config in self._datasets.items()
        }
    
    def list_by_task(self, task_type: str) -> Dict[str, str]:
        """
        List datasets by task type.
        
        Args:
            task_type: 'classification' or 'regression'
        
        Returns:
            Dictionary mapping dataset names to descriptions
        """
        return {
            name: config.get('description', '')
            for name, config in self._datasets.items()
            if config.get('task_type') == task_type
        }
    
    def exists(self, name: str) -> bool:
        """Check if dataset is registered."""
        return name in self._datasets
    
    def clear(self) -> None:
        """Clear all registered datasets."""
        self._datasets.clear()
        logger.info("Dataset registry cleared")


# Global registry instance
_global_registry = DatasetRegistry()


def register_dataset(
    name: str,
    filepath: str | Path,
    target_column: str,
    description: str = "",
    task_type: str = "classification",
    **metadata
) -> None:
    """
    Register a dataset in the global registry.
    
    Args:
        name: Dataset identifier
        filepath: Path to dataset file
        target_column: Target column name
        description: Dataset description
        task_type: 'classification' or 'regression'
        **metadata: Additional metadata
    
    Example:
        >>> register_dataset(
        ...     'breast_cancer',
        ...     'data/breast_cancer.csv',
        ...     'diagnosis',
        ...     description='Breast cancer classification',
        ...     task_type='classification'
        ... )
    """
    _global_registry.register(
        name, filepath, target_column, description, task_type, **metadata
    )


def get_dataset_config(name: str) -> Dict[str, Any]:
    """
    Get dataset configuration from global registry.
    
    Args:
        name: Dataset identifier
    
    Returns:
        Dataset configuration
    """
    return _global_registry.get(name)


def list_datasets() -> Dict[str, str]:
    """List all registered datasets."""
    return _global_registry.list_datasets()


def get_registry() -> DatasetRegistry:
    """Get the global dataset registry."""
    return _global_registry
