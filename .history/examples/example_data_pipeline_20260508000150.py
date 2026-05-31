"""
Example: Dataset loading and preprocessing pipeline.

This example demonstrates:
1. Registering datasets in the framework
2. Loading and exploring datasets
3. Building preprocessing pipelines
4. Handling train/test splits with no data leakage
5. Integrating with future cross-validation workflows

Run with: python examples/example_data_pipeline.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_register_datasets():
    """Example 1: Register datasets in the framework."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Register Datasets")
    logger.info("=" * 60)
    
    from datasets import register_dataset, list_datasets
    
    # Register example datasets
    # In practice, these would point to actual CSV files
    register_dataset(
        name='iris',
        filepath='data/iris.csv',
        target_column='species',
        description='Iris flower classification dataset',
        task_type='classification',
        n_features=4,
    )
    
    register_dataset(
        name='breast_cancer',
        filepath='data/breast_cancer.csv',
        target_column='diagnosis',
        description='Breast cancer classification dataset',
        task_type='classification',
    )
    
    register_dataset(
        name='housing',
        filepath='data/housing.csv',
        target_column='price',
        description='Housing price regression dataset',
        task_type='regression',
    )
    
    logger.info("\nRegistered datasets:")
    for name, description in list_datasets().items():
        logger.info(f"  - {name}: {description}")
    
    print()


def example_load_dataset():
    """Example 2: Load dataset from CSV."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Load Dataset")
    logger.info("=" * 60)
    
    from datasets import load_dataset
    import pandas as pd
    
    # Create example CSV for demonstration
    example_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'income': [30000, 40000, 50000, 60000, 70000, 80000],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'education': ['HS', 'BS', 'MS', 'BS', 'PhD', 'HS'],
        'target': [0, 1, 1, 1, 1, 0],
    })
    
    # Save example data
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    example_path = data_dir / 'example.csv'
    example_data.to_csv(example_path, index=False)
    
    logger.info(f"Created example dataset at {example_path}")
    
    # Load dataset
    X, y, metadata = load_dataset(example_path, target_column='target')
    
    logger.info(f"\nDataset loaded successfully!")
    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Target shape: {y.shape}")
    logger.info(f"\nDataset metadata:")
    for key, value in metadata.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"\nFeatures sample:\n{X.head()}")
    
    print()
    return X, y, metadata


def example_create_pipeline(X):
    """Example 3: Create preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Create Preprocessing Pipeline")
    logger.info("=" * 60)
    
    from preprocessing import create_pipeline
    import numpy as np
    
    # Create pipeline with inferred feature types
    pipeline = create_pipeline(
        X,
        scaling_method='standard',
        encoding_method='onehot',
        random_state=42,
    )
    
    logger.info(f"Pipeline created with configuration:")
    config = pipeline.get_config()
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    print()
    return pipeline


def example_fit_transform(X, y, pipeline):
    """Example 4: Fit and transform with no data leakage."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Fit and Transform (No Data Leakage)")
    logger.info("=" * 60)
    
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"Split data:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    
    # CRITICAL: Fit ONLY on training data
    logger.info(f"\nFitting preprocessing on training data ONLY...")
    X_train_processed = pipeline.fit_and_transform(X_train, y_train)
    
    logger.info(f"✓ Preprocessing fitted on {X_train.shape[0]} training samples")
    logger.info(f"  Output shape: {X_train_processed.shape}")
    
    # Transform test data using fitted parameters
    logger.info(f"\nTransforming test data using fitted parameters...")
    X_test_processed = pipeline.transform(X_test)
    
    logger.info(f"✓ Test data transformed: {X_test_processed.shape}")
    logger.info(f"\nNo data leakage! Test parameters didn't influence preprocessing.")
    
    logger.info(f"\nProcessed training data sample (first 3 rows):")
    logger.info(f"\n{X_train_processed[:3]}")
    
    print()
    return X_train_processed, X_test_processed, y_train, y_test


def example_complete_pipeline():
    """Example 5: Complete integrated pipeline."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Complete Data Preparation Pipeline")
    logger.info("=" * 60)
    
    from preprocessing import prepare_dataset
    from datasets import register_dataset
    import pandas as pd
    
    # Create and register example dataset
    example_data = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / 'demo.csv'
    example_data.to_csv(dataset_path, index=False)
    
    register_dataset(
        name='demo',
        filepath=dataset_path,
        target_column='target',
        description='Demo dataset for pipeline example',
        task_type='classification',
    )
    
    # Complete pipeline
    logger.info("Running complete data preparation pipeline...")
    logger.info("  1. Load registered dataset")
    logger.info("  2. Split into train/test")
    logger.info("  3. Create preprocessing pipeline")
    logger.info("  4. Fit on training data only")
    logger.info("  5. Transform all splits")
    
    X_train, X_test, y_train, y_test, metadata = prepare_dataset(
        dataset_name='demo',
        test_size=0.3,
        random_state=42,
        scaling_method='standard',
        encoding_method='onehot',
    )
    
    logger.info(f"\n✓ Pipeline complete!")
    logger.info(f"\nOutput shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  y_test: {y_test.shape}")
    
    logger.info(f"\nDataset metadata:")
    for key, value in metadata.items():
        if isinstance(value, dict):
            logger.info(f"  {key}: {list(value.keys())}")
        else:
            logger.info(f"  {key}: {value}")
    
    print()
    return X_train, X_test, y_train, y_test


def example_integration_with_cv():
    """Example 6: Integration with future cross-validation."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Integration with Cross-Validation (Preview)")
    logger.info("=" * 60)
    
    from preprocessing import DataPreparation
    from datasets import register_dataset
    import pandas as pd
    
    # Create example dataset
    example_data = pd.DataFrame({
        'x1': range(100),
        'x2': range(100, 200),
        'y': [i % 2 for i in range(100)],
    })
    
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / 'cv_demo.csv'
    example_data.to_csv(dataset_path, index=False)
    
    register_dataset(
        name='cv_demo',
        filepath=dataset_path,
        target_column='y',
        task_type='classification',
    )
    
    # Prepare train/val/test split
    prep = DataPreparation(random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare(
        dataset_name='cv_demo',
        train_size=0.7,
        val_size=0.15,
    )
    
    logger.info("Three-way split prepared (train/val/test):")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Val:   {X_val.shape[0]} samples")
    logger.info(f"  Test:  {X_test.shape[0]} samples")
    
    logger.info("\nKey point: Preprocessing fitted ONLY on training data")
    logger.info("Validation and test data use training parameters (no leakage)")
    
    logger.info("\nThis structure integrates with:")
    logger.info("  - Cross-validation: Use val for hyperparameter tuning")
    logger.info("  - Nested CV: Use test for final evaluation")
    logger.info("  - Multiple folds: Apply same preprocessing to each fold")
    
    print()


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("ML BENCHMARKING FRAMEWORK - DATA PIPELINE EXAMPLES")
    logger.info("=" * 60 + "\n")
    
    try:
        # Run examples
        example_register_datasets()
        X, y, metadata = example_load_dataset()
        pipeline = example_create_pipeline(X)
        example_fit_transform(X, y, pipeline)
        example_complete_pipeline()
        example_integration_with_cv()
        
        logger.info("=" * 60)
        logger.info("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\nKey Takeaways:")
        logger.info("  1. Datasets are registered for easy discovery")
        logger.info("  2. Preprocessing pipelines prevent data leakage")
        logger.info("  3. Fit on train data, transform test/val independently")
        logger.info("  4. Integration ready for model training and cross-validation")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
