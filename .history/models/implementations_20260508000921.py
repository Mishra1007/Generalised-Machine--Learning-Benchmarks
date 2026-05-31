"""Model implementations for benchmarking."""

import logging
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from models.base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression wrapper.
    
    Linear classification model using logistic function.
    
    Properties:
    - Fast training and prediction
    - Works well with scaled features
    - Produces probability estimates
    - Good baseline for comparison
    
    Example:
        >>> model = LogisticRegressionModel(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> y_proba = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        verbose: bool = False,
    ):
        """
        Initialize Logistic Regression.
        
        Args:
            random_state: Random seed (default 42)
            max_iter: Maximum iterations (default 1000)
            verbose: Enable verbose logging
        """
        super().__init__(
            name='LogisticRegression',
            task_type='classification',
            random_state=random_state,
            verbose=verbose,
        )
        self.max_iter = max_iter
    
    def _build_model(self):
        """Build sklearn LogisticRegression model."""
        return SklearnLogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='lbfgs',
            multi_class='auto',
        )


class DecisionTreeModel(BaseModel):
    """
    Decision Tree wrapper.
    
    Tree-based classification model.
    
    Properties:
    - Non-linear decision boundaries
    - Fast prediction
    - Interpretable rules
    - No scaling needed
    - Prone to overfitting
    
    Example:
        >>> model = DecisionTreeModel(max_depth=5, random_state=42)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize Decision Tree.
        
        Args:
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
            verbose: Enable verbose logging
        """
        super().__init__(
            name='DecisionTree',
            task_type='classification',
            random_state=random_state,
            verbose=verbose,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def _build_model(self):
        """Build sklearn DecisionTree model."""
        return SklearnDecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )


class RandomForestModel(BaseModel):
    """
    Random Forest wrapper.
    
    Ensemble of decision trees.
    
    Properties:
    - Strong performance across tasks
    - Handles non-linear relationships
    - Feature importance available
    - Parallelizable training
    - Less prone to overfitting than single tree
    
    Example:
        >>> model = RandomForestModel(n_estimators=100, random_state=42)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> y_proba = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize Random Forest.
        
        Args:
            n_estimators: Number of trees (default 100)
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = all cores)
            verbose: Enable verbose logging
        """
        super().__init__(
            name='RandomForest',
            task_type='classification',
            random_state=random_state,
            verbose=verbose,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
    
    def _build_model(self):
        """Build sklearn RandomForest model."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class SVMModel(BaseModel):
    """
    Support Vector Machine wrapper.
    
    Kernel-based classification model.
    
    Properties:
    - Works well in high dimensions
    - Memory efficient
    - Supports different kernels
    - Slow on large datasets
    - Needs feature scaling
    - Probability estimates available
    
    Example:
        >>> model = SVMModel(kernel='rbf', C=1.0, random_state=42)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> y_proba = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        probability: bool = True,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize SVM.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            probability: Enable probability estimates
            random_state: Random seed
            verbose: Enable verbose logging
        """
        super().__init__(
            name='SVM',
            task_type='classification',
            random_state=random_state,
            verbose=verbose,
        )
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
    
    def _build_model(self):
        """Build sklearn SVM model."""
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state,
        )


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting wrapper.
    
    Sequential ensemble learning model.
    
    Properties:
    - Often best performance in benchmarks
    - Handles both linear and non-linear relationships
    - Feature importance available
    - Slower training than Random Forest
    - Parameter tuning important
    - Produces probability estimates
    
    Example:
        >>> model = GradientBoostingModel(n_estimators=100, learning_rate=0.1, random_state=42)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> y_proba = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize Gradient Boosting.
        
        Args:
            n_estimators: Number of boosting stages (default 100)
            learning_rate: Shrinks contribution of each tree (default 0.1)
            max_depth: Maximum tree depth (default 3)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            subsample: Fraction of samples for training each tree
            random_state: Random seed
            verbose: Enable verbose logging
        """
        super().__init__(
            name='GradientBoosting',
            task_type='classification',
            random_state=random_state,
            verbose=verbose,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
    
    def _build_model(self):
        """Build sklearn GradientBoosting model."""
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
