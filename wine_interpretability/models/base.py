"""
Abstract base classes and interfaces for model training and evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class ModelResult:
    """Container for model training results."""
    
    model: Any
    model_type: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_time: float
    feature_names: Optional[list] = None


class ModelTrainer(ABC):
    """Abstract base class for model training."""
    
    def __init__(self, config: 'ModelConfig'):
        """Initialize trainer with configuration.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Any:
        """Train the model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained model object
        """
        pass
    
    @abstractmethod
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: Optional[np.ndarray] = None,
                                y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize model hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Model predictions
        """
        pass
    
    def get_model(self) -> Any:
        """Get the trained model object.
        
        Returns:
            Trained model object
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before accessing")
        return self.model


class ModelEvaluator:
    """Utility class for model performance evaluation."""
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
        return metrics
    
    @staticmethod
    def compare_models(results: list) -> Dict[str, Any]:
        """Compare multiple model results.
        
        Args:
            results: List of ModelResult objects
            
        Returns:
            Dictionary with comparison results
        """
        if not results:
            raise ValueError("No model results provided for comparison")
        
        comparison = {
            'models': [],
            'best_model': None,
            'best_metric': 'r2',
            'rankings': {}
        }
        
        # Extract model information
        for result in results:
            model_info = {
                'model_type': result.model_type,
                'metrics': result.performance_metrics,
                'training_time': result.training_time
            }
            comparison['models'].append(model_info)
        
        # Find best model based on R² score
        best_r2 = -float('inf')
        best_idx = 0
        
        for i, result in enumerate(results):
            r2 = result.performance_metrics.get('r2', -float('inf'))
            if r2 > best_r2:
                best_r2 = r2
                best_idx = i
        
        comparison['best_model'] = results[best_idx]
        
        # Create rankings for each metric
        metrics = ['rmse', 'mae', 'r2', 'mse']
        for metric in metrics:
            if all(metric in result.performance_metrics for result in results):
                # Sort by metric (ascending for error metrics, descending for R²)
                reverse = metric == 'r2'
                sorted_results = sorted(results, 
                                      key=lambda x: x.performance_metrics[metric],
                                      reverse=reverse)
                comparison['rankings'][metric] = [
                    result.model_type for result in sorted_results
                ]
        
        return comparison