"""
XGBoost model trainer implementation.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

from .base import ModelTrainer, ModelResult, ModelEvaluator
from ..config import ModelConfig, ModelType


logger = logging.getLogger(__name__)


class XGBoostTrainer(ModelTrainer):
    """XGBoost trainer with hyperparameter optimization."""
    
    def __init__(self, config: ModelConfig):
        """Initialize XGBoost trainer.
        
        Args:
            config: Model configuration object
        """
        super().__init__(config)
        if config.model_type != ModelType.XGBOOST:
            raise ValueError("ModelConfig must have model_type=ModelType.XGBOOST")
        
        # Default hyperparameter space if not provided
        if not config.hyperparameter_space:
            config.hyperparameter_space = self._get_default_hyperparameter_space()
    
    def _get_default_hyperparameter_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space for XGBoost.
        
        Returns:
            Dictionary defining hyperparameter search space
        """
        return {
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bylevel': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> xgb.XGBRegressor:
        """Train XGBoost model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained XGBoost model
        """
        start_time = time.time()
        
        # Use optimized hyperparameters if available, otherwise use defaults
        if hasattr(self, '_best_params'):
            params = self._best_params.copy()
        else:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 1.0,
                'reg_alpha': 0.0,
                'reg_lambda': 1.0,
                'min_child_weight': 1,
            }
        
        # Add fixed parameters
        params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbosity': 0,  # Suppress training logs
        })
        
        # Create model (don't add early_stopping_rounds to constructor)
        self.model = xgb.XGBRegressor(**params)
        
        # Prepare fit parameters - use simple approach for compatibility
        fit_params = {}
        
        if X_val is not None and y_val is not None:
            # Just use eval_set for monitoring, skip early stopping for compatibility
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'verbose': False
            }
            logger.info("Training with validation set (early stopping disabled for compatibility)")
        else:
            logger.info("Training without validation set")
        
        # Train model with compatible parameters
        self.model.fit(X_train, y_train, **fit_params)
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
        
        return self.model
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: Optional[np.ndarray] = None,
                                y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        logger.info("Starting hyperparameter optimization for XGBoost")
        
        def objective(trial):
            """Optuna objective function for hyperparameter optimization."""
            # Sample hyperparameters
            params = {}
            for param_name, param_config in self.config.hyperparameter_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Add fixed parameters
            params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': self.config.random_state,
                'n_jobs': 1,  # Use single job for cross-validation
                'verbosity': 0,
            })
            
            # Create model
            model = xgb.XGBRegressor(**params)
            
            # Use validation set if provided, otherwise use cross-validation
            if X_val is not None and y_val is not None:
                # Simple fit with validation set (no early stopping for compatibility)
                fit_params = {'eval_set': [(X_val, y_val)], 'verbose': False}
                
                # Train with validation set
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_val)
                score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
            else:
                # Use cross-validation
                kfold = KFold(
                    n_splits=self.config.cross_validation_folds,
                    shuffle=True,
                    random_state=self.config.random_state
                )
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=kfold,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=1
                )
                score = -scores.mean()  # Convert back to positive RMSE
            
            return score
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.optimization_trials,
            show_progress_bar=False
        )
        
        self._best_params = study.best_params
        
        logger.info(f"Hyperparameter optimization completed. Best RMSE: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self._best_params}")
        
        return self._best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained XGBoost model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Model predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate XGBoost model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing performance metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)
        
        logger.info("XGBoost Model Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """Get feature importance from trained XGBoost model.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', or 'total_gain')
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        if importance_type not in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            raise ValueError("importance_type must be one of: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'")
        
        importance_values = self.model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(importance_values))]
        
        return dict(zip(feature_names, importance_values))
    
    def create_model_result(self, X_test: np.ndarray, y_test: np.ndarray,
                           training_time: float, feature_names: Optional[list] = None) -> ModelResult:
        """Create a ModelResult object with training results.
        
        Args:
            X_test: Test features for evaluation
            y_test: Test targets for evaluation
            training_time: Time taken for training
            feature_names: Names of features (optional)
            
        Returns:
            ModelResult object containing all training information
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before creating results")
        
        # Evaluate model performance
        performance_metrics = self.evaluate(X_test, y_test)
        
        # Get hyperparameters
        hyperparameters = self.model.get_params()
        
        return ModelResult(
            model=self.model,
            model_type="XGBoost",
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters,
            training_time=training_time,
            feature_names=feature_names
        )