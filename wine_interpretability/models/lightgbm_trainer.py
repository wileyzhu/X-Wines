"""
LightGBM model trainer implementation.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

from .base import ModelTrainer, ModelResult, ModelEvaluator
from ..config import ModelConfig, ModelType


logger = logging.getLogger(__name__)


class LightGBMTrainer(ModelTrainer):
    """LightGBM trainer with hyperparameter optimization."""
    
    def __init__(self, config: ModelConfig):
        """Initialize LightGBM trainer.
        
        Args:
            config: Model configuration object
        """
        super().__init__(config)
        if config.model_type != ModelType.LIGHTGBM:
            raise ValueError("ModelConfig must have model_type=ModelType.LIGHTGBM")
        
        # Default hyperparameter space if not provided
        if not config.hyperparameter_space:
            config.hyperparameter_space = self._get_default_hyperparameter_space()
    
    def _get_default_hyperparameter_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space for LightGBM.
        
        Returns:
            Dictionary defining hyperparameter search space
        """
        return {
            'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'bagging_freq': {'type': 'int', 'low': 1, 'high': 7},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> lgb.LGBMRegressor:
        """Train LightGBM model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained LightGBM model
        """
        start_time = time.time()
        
        # Use optimized hyperparameters if available, otherwise use defaults
        if hasattr(self, '_best_params'):
            params = self._best_params.copy()
        else:
            params = {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
            }
        
        # Add fixed parameters
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': self.config.random_state,
            'verbose': -1,
            'n_jobs': -1,
        })
        
        # Create model
        self.model = lgb.LGBMRegressor(**params)
        
        # Prepare validation data for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds or 50),
                lgb.log_evaluation(0)  # Suppress training logs
            ] if eval_set else None
        )
        
        # Validate that training was successful
        if self.model is None:
            raise RuntimeError("Model training failed - model is None")
        
        if not hasattr(self.model, 'booster_') or self.model.booster_ is None:
            logger.warning("Model booster is None - this may cause issues with feature importance")
        
        # Test a simple prediction to ensure model is working
        try:
            test_pred = self.model.predict(X_train[:1])
            if test_pred is None or len(test_pred) == 0:
                raise RuntimeError("Model prediction test failed")
        except Exception as e:
            raise RuntimeError(f"Model validation failed: {str(e)}")
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        logger.info(f"LightGBM training completed in {training_time:.2f} seconds")
        logger.debug(f"Model validation successful - booster available: {hasattr(self.model, 'booster_') and self.model.booster_ is not None}")
        
        return self.model
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: Optional[np.ndarray] = None,
                                y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        logger.info("Starting hyperparameter optimization for LightGBM")
        
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
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'random_state': self.config.random_state,
                'verbose': -1,
                'n_jobs': 1,  # Use single job for cross-validation
            })
            
            # Create model
            model = lgb.LGBMRegressor(**params)
            
            # Use validation set if provided, otherwise use cross-validation
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(self.config.early_stopping_rounds or 50),
                        lgb.log_evaluation(0)
                    ]
                )
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
        """Make predictions using the trained LightGBM model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Model predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate LightGBM model performance.
        
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
        
        logger.info("LightGBM Model Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """Get feature importance from trained LightGBM model.
        
        Args:
            importance_type: Type of importance ('gain', 'split', or 'weight')
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        if importance_type not in ['gain', 'split', 'weight']:
            raise ValueError("importance_type must be one of: 'gain', 'split', 'weight'")
        
        # Check if model exists
        if self.model is None:
            logger.error("Model is None - training may have failed")
            return {}
        
        try:
            importance_values = None
            
            # Method 1: Try LightGBM booster's feature importance (preferred)
            if hasattr(self.model, 'booster_') and self.model.booster_ is not None:
                try:
                    importance_values = self.model.booster_.feature_importance(importance_type=importance_type)
                    logger.debug(f"Successfully extracted {importance_type} importance from booster")
                except Exception as e:
                    logger.warning(f"Booster feature importance failed for {importance_type}: {str(e)}")
                    importance_values = None
            
            # Method 2: Fallback to sklearn-style feature importance (only for 'gain' equivalent)
            if importance_values is None and hasattr(self.model, 'feature_importances_'):
                try:
                    importance_values = self.model.feature_importances_
                    logger.debug("Using sklearn-style feature_importances_")
                except Exception as e:
                    logger.warning(f"Sklearn feature_importances_ failed: {str(e)}")
                    importance_values = None
            
            # Method 3: Try direct model method (last resort)
            if importance_values is None:
                try:
                    importance_values = self.model.feature_importance(importance_type=importance_type)
                    logger.debug(f"Using direct model.feature_importance({importance_type})")
                except Exception as e:
                    logger.warning(f"Direct model feature_importance failed: {str(e)}")
                    importance_values = None
            
            # Check if we got valid importance values
            if importance_values is None:
                logger.error(f"All feature importance extraction methods failed for {importance_type}")
                return {}
            
            # Validate importance values
            if len(importance_values) == 0:
                logger.error("Feature importance array is empty")
                return {}
            
            # Check for all zeros (indicates potential training issue)
            if np.all(importance_values == 0):
                logger.warning("All feature importance values are zero - model may not have trained properly")
                logger.warning("This can happen with very small datasets, constant features, or training failures")
                
                # Try to get alternative importance if available
                if importance_type == 'gain' and hasattr(self.model, 'feature_importances_'):
                    try:
                        fallback_values = self.model.feature_importances_
                        if not np.all(fallback_values == 0):
                            logger.info("Using sklearn feature_importances_ as fallback")
                            importance_values = fallback_values
                    except Exception:
                        pass
            
            # Create feature names
            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
            
            result = dict(zip(feature_names, importance_values))
            logger.debug(f"Feature importance extraction successful: {len(result)} features, sum={sum(result.values()):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in feature importance extraction: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
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
            model_type="LightGBM",
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters,
            training_time=training_time,
            feature_names=feature_names
        )