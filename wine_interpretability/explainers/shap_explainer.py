"""
SHAP explainer implementation.
"""

import numpy as np
import shap
from typing import Any, List, Optional, Union, TYPE_CHECKING
import logging

from .base import BaseExplainer, SHAPExplanation

if TYPE_CHECKING:
    from ..config import ExplanationConfig

# Import enum at runtime for comparison
try:
    from ..config import SHAPExplainerType
except ImportError:
    # Fallback for testing
    from enum import Enum
    class SHAPExplainerType(Enum):
        TREE = "tree"
        LINEAR = "linear"
        KERNEL = "kernel"

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """SHAP-based model explainer with TreeExplainer and KernelExplainer support."""
    
    def __init__(self, config: 'ExplanationConfig'):
        """Initialize SHAP explainer with configuration.
        
        Args:
            config: Explanation configuration object
        """
        super().__init__(config)
        self.model = None
        self.feature_names = None
        self.background_data = None
        
    def fit(self, model: Any, X_background: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> None:
        """Fit the SHAP explainer to the model and background data.
        
        Args:
            model: Trained model to explain (LightGBM, XGBoost, or sklearn-compatible)
            X_background: Background dataset for explanation
            feature_names: Names of features (optional)
            
        Raises:
            ValueError: If explainer type is not supported or model is incompatible
            RuntimeError: If explainer initialization fails
        """
        try:
            self.model = model
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]
            
            # Sample background data if too large
            if X_background.shape[0] > self.config.background_samples:
                indices = np.random.choice(
                    X_background.shape[0], 
                    self.config.background_samples, 
                    replace=False
                )
                self.background_data = X_background[indices]
            else:
                self.background_data = X_background.copy()
            
            # Initialize appropriate SHAP explainer based on configuration
            if self.config.shap_explainer_type == SHAPExplainerType.TREE:
                self.explainer = self._create_tree_explainer(model, self.background_data)
            elif self.config.shap_explainer_type == SHAPExplainerType.LINEAR:
                self.explainer = self._create_linear_explainer(model, self.background_data)
            elif self.config.shap_explainer_type == SHAPExplainerType.KERNEL:
                self.explainer = self._create_kernel_explainer(model, self.background_data)
            else:
                raise ValueError(f"Unsupported SHAP explainer type: {self.config.shap_explainer_type}")
            
            self.is_fitted = True
            logger.info(f"SHAP explainer fitted with {self.config.shap_explainer_type.value} explainer")
            
        except Exception as e:
            logger.error(f"Failed to fit SHAP explainer: {str(e)}")
            raise RuntimeError(f"SHAP explainer fitting failed: {str(e)}")
    
    def _create_tree_explainer(self, model: Any, background_data: np.ndarray) -> shap.TreeExplainer:
        """Create TreeExplainer for tree-based models.
        
        Args:
            model: Tree-based model (LightGBM, XGBoost, etc.)
            background_data: Background dataset
            
        Returns:
            Configured TreeExplainer
        """
        try:
            return shap.TreeExplainer(model, background_data)
        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to KernelExplainer: {str(e)}")
            return self._create_kernel_explainer(model, background_data)
    
    def _create_linear_explainer(self, model: Any, background_data: np.ndarray) -> shap.LinearExplainer:
        """Create LinearExplainer for linear models.
        
        Args:
            model: Linear model
            background_data: Background dataset
            
        Returns:
            Configured LinearExplainer
        """
        return shap.LinearExplainer(model, background_data)
    
    def _create_kernel_explainer(self, model: Any, background_data: np.ndarray) -> shap.KernelExplainer:
        """Create KernelExplainer for any model type.
        
        Args:
            model: Any model with predict method
            background_data: Background dataset
            
        Returns:
            Configured KernelExplainer
        """
        # Create prediction function that works with the model
        if hasattr(model, 'predict'):
            predict_fn = model.predict
        elif hasattr(model, 'predict_proba'):
            predict_fn = lambda x: model.predict_proba(x)[:, 1]  # For binary classification
        else:
            raise ValueError("Model must have 'predict' or 'predict_proba' method")
        
        return shap.KernelExplainer(predict_fn, background_data)
    
    def explain_global(self, X_data: np.ndarray) -> SHAPExplanation:
        """Generate global SHAP explanations for the entire dataset.
        
        Args:
            X_data: Dataset to explain
            
        Returns:
            SHAPExplanation object with global explanation results
            
        Raises:
            RuntimeError: If explainer hasn't been fitted
        """
        self._validate_fitted()
        
        try:
            logger.info(f"Computing global SHAP values for {X_data.shape[0]} samples")
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X_data)
            expected_value = self.explainer.expected_value
            
            # Handle multi-output models (e.g., multi-class classification)
            if isinstance(shap_values, list):
                # For multi-class, take the first class or average
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
                expected_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            
            return SHAPExplanation(
                shap_values=shap_values,
                expected_value=expected_value,
                feature_names=self.feature_names,
                data=X_data,
                explanation_type='global',
                model_output='raw'
            )
            
        except Exception as e:
            logger.error(f"Failed to compute global SHAP values: {str(e)}")
            raise RuntimeError(f"Global SHAP explanation failed: {str(e)}")
    
    def explain_local(self, X_sample: np.ndarray) -> SHAPExplanation:
        """Generate local SHAP explanation for individual predictions.
        
        Args:
            X_sample: Single sample to explain (1D array) or batch of samples (2D array)
            
        Returns:
            SHAPExplanation object with local explanation results
            
        Raises:
            RuntimeError: If explainer hasn't been fitted
            ValueError: If sample shape is invalid
        """
        self._validate_fitted()
        
        # Ensure X_sample is 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        elif X_sample.ndim != 2:
            raise ValueError("X_sample must be 1D or 2D array")
        
        try:
            logger.info(f"Computing local SHAP values for {X_sample.shape[0]} sample(s)")
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            expected_value = self.explainer.expected_value
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
                expected_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            
            return SHAPExplanation(
                shap_values=shap_values,
                expected_value=expected_value,
                feature_names=self.feature_names,
                data=X_sample,
                explanation_type='local',
                model_output='raw'
            )
            
        except Exception as e:
            logger.error(f"Failed to compute local SHAP values: {str(e)}")
            raise RuntimeError(f"Local SHAP explanation failed: {str(e)}")
    
    def compute_interaction_values(self, X_data: np.ndarray) -> np.ndarray:
        """Compute SHAP interaction values for feature interaction analysis.
        
        Args:
            X_data: Dataset to compute interactions for
            
        Returns:
            SHAP interaction values array
            
        Raises:
            RuntimeError: If explainer doesn't support interaction values
        """
        self._validate_fitted()
        
        if not hasattr(self.explainer, 'shap_interaction_values'):
            raise RuntimeError("Current explainer does not support interaction values")
        
        try:
            logger.info(f"Computing SHAP interaction values for {X_data.shape[0]} samples")
            interaction_values = self.explainer.shap_interaction_values(X_data)
            
            # Handle multi-output models
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[0] if len(interaction_values) > 0 else interaction_values
            
            return interaction_values
            
        except Exception as e:
            logger.error(f"Failed to compute SHAP interaction values: {str(e)}")
            raise RuntimeError(f"SHAP interaction computation failed: {str(e)}")
    
    def get_feature_importance(self, shap_explanation: SHAPExplanation, 
                             method: str = 'mean_abs') -> dict:
        """Extract feature importance from SHAP values.
        
        Args:
            shap_explanation: SHAP explanation object
            method: Method to compute importance ('mean_abs', 'mean', 'std')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        shap_values = shap_explanation.shap_values
        
        if method == 'mean_abs':
            importance_scores = np.mean(np.abs(shap_values), axis=0)
        elif method == 'mean':
            importance_scores = np.mean(shap_values, axis=0)
        elif method == 'std':
            importance_scores = np.std(shap_values, axis=0)
        else:
            raise ValueError(f"Unsupported importance method: {method}")
        
        return dict(zip(self.feature_names, importance_scores))