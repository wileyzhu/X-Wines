"""
LIME explainer implementation for tabular data explanations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from lime import lime_tabular
import logging

from .base import BaseExplainer, LIMEExplanation
from ..config import ExplanationConfig

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """LIME-based model explainer for tabular data."""
    
    def __init__(self, config: ExplanationConfig):
        """Initialize LIME explainer with configuration.
        
        Args:
            config: Explanation configuration object
        """
        super().__init__(config)
        self.lime_explainer = None
        self.feature_names = None
        self.model = None
        
    def fit(self, model: Any, X_background: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> None:
        """Fit the LIME explainer to the model and background data.
        
        Args:
            model: Trained model to explain
            X_background: Background dataset for explanation
            feature_names: Names of features (optional)
        
        Raises:
            ValueError: If background data is empty or invalid
        """
        if X_background is None or len(X_background) == 0:
            raise ValueError("Background data cannot be empty")
            
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]
        
        # Initialize LIME tabular explainer
        try:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_background,
                feature_names=self.feature_names,
                mode='regression',  # Assuming wine quality prediction is regression
                discretize_continuous=True,
                random_state=42
            )
            self.is_fitted = True
            logger.info(f"LIME explainer fitted with {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Failed to fit LIME explainer: {str(e)}")
            raise RuntimeError(f"LIME explainer fitting failed: {str(e)}")
    
    def explain_global(self, X_data: np.ndarray) -> List[LIMEExplanation]:
        """Generate global explanations by explaining multiple samples.
        
        Note: LIME is inherently local, so global explanation is achieved
        by aggregating multiple local explanations.
        
        Args:
            X_data: Dataset to explain
            
        Returns:
            List of LIME explanations for each sample
        """
        self._validate_fitted()
        
        if len(X_data) == 0:
            raise ValueError("Input data cannot be empty")
            
        explanations = []
        
        # Sample a subset for global analysis if dataset is large
        sample_size = min(len(X_data), 100)  # Limit for performance
        if len(X_data) > sample_size:
            indices = np.random.choice(len(X_data), sample_size, replace=False)
            X_sample = X_data[indices]
            logger.info(f"Sampling {sample_size} instances from {len(X_data)} for global analysis")
        else:
            X_sample = X_data
            
        for i, instance in enumerate(X_sample):
            try:
                explanation = self.explain_local(instance.reshape(1, -1))
                explanations.append(explanation)
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {str(e)}")
                continue
                
        logger.info(f"Generated {len(explanations)} LIME explanations for global analysis")
        return explanations
    
    def explain_local(self, X_sample: np.ndarray) -> LIMEExplanation:
        """Generate local explanation for a single sample.
        
        Args:
            X_sample: Single sample to explain (shape: 1 x n_features)
            
        Returns:
            LIME explanation object
        """
        self._validate_fitted()
        
        if X_sample.shape[0] != 1:
            raise ValueError("X_sample must contain exactly one instance")
            
        instance = X_sample.flatten()
        
        try:
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                data_row=instance,
                predict_fn=self._predict_fn,
                num_features=self.config.lime_num_features,
                num_samples=self.config.lime_num_samples
            )
            
            # Extract feature importance
# FIXED: Use proper feature names instead of values
            raw_explanation = explanation.as_list()
            feature_importance = {}
            
            # Map LIME's feature descriptions to actual feature names
            for i, (feature_desc, importance) in enumerate(raw_explanation):
                if i < len(self.feature_names):
                    # Use the actual feature name
                    feature_name = self.feature_names[i]
                    feature_importance[feature_name] = importance
                else:
                    # Fallback: parse feature name from description
                    if ' <= ' in feature_desc:
                        feature_name = feature_desc.split(' <= ')[0].strip()
                    elif ' > ' in feature_desc:
                        feature_name = feature_desc.split(' > ')[0].strip()
                    elif '=' in feature_desc:
                        feature_name = feature_desc.split('=')[0].strip()
                    else:
                        feature_name = feature_desc.strip()
                    
                    feature_importance[feature_name] = importance
            
            # Get prediction and confidence interval
            prediction = self._predict_fn(instance.reshape(1, -1))[0]
            
            # Calculate confidence interval (approximate)
            local_prediction = explanation.local_pred[0] if hasattr(explanation, 'local_pred') else prediction
            intercept = explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0
            
            # Estimate confidence interval based on explanation variance
            importance_values = list(feature_importance.values())
            std_error = np.std(importance_values) if importance_values else 0.1
            confidence_interval = (prediction - 1.96 * std_error, prediction + 1.96 * std_error)
            
            return LIMEExplanation(
                feature_importance=feature_importance,
                prediction=prediction,
                confidence_interval=confidence_interval,
                local_prediction=local_prediction,
                intercept=intercept,
                feature_names=self.feature_names,
                instance_data=instance
            )
            
        except Exception as e:
            logger.error(f"Failed to generate LIME explanation: {str(e)}")
            raise RuntimeError(f"LIME explanation failed: {str(e)}")
    
    def explain_batch(self, X_samples: np.ndarray) -> List[LIMEExplanation]:
        """Generate explanations for multiple wine samples.
        
        Args:
            X_samples: Multiple samples to explain
            
        Returns:
            List of LIME explanations
        """
        self._validate_fitted()
        
        if len(X_samples) == 0:
            raise ValueError("Input samples cannot be empty")
            
        explanations = []
        
        for i, sample in enumerate(X_samples):
            try:
                explanation = self.explain_local(sample.reshape(1, -1))
                explanations.append(explanation)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(X_samples)} samples")
                    
            except Exception as e:
                logger.warning(f"Failed to explain sample {i}: {str(e)}")
                continue
                
        logger.info(f"Generated {len(explanations)} LIME explanations from {len(X_samples)} samples")
        return explanations
    
    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction function wrapper for LIME.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise RuntimeError("Model not available for prediction")
            
        try:
            # Handle different model interfaces
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            elif callable(self.model):
                return self.model(X)
            else:
                raise ValueError("Model must have predict method or be callable")
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    
    def get_feature_importance_summary(self, explanations: List[LIMEExplanation]) -> Dict[str, Dict[str, float]]:
        """Aggregate feature importance across multiple explanations.
        
        Args:
            explanations: List of LIME explanations
            
        Returns:
            Dictionary with aggregated feature importance statistics
        """
        if not explanations:
            return {}
            
        # Collect all feature importance values
        feature_values = {}
        for explanation in explanations:
            for feature, importance in explanation.feature_importance.items():
                if feature not in feature_values:
                    feature_values[feature] = []
                feature_values[feature].append(importance)
        
        # Calculate statistics for each feature
        summary = {}
        for feature, values in feature_values.items():
            summary[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
            
        return summary