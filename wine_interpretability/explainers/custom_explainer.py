"""
Custom model explainer implementation without external SHAP dependency.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .base import BaseExplainer


@dataclass
class CustomExplanation:
    """Container for custom explanation results."""
    
    feature_importance: Dict[str, float]
    feature_contributions: np.ndarray
    baseline_prediction: float
    instance_prediction: float
    feature_names: List[str]
    explanation_method: str
    confidence_score: float


class PermutationExplainer(BaseExplainer):
    """Custom explainer using permutation importance and local perturbations."""
    
    def __init__(self, config):
        super().__init__(config)
        self.baseline_prediction = None
        self.feature_names = None
        self.model = None
        
    def fit(self, model: Any, X_background: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> None:
        """Fit the explainer to the model and background data."""
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]
        
        # Calculate baseline prediction (mean of background predictions)
        baseline_preds = model.predict(X_background)
        self.baseline_prediction = np.mean(baseline_preds)
        self.is_fitted = True
        
    def explain_global(self, X_data: np.ndarray) -> CustomExplanation:
        """Generate global explanations using permutation importance."""
        self._validate_fitted()
        
        # Get baseline predictions
        original_preds = self.model.predict(X_data)
        original_score = np.mean(original_preds)
        
        feature_importance = {}
        
        # Calculate permutation importance for each feature
        for i, feature_name in enumerate(self.feature_names):
            # Create permuted dataset
            X_permuted = X_data.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get predictions on permuted data
            permuted_preds = self.model.predict(X_permuted)
            permuted_score = np.mean(permuted_preds)
            
            # Importance is the decrease in performance
            importance = abs(original_score - permuted_score)
            feature_importance[feature_name] = importance
        
        # Normalize importance scores
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: v / total_importance 
                for k, v in feature_importance.items()
            }
        
        # Calculate average contributions
        contributions = np.array([feature_importance[name] for name in self.feature_names])
        
        return CustomExplanation(
            feature_importance=feature_importance,
            feature_contributions=contributions,
            baseline_prediction=self.baseline_prediction,
            instance_prediction=original_score,
            feature_names=self.feature_names,
            explanation_method="permutation_importance",
            confidence_score=self._calculate_confidence(feature_importance)
        )
    
    def explain_local(self, X_sample: np.ndarray) -> CustomExplanation:
        """Generate local explanation for a single sample using perturbations."""
        self._validate_fitted()
        
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # Get original prediction
        original_pred = self.model.predict(X_sample)[0]
        
        feature_contributions = {}
        
        # For each feature, measure impact by setting it to baseline values
        for i, feature_name in enumerate(self.feature_names):
            # Create modified sample with feature set to mean value
            X_modified = X_sample.copy()
            X_modified[0, i] = 0  # or use feature mean from background data
            
            # Get prediction without this feature's contribution
            modified_pred = self.model.predict(X_modified)[0]
            
            # Contribution is the difference
            contribution = original_pred - modified_pred
            feature_contributions[feature_name] = contribution
        
        # Convert to array for consistency
        contributions_array = np.array([
            feature_contributions[name] for name in self.feature_names
        ])
        
        return CustomExplanation(
            feature_importance=feature_contributions,
            feature_contributions=contributions_array,
            baseline_prediction=self.baseline_prediction,
            instance_prediction=original_pred,
            feature_names=self.feature_names,
            explanation_method="local_perturbation",
            confidence_score=self._calculate_confidence(feature_contributions)
        )
    
    def _calculate_confidence(self, importance_dict: Dict[str, float]) -> float:
        """Calculate confidence score based on importance distribution."""
        values = list(importance_dict.values())
        if not values:
            return 0.0
        
        # Higher confidence when there are clear dominant features
        max_importance = max(values)
        mean_importance = np.mean(values)
        
        if mean_importance == 0:
            return 0.0
        
        # Confidence based on how much the top feature dominates
        confidence = min(1.0, max_importance / (mean_importance * len(values)))
        return confidence


class GradientExplainer(BaseExplainer):
    """Custom explainer using numerical gradients for tree models."""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.feature_names = None
        self.baseline_prediction = None
        
    def fit(self, model: Any, X_background: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> None:
        """Fit the explainer to the model and background data."""
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]
        
        # Calculate baseline
        baseline_preds = model.predict(X_background)
        self.baseline_prediction = np.mean(baseline_preds)
        self.is_fitted = True
        
    def explain_local(self, X_sample: np.ndarray) -> CustomExplanation:
        """Generate local explanation using numerical gradients."""
        self._validate_fitted()
        
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        original_pred = self.model.predict(X_sample)[0]
        
        # Calculate numerical gradients
        gradients = {}
        epsilon = 1e-4
        
        for i, feature_name in enumerate(self.feature_names):
            # Forward difference
            X_plus = X_sample.copy()
            X_plus[0, i] += epsilon
            pred_plus = self.model.predict(X_plus)[0]
            
            # Backward difference  
            X_minus = X_sample.copy()
            X_minus[0, i] -= epsilon
            pred_minus = self.model.predict(X_minus)[0]
            
            # Numerical gradient
            gradient = (pred_plus - pred_minus) / (2 * epsilon)
            
            # Feature contribution = gradient * feature_value
            contribution = gradient * X_sample[0, i]
            gradients[feature_name] = contribution
        
        contributions_array = np.array([gradients[name] for name in self.feature_names])
        
        return CustomExplanation(
            feature_importance=gradients,
            feature_contributions=contributions_array,
            baseline_prediction=self.baseline_prediction,
            instance_prediction=original_pred,
            feature_names=self.feature_names,
            explanation_method="numerical_gradient",
            confidence_score=self._calculate_gradient_confidence(gradients)
        )
    
    def explain_global(self, X_data: np.ndarray) -> CustomExplanation:
        """Generate global explanation by averaging local explanations."""
        self._validate_fitted()
        
        # Get local explanations for all samples
        all_contributions = []
        for i in range(X_data.shape[0]):
            local_exp = self.explain_local(X_data[i:i+1])
            all_contributions.append(local_exp.feature_contributions)
        
        # Average contributions across all samples
        avg_contributions = np.mean(all_contributions, axis=0)
        
        # Convert to importance dictionary
        feature_importance = {
            name: abs(contrib) for name, contrib in 
            zip(self.feature_names, avg_contributions)
        }
        
        # Normalize
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: v / total_importance 
                for k, v in feature_importance.items()
            }
        
        avg_pred = np.mean(self.model.predict(X_data))
        
        return CustomExplanation(
            feature_importance=feature_importance,
            feature_contributions=avg_contributions,
            baseline_prediction=self.baseline_prediction,
            instance_prediction=avg_pred,
            feature_names=self.feature_names,
            explanation_method="averaged_gradients",
            confidence_score=self._calculate_gradient_confidence(feature_importance)
        )
    
    def _calculate_gradient_confidence(self, gradients: Dict[str, float]) -> float:
        """Calculate confidence based on gradient magnitudes."""
        values = [abs(v) for v in gradients.values()]
        if not values:
            return 0.0
        
        # Higher confidence when gradients are more consistent
        std_gradients = np.std(values)
        mean_gradients = np.mean(values)
        
        if mean_gradients == 0:
            return 0.0
        
        # Lower coefficient of variation = higher confidence
        cv = std_gradients / mean_gradients
        confidence = max(0.0, 1.0 - cv)
        return confidence