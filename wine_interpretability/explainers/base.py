"""
Abstract base classes for model explanation components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..config import ExplanationConfig, VisualizationConfig


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results."""
    
    shap_values: np.ndarray
    expected_value: Union[float, np.ndarray]
    feature_names: List[str]
    data: np.ndarray
    explanation_type: str  # 'global' or 'local'
    model_output: Optional[str] = None  # 'probability' or 'raw'


@dataclass
class LIMEExplanation:
    """Container for LIME explanation results."""
    
    feature_importance: Dict[str, float]
    prediction: float
    confidence_interval: tuple
    local_prediction: float
    intercept: float
    feature_names: List[str]
    instance_data: np.ndarray


@dataclass
class InteractionAnalysis:
    """Container for feature interaction analysis results."""
    
    interaction_values: np.ndarray
    feature_pairs: List[tuple]
    top_interactions: List[Dict[str, Any]]
    interaction_strength: Dict[tuple, float]


@dataclass
class ComparisonResult:
    """Container for explanation comparison results."""
    
    shap_importance: Dict[str, float]
    lime_importance: Dict[str, float]
    correlation_score: float
    agreement_features: List[str]
    disagreement_features: List[str]
    insights: List[str]
    consistency_score: float


class BaseExplainer(ABC):
    """Abstract base class for model explainers."""
    
    def __init__(self, config: 'ExplanationConfig'):
        """Initialize explainer with configuration.
        
        Args:
            config: Explanation configuration object
        """
        self.config = config
        self.explainer = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, model: Any, X_background: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> None:
        """Fit the explainer to the model and background data.
        
        Args:
            model: Trained model to explain
            X_background: Background dataset for explanation
            feature_names: Names of features (optional)
        """
        pass
    
    @abstractmethod
    def explain_global(self, X_data: np.ndarray) -> Any:
        """Generate global explanations for the dataset.
        
        Args:
            X_data: Dataset to explain
            
        Returns:
            Global explanation object
        """
        pass
    
    @abstractmethod
    def explain_local(self, X_sample: np.ndarray) -> Any:
        """Generate local explanation for a single sample.
        
        Args:
            X_sample: Single sample to explain
            
        Returns:
            Local explanation object
        """
        pass
    
    def _validate_fitted(self) -> None:
        """Check if explainer has been fitted.
        
        Raises:
            RuntimeError: If explainer hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Explainer must be fitted before generating explanations")


class BaseVisualizer(ABC):
    """Abstract base class for explanation visualizers."""
    
    def __init__(self, config: 'VisualizationConfig'):
        """Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration object
        """
        self.config = config
    
    @abstractmethod
    def create_summary_plot(self, explanation: Any, **kwargs) -> Any:
        """Create summary visualization of explanations.
        
        Args:
            explanation: Explanation object to visualize
            **kwargs: Additional visualization parameters
            
        Returns:
            Matplotlib figure or similar visualization object
        """
        pass
    
    @abstractmethod
    def create_feature_importance_plot(self, explanation: Any, **kwargs) -> Any:
        """Create feature importance visualization.
        
        Args:
            explanation: Explanation object to visualize
            **kwargs: Additional visualization parameters
            
        Returns:
            Matplotlib figure or similar visualization object
        """
        pass
    
    @abstractmethod
    def save_plot(self, figure: Any, filename: str, **kwargs) -> None:
        """Save visualization to file.
        
        Args:
            figure: Figure object to save
            filename: Output filename
            **kwargs: Additional save parameters
        """
        pass


class BaseAnalyzer(ABC):
    """Abstract base class for explanation analysis."""
    
    @abstractmethod
    def analyze_feature_importance(self, explanations: List[Any]) -> Dict[str, Any]:
        """Analyze feature importance patterns across explanations.
        
        Args:
            explanations: List of explanation objects
            
        Returns:
            Analysis results dictionary
        """
        pass
    
    @abstractmethod
    def detect_interactions(self, explanation: Any) -> InteractionAnalysis:
        """Detect and analyze feature interactions.
        
        Args:
            explanation: Explanation object to analyze
            
        Returns:
            Interaction analysis results
        """
        pass
    
    @abstractmethod
    def generate_insights(self, explanations: List[Any]) -> List[str]:
        """Generate actionable insights from explanations.
        
        Args:
            explanations: List of explanation objects
            
        Returns:
            List of insight strings
        """
        pass