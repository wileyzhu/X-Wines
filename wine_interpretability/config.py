"""
Configuration classes for model training and explanation settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"


class SHAPExplainerType(Enum):
    """Supported SHAP explainer types."""
    TREE = "tree"
    LINEAR = "linear" 
    KERNEL = "kernel"


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    model_type: ModelType
    hyperparameter_space: Dict[str, Any]
    optimization_trials: int = 100
    cross_validation_folds: int = 5
    random_state: int = 42
    early_stopping_rounds: Optional[int] = 50
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.optimization_trials <= 0:
            raise ValueError("optimization_trials must be positive")
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least 2")


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation."""
    
    shap_explainer_type: SHAPExplainerType = SHAPExplainerType.TREE
    lime_num_features: int = 10
    lime_num_samples: int = 5000
    background_samples: int = 100
    max_display_features: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.lime_num_features <= 0:
            raise ValueError("lime_num_features must be positive")
        if self.lime_num_samples <= 0:
            raise ValueError("lime_num_samples must be positive")
        if self.background_samples <= 0:
            raise ValueError("background_samples must be positive")


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    figure_size: tuple = (12, 8)
    dpi: int = 300
    save_format: str = "png"
    color_palette: str = "viridis"
    font_size: int = 12
    show_plots: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.save_format not in ["png", "svg", "pdf", "jpg"]:
            raise ValueError("save_format must be one of: png, svg, pdf, jpg")


@dataclass
class ExportConfig:
    """Configuration for exporting results."""
    
    output_directory: str = "results"
    export_plots: bool = True
    export_data: bool = True
    export_html_report: bool = True
    data_formats: List[str] = None
    create_subdirectories: bool = True
    timestamp_files: bool = True
    
    def __post_init__(self):
        """Set default data formats if not provided."""
        if self.data_formats is None:
            self.data_formats = ["csv", "json"]


@dataclass
class PipelineConfig:
    """Main configuration class that combines all configuration components."""
    
    model_config: ModelConfig
    explanation_config: ExplanationConfig
    visualization_config: VisualizationConfig
    export_config: ExportConfig
    
    # Global pipeline settings
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        """Validate the complete pipeline configuration."""
        # Ensure consistent random states across components
        if hasattr(self.model_config, 'random_state'):
            self.model_config.random_state = self.random_state