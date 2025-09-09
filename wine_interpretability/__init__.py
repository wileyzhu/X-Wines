"""
Wine Model Interpretability System

A comprehensive system for training machine learning models on wine quality data
and generating interpretable explanations using SHAP and LIME.
"""

__version__ = "1.0.0"
__author__ = "Wine Interpretability Team"

# Import main components for easy access
from .config import (
    ModelConfig, ExplanationConfig, VisualizationConfig, 
    ExportConfig, PipelineConfig, ModelType, SHAPExplainerType
)

__all__ = [
    'ModelConfig', 'ExplanationConfig', 'VisualizationConfig',
    'ExportConfig', 'PipelineConfig', 'ModelType', 'SHAPExplainerType'
]