#!/usr/bin/env python3
"""
Example usage scripts for the Wine Model Interpretability system.

This module provides example code showing how to use the interpretability
system for different use cases.
"""

import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from main import WineInterpretabilityApp, create_default_config
from config import ModelType, SHAPExplainerType


def example_basic_analysis():
    """Example: Basic interpretability analysis with default settings."""
    print("Example: Basic Interpretability Analysis")
    print("=" * 50)
    
    # Create default configuration
    config = create_default_config()
    
    # Initialize the application
    app = WineInterpretabilityApp(config)
    
    # Note: This is a placeholder example - actual data loading will be implemented in task 2.1
    print("Configuration created successfully:")
    print(f"  Model type: {config.model_config.model_type.value}")
    print(f"  SHAP explainer: {config.explanation_config.shap_explainer_type.value}")
    print(f"  Output directory: {config.export_config.output_directory}")
    
    # In a real scenario, you would call:
    # results = app.run_complete_analysis('path/to/wine_data.csv')
    print("\nTo run actual analysis, use:")
    print("  results = app.run_complete_analysis('path/to/wine_data.csv')")


def example_custom_configuration():
    """Example: Custom configuration for specific analysis needs."""
    print("\nExample: Custom Configuration")
    print("=" * 50)
    
    from config import (
        ModelConfig, ExplanationConfig, VisualizationConfig, 
        ExportConfig, PipelineConfig
    )
    
    # Create custom model configuration for XGBoost
    model_config = ModelConfig(
        model_type=ModelType.XGBOOST,
        hyperparameter_space={
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        },
        optimization_trials=100,
        cross_validation_folds=10
    )
    
    # Custom explanation configuration
    explanation_config = ExplanationConfig(
        shap_explainer_type=SHAPExplainerType.KERNEL,  # Use kernel explainer
        lime_num_features=15,  # Show more features
        lime_num_samples=10000,  # More samples for better accuracy
        background_samples=200  # Larger background dataset
    )
    
    # Custom visualization settings
    visualization_config = VisualizationConfig(
        figure_size=(16, 10),  # Larger figures
        dpi=600,  # Higher resolution
        save_format="svg",  # Vector format
        color_palette="plasma"
    )
    
    # Custom export settings
    export_config = ExportConfig(
        output_directory="custom_wine_analysis",
        export_plots=True,
        export_data=True,
        export_html_report=True,
        data_formats=["csv", "json", "parquet"],
        timestamp_files=True
    )
    
    # Combine into pipeline configuration
    config = PipelineConfig(
        model_config=model_config,
        explanation_config=explanation_config,
        visualization_config=visualization_config,
        export_config=export_config,
        verbose=True
    )
    
    print("Custom configuration created:")
    print(f"  Model: {config.model_config.model_type.value}")
    print(f"  Optimization trials: {config.model_config.optimization_trials}")
    print(f"  SHAP explainer: {config.explanation_config.shap_explainer_type.value}")
    print(f"  LIME features: {config.explanation_config.lime_num_features}")
    print(f"  Figure format: {config.visualization_config.save_format}")
    print(f"  Output directory: {config.export_config.output_directory}")


def example_model_comparison():
    """Example: Comparing different models and their interpretations."""
    print("\nExample: Model Comparison")
    print("=" * 50)
    
    # This example shows how you would compare LightGBM and XGBoost
    models_to_compare = [ModelType.LIGHTGBM, ModelType.XGBOOST]
    
    print("Models to compare:")
    for model_type in models_to_compare:
        print(f"  - {model_type.value}")
    
    print("\nFor each model, the system will:")
    print("  1. Train with hyperparameter optimization")
    print("  2. Generate SHAP explanations")
    print("  3. Generate LIME explanations")
    print("  4. Create comparison visualizations")
    print("  5. Export results to separate directories")
    
    # In actual implementation (future tasks), you would:
    # for model_type in models_to_compare:
    #     config = create_default_config()
    #     config.model_config.model_type = model_type
    #     config.export_config.output_directory = f"results_{model_type.value}"
    #     
    #     app = WineInterpretabilityApp(config)
    #     results = app.run_complete_analysis('wine_data.csv')


def example_feature_analysis():
    """Example: Focus on specific feature analysis."""
    print("\nExample: Feature-Focused Analysis")
    print("=" * 50)
    
    # Configuration optimized for detailed feature analysis
    config = create_default_config()
    
    # Increase the number of features analyzed
    config.explanation_config.lime_num_features = 20
    config.explanation_config.max_display_features = 25
    
    # Use more background samples for better SHAP accuracy
    config.explanation_config.background_samples = 500
    
    # Enable all export formats for detailed analysis
    config.export_config.data_formats = ["csv", "json"]
    config.export_config.export_data = True
    
    print("Feature analysis configuration:")
    print(f"  LIME features to analyze: {config.explanation_config.lime_num_features}")
    print(f"  Max features to display: {config.explanation_config.max_display_features}")
    print(f"  Background samples: {config.explanation_config.background_samples}")
    print(f"  Export formats: {config.export_config.data_formats}")
    
    print("\nThis configuration will provide:")
    print("  - Detailed feature importance rankings")
    print("  - Feature interaction analysis")
    print("  - Individual wine sample explanations")
    print("  - Exportable data for further analysis")


def main():
    """Run all examples."""
    print("Wine Model Interpretability - Usage Examples")
    print("=" * 60)
    
    example_basic_analysis()
    example_custom_configuration()
    example_model_comparison()
    example_feature_analysis()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo run the actual system:")
    print("  python -m wine_interpretability.main --data your_wine_data.csv")
    print("\nFor help:")
    print("  python -m wine_interpretability.main --help")


if __name__ == "__main__":
    main()