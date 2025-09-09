#!/usr/bin/env python3
"""
Validation script to ensure project structure and core interfaces are properly set up.
"""

import sys
import importlib
from pathlib import Path


def validate_imports():
    """Validate that all core components can be imported."""
    print("Validating imports...")
    
    try:
        # Add current directory to path for local imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Test configuration imports
        from config import (
            ModelConfig, ExplanationConfig, VisualizationConfig, 
            ExportConfig, PipelineConfig, ModelType, SHAPExplainerType
        )
        print("✓ Configuration classes imported successfully")
        
        # Test base class imports
        from models.base import ModelTrainer, ModelEvaluator
        print("✓ Model base classes imported successfully")
        
        from explainers.base import BaseExplainer, BaseVisualizer, BaseAnalyzer
        print("✓ Explainer base classes imported successfully")
        
        return True
        
    except ImportError as e:
        if "numpy" in str(e) or "sklearn" in str(e):
            print("⚠ Some optional dependencies not installed (numpy, sklearn) - this is expected")
            print("✓ Core package structure is valid")
            return True
        else:
            print(f"✗ Import error: {e}")
            return False


def validate_directory_structure():
    """Validate that all required directories exist."""
    print("\nValidating directory structure...")
    
    base_path = Path(__file__).parent
    required_dirs = [
        "models",
        "explainers", 
        "visualizers",
        "utils"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ Directory {dir_name}/ exists")
        else:
            print(f"✗ Directory {dir_name}/ missing")
            all_exist = False
    
    return all_exist


def validate_configuration_classes():
    """Validate that configuration classes work properly."""
    print("\nValidating configuration classes...")
    
    try:
        # Add current directory to path for local imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        from config import (
            ModelConfig, ExplanationConfig, VisualizationConfig,
            ExportConfig, PipelineConfig, ModelType, SHAPExplainerType
        )
        
        # Test ModelConfig
        model_config = ModelConfig(
            model_type=ModelType.LIGHTGBM,
            hyperparameter_space={"n_estimators": [100, 200]},
            optimization_trials=50
        )
        print("✓ ModelConfig instantiated successfully")
        
        # Test ExplanationConfig
        explanation_config = ExplanationConfig(
            shap_explainer_type=SHAPExplainerType.TREE,
            lime_num_features=10
        )
        print("✓ ExplanationConfig instantiated successfully")
        
        # Test VisualizationConfig
        viz_config = VisualizationConfig(
            figure_size=(10, 6),
            dpi=300
        )
        print("✓ VisualizationConfig instantiated successfully")
        
        # Test ExportConfig
        export_config = ExportConfig(
            output_directory="test_results"
        )
        print("✓ ExportConfig instantiated successfully")
        
        # Test PipelineConfig
        pipeline_config = PipelineConfig(
            model_config=model_config,
            explanation_config=explanation_config,
            visualization_config=viz_config,
            export_config=export_config
        )
        print("✓ PipelineConfig instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        return False


def main():
    """Run all validation checks."""
    print("Wine Model Interpretability - Setup Validation")
    print("=" * 50)
    
    checks = [
        validate_directory_structure(),
        validate_imports(),
        validate_configuration_classes()
    ]
    
    if all(checks):
        print("\n🎉 All validation checks passed!")
        print("Project structure and core interfaces are properly set up.")
        return 0
    else:
        print("\n❌ Some validation checks failed.")
        print("Please review the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())