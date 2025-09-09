# Wine Model Interpretability

A comprehensive system for training black-box models on wine datasets and generating interpretable explanations using SHAP and LIME.

## Project Structure

```
wine_interpretability/
├── __init__.py              # Main package interface
├── main.py                  # Main application orchestrator
├── config.py                # Configuration classes
├── examples.py              # Usage examples
├── validate_setup.py        # Setup validation script
├── README.md               # This file
├── models/                 # Model training components
│   ├── __init__.py
│   ├── base.py            # Abstract base classes for model training
│   ├── lightgbm_trainer.py
│   └── xgboost_trainer.py
├── explainers/            # Explanation generation components
│   ├── __init__.py
│   ├── base.py           # Abstract base classes for explainers
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   └── comparison.py
├── visualizers/           # Visualization components
│   ├── __init__.py
│   ├── base.py
│   ├── shap_visualizer.py
│   ├── lime_visualizer.py
│   └── export.py
└── utils/                 # Utility components
    ├── __init__.py
    ├── data_loader.py
    ├── feature_processor.py
    └── validation.py
```

## Core Interfaces

### Configuration Classes
- `ModelConfig`: Configuration for model training and hyperparameter optimization
- `ExplanationConfig`: Configuration for SHAP and LIME explanation generation
- `VisualizationConfig`: Configuration for plot generation and styling
- `ExportConfig`: Configuration for result export and file formats
- `PipelineConfig`: Main configuration combining all components

### Abstract Base Classes
- `ModelTrainer`: Abstract base for model training implementations
- `ModelEvaluator`: Utility class for model performance evaluation
- `BaseExplainer`: Abstract base for explanation generation
- `BaseVisualizer`: Abstract base for visualization creation
- `BaseAnalyzer`: Abstract base for explanation analysis

### Main Application
- `WineInterpretabilityApp`: Main orchestrator class for complete analysis pipeline
- `create_default_config()`: Factory function for default configuration

## Usage

### Basic Usage
```python
from wine_interpretability import WineInterpretabilityApp, create_default_config

# Create default configuration
config = create_default_config()

# Initialize application
app = WineInterpretabilityApp(config)

# Run complete analysis
results = app.run_complete_analysis('path/to/wine_data.csv')
```

### Command Line Interface
```bash
# Basic analysis
python -m wine_interpretability.main --data wine_data.csv

# With custom settings
python -m wine_interpretability.main --data wine_data.csv --model xgboost --output results/

# Show help
python -m wine_interpretability.main --help
```

### Examples
Run the examples script to see different usage patterns:
```bash
python wine_interpretability/examples.py
```

## Validation

To validate the project setup:
```bash
python wine_interpretability/validate_setup.py
```

## Implementation Status

✅ **Task 1 Complete**: Project structure and core interfaces
- Directory structure created
- Abstract base classes defined
- Configuration classes implemented
- Main application interface created
- Validation and examples provided

🔄 **Next Tasks**: Implementation of specific components (data loading, model training, explanation generation, etc.)

## Requirements

The core interfaces are designed to work with:
- Python 3.8+
- NumPy (for data handling)
- scikit-learn (for model evaluation)
- LightGBM and XGBoost (for model training)
- SHAP and LIME (for explanations)
- Matplotlib/Seaborn (for visualizations)

Note: Dependencies will be installed as needed during implementation of specific tasks.

## Features

- **Model Training**: Support for LightGBM and XGBoost with hyperparameter optimization
- **SHAP Explanations**: Global and local explanations using SHAP values
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Comparison Analysis**: Compare explanations between different methods
- **Rich Visualizations**: Comprehensive plotting and export capabilities
- **Flexible Configuration**: Configurable settings for all components

## Project Structure

```
wine_interpretability/
├── __init__.py              # Package initialization
├── config.py                # Configuration classes
├── models/                  # Model training components
│   ├── __init__.py
│   ├── base.py             # Abstract base classes
│   ├── lightgbm_trainer.py # LightGBM implementation
│   └── xgboost_trainer.py  # XGBoost implementation
├── explainers/             # Explanation components
│   ├── __init__.py
│   ├── base.py             # Abstract base classes
│   ├── shap_explainer.py   # SHAP implementation
│   ├── lime_explainer.py   # LIME implementation
│   └── comparison.py       # Explanation comparison
├── visualizers/            # Visualization components
│   ├── __init__.py
│   ├── base.py             # Base visualization classes
│   ├── shap_visualizer.py  # SHAP visualizations
│   ├── lime_visualizer.py  # LIME visualizations
│   └── export.py           # Export utilities
└── utils/                  # Utility components
    ├── __init__.py
    ├── data_loader.py      # Data loading utilities
    ├── feature_processor.py # Feature processing
    └── validation.py       # Data validation
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from wine_interpretability import ModelConfig, ExplanationConfig
from wine_interpretability.models import LightGBMTrainer
from wine_interpretability.explainers import SHAPExplainer

# Configure and train model
config = ModelConfig(model_type="lightgbm", optimization_trials=100)
trainer = LightGBMTrainer(config)
model = trainer.train(X_train, y_train, X_val, y_val)

# Generate explanations
exp_config = ExplanationConfig(shap_explainer_type="tree")
explainer = SHAPExplainer(exp_config)
explainer.fit(model, X_background)
explanations = explainer.explain_global(X_test)
```

## Requirements

See `requirements.txt` for detailed dependencies. Key requirements:
- Python >= 3.8
- scikit-learn >= 1.0.0
- lightgbm >= 3.3.0
- xgboost >= 1.6.0
- shap >= 0.41.0
- lime >= 0.2.0