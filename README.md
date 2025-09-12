# Wine Model Interpretability

A comprehensive framework for interpreting machine learning models in wine quality prediction using SHAP and LIME explanations.

## Features

- 🍷 **Enhanced Wine Dataset**: 21 features including grape varieties, regions, countries, and food pairings
- 🔍 **SHAP Explanations**: Global and local model interpretability using SHAP values
- 🎯 **LIME Explanations**: Local interpretable model-agnostic explanations
- 📊 **Rich Visualizations**: Interactive plots showing feature importance and model behavior
- 🛠️ **Preprocessing Pipeline**: Automated data cleaning, imputation, and standardization

## Project Structure

```
wine_interpretability/
├── wine_interpretability/    # Main package
│   ├── models/              # ML model trainers (LightGBM, XGBoost)
│   ├── explainers/          # SHAP and LIME explainers
│   ├── visualizers/         # Plotting and visualization tools
│   └── utils/               # Data processing utilities
├── examples/                # Example notebooks and workflows
├── utils/                   # Utility scripts
├── data/                    # Processed datasets
├── Dataset/                 # Original raw datasets
└── docs/                    # Documentation and images
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the example notebook:**
   ```bash
   jupyter notebook examples/fixed_interpretability_workflow.ipynb
   ```

3. **Use the enhanced data loader:**
   ```python
   from utils.enhanced_data_loader import create_enhanced_wine_features_from_merged
   
   # Load enhanced dataset with 21 features
   data = create_enhanced_wine_features_from_merged(
       data_path="data/wine_analysis_data_fixed.csv"
   )
   ```

## Enhanced Features

The framework transforms basic wine data into rich, interpretable features:

- **Original**: 5 basic features (ABV, wine_type_encoded, body_encoded, acidity_encoded, rating_count)
- **Enhanced**: 21 features including:
  - 🍇 Grape varieties (Chardonnay, Merlot, Cabernet Sauvignon, etc.)
  - 🌍 Wine regions (Napa Valley, Bordeaux, Tuscany, etc.)
  - 🏳️ Countries (France, Italy, Spain, USA, etc.)
  - 🍽️ Food pairings (Beef, Seafood, Cheese, etc.)
  - 📊 Derived features (vintage counts, blend indicators, etc.)

## Results

Your LIME and SHAP explanations will show meaningful feature names like:
- "Chardonnay" instead of "13.0"
- "France" instead of "1.0"
- "Beef pairing" instead of "4.0"

## License

MIT License - see LICENSE file for details.
