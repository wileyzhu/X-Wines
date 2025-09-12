# 📓 Examples Directory

This directory contains example notebooks and scripts demonstrating wine model interpretability with enhanced features.

## 🚀 Quick Start

### Main Notebook (Recommended)
```bash
jupyter notebook fixed_interpretability_workflow.ipynb
```

This notebook demonstrates the complete workflow with **21 enhanced features** instead of 5 basic ones.

## 📁 Files Overview

### 📓 Notebooks
- **`fixed_interpretability_workflow.ipynb`** - **Main notebook** with enhanced 21-feature analysis
- **`complete_interpretability_workflow.ipynb`** - Alternative comprehensive workflow

### 🐍 Python Scripts
- **`global_explanation_example.py`** - SHAP global explanations demo
- **`local_explanation_example.py`** - LIME local explanations demo  
- **`sample_wine_analysis.py`** - Simple analysis script for quick testing

### 🔧 Helper Files (For Notebook Updates)
- **`updated_notebook_cell.py`** - Fixed import cell with proper paths
- **`updated_data_loading_cell.py`** - Enhanced data loading with 21 features
- **`updated_analysis_cell.py`** - Updated analysis configuration
- **`notebook_update_guide.md`** - Step-by-step update instructions

## 🎯 What You'll Get

### Before Enhancement (5 features)
```
LIME Feature Importance:
- 13.0: 0.045    # Cryptic ABV value
- 1.0: 0.032     # Encoded wine type  
- 4.0: 0.028     # Encoded body
```

### After Enhancement (21 features)
```
LIME Feature Importance:
- Chardonnay: 0.045      # Clear grape variety
- France: 0.032          # Interpretable country
- Beef pairing: 0.028    # Meaningful food pairing
- Bordeaux: 0.025        # Recognizable region
- Single varietal: 0.022 # Wine characteristic
```

## 🔄 How to Update Your Notebook

If your notebook shows only 5 features instead of 21, follow these steps:

### Option 1: Quick Fix
1. **Replace import cell** with code from `updated_notebook_cell.py`
2. **Replace data loading** with code from `updated_data_loading_cell.py`  
3. **Replace analysis cell** with code from `updated_analysis_cell.py`

### Option 2: Manual Updates
1. **Change data path**: `'wine_analysis_data_fixed.csv'` → `'../data/enhanced_wine_analysis_data.csv'`
2. **Update imports**: Add `from utils.enhanced_data_loader import create_enhanced_wine_features_from_merged`
3. **Increase LIME features**: Set `lime_num_features=10` in configuration

## 📊 Enhanced Features Included

| Category | Features | Count |
|----------|----------|-------|
| **🍇 Grape Varieties** | Primary grape, blend indicator | 2 |
| **🌍 Geographic** | Country, region, popularity | 3 |
| **🍽️ Food Pairings** | Primary pairing, pairing count | 2 |
| **🍷 Wine Characteristics** | Type, body, acidity, elaboration | 4 |
| **📊 Derived Features** | ABV categories, vintage info, counts | 6 |
| **📈 Original Features** | ABV, rating count, encoded features | 4 |
| **Total** | **Enhanced interpretable features** | **21** |

## 📁 Outputs Directory

All analysis results are saved to `outputs/`:

```
outputs/
├── lightgbm_enhanced_results/     # LightGBM analysis
│   ├── visualizations/            # SHAP & LIME plots
│   ├── shap_data/                 # SHAP values & explanations
│   └── reports/                   # HTML reports
├── xgboost_enhanced_results/      # XGBoost analysis
│   ├── visualizations/            # Model visualizations
│   └── model_data/                # Model metrics & importance
└── enhanced_analysis_results/     # Combined results
    ├── comparison_plots/          # SHAP vs LIME comparisons
    └── feature_analysis/          # Feature importance analysis
```

## 🎨 Generated Visualizations

- **🎯 SHAP Beeswarm Plots**: Global feature importance with value distributions
- **📊 LIME Bar Charts**: Local explanations for individual wine predictions
- **🌊 SHAP Waterfall**: Step-by-step prediction breakdown
- **📈 Feature Importance**: Model-based rankings with meaningful names
- **🔄 Comparison Charts**: SHAP vs LIME consistency analysis

## 🚨 Troubleshooting

### Issue: Only 5 features showing
**Solution**: Update notebook using helper files or manual steps above

### Issue: Import errors
**Solution**: Run from examples directory and ensure parent directory is in path:
```python
import sys
sys.path.insert(0, "..")
```

### Issue: Data file not found
**Solution**: Check data paths point to `../data/` directory

### Issue: LIME shows numbers instead of names
**Solution**: Ensure using `enhanced_wine_analysis_data.csv` not the original file

## 🎯 Expected Results

After running the enhanced notebook, you should see:
- ✅ **21 features** in analysis instead of 5
- ✅ **Meaningful LIME names** like "Chardonnay", "France"
- ✅ **Rich SHAP visualizations** with interpretable features
- ✅ **Model comparison** between LightGBM and XGBoost
- ✅ **Professional outputs** saved to organized directories

## 💡 Tips

- **Start with the main notebook**: `fixed_interpretability_workflow.ipynb`
- **Check outputs directory**: All results are automatically saved
- **Compare models**: Run both LightGBM and XGBoost for comparison
- **Explore features**: The 21 enhanced features provide much richer insights
- **Use helper files**: If notebook needs updates, use the provided helper code

---

**Transform your wine analysis from cryptic numbers to meaningful insights! 🍷✨**
