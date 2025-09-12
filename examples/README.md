# Examples Directory

This directory contains example notebooks and scripts for wine model interpretability.

## Files

- `fixed_interpretability_workflow.ipynb` - Main notebook with enhanced features
- `complete_interpretability_workflow.ipynb` - Alternative workflow
- `global_explanation_example.py` - Global SHAP explanations example
- `local_explanation_example.py` - Local LIME explanations example
- `sample_wine_analysis.py` - Simple analysis script

## Helper Files

- `updated_notebook_cell.py` - Updated import cell for notebook
- `updated_data_loading_cell.py` - Updated data loading cell
- `updated_analysis_cell.py` - Updated analysis cell
- `notebook_update_guide.md` - Guide for updating the notebook

## Outputs

All analysis outputs will be saved to the `outputs/` directory:
- Model results and metrics
- SHAP and LIME visualizations  
- Feature importance plots
- HTML reports

## Usage

1. Open `fixed_interpretability_workflow.ipynb`
2. Update cells using the helper files if needed
3. Run the notebook
4. Check `outputs/` for results

The notebook will show enhanced features with meaningful names like "Chardonnay", "France", "Beef pairing" instead of cryptic numbers.
