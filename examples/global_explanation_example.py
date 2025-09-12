#!/usr/bin/env python3
"""
Global Explanation Example

This script demonstrates how to run global interpretability analysis using SHAP
to understand overall feature importance patterns in wine quality prediction.

Usage:
    python global_explanation_example.py --data wine_data.csv
    python global_explanation_example.py --data wine_data.csv --model xgboost --output global_results
"""

import argparse
import sys
from pathlib import Path

# Import wine interpretability components
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from wine_interpretability.main import WineInterpretabilityApp, create_default_config
    from wine_interpretability.config import ModelType, SHAPExplainerType
except ImportError:
    print("Error: Could not import wine_interpretability module.")
    print("Make sure you're running from the project root directory or install the package with:")
    print("  pip install -e .")
    sys.exit(1)


def create_global_analysis_config(model_type='lightgbm', output_dir='global_analysis_results'):
    """Create configuration optimized for global analysis.
    
    Args:
        model_type: Model type to use ('lightgbm' or 'xgboost')
        output_dir: Output directory for results
        
    Returns:
        PipelineConfig optimized for global analysis
    """
    config = create_default_config()
    
    # Set model type
    if model_type == 'xgboost':
        config.model_config.model_type = ModelType.XGBOOST
    else:
        config.model_config.model_type = ModelType.LIGHTGBM
    
    # Optimize for global analysis
    config.explanation_config.shap_explainer_type = SHAPExplainerType.TREE
    config.explanation_config.background_samples = 200  # More samples for better global understanding
    config.explanation_config.max_display_features = 15  # Show more features in global plots
    
    # Configure visualization for global analysis
    config.visualization_config.figure_size = (14, 10)  # Larger plots for global analysis
    config.visualization_config.dpi = 300
    
    # Set output directory
    config.export_config.output_directory = output_dir
    config.export_config.create_subdirectories = True
    
    return config


def main():
    """Run global interpretability analysis."""
    parser = argparse.ArgumentParser(
        description="Global Wine Quality Interpretability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script focuses on global interpretability using SHAP values to understand:
- Overall feature importance across all wine samples
- Feature interaction patterns
- Global trends in wine quality prediction

Examples:
  python global_explanation_example.py --data wine_data.csv
  python global_explanation_example.py --data wine_data.csv --model xgboost
  python global_explanation_example.py --data wine_data.csv --output my_global_results
        """
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='Path to the wine dataset CSV file'
    )
    
    parser.add_argument(
        '--target',
        default='quality',
        help='Name of the target column (default: quality)'
    )
    
    parser.add_argument(
        '--model',
        choices=['lightgbm', 'xgboost'],
        default='lightgbm',
        help='Model type to use (default: lightgbm)'
    )
    
    parser.add_argument(
        '--output',
        default='global_analysis_results',
        help='Output directory for results (default: global_analysis_results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate data file
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Create configuration
    config = create_global_analysis_config(args.model, args.output)
    config.verbose = args.verbose
    
    print("="*60)
    print("WINE QUALITY GLOBAL INTERPRETABILITY ANALYSIS")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Output: {args.output}")
    print()
    
    # Run global analysis
    try:
        app = WineInterpretabilityApp(config)
        
        print("Starting global interpretability analysis...")
        print("This will generate SHAP explanations for understanding global feature importance.")
        print()
        
        results = app.run_global_analysis_only(str(data_path), args.target)
        
        print("✓ Global analysis completed successfully!")
        print()
        print("Results Summary:")
        print("-" * 40)
        
        for section, data in results.items():
            if isinstance(data, dict) and 'status' in data:
                print(f"  {section}: {data['status']}")
        
        print()
        print(f"📁 Results saved to: {config.export_config.output_directory}")
        print()
        print("Generated outputs include:")
        print("  • SHAP summary plots showing global feature importance")
        print("  • SHAP waterfall plots for feature contributions")
        print("  • Feature interaction analysis")
        print("  • Global explanation data exports (CSV/JSON)")
        print("  • HTML report with all visualizations")
        
        # Check for specific output files
        output_dir = Path(config.export_config.output_directory)
        if output_dir.exists():
            print()
            print("Key files to examine:")
            
            # Look for common SHAP output files
            shap_files = list(output_dir.glob("*shap*"))
            if shap_files:
                for file_path in shap_files[:5]:  # Show first 5 files
                    print(f"  📊 {file_path.name}")
            
            # Look for HTML report
            html_files = list(output_dir.glob("*.html"))
            if html_files:
                print(f"  📄 {html_files[0].name} (comprehensive report)")
        
    except Exception as e:
        print(f"✗ Global analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()