#!/usr/bin/env python3
"""
Local Explanation Example

This script demonstrates how to run local interpretability analysis using LIME
to understand individual wine quality predictions and their explanations.

Usage:
    python local_explanation_example.py --data wine_data.csv
    python local_explanation_example.py --data wine_data.csv --samples 0,5,10 --model xgboost
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

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


def create_local_analysis_config(model_type='lightgbm', output_dir='local_analysis_results'):
    """Create configuration optimized for local analysis.
    
    Args:
        model_type: Model type to use ('lightgbm' or 'xgboost')
        output_dir: Output directory for results
        
    Returns:
        PipelineConfig optimized for local analysis
    """
    config = create_default_config()
    
    # Set model type
    if model_type == 'xgboost':
        config.model_config.model_type = ModelType.XGBOOST
    else:
        config.model_config.model_type = ModelType.LIGHTGBM
    
    # Optimize for local analysis
    config.explanation_config.lime_num_features = 8  # Focus on top features
    config.explanation_config.lime_num_samples = 2000  # Good balance of accuracy and speed
    config.explanation_config.background_samples = 100  # Sufficient for local analysis
    
    # Configure visualization for local analysis
    config.visualization_config.figure_size = (12, 8)  # Good size for individual explanations
    config.visualization_config.dpi = 300
    config.visualization_config.show_plots = True
    
    # Set output directory
    config.export_config.output_directory = output_dir
    config.export_config.create_subdirectories = True
    
    return config


def parse_sample_indices(samples_str: str) -> Optional[List[int]]:
    """Parse comma-separated sample indices.
    
    Args:
        samples_str: Comma-separated string of sample indices
        
    Returns:
        List of sample indices or None if parsing fails
    """
    if not samples_str:
        return None
    
    try:
        indices = [int(idx.strip()) for idx in samples_str.split(',')]
        return indices
    except ValueError:
        return None


def main():
    """Run local interpretability analysis."""
    parser = argparse.ArgumentParser(
        description="Local Wine Quality Interpretability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script focuses on local interpretability using LIME to understand:
- Individual wine quality predictions
- Feature contributions for specific wines
- Local decision boundaries and explanations

Examples:
  # Analyze random samples
  python local_explanation_example.py --data wine_data.csv
  
  # Analyze specific wine samples
  python local_explanation_example.py --data wine_data.csv --samples 0,5,10,15
  
  # Use XGBoost model
  python local_explanation_example.py --data wine_data.csv --model xgboost --samples 1,2,3
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
        '--samples',
        help='Comma-separated list of sample indices to analyze (e.g., "0,5,10"). If not provided, random samples will be selected.'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of random samples to analyze if --samples not provided (default: 5)'
    )
    
    parser.add_argument(
        '--output',
        default='local_analysis_results',
        help='Output directory for results (default: local_analysis_results)'
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
    
    # Parse sample indices
    sample_indices = None
    if args.samples:
        sample_indices = parse_sample_indices(args.samples)
        if sample_indices is None:
            print(f"Error: Invalid sample indices format: {args.samples}")
            print("Use comma-separated integers, e.g., '0,5,10'")
            sys.exit(1)
    
    # Create configuration
    config = create_local_analysis_config(args.model, args.output)
    config.verbose = args.verbose
    
    print("="*60)
    print("WINE QUALITY LOCAL INTERPRETABILITY ANALYSIS")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Output: {args.output}")
    
    if sample_indices:
        print(f"Analyzing samples: {sample_indices}")
    else:
        print(f"Analyzing {args.num_samples} random samples")
    
    print()
    
    # Run local analysis
    try:
        app = WineInterpretabilityApp(config)
        
        print("Starting local interpretability analysis...")
        print("This will generate LIME explanations for individual wine predictions.")
        print()
        
        results = app.run_local_analysis_only(
            str(data_path), 
            args.target,
            sample_indices=sample_indices
        )
        
        print("✓ Local analysis completed successfully!")
        print()
        print("Results Summary:")
        print("-" * 40)
        
        for section, data in results.items():
            if isinstance(data, dict) and 'status' in data:
                print(f"  {section}: {data['status']}")
            elif section == 'sample_indices' and data:
                print(f"  analyzed_samples: {data}")
        
        print()
        print(f"📁 Results saved to: {config.export_config.output_directory}")
        print()
        print("Generated outputs include:")
        print("  • LIME explanation plots for each analyzed wine")
        print("  • Feature contribution charts with confidence intervals")
        print("  • Individual prediction breakdowns")
        print("  • Local explanation data exports (CSV/JSON)")
        print("  • HTML report with all individual explanations")
        
        # Check for specific output files
        output_dir = Path(config.export_config.output_directory)
        if output_dir.exists():
            print()
            print("Key files to examine:")
            
            # Look for LIME output files
            lime_files = list(output_dir.glob("*lime*"))
            if lime_files:
                for file_path in lime_files[:5]:  # Show first 5 files
                    print(f"  📊 {file_path.name}")
            
            # Look for individual explanation files
            explanation_files = list(output_dir.glob("*explanation*"))
            if explanation_files:
                for file_path in explanation_files[:3]:  # Show first 3 files
                    print(f"  📋 {file_path.name}")
            
            # Look for HTML report
            html_files = list(output_dir.glob("*.html"))
            if html_files:
                print(f"  📄 {html_files[0].name} (comprehensive report)")
        
        print()
        print("💡 Tips for interpreting results:")
        print("  • Positive feature contributions increase predicted quality")
        print("  • Negative feature contributions decrease predicted quality")
        print("  • Confidence intervals show uncertainty in explanations")
        print("  • Compare explanations across different wine samples")
        
    except Exception as e:
        print(f"✗ Local analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()