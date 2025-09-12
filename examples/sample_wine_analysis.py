#!/usr/bin/env python3
"""
Sample Wine Dataset Analysis

This script demonstrates a complete interpretability analysis using a sample wine dataset.
It creates synthetic wine data and runs the full interpretability pipeline to showcase
all features of the Wine Model Interpretability system.

Usage:
    python sample_wine_analysis.py
    python sample_wine_analysis.py --samples 1000 --model xgboost
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

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


def create_sample_wine_dataset(n_samples=1000, random_state=42):
    """Create a realistic synthetic wine quality dataset.
    
    Args:
        n_samples: Number of wine samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        pandas.DataFrame with wine features and quality scores
    """
    np.random.seed(random_state)
    
    print(f"Creating synthetic wine dataset with {n_samples} samples...")
    
    # Generate realistic wine chemistry features
    data = {
        # Acidity features
        'fixed_acidity': np.random.normal(8.3, 1.7, n_samples),
        'volatile_acidity': np.random.gamma(2, 0.25, n_samples),  # Skewed distribution
        'citric_acid': np.random.beta(2, 5, n_samples) * 0.8,  # Bounded between 0-0.8
        
        # Sugar and minerals
        'residual_sugar': np.random.lognormal(1.2, 1.0, n_samples),
        'chlorides': np.random.gamma(1.5, 0.05, n_samples),
        
        # Sulfur compounds
        'free_sulfur_dioxide': np.random.gamma(2, 7.5, n_samples),
        'total_sulfur_dioxide': np.random.gamma(3, 15, n_samples),
        
        # Physical properties
        'density': np.random.normal(0.996, 0.003, n_samples),
        'pH': np.random.normal(3.3, 0.15, n_samples),
        
        # Chemical additives
        'sulphates': np.random.gamma(3, 0.22, n_samples),
        'alcohol': np.random.normal(10.4, 1.1, n_samples)
    }
    
    # Ensure realistic bounds
    data['fixed_acidity'] = np.clip(data['fixed_acidity'], 4.6, 15.9)
    data['volatile_acidity'] = np.clip(data['volatile_acidity'], 0.12, 1.58)
    data['citric_acid'] = np.clip(data['citric_acid'], 0.0, 1.0)
    data['residual_sugar'] = np.clip(data['residual_sugar'], 0.9, 15.5)
    data['chlorides'] = np.clip(data['chlorides'], 0.012, 0.611)
    data['free_sulfur_dioxide'] = np.clip(data['free_sulfur_dioxide'], 1, 72)
    data['total_sulfur_dioxide'] = np.clip(data['total_sulfur_dioxide'], 6, 289)
    data['density'] = np.clip(data['density'], 0.99007, 1.00369)
    data['pH'] = np.clip(data['pH'], 2.74, 4.01)
    data['sulphates'] = np.clip(data['sulphates'], 0.33, 2.0)
    data['alcohol'] = np.clip(data['alcohol'], 8.4, 14.9)
    
    # Create realistic quality scores based on wine chemistry knowledge
    quality_score = (
        # Positive factors (higher values improve quality)
        2.0 * (data['alcohol'] - 8.4) / (14.9 - 8.4) +  # Normalized alcohol
        1.5 * (data['sulphates'] - 0.33) / (2.0 - 0.33) +  # Normalized sulphates
        0.8 * (data['citric_acid'] - 0.0) / (1.0 - 0.0) +  # Normalized citric acid
        
        # Negative factors (higher values decrease quality)
        -2.5 * (data['volatile_acidity'] - 0.12) / (1.58 - 0.12) +  # Normalized volatile acidity
        -1.2 * (data['chlorides'] - 0.012) / (0.611 - 0.012) +  # Normalized chlorides
        
        # Optimal range factors (deviation from optimal decreases quality)
        -0.5 * np.abs(data['pH'] - 3.3) / 0.7 +  # pH deviation from 3.3
        -0.3 * np.abs(data['density'] - 0.996) / 0.013 +  # Density deviation
        
        # Add some noise
        np.random.normal(0, 0.3, n_samples)
    )
    
    # Convert to 3-9 quality scale (typical wine rating scale)
    quality_min, quality_max = quality_score.min(), quality_score.max()
    quality_normalized = (quality_score - quality_min) / (quality_max - quality_min)
    quality = np.round(3 + 6 * quality_normalized).astype(int)
    quality = np.clip(quality, 3, 9)
    
    # Create DataFrame
    wine_df = pd.DataFrame(data)
    wine_df['quality'] = quality
    
    print(f"✓ Dataset created with shape: {wine_df.shape}")
    print(f"Quality distribution:")
    quality_counts = wine_df['quality'].value_counts().sort_index()
    for q, count in quality_counts.items():
        print(f"  Quality {q}: {count} wines ({count/len(wine_df)*100:.1f}%)")
    
    return wine_df


def create_analysis_config(model_type='lightgbm', output_dir='sample_analysis_results'):
    """Create configuration for sample analysis.
    
    Args:
        model_type: Model type to use
        output_dir: Output directory
        
    Returns:
        Configured PipelineConfig
    """
    config = create_default_config()
    
    # Set model type
    if model_type == 'xgboost':
        config.model_config.model_type = ModelType.XGBOOST
        # XGBoost hyperparameter space
        config.model_config.hyperparameter_space = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    else:
        config.model_config.model_type = ModelType.LIGHTGBM
        # LightGBM hyperparameter space
        config.model_config.hyperparameter_space = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [20, 30, 50]
        }
    
    # Optimize settings for demo
    config.model_config.optimization_trials = 30  # Reduced for faster demo
    config.model_config.cross_validation_folds = 3
    
    # Configure explanations
    config.explanation_config.shap_explainer_type = SHAPExplainerType.TREE
    config.explanation_config.lime_num_features = 8
    config.explanation_config.lime_num_samples = 1000  # Reduced for demo
    config.explanation_config.background_samples = 100
    
    # Configure output
    config.export_config.output_directory = output_dir
    config.export_config.export_plots = True
    config.export_config.export_data = True
    config.export_config.export_html_report = True
    
    return config


def main():
    """Run sample wine analysis demonstration."""
    parser = argparse.ArgumentParser(
        description="Sample Wine Dataset Interpretability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script demonstrates the complete Wine Model Interpretability system using
synthetic wine data. It showcases all features including:

- Model training with hyperparameter optimization
- SHAP global and local explanations
- LIME local explanations
- Explanation comparison and analysis
- Comprehensive result export

The synthetic dataset mimics real wine chemistry and quality relationships.
        """
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of wine samples to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--model',
        choices=['lightgbm', 'xgboost'],
        default='lightgbm',
        help='Model type to use (default: lightgbm)'
    )
    
    parser.add_argument(
        '--output',
        default='sample_analysis_results',
        help='Output directory for results (default: sample_analysis_results)'
    )
    
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save the generated dataset to CSV file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("WINE MODEL INTERPRETABILITY - SAMPLE ANALYSIS DEMONSTRATION")
    print("="*70)
    print(f"Samples: {args.samples}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print()
    
    try:
        # Step 1: Create sample dataset
        wine_df = create_sample_wine_dataset(args.samples)
        
        # Save dataset if requested
        data_path = "sample_wine_data.csv"
        wine_df.to_csv(data_path, index=False)
        print(f"✓ Dataset saved to: {data_path}")
        
        if args.save_data:
            permanent_path = f"wine_sample_{args.samples}.csv"
            wine_df.to_csv(permanent_path, index=False)
            print(f"✓ Dataset also saved to: {permanent_path}")
        
        print()
        
        # Step 2: Create configuration
        config = create_analysis_config(args.model, args.output)
        config.verbose = args.verbose
        
        # Step 3: Run complete analysis
        app = WineInterpretabilityApp(config)
        
        print("Starting complete interpretability analysis...")
        print("This demonstration will:")
        print("  1. Train and optimize the selected model")
        print("  2. Generate SHAP explanations (global feature importance)")
        print("  3. Generate LIME explanations (local instance explanations)")
        print("  4. Compare explanation methods")
        print("  5. Export all results and visualizations")
        print()
        
        results = app.run_complete_analysis(data_path, target_column='quality')
        
        print("✓ Complete analysis finished successfully!")
        print()
        
        # Step 4: Display results summary
        print("Analysis Results Summary:")
        print("-" * 50)
        
        for section, data in results.items():
            if section != 'pipeline_config':
                if isinstance(data, dict) and 'status' in data:
                    print(f"  {section}: {data['status']}")
        
        print()
        print(f"📁 All results saved to: {config.export_config.output_directory}")
        
        # Check output directory
        output_dir = Path(config.export_config.output_directory)
        if output_dir.exists():
            print()
            print("Generated files:")
            
            file_count = 0
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    if file_count <= 10:  # Show first 10 files
                        rel_path = file_path.relative_to(output_dir)
                        print(f"  📄 {rel_path}")
            
            if file_count > 10:
                print(f"  ... and {file_count - 10} more files")
        
        print()
        print("🎯 Key insights to explore:")
        print("  • Check SHAP summary plots for global feature importance")
        print("  • Examine LIME explanations for individual wine predictions")
        print("  • Review the HTML report for comprehensive analysis")
        print("  • Compare how different explanation methods agree/disagree")
        print()
        print("📊 Recommended next steps:")
        print("  1. Open the HTML report in your browser")
        print("  2. Examine feature importance rankings")
        print("  3. Look at individual wine explanations")
        print("  4. Try different model types and compare results")
        
        # Clean up temporary data file if not saving permanently
        if not args.save_data:
            Path(data_path).unlink(missing_ok=True)
            print(f"\n🧹 Cleaned up temporary file: {data_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()