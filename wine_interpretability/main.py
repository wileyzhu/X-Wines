#!/usr/bin/env python3
"""
Main application interface for Wine Model Interpretability system.

This module provides the primary entry point and orchestrator for the complete
interpretability workflow, from data loading to report generation.
"""

import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# Optional yaml import - graceful fallback if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    # Try relative import first (when used as module)
    from .config import (
        ModelConfig, ExplanationConfig, VisualizationConfig, 
        ExportConfig, PipelineConfig, ModelType, SHAPExplainerType
    )
    from .utils.data_loader import DataLoader
    from .utils.feature_processor import FeatureProcessor
    from .models.lightgbm_trainer import LightGBMTrainer
    from .models.xgboost_trainer import XGBoostTrainer
    from .explainers.shap_explainer import SHAPExplainer
    from .explainers.lime_explainer import LIMEExplainer
    from .explainers.comparison import ExplanationComparator
    from .visualizers.export import VisualizationExporter
    from .utils.data_exporter import DataExporter
    from .utils.report_generator import HTMLReportGenerator
except ImportError:
    # Fall back to direct import (when run directly)
    from config import (
        ModelConfig, ExplanationConfig, VisualizationConfig, 
        ExportConfig, PipelineConfig, ModelType, SHAPExplainerType
    )
    from utils.data_loader import DataLoader
    from utils.feature_processor import FeatureProcessor
    from models.lightgbm_trainer import LightGBMTrainer
    from models.xgboost_trainer import XGBoostTrainer
    from explainers.shap_explainer import SHAPExplainer
    from explainers.lime_explainer import LIMEExplainer
    from explainers.comparison import ExplanationComparator
    from visualizers.export import VisualizationExporter
    from utils.data_exporter import DataExporter
    from utils.report_generator import HTMLReportGenerator


class WineInterpretabilityApp:
    """Main application class that orchestrates all interpretability components."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the application with configuration.
        
        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = None  # Will be initialized with data paths
        self.feature_processor = FeatureProcessor()
        
        # Model trainer will be selected based on config
        self.model_trainer = None
        self.trained_model = None
        
        # Explainers - will be initialized after model training
        self.shap_explainer = None
        self.lime_explainer = None
        self.comparator = None
        
        # Export components
        self.viz_exporter = VisualizationExporter(config.visualization_config, config.export_config)
        self.data_exporter = DataExporter(config.export_config)
        self.report_generator = HTMLReportGenerator(config.export_config, config.visualization_config)
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        self.logger.info("Wine Interpretability App initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger('wine_interpretability')
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        return logger
    
    def _get_config_summary(self) -> str:
        """Get a summary of the current configuration.
        
        Returns:
            String summary of configuration
        """
        return (
            f"Model: {self.config.model_config.model_type.value}, "
            f"SHAP: {self.config.explanation_config.shap_explainer_type.value}, "
            f"Output: {self.config.export_config.output_directory}"
        )
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize the configuration for saving.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'model_config': {
                'model_type': self.config.model_config.model_type.value,
                'optimization_trials': self.config.model_config.optimization_trials,
                'cross_validation_folds': self.config.model_config.cross_validation_folds,
                'random_state': self.config.model_config.random_state
            },
            'explanation_config': {
                'shap_explainer_type': self.config.explanation_config.shap_explainer_type.value,
                'lime_num_features': self.config.explanation_config.lime_num_features,
                'lime_num_samples': self.config.explanation_config.lime_num_samples,
                'background_samples': self.config.explanation_config.background_samples
            },
            'visualization_config': {
                'figure_size': self.config.visualization_config.figure_size,
                'dpi': self.config.visualization_config.dpi,
                'save_format': self.config.visualization_config.save_format
            },
            'export_config': {
                'output_directory': self.config.export_config.output_directory,
                'export_plots': self.config.export_config.export_plots,
                'export_data': self.config.export_config.export_data,
                'export_html_report': self.config.export_config.export_html_report
            }
        }
    
    def _identify_target_leakage_features(self, columns: List[str], target_column: str) -> List[str]:
        """Identify features that represent target leakage.
        
        Args:
            columns: List of all column names
            target_column: Name of the target column
            
        Returns:
            List of column names that should be excluded due to target leakage
        """
        leakage_features = []
        
        # Common patterns for target leakage in wine datasets
        leakage_patterns = [
            'avg_rating',      # Average rating is essentially the target
            'rating_std',      # Standard deviation derived from ratings
            'mean_rating',     # Another form of average rating
            'rating_mean',     # Another form of average rating
            'score',           # Generic score that might be the target
            'rating_avg',      # Another form of average rating
        ]
        
        # Check for exact matches
        for col in columns:
            if col.lower() in [pattern.lower() for pattern in leakage_patterns]:
                leakage_features.append(col)
        
        # Check for columns that are highly correlated with target name
        # (e.g., if target is 'quality', exclude 'quality_score', 'quality_rating', etc.)
        if target_column:
            target_base = target_column.lower()
            for col in columns:
                col_lower = col.lower()
                if (col_lower != target_base and 
                    (target_base in col_lower or col_lower in target_base) and
                    any(keyword in col_lower for keyword in ['rating', 'score', 'quality', 'avg', 'mean'])):
                    leakage_features.append(col)
        
        return leakage_features
    
    def _ensure_output_directory(self):
        """Ensure the output directory exists."""
        output_path = Path(self.config.export_config.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {output_path.absolute()}")
    
    def _save_pipeline_config(self, results: Dict[str, Any]):
        """Save the pipeline configuration to the output directory.
        
        Args:
            results: Analysis results containing configuration
        """
        config_path = Path(self.config.export_config.output_directory) / "pipeline_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(results['pipeline_config'], f, indent=2)
        
        self.logger.info(f"Pipeline configuration saved to: {config_path}")
    
    def run_global_analysis_only(self, data_path: str, target_column: str = 'quality') -> Dict[str, Any]:
        """Run only global interpretability analysis (SHAP global explanations).
        
        Args:
            data_path: Path to the wine dataset
            target_column: Name of the target column
            
        Returns:
            Dictionary containing global analysis results
        """
        self.logger.info("Starting global interpretability analysis")
        
        results = {
            'data_info': {},
            'model_results': {},
            'shap_results': {},
            'export_paths': {}
        }
        
        try:
            self._ensure_output_directory()
            
            results['data_info'] = self._load_and_preprocess_data(data_path, target_column)
            results['model_results'] = self._train_models()
            self.shap_results = self._generate_shap_explanations()
            results['shap_results'] = self.shap_results
            results['export_paths'] = self._export_results(results)
            
            self.logger.info("Global analysis finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Global analysis failed: {str(e)}")
            raise
    
    def run_local_analysis_only(self, data_path: str, target_column: str = 'quality', 
                               sample_indices: Optional[list] = None) -> Dict[str, Any]:
        """Run only local interpretability analysis (LIME explanations).
        
        Args:
            data_path: Path to the wine dataset
            target_column: Name of the target column
            sample_indices: Specific sample indices to analyze (if None, uses random samples)
            
        Returns:
            Dictionary containing local analysis results
        """
        self.logger.info("Starting local interpretability analysis")
        
        results = {
            'data_info': {},
            'model_results': {},
            'lime_results': {},
            'export_paths': {},
            'sample_indices': sample_indices
        }
        
        try:
            self._ensure_output_directory()
            
            results['data_info'] = self._load_and_preprocess_data(data_path, target_column)
            results['model_results'] = self._train_models()
            self.lime_results = self._generate_lime_explanations()
            results['lime_results'] = self.lime_results
            results['export_paths'] = self._export_results(results)
            
            self.logger.info("Local analysis finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Local analysis failed: {str(e)}")
            raise
    
    def run_complete_analysis(self, data_path: str, target_column: str = 'quality') -> Dict[str, Any]:
        """Run the complete interpretability analysis pipeline.
        
        Args:
            data_path: Path to the wine dataset
            target_column: Name of the target column
            
        Returns:
            Dictionary containing analysis results and file paths
        """
        self.logger.info("Starting complete interpretability analysis")
        self.logger.info(f"Configuration: {self._get_config_summary()}")
        
        results = {
            'data_info': {},
            'model_results': {},
            'shap_results': {},
            'lime_results': {},
            'comparison_results': {},
            'export_paths': {},
            'pipeline_config': self._serialize_config()
        }
        
        try:
            # Create output directory
            self._ensure_output_directory()
            
            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Loading and preprocessing data")
            results['data_info'] = self._load_and_preprocess_data(data_path, target_column)
            
            # Step 2: Train models
            self.logger.info("Step 2: Training models")
            results['model_results'] = self._train_models()
            
            # Step 3: Generate SHAP explanations
            self.logger.info("Step 3: Generating SHAP explanations")
            self.shap_results = self._generate_shap_explanations()
            results['shap_results'] = self.shap_results
            
            # Step 4: Generate LIME explanations
            self.logger.info("Step 4: Generating LIME explanations")
            self.lime_results = self._generate_lime_explanations()
            results['lime_results'] = self.lime_results
            
            # Step 5: Compare explanations
            self.logger.info("Step 5: Comparing explanations")
            results['comparison_results'] = self._compare_explanations()
            
            # Step 6: Export results
            self.logger.info("Step 6: Exporting results")
            results['export_paths'] = self._export_results(results)
            
            # Save pipeline configuration
            self._save_pipeline_config(results)
            
            self.logger.info("Complete analysis finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _load_and_preprocess_data(self, data_path: str, target_column: str) -> Dict[str, Any]:
        """Load and preprocess the wine dataset.
        
        Args:
            data_path: Path to the dataset
            target_column: Name of the target column
            
        Returns:
            Dictionary with data information
        """
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            # Load the dataset directly
            df = pd.read_csv(data_path)
            
            # Separate features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Identify and exclude target leakage features
            target_leakage_features = self._identify_target_leakage_features(df.columns, target_column)
            if target_leakage_features:
                self.logger.warning(f"Excluding target leakage features: {target_leakage_features}")
            
            # Get feature columns (exclude target and leakage features)
            excluded_columns = {target_column} | set(target_leakage_features)
            feature_columns = [col for col in df.columns if col not in excluded_columns]
            
            X = df[feature_columns]
            y = df[target_column]
            
            # Identify categorical and numerical columns
            categorical_cols = []
            numerical_cols = []
            
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    categorical_cols.append(col)
                else:
                    # Check if it's actually categorical (low cardinality integers)
                    unique_vals = X[col].nunique()
                    if X[col].dtype in ['int64', 'int32'] and unique_vals < 20:
                        categorical_cols.append(col)
                    else:
                        numerical_cols.append(col)
            
            self.logger.info(f"Identified {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")
            self.logger.info(f"Categorical features: {categorical_cols}")
            
            # Preprocess features using FeatureProcessor
            if categorical_cols:
                X_processed, preprocessing_info = self.feature_processor.prepare_features_for_modeling(
                    X, categorical_cols, numerical_cols, encoding_method='label'
                )
                self.logger.info(f"Feature preprocessing completed: {X.shape} -> {X_processed.shape}")
            else:
                X_processed = X
                preprocessing_info = {'encoding_method': 'none', 'categorical_columns': []}
            
            # Update feature names after preprocessing
            self.feature_names = list(X_processed.columns)
            
            # Split the data
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=self.config.random_state
            )
            
            data_info = {
                'data_path': data_path,
                'target_column': target_column,
                'n_samples': len(df),
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'categorical_features': categorical_cols,
                'numerical_features': numerical_cols,
                'excluded_features': target_leakage_features,
                'preprocessing_info': preprocessing_info,
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'target_stats': {
                    'mean': float(self.y_train.mean()),
                    'std': float(self.y_train.std()),
                    'min': float(self.y_train.min()),
                    'max': float(self.y_train.max())
                },
                'status': 'completed'
            }
            
            self.logger.info(f"Data loaded successfully: {data_info['n_samples']} samples, {data_info['n_features']} features")
            return data_info
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _train_models(self) -> Dict[str, Any]:
        """Train the configured models.
        
        Returns:
            Dictionary with model training results
        """
        self.logger.info("Training models")
        
        try:
            # Select trainer based on configuration
            if self.config.model_config.model_type == ModelType.LIGHTGBM:
                self.model_trainer = LightGBMTrainer(self.config.model_config)
            elif self.config.model_config.model_type == ModelType.XGBOOST:
                self.model_trainer = XGBoostTrainer(self.config.model_config)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_config.model_type}")
            self.model_trainer._best_params = self.model_trainer.optimize_hyperparameters(
                self.X_train.values, self.y_train.values
            )
            # Train the model
            self.trained_model = self.model_trainer.train(
                self.X_train.values, self.y_train.values, self.X_test.values, self.y_test.values
            )
            
            # Calculate performance metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            train_pred = self.trained_model.predict(self.X_train.values)
            test_pred = self.trained_model.predict(self.X_test.values)
            
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_r2 = r2_score(self.y_test, test_pred)
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(self.trained_model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.trained_model.feature_importances_))
            
            model_results = {
                'model_type': self.config.model_config.model_type.value,
                'best_params': getattr(self.model_trainer, '_best_params', {}),
                'training_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'test_r2': float(test_r2),
                'feature_importance': feature_importance,
                'n_features': len(self.feature_names),
                'status': 'completed'
            }
            
            self.logger.info(f"Model training completed. Test RMSE: {model_results['test_rmse']:.4f}, R²: {model_results['test_r2']:.4f}")
            return model_results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _generate_shap_explanations(self) -> Dict[str, Any]:
        """Generate SHAP explanations.
        
        Returns:
            Dictionary with SHAP explanation results
        """
        self.logger.info("Generating SHAP explanations")
        
        try:
            # Initialize SHAP explainer
            self.shap_explainer = SHAPExplainer(self.config.explanation_config)
            
            # Fit the explainer with background data and model
            background_data = self.X_train.values[:self.config.explanation_config.background_samples]
            self.shap_explainer.fit(
                model=self.trained_model,
                X_background=background_data,
                feature_names=self.feature_names
            )
            
            # Generate global explanations
            global_explanations = self.shap_explainer.explain_global(self.X_test.values)
            
            # Generate local explanations for a sample of test data
            sample_size = min(10, len(self.X_test))
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            local_explanations = []
            
            for idx in sample_indices:
                local_exp = self.shap_explainer.explain_local(
                    self.X_test.iloc[[idx]].values
                )
                local_explanations.append({
                    'sample_index': int(idx),
                    'explanation': local_exp
                })
            
            shap_results = {
                'explainer_type': self.config.explanation_config.shap_explainer_type.value,
                'global_explanations': global_explanations,
                'local_explanations': local_explanations,
                'feature_importance': global_explanations.feature_importance if hasattr(global_explanations, 'feature_importance') else {},
                'n_samples_explained': len(self.X_test),
                'n_local_samples': len(local_explanations),
                'status': 'completed'
            }
            
            self.logger.info(f"SHAP explanations generated for {len(self.X_test)} samples")
            return shap_results
            
        except Exception as e:
            self.logger.error(f"SHAP explanation generation failed: {str(e)}")
            raise
    
    def _generate_lime_explanations(self) -> Dict[str, Any]:
        """Generate LIME explanations.
        
        Returns:
            Dictionary with LIME explanation results
        """
        self.logger.info("Generating LIME explanations")
        
        try:
            # Initialize LIME explainer
            self.lime_explainer = LIMEExplainer(self.config.explanation_config)
            
            # Fit the explainer with background data and model
            background_data = self.X_train.values[:self.config.explanation_config.background_samples]
            self.lime_explainer.fit(
                model=self.trained_model,
                X_background=background_data,
                feature_names=self.feature_names
            )
            
            # Generate LIME explanations for a sample of test data
            sample_size = min(10, len(self.X_test))
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            lime_explanations = []
            
            for idx in sample_indices:
                explanation = self.lime_explainer.explain_local(
                    self.X_test.iloc[[idx]].values
                )
                
                lime_explanations.append({
                    'sample_index': int(idx),
                    'explanation': explanation,
                    'prediction': explanation.prediction,
                    'actual': float(self.y_test.iloc[idx]),
                    'feature_importance': explanation.feature_importance,
                    'confidence_interval': explanation.confidence_interval
                })
            
            lime_results = {
                'num_features': self.config.explanation_config.lime_num_features,
                'num_samples': self.config.explanation_config.lime_num_samples,
                'explanations': lime_explanations,
                'n_samples_explained': len(lime_explanations),
                'status': 'completed'
            }
            
            self.logger.info(f"LIME explanations generated for {len(lime_explanations)} samples")
            return lime_results
            
        except Exception as e:
            self.logger.error(f"LIME explanation generation failed: {str(e)}")
            raise
    
    def _compare_explanations(self) -> Dict[str, Any]:
        """Compare SHAP and LIME explanations.
        
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Comparing explanations")
        
        try:
            # Initialize comparison component
            self.comparator = ExplanationComparator()
            
            # Get SHAP and LIME results (assuming they were generated)
            if not hasattr(self, 'shap_results') or not hasattr(self, 'lime_results'):
                self.logger.warning("SHAP or LIME results not available for comparison")
                return {
                    'status': 'skipped - missing explanation results'
                }
            
            # Compare SHAP and LIME explanations
            comparison_results = self.comparator.compare_explanations(
                shap_explanation=self.shap_results.get('global_explanations'),
                lime_explanations=self.lime_results.get('explanations', [])
            )
            
            # Compare feature rankings
            ranking_analysis = self.comparator.compare_feature_rankings(
                shap_explanation=self.shap_results.get('global_explanations'),
                lime_explanations=self.lime_results.get('explanations', [])
            )
            
            comparison_results.update({
                'ranking_analysis': ranking_analysis,
                'n_samples_compared': min(
                    len(self.shap_results.get('local_explanations', [])),
                    len(self.lime_results.get('explanations', []))
                ),
                'status': 'completed'
            })
            
            self.logger.info("Explanation comparison completed")
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Explanation comparison failed: {str(e)}")
            # Return partial results instead of failing completely
            return {
                'status': f'failed - {str(e)}',
                'error': str(e)
            }
    
    def _export_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Export all results and visualizations.
        
        Args:
            results: Complete analysis results
            
        Returns:
            Dictionary with export file paths
        """
        self.logger.info("Exporting results")
        
        try:
            output_dir = Path(self.config.export_config.output_directory)
            export_paths = {}
            
            # Export visualizations if requested
            if self.config.export_config.export_plots:
                # Create some basic plots and export them
                import matplotlib.pyplot as plt
                
                # Create a simple feature importance plot
                if 'feature_importance' in results.get('model_results', {}):
                    fig, ax = plt.subplots(figsize=self.config.visualization_config.figure_size)
                    importance = results['model_results']['feature_importance']
                    if importance:
                        features = list(importance.keys())
                        values = list(importance.values())
                        ax.barh(features, values)
                        ax.set_title('Feature Importance')
                        ax.set_xlabel('Importance')
                        
                        viz_paths = self.viz_exporter.save_plot(
                            figure=fig,
                            filename='feature_importance',
                            subdirectory='visualizations'
                        )
                        export_paths.update(viz_paths)
                        plt.close(fig)
            
            # Export data if requested
            if self.config.export_config.export_data:
                data_paths = {}
                
                # Export SHAP values if available
                if 'shap_results' in results and 'global_explanations' in results['shap_results']:
                    shap_paths = self.data_exporter.export_shap_values(
                        explanation=results['shap_results']['global_explanations'],
                        filename="shap_values",
                        formats=self.config.export_config.data_formats
                    )
                    data_paths.update(shap_paths)
                
                # Export LIME explanations if available
                if 'lime_results' in results and 'explanations' in results['lime_results']:
                    lime_paths = self.data_exporter.export_lime_explanations(
                        explanations=results['lime_results']['explanations'],
                        filename="lime_explanations",
                        formats=self.config.export_config.data_formats
                    )
                    data_paths.update(lime_paths)
                
                export_paths.update(data_paths)
            
            # Generate HTML report if requested
            if self.config.export_config.export_html_report:
                report_path = self.report_generator.generate_html_report(
                    results=results,
                    output_path=output_dir / "report.html",
                    config=self.config
                )
                export_paths['html_report'] = str(report_path)
            
            export_summary = {
                'output_directory': str(output_dir.absolute()),
                'exported_files': export_paths,
                'n_files_exported': len(export_paths),
                'status': 'completed'
            }
            
            self.logger.info(f"Results exported to {output_dir.absolute()}")
            self.logger.info(f"Exported {len(export_paths)} files")
            
            return export_summary
            
        except Exception as e:
            self.logger.error(f"Results export failed: {str(e)}")
            return {
                'output_directory': self.config.export_config.output_directory,
                'status': f'failed - {str(e)}',
                'error': str(e)
            }


def load_config_from_file(config_path: str) -> PipelineConfig:
    """Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Pipeline configuration loaded from file
        
    Raises:
        ValueError: If file format is not supported or configuration is invalid
        FileNotFoundError: If configuration file doesn't exist
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format
    if config_file.suffix.lower() == '.json':
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    elif config_file.suffix.lower() in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ValueError("YAML support not available. Install PyYAML: pip install pyyaml")
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}. Supported: .json, .yaml, .yml")
    
    return _create_config_from_dict(config_data)


def _create_config_from_dict(config_data: Dict[str, Any]) -> PipelineConfig:
    """Create PipelineConfig from dictionary data.
    
    Args:
        config_data: Dictionary containing configuration data
        
    Returns:
        Pipeline configuration object
    """
    # Model configuration
    model_data = config_data.get('model_config', {})
    model_config = ModelConfig(
        model_type=ModelType(model_data.get('model_type', 'lightgbm')),
        hyperparameter_space=model_data.get('hyperparameter_space', {
            'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100}
        }),
        optimization_trials=model_data.get('optimization_trials', 50),
        cross_validation_folds=model_data.get('cross_validation_folds', 5),
        random_state=model_data.get('random_state', 42)
    )
    
    # Explanation configuration
    explanation_data = config_data.get('explanation_config', {})
    explanation_config = ExplanationConfig(
        shap_explainer_type=SHAPExplainerType(explanation_data.get('shap_explainer_type', 'tree')),
        lime_num_features=explanation_data.get('lime_num_features', 10),
        lime_num_samples=explanation_data.get('lime_num_samples', 5000),
        background_samples=explanation_data.get('background_samples', 100)
    )
    
    # Visualization configuration
    viz_data = config_data.get('visualization_config', {})
    visualization_config = VisualizationConfig(
        figure_size=tuple(viz_data.get('figure_size', [12, 8])),
        dpi=viz_data.get('dpi', 300),
        save_format=viz_data.get('save_format', 'png'),
        color_palette=viz_data.get('color_palette', 'viridis'),
        font_size=viz_data.get('font_size', 12),
        show_plots=viz_data.get('show_plots', True)
    )
    
    # Export configuration
    export_data = config_data.get('export_config', {})
    export_config = ExportConfig(
        output_directory=export_data.get('output_directory', 'wine_interpretability_results'),
        export_plots=export_data.get('export_plots', True),
        export_data=export_data.get('export_data', True),
        export_html_report=export_data.get('export_html_report', True),
        data_formats=export_data.get('data_formats', ['csv', 'json'])
    )
    
    return PipelineConfig(
        model_config=model_config,
        explanation_config=explanation_config,
        visualization_config=visualization_config,
        export_config=export_config,
        verbose=config_data.get('verbose', True),
        random_state=config_data.get('random_state', 42)
    )


def save_config_template(output_path: str = "wine_interpretability_config.json"):
    """Save a template configuration file.
    
    Args:
        output_path: Path where to save the template configuration
    """
    template_config = {
        "model_config": {
            "model_type": "lightgbm",
            "hyperparameter_space": {
                "num_leaves": {"type": "int", "low": 10, "high": 300},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                "feature_fraction": {"type": "float", "low": 0.4, "high": 1.0},
                "bagging_fraction": {"type": "float", "low": 0.4, "high": 1.0},
                "min_child_samples": {"type": "int", "low": 5, "high": 100}
            },
            "optimization_trials": 100,
            "cross_validation_folds": 5,
            "random_state": 42
        },
        "explanation_config": {
            "shap_explainer_type": "tree",
            "lime_num_features": 10,
            "lime_num_samples": 5000,
            "background_samples": 100,
            "max_display_features": 20
        },
        "visualization_config": {
            "figure_size": [12, 8],
            "dpi": 300,
            "save_format": "png",
            "color_palette": "viridis",
            "font_size": 12,
            "show_plots": True
        },
        "export_config": {
            "output_directory": "wine_interpretability_results",
            "export_plots": True,
            "export_data": True,
            "export_html_report": True,
            "data_formats": ["csv", "json"],
            "create_subdirectories": True,
            "timestamp_files": True
        },
        "verbose": True,
        "random_state": 42,
        "n_jobs": -1
    }
    
    with open(output_path, 'w') as f:
        json.dump(template_config, f, indent=2)
    
    print(f"Configuration template saved to: {output_path}")


def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration.
    
    Returns:
        Default pipeline configuration
    """
    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        hyperparameter_space={
            'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100}
        },
        optimization_trials=50,
        cross_validation_folds=5
    )
    
    explanation_config = ExplanationConfig(
        shap_explainer_type=SHAPExplainerType.TREE,
        lime_num_features=10,
        lime_num_samples=5000,
        background_samples=100
    )
    
    visualization_config = VisualizationConfig(
        figure_size=(12, 8),
        dpi=300,
        save_format="png"
    )
    
    export_config = ExportConfig(
        output_directory="wine_interpretability_results",
        export_plots=True,
        export_data=True,
        export_html_report=True
    )
    
    return PipelineConfig(
        model_config=model_config,
        explanation_config=explanation_config,
        visualization_config=visualization_config,
        export_config=export_config,
        verbose=True
    )


def main():
    """Command-line interface for the Wine Interpretability system."""
    parser = argparse.ArgumentParser(
        description="Wine Model Interpretability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis with default settings
  python -m wine_interpretability.main --data wine_data.csv
  
  # Use XGBoost model with custom target column
  python -m wine_interpretability.main --data wine_data.csv --target quality --model xgboost
  
  # Use custom configuration file
  python -m wine_interpretability.main --data wine_data.csv --config custom_config.json
  
  # Run only global analysis (SHAP)
  python -m wine_interpretability.main --data wine_data.csv --mode global
  
  # Run only local analysis (LIME)
  python -m wine_interpretability.main --data wine_data.csv --mode local
  
  # Generate configuration template
  python -m wine_interpretability.main --save-config-template my_config.json
        """
    )
    
    # Main command group
    parser.add_argument(
        '--data', 
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
        '--config',
        help='Path to custom configuration JSON/YAML file'
    )
    
    parser.add_argument(
        '--output',
        default='wine_interpretability_results',
        help='Output directory for results (default: wine_interpretability_results)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['complete', 'global', 'local'],
        default='complete',
        help='Analysis mode: complete (SHAP+LIME), global (SHAP only), local (LIME only)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    # Utility commands
    parser.add_argument(
        '--save-config-template',
        metavar='PATH',
        help='Save a configuration template file and exit'
    )
    
    parser.add_argument(
        '--validate-config',
        metavar='PATH',
        help='Validate a configuration file and exit'
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.save_config_template:
        save_config_template(args.save_config_template)
        return
    
    if args.validate_config:
        try:
            config = load_config_from_file(args.validate_config)
            print(f"✓ Configuration file is valid: {args.validate_config}")
            print(f"  Model: {config.model_config.model_type.value}")
            print(f"  SHAP Explainer: {config.explanation_config.shap_explainer_type.value}")
            print(f"  Output Directory: {config.export_config.output_directory}")
        except Exception as e:
            print(f"✗ Configuration file is invalid: {e}")
            sys.exit(1)
        return
    
    # Validate required arguments for analysis
    if not args.data:
        parser.error("--data is required for analysis (use --save-config-template for utility)")
    
    # Create configuration
    if args.config:
        try:
            config = load_config_from_file(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.model == 'xgboost':
        config.model_config.model_type = ModelType.XGBOOST
    
    config.export_config.output_directory = args.output
    
    # Handle verbosity settings
    if args.quiet:
        config.verbose = False
    elif args.verbose:
        config.verbose = True
    
    # Validate data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Run analysis based on mode
    try:
        app = WineInterpretabilityApp(config)
        
        if args.mode == 'complete':
            results = app.run_complete_analysis(str(data_path), args.target)
        elif args.mode == 'global':
            results = app.run_global_analysis_only(str(data_path), args.target)
        elif args.mode == 'local':
            results = app.run_local_analysis_only(str(data_path), args.target)
        
        if not args.quiet:
            print("\n" + "="*60)
            print(f"WINE INTERPRETABILITY ANALYSIS COMPLETE ({args.mode.upper()} MODE)")
            print("="*60)
            print(f"Results saved to: {config.export_config.output_directory}")
            print(f"Configuration saved to: {config.export_config.output_directory}/pipeline_config.json")
            
            print("\nAnalysis Summary:")
            for section, data in results.items():
                if isinstance(data, dict) and 'status' in data:
                    print(f"  {section}: {data['status']}")
                elif section == 'export_paths' and isinstance(data, dict):
                    print(f"  {section}: {len(data)} files exported")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Analysis failed - {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()