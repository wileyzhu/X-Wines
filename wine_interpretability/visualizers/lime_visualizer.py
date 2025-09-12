"""
LIME visualization components for creating explanation plots.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import logging

from ..explainers.base import BaseVisualizer, LIMEExplanation
from ..config import VisualizationConfig

logger = logging.getLogger(__name__)


class LIMEVisualizer(BaseVisualizer):
    """LIME-specific visualizations with confidence intervals."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize LIME visualizer with configuration.
        
        Args:
            config: Visualization configuration object
        """
        super().__init__(config)
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Set up matplotlib style for consistent visualizations."""
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi,
            'figure.figsize': self.config.figure_size
        })
    
    def create_summary_plot(self, explanation: LIMEExplanation, **kwargs) -> plt.Figure:
        """Create summary visualization of LIME explanation.
        
        Args:
            explanation: LIME explanation object to visualize
            **kwargs: Additional visualization parameters
                - title: Custom plot title
                - max_features: Maximum number of features to display
                - show_confidence: Whether to show confidence intervals
            
        Returns:
            Matplotlib figure object
        """
        max_features = kwargs.get('max_features', 10)
        title = kwargs.get('title', 'LIME Feature Importance')
        show_confidence = kwargs.get('show_confidence', True)
        
        # Sort features by absolute importance
        sorted_features = sorted(
            explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_features]
        
        features, importances = zip(*sorted_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create horizontal bar plot
        colors = ['red' if imp < 0 else 'green' for imp in importances]
        bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
        
        # Add confidence interval if requested
        if show_confidence and hasattr(explanation, 'confidence_interval'):
            ci_lower, ci_upper = explanation.confidence_interval
            ci_width = ci_upper - ci_lower
            # Add error bars (simplified representation)
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                error = ci_width * 0.1  # Approximate error for each feature
                ax.errorbar(imp, i, xerr=error, fmt='none', color='black', alpha=0.5)
        
        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(title)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add prediction info
        pred_text = f'Prediction: {explanation.prediction:.3f}'
        if show_confidence:
            ci_lower, ci_upper = explanation.confidence_interval
            pred_text += f'\nCI: [{ci_lower:.3f}, {ci_upper:.3f}]'
        
        ax.text(0.02, 0.98, pred_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add value labels on bars for better readability
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            label_x = imp + (0.01 * max(abs(x) for x in importances))
            if imp < 0:
                label_x = imp - (0.01 * max(abs(x) for x in importances))
            ax.text(label_x, i, f'{imp:.3f}', 
                   va='center', ha='left' if imp >= 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def create_feature_importance_plot(self, explanation: LIMEExplanation, **kwargs) -> plt.Figure:
        """Create detailed feature importance visualization with confidence intervals.
        
        Args:
            explanation: LIME explanation object to visualize
            **kwargs: Additional visualization parameters
                - title: Custom plot title
                - show_values: Whether to show importance values on bars
                - orientation: 'horizontal' or 'vertical'
            
        Returns:
            Matplotlib figure object
        """
        title = kwargs.get('title', 'LIME Feature Contributions')
        show_values = kwargs.get('show_values', True)
        orientation = kwargs.get('orientation', 'horizontal')
        
        # Prepare data
        features = list(explanation.feature_importance.keys())
        importances = list(explanation.feature_importance.values())
        
        # Sort by absolute importance
        sorted_data = sorted(zip(features, importances), key=lambda x: abs(x[1]), reverse=True)
        features, importances = zip(*sorted_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create bar plot based on orientation
        if orientation == 'horizontal':
            colors = ['red' if imp < 0 else 'green' for imp in importances]
            bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            
            # Add value labels if requested
            if show_values:
                for i, (bar, imp) in enumerate(zip(bars, importances)):
                    ax.text(imp + (0.01 if imp >= 0 else -0.01), i, f'{imp:.3f}',
                           ha='left' if imp >= 0 else 'right', va='center')
        else:
            colors = ['red' if imp < 0 else 'green' for imp in importances]
            bars = ax.bar(range(len(features)), importances, color=colors, alpha=0.7)
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_ylabel('Feature Importance')
            
            # Add value labels if requested
            if show_values:
                for i, (bar, imp) in enumerate(zip(bars, importances)):
                    ax.text(i, imp + (0.01 if imp >= 0 else -0.01), f'{imp:.3f}',
                           ha='center', va='bottom' if imp >= 0 else 'top')
        
        ax.set_title(title)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3) if orientation == 'vertical' else ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, explanations: List[LIMEExplanation], **kwargs) -> plt.Figure:
        """Create comparison visualization for multiple LIME explanations.
        
        Args:
            explanations: List of LIME explanations to compare
            **kwargs: Additional visualization parameters
                - title: Custom plot title
                - max_features: Maximum number of features to display
                - labels: Labels for each explanation
            
        Returns:
            Matplotlib figure object
        """
        if not explanations:
            raise ValueError("At least one explanation is required")
            
        title = kwargs.get('title', 'LIME Explanations Comparison')
        max_features = kwargs.get('max_features', 10)
        labels = kwargs.get('labels', [f'Sample {i+1}' for i in range(len(explanations))])
        
        # Get common features across all explanations
        all_features = set()
        for exp in explanations:
            all_features.update(exp.feature_importance.keys())
        
        # Calculate average importance for feature selection
        feature_avg_importance = {}
        for feature in all_features:
            importances = []
            for exp in explanations:
                if feature in exp.feature_importance:
                    importances.append(abs(exp.feature_importance[feature]))
                else:
                    importances.append(0)
            feature_avg_importance[feature] = np.mean(importances)
        
        # Select top features
        top_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_features = [f[0] for f in top_features]
        
        # Prepare data matrix
        data_matrix = []
        for exp in explanations:
            row = [exp.feature_importance.get(feature, 0) for feature in selected_features]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(selected_features) * 0.8), max(6, len(explanations) * 0.5)))
        
        im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(selected_features)))
        ax.set_xticklabels(selected_features, rotation=45, ha='right')
        ax.set_yticks(range(len(explanations)))
        ax.set_yticklabels(labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Feature Importance')
        
        # Add text annotations
        for i in range(len(explanations)):
            for j in range(len(selected_features)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(data_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    def create_confidence_interval_plot(self, explanations: List[LIMEExplanation], **kwargs) -> plt.Figure:
        """Create visualization showing confidence intervals across multiple explanations.
        
        Args:
            explanations: List of LIME explanations
            **kwargs: Additional visualization parameters
                - title: Custom plot title
                - feature: Specific feature to analyze (if None, shows all)
            
        Returns:
            Matplotlib figure object
        """
        if not explanations:
            raise ValueError("At least one explanation is required")
            
        title = kwargs.get('title', 'LIME Prediction Confidence Intervals')
        specific_feature = kwargs.get('feature', None)
        
        # Extract predictions and confidence intervals
        predictions = [exp.prediction for exp in explanations]
        ci_lowers = [exp.confidence_interval[0] for exp in explanations]
        ci_uppers = [exp.confidence_interval[1] for exp in explanations]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figure_size[0], self.config.figure_size[1] * 1.2))
        
        # Plot 1: Predictions with confidence intervals
        x_pos = range(len(predictions))
        ax1.errorbar(x_pos, predictions, 
                    yerr=[np.array(predictions) - np.array(ci_lowers), 
                          np.array(ci_uppers) - np.array(predictions)],
                    fmt='o', capsize=5, capthick=2, alpha=0.7)
        
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Predicted Value')
        ax1.set_title('Predictions with Confidence Intervals')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature importance consistency (if specific feature not specified)
        if specific_feature is None:
            # Show feature importance variance across explanations
            all_features = set()
            for exp in explanations:
                all_features.update(exp.feature_importance.keys())
            
            feature_stats = {}
            for feature in all_features:
                importances = []
                for exp in explanations:
                    if feature in exp.feature_importance:
                        importances.append(exp.feature_importance[feature])
                
                if importances:
                    feature_stats[feature] = {
                        'mean': np.mean(importances),
                        'std': np.std(importances)
                    }
            
            # Plot top features by mean importance
            sorted_features = sorted(feature_stats.items(), 
                                   key=lambda x: abs(x[1]['mean']), reverse=True)[:10]
            
            features, stats = zip(*sorted_features)
            means = [s['mean'] for s in stats]
            stds = [s['std'] for s in stats]
            
            ax2.errorbar(range(len(features)), means, yerr=stds, 
                        fmt='s', capsize=5, capthick=2, alpha=0.7)
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels(features, rotation=45, ha='right')
            ax2.set_ylabel('Feature Importance')
            ax2.set_title('Feature Importance Consistency Across Samples')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
        else:
            # Show specific feature importance across samples
            feature_importances = []
            for exp in explanations:
                feature_importances.append(exp.feature_importance.get(specific_feature, 0))
            
            ax2.plot(x_pos, feature_importances, 'o-', alpha=0.7)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Feature Importance')
            ax2.set_title(f'Feature "{specific_feature}" Importance Across Samples')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def save_plot(self, figure: plt.Figure, filename: str, **kwargs) -> None:
        """Save visualization to file.
        
        Args:
            figure: Matplotlib figure to save
            filename: Output filename (without extension)
            **kwargs: Additional save parameters
                - directory: Output directory
                - format: File format (overrides config)
                - dpi: DPI setting (overrides config)
        """
        output_dir = Path(kwargs.get('directory', 'results/lime_plots'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_format = kwargs.get('format', self.config.save_format)
        dpi = kwargs.get('dpi', self.config.dpi)
        
        filepath = output_dir / f"{filename}.{file_format}"
        
        try:
            figure.savefig(filepath, format=file_format, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved LIME plot to {filepath}")
            
            if not self.config.show_plots:
                plt.close(figure)
                
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {str(e)}")
            raise RuntimeError(f"Plot saving failed: {str(e)}")
    
    def create_feature_distribution_plot(self, explanations: List[LIMEExplanation], 
                                       feature_name: str, **kwargs) -> plt.Figure:
        """Create distribution plot for a specific feature's importance across explanations.
        
        Args:
            explanations: List of LIME explanations
            feature_name: Name of feature to analyze
            **kwargs: Additional visualization parameters
            
        Returns:
            Matplotlib figure object
        """
        if not explanations:
            raise ValueError("At least one explanation is required")
            
        # Extract feature importance values
        importances = []
        for exp in explanations:
            if feature_name in exp.feature_importance:
                importances.append(exp.feature_importance[feature_name])
        
        if not importances:
            raise ValueError(f"Feature '{feature_name}' not found in any explanation")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figure_size[0] * 1.5, self.config.figure_size[1]))
        
        # Histogram
        ax1.hist(importances, bins=min(20, len(importances)//2 + 1), alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Feature Importance')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of "{feature_name}" Importance')
        ax1.axvline(x=np.mean(importances), color='red', linestyle='--', label=f'Mean: {np.mean(importances):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(importances, vert=True)
        ax2.set_ylabel('Feature Importance')
        ax2.set_title(f'Box Plot of "{feature_name}" Importance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig