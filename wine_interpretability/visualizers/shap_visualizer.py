"""
SHAP visualization components.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Any, Optional, List, Dict, Union, TYPE_CHECKING
import logging
from pathlib import Path

try:
    from ..explainers.base import BaseVisualizer, SHAPExplanation, InteractionAnalysis
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from explainers.base import BaseVisualizer, SHAPExplanation, InteractionAnalysis

if TYPE_CHECKING:
    from ..config import VisualizationConfig

logger = logging.getLogger(__name__)


class SHAPVisualizer(BaseVisualizer):
    """SHAP-specific visualizations for summary plots, waterfall charts, and interaction plots."""
    
    def __init__(self, config: 'VisualizationConfig'):
        """Initialize SHAP visualizer with configuration.
        
        Args:
            config: Visualization configuration object
        """
        super().__init__(config)
        # Set matplotlib and seaborn styling
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi,
            'figure.figsize': self.config.figure_size
        })
    
    def create_summary_plot(self, explanation: SHAPExplanation, 
                          max_display: Optional[int] = None, **kwargs) -> plt.Figure:
        """Create SHAP summary plot showing feature importance and value distributions.
        
        Args:
            explanation: SHAP explanation object
            max_display: Maximum number of features to display
            **kwargs: Additional parameters for SHAP summary plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            max_display = max_display or self.config.max_display_features
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Create SHAP summary plot
            shap.summary_plot(
                explanation.shap_values,
                explanation.data,
                feature_names=explanation.feature_names,
                max_display=max_display,
                show=False,
                **kwargs
            )
            
            plt.title(f'SHAP Summary Plot ({explanation.explanation_type.title()})', 
                     fontsize=self.config.font_size + 2, fontweight='bold')
            plt.tight_layout()
            
            logger.info(f"Created SHAP summary plot for {explanation.explanation_type} explanation")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create SHAP summary plot: {str(e)}")
            raise RuntimeError(f"SHAP summary plot creation failed: {str(e)}")
    
    def create_enhanced_beeswarm_plot(self, explanation: SHAPExplanation, 
                                    max_display: Optional[int] = None, **kwargs) -> plt.Figure:
        """Create enhanced SHAP beeswarm plot with better styling.
        
        Args:
            explanation: SHAP explanation object
            max_display: Maximum number of features to display
            **kwargs: Additional parameters for beeswarm plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            max_display = max_display or self.config.max_display_features
            
            fig, ax = plt.subplots(figsize=(self.config.figure_size[0], self.config.figure_size[1] + 2))
            
            # Create enhanced beeswarm plot
            shap.plots.beeswarm(
                shap.Explanation(
                    values=explanation.shap_values,
                    base_values=explanation.expected_value,
                    data=explanation.data,
                    feature_names=explanation.feature_names
                ),
                max_display=max_display,
                show=False,
                **kwargs
            )
            
            plt.title(f'Enhanced SHAP Beeswarm Plot\n{explanation.explanation_type.title()} Explanations', 
                     fontsize=self.config.font_size + 4, fontweight='bold', pad=20)
            plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=self.config.font_size + 1)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='x')
            
            # Improve layout
            plt.tight_layout()
            
            logger.info(f"Created enhanced SHAP beeswarm plot for {explanation.explanation_type} explanation")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create enhanced SHAP beeswarm plot: {str(e)}")
            raise RuntimeError(f"Enhanced SHAP beeswarm plot creation failed: {str(e)}")
    
    def create_bar_plot(self, explanation: SHAPExplanation, 
                       max_display: Optional[int] = None, **kwargs) -> plt.Figure:
        """Create SHAP bar plot showing feature importance ranking.
        
        Args:
            explanation: SHAP explanation object
            max_display: Maximum number of features to display
            **kwargs: Additional parameters for bar plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            max_display = max_display or self.config.max_display_features
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Create SHAP bar plot
            shap.plots.bar(
                shap.Explanation(
                    values=explanation.shap_values,
                    base_values=explanation.expected_value,
                    data=explanation.data,
                    feature_names=explanation.feature_names
                ),
                max_display=max_display,
                show=False,
                **kwargs
            )
            
            plt.title(f'SHAP Feature Importance Ranking\n{explanation.explanation_type.title()} Analysis', 
                     fontsize=self.config.font_size + 2, fontweight='bold')
            plt.xlabel('Mean |SHAP Value|', fontsize=self.config.font_size)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            logger.info(f"Created SHAP bar plot for {explanation.explanation_type} explanation")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create SHAP bar plot: {str(e)}")
            raise RuntimeError(f"SHAP bar plot creation failed: {str(e)}")
    
    def create_waterfall_plot(self, explanation: SHAPExplanation, 
                            sample_idx: int = 0, **kwargs) -> plt.Figure:
        """Create SHAP waterfall plot for individual prediction explanation.
        
        Args:
            explanation: SHAP explanation object
            sample_idx: Index of sample to create waterfall plot for
            **kwargs: Additional parameters for waterfall plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            if explanation.explanation_type != 'local':
                logger.warning("Waterfall plots are most meaningful for local explanations")
            
            if sample_idx >= explanation.shap_values.shape[0]:
                raise ValueError(f"Sample index {sample_idx} out of range")
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Create waterfall plot using SHAP
            shap.waterfall_plot(
                shap.Explanation(
                    values=explanation.shap_values[sample_idx],
                    base_values=explanation.expected_value,
                    data=explanation.data[sample_idx],
                    feature_names=explanation.feature_names
                ),
                show=False,
                **kwargs
            )
            
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', 
                     fontsize=self.config.font_size + 2, fontweight='bold')
            plt.tight_layout()
            
            logger.info(f"Created SHAP waterfall plot for sample {sample_idx}")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create SHAP waterfall plot: {str(e)}")
            raise RuntimeError(f"SHAP waterfall plot creation failed: {str(e)}")
    
    def create_feature_importance_plot(self, explanation: SHAPExplanation, 
                                     importance_type: str = 'mean_abs',
                                     max_features: Optional[int] = None, **kwargs) -> plt.Figure:
        """Create global feature importance plot from SHAP values.
        
        Args:
            explanation: SHAP explanation object
            importance_type: Type of importance calculation ('mean_abs', 'mean', 'std')
            max_features: Maximum number of features to display
            **kwargs: Additional visualization parameters
            
        Returns:
            Matplotlib figure object
        """
        try:
            max_features = max_features or self.config.max_display_features
            
            # Calculate feature importance
            if importance_type == 'mean_abs':
                importance = np.mean(np.abs(explanation.shap_values), axis=0)
            elif importance_type == 'mean':
                importance = np.mean(explanation.shap_values, axis=0)
            elif importance_type == 'std':
                importance = np.std(explanation.shap_values, axis=0)
            else:
                raise ValueError(f"Unsupported importance type: {importance_type}")
            
            # Sort features by importance
            feature_importance = list(zip(explanation.feature_names, importance))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Take top features
            top_features = feature_importance[:max_features]
            features, scores = zip(*top_features)
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            colors = ['red' if score < 0 else 'blue' for score in scores]
            bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel(f'SHAP Feature Importance ({importance_type})')
            ax.set_title(f'Global Feature Importance - {importance_type.replace("_", " ").title()}',
                        fontsize=self.config.font_size + 2, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(score + (0.01 * max(abs(s) for s in scores)), i, 
                       f'{score:.3f}', va='center', ha='left' if score >= 0 else 'right')
            
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            logger.info(f"Created feature importance plot with {len(top_features)} features")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create feature importance plot: {str(e)}")
            raise RuntimeError(f"Feature importance plot creation failed: {str(e)}")
    
    def create_interaction_plot(self, interaction_analysis: InteractionAnalysis,
                              top_k: int = 10, **kwargs) -> plt.Figure:
        """Create SHAP interaction plot showing feature pair interactions.
        
        Args:
            interaction_analysis: Interaction analysis results
            top_k: Number of top interactions to display
            **kwargs: Additional visualization parameters
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Get top interactions
            top_interactions = interaction_analysis.top_interactions[:top_k]
            
            if not top_interactions:
                raise ValueError("No interactions found in analysis")
            
            fig, axes = plt.subplots(2, 2, figsize=(self.config.figure_size[0] * 1.5, 
                                                   self.config.figure_size[1] * 1.5))
            axes = axes.flatten()
            
            # Plot 1: Interaction strength heatmap
            if len(interaction_analysis.feature_pairs) > 1:
                # Create interaction matrix for heatmap
                n_features = int(np.sqrt(len(interaction_analysis.feature_pairs))) + 1
                interaction_matrix = np.zeros((n_features, n_features))
                
                for i, (feat1, feat2) in enumerate(interaction_analysis.feature_pairs[:n_features**2]):
                    strength = interaction_analysis.interaction_strength.get((feat1, feat2), 0)
                    row, col = divmod(i, n_features)
                    if row < n_features and col < n_features:
                        interaction_matrix[row, col] = strength
                
                sns.heatmap(interaction_matrix, annot=True, cmap='RdBu_r', center=0,
                           ax=axes[0], cbar_kws={'label': 'Interaction Strength'})
                axes[0].set_title('Feature Interaction Heatmap')
            
            # Plot 2: Top interactions bar plot
            interaction_names = [f"{inter['feature_1']} × {inter['feature_2']}" 
                               for inter in top_interactions]
            interaction_strengths = [inter['strength'] for inter in top_interactions]
            
            axes[1].barh(range(len(interaction_names)), interaction_strengths, 
                        color=sns.color_palette(self.config.color_palette, len(interaction_names)))
            axes[1].set_yticks(range(len(interaction_names)))
            axes[1].set_yticklabels(interaction_names)
            axes[1].set_xlabel('Interaction Strength')
            axes[1].set_title(f'Top {len(top_interactions)} Feature Interactions')
            
            # Plot 3: Interaction values distribution
            if interaction_analysis.interaction_values.size > 0:
                flat_values = interaction_analysis.interaction_values.flatten()
                axes[2].hist(flat_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[2].set_xlabel('Interaction Values')
                axes[2].set_ylabel('Frequency')
                axes[2].set_title('Distribution of Interaction Values')
            
            # Plot 4: Feature pair scatter (if available)
            if top_interactions:
                top_inter = top_interactions[0]
                feat1_idx = top_inter.get('feature_1_idx', 0)
                feat2_idx = top_inter.get('feature_2_idx', 1)
                
                if (interaction_analysis.interaction_values.ndim >= 3 and 
                    feat1_idx < interaction_analysis.interaction_values.shape[1] and
                    feat2_idx < interaction_analysis.interaction_values.shape[2]):
                    
                    inter_vals = interaction_analysis.interaction_values[:, feat1_idx, feat2_idx]
                    axes[3].scatter(range(len(inter_vals)), inter_vals, alpha=0.6)
                    axes[3].set_xlabel('Sample Index')
                    axes[3].set_ylabel('Interaction Value')
                    axes[3].set_title(f'Interaction: {top_inter["feature_1"]} × {top_inter["feature_2"]}')
                else:
                    axes[3].text(0.5, 0.5, 'Interaction data\nnot available', 
                               ha='center', va='center', transform=axes[3].transAxes)
                    axes[3].set_title('Interaction Values')
            
            plt.suptitle('SHAP Feature Interaction Analysis', 
                        fontsize=self.config.font_size + 4, fontweight='bold')
            plt.tight_layout()
            
            logger.info(f"Created interaction plot with {len(top_interactions)} interactions")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create interaction plot: {str(e)}")
            raise RuntimeError(f"Interaction plot creation failed: {str(e)}")
    
    def create_partial_dependence_plot(self, explanation: SHAPExplanation,
                                     feature_names: List[str],
                                     interaction: bool = False, **kwargs) -> plt.Figure:
        """Create partial dependence plots for specified features.
        
        Args:
            explanation: SHAP explanation object
            feature_names: List of feature names to create plots for
            interaction: Whether to create interaction plots between feature pairs
            **kwargs: Additional visualization parameters
            
        Returns:
            Matplotlib figure object
        """
        try:
            n_features = len(feature_names)
            if n_features == 0:
                raise ValueError("No features specified for partial dependence plot")
            
            # Determine subplot layout
            if interaction and n_features >= 2:
                n_plots = min(4, n_features * (n_features - 1) // 2)  # Max 4 interaction plots
                cols = 2
                rows = (n_plots + 1) // 2
            else:
                n_plots = min(6, n_features)  # Max 6 individual plots
                cols = 3 if n_plots > 2 else n_plots
                rows = (n_plots + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            if n_plots == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            else:
                axes = axes.flatten()
            
            plot_idx = 0
            
            if interaction and n_features >= 2:
                # Create interaction plots
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        if plot_idx >= n_plots:
                            break
                        
                        feat1_name = feature_names[i]
                        feat2_name = feature_names[j]
                        
                        # Find feature indices
                        try:
                            feat1_idx = explanation.feature_names.index(feat1_name)
                            feat2_idx = explanation.feature_names.index(feat2_name)
                        except ValueError:
                            continue
                        
                        # Create scatter plot of SHAP values vs feature values
                        feat1_values = explanation.data[:, feat1_idx]
                        feat2_values = explanation.data[:, feat2_idx]
                        shap_vals = explanation.shap_values[:, feat1_idx] + explanation.shap_values[:, feat2_idx]
                        
                        scatter = axes[plot_idx].scatter(feat1_values, feat2_values, 
                                                       c=shap_vals, cmap='RdBu_r', alpha=0.6)
                        axes[plot_idx].set_xlabel(feat1_name)
                        axes[plot_idx].set_ylabel(feat2_name)
                        axes[plot_idx].set_title(f'Interaction: {feat1_name} × {feat2_name}')
                        
                        # Add colorbar
                        plt.colorbar(scatter, ax=axes[plot_idx], label='Combined SHAP Value')
                        
                        plot_idx += 1
            else:
                # Create individual feature plots
                for feat_name in feature_names[:n_plots]:
                    try:
                        feat_idx = explanation.feature_names.index(feat_name)
                    except ValueError:
                        continue
                    
                    feat_values = explanation.data[:, feat_idx]
                    shap_vals = explanation.shap_values[:, feat_idx]
                    
                    # Create scatter plot
                    axes[plot_idx].scatter(feat_values, shap_vals, alpha=0.6, 
                                         color=sns.color_palette(self.config.color_palette)[plot_idx % 10])
                    axes[plot_idx].set_xlabel(f'{feat_name} Value')
                    axes[plot_idx].set_ylabel('SHAP Value')
                    axes[plot_idx].set_title(f'Partial Dependence: {feat_name}')
                    axes[plot_idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('SHAP Partial Dependence Analysis', 
                        fontsize=self.config.font_size + 4, fontweight='bold')
            plt.tight_layout()
            
            logger.info(f"Created partial dependence plot for {len(feature_names)} features")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create partial dependence plot: {str(e)}")
            raise RuntimeError(f"Partial dependence plot creation failed: {str(e)}")
    
    def save_plot(self, figure: plt.Figure, filename: str, 
                  output_dir: Optional[str] = None, **kwargs) -> None:
        """Save visualization to file.
        
        Args:
            figure: Matplotlib figure to save
            filename: Output filename (without extension)
            output_dir: Output directory (optional)
            **kwargs: Additional save parameters
        """
        try:
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                full_path = output_path / f"{filename}.{self.config.save_format}"
            else:
                full_path = f"{filename}.{self.config.save_format}"
            
            figure.savefig(
                full_path,
                format=self.config.save_format,
                dpi=self.config.dpi,
                bbox_inches='tight',
                **kwargs
            )
            
            logger.info(f"Saved plot to {full_path}")
            
            if not self.config.show_plots:
                plt.close(figure)
                
        except Exception as e:
            logger.error(f"Failed to save plot: {str(e)}")
            raise RuntimeError(f"Plot saving failed: {str(e)}")
    
    def create_force_plot(self, explanation: SHAPExplanation, 
                         sample_idx: int = 0, **kwargs) -> Any:
        """Create SHAP force plot for individual prediction.
        
        Args:
            explanation: SHAP explanation object
            sample_idx: Index of sample to create force plot for
            **kwargs: Additional parameters for force plot
            
        Returns:
            SHAP force plot object (for Jupyter display)
        """
        try:
            if sample_idx >= explanation.shap_values.shape[0]:
                raise ValueError(f"Sample index {sample_idx} out of range")
            
            force_plot = shap.force_plot(
                explanation.expected_value,
                explanation.shap_values[sample_idx],
                explanation.data[sample_idx],
                feature_names=explanation.feature_names,
                **kwargs
            )
            
            logger.info(f"Created SHAP force plot for sample {sample_idx}")
            return force_plot
            
        except Exception as e:
            logger.error(f"Failed to create SHAP force plot: {str(e)}")
            raise RuntimeError(f"SHAP force plot creation failed: {str(e)}")