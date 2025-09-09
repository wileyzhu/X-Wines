"""
Visualization export utilities for saving plots in multiple formats with customizable styling.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from ..config import VisualizationConfig, ExportConfig
from ..utils.data_exporter import DataExporter
from ..utils.report_generator import HTMLReportGenerator

logger = logging.getLogger(__name__)


class PlotStyleManager:
    """Manages customizable plot styling and formatting options."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize style manager with visualization configuration.
        
        Args:
            config: Visualization configuration object
        """
        self.config = config
        self._custom_styles = {}
        self._setup_default_styles()
    
    def _setup_default_styles(self) -> None:
        """Set up default plot styles."""
        self._custom_styles = {
            'publication': {
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'lines.linewidth': 2,
                'axes.linewidth': 1.5,
                'grid.alpha': 0.3,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            },
            'presentation': {
                'font.size': 16,
                'axes.titlesize': 20,
                'axes.labelsize': 18,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 14,
                'figure.titlesize': 24,
                'lines.linewidth': 3,
                'axes.linewidth': 2,
                'grid.alpha': 0.4,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            },
            'web': {
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'lines.linewidth': 1.5,
                'axes.linewidth': 1,
                'grid.alpha': 0.2,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            }
        }
    
    def apply_style(self, style_name: str = 'default') -> None:
        """Apply a specific style to matplotlib.
        
        Args:
            style_name: Name of style to apply ('default', 'publication', 'presentation', 'web')
        """
        try:
            if style_name == 'default':
                plt.rcParams.update({
                    'font.size': self.config.font_size,
                    'figure.dpi': self.config.dpi,
                    'savefig.dpi': self.config.dpi,
                    'figure.figsize': self.config.figure_size
                })
            elif style_name in self._custom_styles:
                plt.rcParams.update(self._custom_styles[style_name])
                plt.rcParams.update({
                    'figure.dpi': self.config.dpi,
                    'savefig.dpi': self.config.dpi,
                    'figure.figsize': self.config.figure_size
                })
            else:
                logger.warning(f"Unknown style '{style_name}', using default")
                self.apply_style('default')
                
            logger.info(f"Applied plot style: {style_name}")
            
        except Exception as e:
            logger.error(f"Failed to apply style {style_name}: {str(e)}")
            raise RuntimeError(f"Style application failed: {str(e)}")
    
    def create_custom_style(self, name: str, style_params: Dict[str, Any]) -> None:
        """Create a custom style configuration.
        
        Args:
            name: Name for the custom style
            style_params: Dictionary of matplotlib rcParams
        """
        self._custom_styles[name] = style_params
        logger.info(f"Created custom style: {name}")
    
    def get_available_styles(self) -> List[str]:
        """Get list of available style names.
        
        Returns:
            List of available style names
        """
        return ['default'] + list(self._custom_styles.keys())


class VisualizationExporter:
    """Export visualizations in various formats with batch processing capabilities."""
    
    def __init__(self, config: VisualizationConfig, export_config: ExportConfig):
        """Initialize visualization exporter.
        
        Args:
            config: Visualization configuration
            export_config: Export configuration
        """
        self.config = config
        self.export_config = export_config
        self.style_manager = PlotStyleManager(config)
        self._supported_formats = ['png', 'svg', 'pdf', 'jpg', 'eps', 'tiff']
        
        # Create output directory
        self.output_dir = Path(export_config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized VisualizationExporter with output directory: {self.output_dir}")
    
    def save_plot(self, figure: plt.Figure, filename: str, 
                  formats: Optional[List[str]] = None,
                  subdirectory: Optional[str] = None,
                  style: str = 'default',
                  **kwargs) -> Dict[str, Path]:
        """Save a single plot in specified formats.
        
        Args:
            figure: Matplotlib figure to save
            filename: Base filename (without extension)
            formats: List of formats to save in (defaults to config format)
            subdirectory: Optional subdirectory within output directory
            style: Plot style to apply before saving
            **kwargs: Additional parameters for savefig
            
        Returns:
            Dictionary mapping format to saved file path
        """
        if formats is None:
            formats = [self.config.save_format]
        
        # Validate formats
        invalid_formats = [f for f in formats if f not in self._supported_formats]
        if invalid_formats:
            raise ValueError(f"Unsupported formats: {invalid_formats}")
        
        # Apply style
        self.style_manager.apply_style(style)
        
        # Determine output directory
        if subdirectory and self.export_config.create_subdirectories:
            output_path = self.output_dir / subdirectory
        else:
            output_path = self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp if configured
        if self.export_config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        saved_files = {}
        
        # Save in each format
        for fmt in formats:
            try:
                filepath = output_path / f"{filename}.{fmt}"
                
                # Format-specific parameters
                save_params = {
                    'format': fmt,
                    'dpi': self.config.dpi,
                    'bbox_inches': 'tight',
                    'facecolor': 'white',
                    'edgecolor': 'none'
                }
                save_params.update(kwargs)
                
                # Special handling for SVG
                if fmt == 'svg':
                    save_params['transparent'] = True
                
                figure.savefig(filepath, **save_params)
                saved_files[fmt] = filepath
                logger.info(f"Saved plot as {fmt}: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save plot as {fmt}: {str(e)}")
                continue
        
        if not saved_files:
            raise RuntimeError("Failed to save plot in any format")
        
        # Close figure if not showing plots
        if not self.config.show_plots:
            plt.close(figure)
        
        return saved_files
    
    def batch_export_plots(self, figures: Dict[str, plt.Figure],
                          formats: Optional[List[str]] = None,
                          subdirectory: Optional[str] = None,
                          style: str = 'default',
                          **kwargs) -> Dict[str, Dict[str, Path]]:
        """Export multiple plots in batch.
        
        Args:
            figures: Dictionary mapping filename to matplotlib figure
            formats: List of formats to save in
            subdirectory: Optional subdirectory for all plots
            style: Plot style to apply
            **kwargs: Additional parameters for savefig
            
        Returns:
            Dictionary mapping filename to format-path dictionary
        """
        if not figures:
            raise ValueError("No figures provided for batch export")
        
        logger.info(f"Starting batch export of {len(figures)} plots")
        
        results = {}
        failed_exports = []
        
        for filename, figure in figures.items():
            try:
                saved_files = self.save_plot(
                    figure, filename, formats, subdirectory, style, **kwargs
                )
                results[filename] = saved_files
            except Exception as e:
                logger.error(f"Failed to export plot '{filename}': {str(e)}")
                failed_exports.append(filename)
                continue
        
        if failed_exports:
            logger.warning(f"Failed to export {len(failed_exports)} plots: {failed_exports}")
        
        logger.info(f"Batch export completed. Successfully exported {len(results)} plots")
        return results
    
    def create_pdf_report(self, figures: Dict[str, plt.Figure],
                         filename: str = "visualization_report",
                         title: str = "Wine Model Interpretability Report",
                         style: str = 'publication') -> Path:
        """Create a multi-page PDF report with all visualizations.
        
        Args:
            figures: Dictionary mapping plot names to matplotlib figures
            filename: Output PDF filename (without extension)
            title: Report title
            style: Plot style to apply
            
        Returns:
            Path to created PDF file
        """
        if not figures:
            raise ValueError("No figures provided for PDF report")
        
        # Apply style
        self.style_manager.apply_style(style)
        
        # Add timestamp if configured
        if self.export_config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        pdf_path = self.output_dir / f"{filename}.pdf"
        
        try:
            with PdfPages(pdf_path) as pdf:
                # Create title page
                title_fig = self._create_title_page(title)
                pdf.savefig(title_fig, bbox_inches='tight')
                plt.close(title_fig)
                
                # Add each figure
                for plot_name, figure in figures.items():
                    # Add plot title if not present
                    if not figure.axes or not figure.axes[0].get_title():
                        figure.suptitle(plot_name, fontsize=16, fontweight='bold')
                    
                    pdf.savefig(figure, bbox_inches='tight')
                    
                    if not self.config.show_plots:
                        plt.close(figure)
                
                # Set PDF metadata
                pdf_info = pdf.infodict()
                pdf_info['Title'] = title
                pdf_info['Author'] = 'Wine Interpretability System'
                pdf_info['Subject'] = 'Model Interpretability Analysis'
                pdf_info['Creator'] = 'Wine Interpretability Toolkit'
                pdf_info['CreationDate'] = datetime.now()
            
            logger.info(f"Created PDF report: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to create PDF report: {str(e)}")
            raise RuntimeError(f"PDF report creation failed: {str(e)}")
    
    def _create_title_page(self, title: str) -> plt.Figure:
        """Create a title page for PDF reports.
        
        Args:
            title: Report title
            
        Returns:
            Matplotlib figure for title page
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.axis('off')
        
        # Main title
        ax.text(0.5, 0.7, title, ha='center', va='center', 
                fontsize=24, fontweight='bold', transform=ax.transAxes)
        
        # Subtitle
        ax.text(0.5, 0.6, 'Explainable AI Analysis for Wine Quality Prediction', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        
        # Generation info
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.4, f'Generated on: {generation_time}', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        # System info
        ax.text(0.5, 0.3, 'Wine Interpretability Toolkit', 
                ha='center', va='center', fontsize=10, 
                style='italic', transform=ax.transAxes)
        
        return fig
    
    def export_plot_metadata(self, figures: Dict[str, plt.Figure],
                           filename: str = "plot_metadata") -> Path:
        """Export metadata about the plots (dimensions, formats, etc.).
        
        Args:
            figures: Dictionary of figures to analyze
            filename: Output filename for metadata
            
        Returns:
            Path to metadata file
        """
        metadata = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_plots': len(figures),
                'export_config': {
                    'output_directory': str(self.output_dir),
                    'formats': [self.config.save_format],
                    'dpi': self.config.dpi,
                    'figure_size': self.config.figure_size
                }
            },
            'plots': {}
        }
        
        for plot_name, figure in figures.items():
            try:
                # Get figure information
                fig_info = {
                    'size_inches': figure.get_size_inches().tolist(),
                    'dpi': figure.dpi,
                    'num_axes': len(figure.axes),
                    'has_title': bool(figure._suptitle),
                    'axes_info': []
                }
                
                # Get axes information
                for i, ax in enumerate(figure.axes):
                    ax_info = {
                        'index': i,
                        'title': ax.get_title(),
                        'xlabel': ax.get_xlabel(),
                        'ylabel': ax.get_ylabel(),
                        'has_legend': ax.get_legend() is not None,
                        'xlim': ax.get_xlim(),
                        'ylim': ax.get_ylim()
                    }
                    fig_info['axes_info'].append(ax_info)
                
                metadata['plots'][plot_name] = fig_info
                
            except Exception as e:
                logger.warning(f"Failed to extract metadata for plot '{plot_name}': {str(e)}")
                metadata['plots'][plot_name] = {'error': str(e)}
        
        # Save metadata
        metadata_path = self.output_dir / f"{filename}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Exported plot metadata: {metadata_path}")
        return metadata_path
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of export configuration and capabilities.
        
        Returns:
            Dictionary with export summary information
        """
        return {
            'output_directory': str(self.output_dir),
            'supported_formats': self._supported_formats,
            'current_format': self.config.save_format,
            'dpi': self.config.dpi,
            'figure_size': self.config.figure_size,
            'available_styles': self.style_manager.get_available_styles(),
            'export_settings': {
                'create_subdirectories': self.export_config.create_subdirectories,
                'timestamp_files': self.export_config.timestamp_files,
                'export_plots': self.export_config.export_plots
            }
        }


class ComprehensiveExporter:
    """Comprehensive export manager that combines visualization and data export capabilities."""
    
    def __init__(self, viz_config: VisualizationConfig, export_config: ExportConfig):
        """Initialize comprehensive exporter.
        
        Args:
            viz_config: Visualization configuration
            export_config: Export configuration
        """
        self.viz_config = viz_config
        self.export_config = export_config
        
        # Initialize component exporters
        self.viz_exporter = VisualizationExporter(viz_config, export_config)
        self.data_exporter = DataExporter(export_config)
        self.report_generator = HTMLReportGenerator(export_config, viz_config)
        
        logger.info("Initialized ComprehensiveExporter with all export capabilities")
    
    def export_complete_analysis(self,
                               model_results: Dict[str, Any],
                               shap_explanation: Optional[Any] = None,
                               lime_explanations: Optional[List[Any]] = None,
                               interaction_analysis: Optional[Any] = None,
                               figures: Optional[Dict[str, plt.Figure]] = None,
                               base_filename: str = "wine_interpretability_analysis") -> Dict[str, Any]:
        """Export complete interpretability analysis including plots, data, and report.
        
        Args:
            model_results: Model training results and metrics
            shap_explanation: SHAP explanation results
            lime_explanations: List of LIME explanations
            interaction_analysis: Feature interaction analysis results
            figures: Dictionary of matplotlib figures
            base_filename: Base filename for all exports
            
        Returns:
            Dictionary with paths to all exported files
        """
        export_results = {
            'plots': {},
            'data': {},
            'reports': {},
            'metadata': {}
        }
        
        try:
            # Export visualizations
            if figures and self.export_config.export_plots:
                logger.info("Exporting visualizations...")
                plot_results = self.viz_exporter.batch_export_plots(
                    figures, 
                    formats=['png', 'svg'],
                    subdirectory='plots'
                )
                export_results['plots'] = plot_results
                
                # Create PDF report with all plots
                pdf_path = self.viz_exporter.create_pdf_report(
                    figures, 
                    filename=f"{base_filename}_plots"
                )
                export_results['reports']['pdf_plots'] = pdf_path
                
                # Export plot metadata
                metadata_path = self.viz_exporter.export_plot_metadata(
                    figures, 
                    filename=f"{base_filename}_plot_metadata"
                )
                export_results['metadata']['plots'] = metadata_path
            
            # Export data
            if self.export_config.export_data:
                logger.info("Exporting data...")
                
                # Export SHAP data
                if shap_explanation:
                    shap_files = self.data_exporter.export_shap_values(
                        shap_explanation,
                        filename=f"{base_filename}_shap"
                    )
                    export_results['data']['shap'] = shap_files
                
                # Export LIME data
                if lime_explanations:
                    lime_files = self.data_exporter.export_lime_explanations(
                        lime_explanations,
                        filename=f"{base_filename}_lime"
                    )
                    export_results['data']['lime'] = lime_files
                
                # Export feature importance rankings
                if shap_explanation or lime_explanations:
                    ranking_files = self.data_exporter.export_feature_importance_ranking(
                        shap_explanation=shap_explanation,
                        lime_explanations=lime_explanations,
                        filename=f"{base_filename}_rankings"
                    )
                    export_results['data']['rankings'] = ranking_files
            
            # Generate HTML report
            if self.export_config.export_html_report:
                logger.info("Generating HTML report...")
                html_report_path = self.report_generator.generate_comprehensive_report(
                    model_results=model_results,
                    shap_explanation=shap_explanation,
                    lime_explanations=lime_explanations,
                    interaction_analysis=interaction_analysis,
                    figures=figures,
                    filename=f"{base_filename}_report"
                )
                export_results['reports']['html'] = html_report_path
            
            # Export summary metadata
            summary_metadata = self._create_export_summary(export_results)
            summary_path = Path(self.export_config.output_directory) / f"{base_filename}_export_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_metadata, f, indent=2, default=str)
            export_results['metadata']['summary'] = summary_path
            
            logger.info(f"Complete analysis export finished. Results: {len(export_results)} categories")
            return export_results
            
        except Exception as e:
            logger.error(f"Failed to export complete analysis: {str(e)}")
            raise RuntimeError(f"Complete analysis export failed: {str(e)}")
    
    def _create_export_summary(self, export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary metadata for the export.
        
        Args:
            export_results: Dictionary with export results
            
        Returns:
            Summary metadata dictionary
        """
        return {
            'export_timestamp': datetime.now().isoformat(),
            'export_configuration': {
                'output_directory': self.export_config.output_directory,
                'export_plots': self.export_config.export_plots,
                'export_data': self.export_config.export_data,
                'export_html_report': self.export_config.export_html_report,
                'data_formats': self.export_config.data_formats,
                'create_subdirectories': self.export_config.create_subdirectories,
                'timestamp_files': self.export_config.timestamp_files
            },
            'visualization_configuration': {
                'figure_size': self.viz_config.figure_size,
                'dpi': self.viz_config.dpi,
                'save_format': self.viz_config.save_format,
                'color_palette': self.viz_config.color_palette,
                'font_size': self.viz_config.font_size
            },
            'export_summary': {
                'plots_exported': len(export_results.get('plots', {})),
                'data_files_exported': sum(len(files) for files in export_results.get('data', {}).values()),
                'reports_generated': len(export_results.get('reports', {})),
                'metadata_files': len(export_results.get('metadata', {}))
            },
            'file_paths': export_results
        }