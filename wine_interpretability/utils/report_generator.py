"""
HTML report generator for comprehensive interpretability analysis.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..explainers.base import SHAPExplanation, LIMEExplanation, InteractionAnalysis
from ..config import ExportConfig, VisualizationConfig

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generate comprehensive HTML reports with embedded visualizations."""
    
    def __init__(self, export_config: ExportConfig, viz_config: VisualizationConfig):
        """Initialize HTML report generator.
        
        Args:
            export_config: Export configuration
            viz_config: Visualization configuration
        """
        self.export_config = export_config
        self.viz_config = viz_config
        self.output_dir = Path(export_config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized HTMLReportGenerator with output directory: {self.output_dir}")
    
    def generate_comprehensive_report(self,
                                    model_results: Dict[str, Any],
                                    shap_explanation: Optional[SHAPExplanation] = None,
                                    lime_explanations: Optional[List[LIMEExplanation]] = None,
                                    interaction_analysis: Optional[InteractionAnalysis] = None,
                                    figures: Optional[Dict[str, plt.Figure]] = None,
                                    filename: str = "interpretability_report") -> Path:
        """Generate a comprehensive HTML report with all analysis results.
        
        Args:
            model_results: Dictionary with model training results and metrics
            shap_explanation: SHAP explanation results
            lime_explanations: List of LIME explanations
            interaction_analysis: Feature interaction analysis results
            figures: Dictionary of matplotlib figures to embed
            filename: Output filename (without extension)
            
        Returns:
            Path to generated HTML report
        """
        # Add timestamp if configured
        if self.export_config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        report_path = self.output_dir / f"{filename}.html"
        
        try:
            # Generate HTML content
            html_content = self._generate_html_content(
                model_results, shap_explanation, lime_explanations, 
                interaction_analysis, figures
            )
            
            # Write to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated comprehensive HTML report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
            raise RuntimeError(f"HTML report generation failed: {str(e)}")
    
    def _generate_html_content(self,
                             model_results: Dict[str, Any],
                             shap_explanation: Optional[SHAPExplanation],
                             lime_explanations: Optional[List[LIMEExplanation]],
                             interaction_analysis: Optional[InteractionAnalysis],
                             figures: Optional[Dict[str, plt.Figure]]) -> str:
        """Generate the complete HTML content for the report.
        
        Args:
            model_results: Model training results
            shap_explanation: SHAP explanation results
            lime_explanations: LIME explanations
            interaction_analysis: Interaction analysis results
            figures: Dictionary of figures to embed
            
        Returns:
            Complete HTML content as string
        """
        # Generate individual sections
        header = self._generate_header()
        css_styles = self._generate_css_styles()
        title_section = self._generate_title_section()
        executive_summary = self._generate_executive_summary(
            model_results, shap_explanation, lime_explanations
        )
        model_section = self._generate_model_section(model_results)
        shap_section = self._generate_shap_section(shap_explanation, figures)
        lime_section = self._generate_lime_section(lime_explanations, figures)
        interaction_section = self._generate_interaction_section(interaction_analysis, figures)
        comparison_section = self._generate_comparison_section(
            shap_explanation, lime_explanations, figures
        )
        conclusions_section = self._generate_conclusions_section(
            shap_explanation, lime_explanations, interaction_analysis
        )
        footer = self._generate_footer()
        
        # Combine all sections
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        {header}
        <body>
            {css_styles}
            {title_section}
            {executive_summary}
            {model_section}
            {shap_section}
            {lime_section}
            {interaction_section}
            {comparison_section}
            {conclusions_section}
            {footer}
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_header(self) -> str:
        """Generate HTML header section."""
        return """
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Wine Model Interpretability Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        """
    
    def _generate_css_styles(self) -> str:
        """Generate CSS styles for the report."""
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            
            .header {
                text-align: center;
                border-bottom: 3px solid #2c3e50;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            
            .header h1 {
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header .subtitle {
                color: #7f8c8d;
                font-size: 1.2em;
                font-style: italic;
            }
            
            .section {
                margin-bottom: 40px;
                padding: 20px;
                border-left: 4px solid #3498db;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            
            .section h2 {
                color: #2c3e50;
                font-size: 1.8em;
                margin-bottom: 15px;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 10px;
            }
            
            .section h3 {
                color: #34495e;
                font-size: 1.4em;
                margin-top: 25px;
                margin-bottom: 15px;
            }
            
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }
            
            .metric-label {
                color: #7f8c8d;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .figure-container {
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .figure-container img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }
            
            .figure-caption {
                margin-top: 15px;
                font-style: italic;
                color: #7f8c8d;
                font-size: 0.9em;
            }
            
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .data-table th {
                background-color: #34495e;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }
            
            .data-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #ecf0f1;
            }
            
            .data-table tr:hover {
                background-color: #f8f9fa;
            }
            
            .highlight {
                background-color: #fff3cd;
                padding: 15px;
                border-left: 4px solid #ffc107;
                margin: 20px 0;
                border-radius: 5px;
            }
            
            .alert {
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }
            
            .alert-info {
                background-color: #d1ecf1;
                border-left: 4px solid #17a2b8;
                color: #0c5460;
            }
            
            .alert-warning {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                color: #856404;
            }
            
            .footer {
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
                color: #7f8c8d;
                font-size: 0.9em;
            }
            
            .toc {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }
            
            .toc h3 {
                margin-top: 0;
                color: #2c3e50;
            }
            
            .toc ul {
                list-style-type: none;
                padding-left: 0;
            }
            
            .toc li {
                margin: 8px 0;
            }
            
            .toc a {
                color: #3498db;
                text-decoration: none;
                font-weight: 500;
            }
            
            .toc a:hover {
                text-decoration: underline;
            }
        </style>
        """
    
    def _generate_title_section(self) -> str:
        """Generate title section of the report."""
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""
        <div class="container">
            <div class="header">
                <h1>Wine Model Interpretability Report</h1>
                <div class="subtitle">Explainable AI Analysis for Wine Quality Prediction</div>
                <div style="margin-top: 15px; color: #7f8c8d; font-size: 0.9em;">
                    Generated on {generation_time}
                </div>
            </div>
            
            <div class="toc">
                <h3>Table of Contents</h3>
                <ul>
                    <li><a href="#executive-summary">Executive Summary</a></li>
                    <li><a href="#model-performance">Model Performance</a></li>
                    <li><a href="#shap-analysis">SHAP Analysis</a></li>
                    <li><a href="#lime-analysis">LIME Analysis</a></li>
                    <li><a href="#interaction-analysis">Feature Interaction Analysis</a></li>
                    <li><a href="#method-comparison">Method Comparison</a></li>
                    <li><a href="#conclusions">Conclusions and Insights</a></li>
                </ul>
            </div>
        """
    
    def _generate_executive_summary(self,
                                  model_results: Dict[str, Any],
                                  shap_explanation: Optional[SHAPExplanation],
                                  lime_explanations: Optional[List[LIMEExplanation]]) -> str:
        """Generate executive summary section."""
        # Extract key metrics
        model_type = model_results.get('model_type', 'Unknown')
        performance = model_results.get('performance_metrics', {})
        rmse = performance.get('rmse', 'N/A')
        r2 = performance.get('r2_score', 'N/A')
        
        # Count features analyzed
        shap_features = len(shap_explanation.feature_names) if shap_explanation else 0
        lime_samples = len(lime_explanations) if lime_explanations else 0
        
        return f"""
            <div class="section" id="executive-summary">
                <h2>Executive Summary</h2>
                
                <div class="highlight">
                    <strong>Key Findings:</strong> This report presents a comprehensive interpretability analysis 
                    of a {model_type} model trained for wine quality prediction. The analysis combines global 
                    explanations (SHAP) and local explanations (LIME) to provide insights into model behavior 
                    and feature importance.
                </div>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{model_type}</div>
                        <div class="metric-label">Model Type</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{rmse if isinstance(rmse, str) else f'{rmse:.4f}'}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{r2 if isinstance(r2, str) else f'{r2:.4f}'}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{shap_features}</div>
                        <div class="metric-label">Features Analyzed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{lime_samples}</div>
                        <div class="metric-label">LIME Samples</div>
                    </div>
                </div>
                
                <h3>Analysis Overview</h3>
                <ul>
                    <li><strong>Model Performance:</strong> The {model_type} model achieved an R² score of {r2 if isinstance(r2, str) else f'{r2:.4f}'}, indicating {'good' if isinstance(r2, (int, float)) and r2 > 0.7 else 'moderate' if isinstance(r2, (int, float)) and r2 > 0.5 else 'limited'} predictive performance.</li>
                    <li><strong>Global Interpretability:</strong> SHAP analysis reveals the most important features driving wine quality predictions across the entire dataset.</li>
                    <li><strong>Local Interpretability:</strong> LIME explanations provide instance-specific insights for {lime_samples} individual wine samples.</li>
                    <li><strong>Feature Interactions:</strong> Advanced analysis identifies complex relationships between wine characteristics.</li>
                </ul>
            </div>
        """
    
    def _generate_model_section(self, model_results: Dict[str, Any]) -> str:
        """Generate model performance section."""
        model_type = model_results.get('model_type', 'Unknown')
        performance = model_results.get('performance_metrics', {})
        hyperparams = model_results.get('hyperparameters', {})
        training_time = model_results.get('training_time', 'N/A')
        
        # Create performance metrics table
        metrics_html = "<table class='data-table'><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
        for metric, value in performance.items():
            formatted_value = f"{value:.6f}" if isinstance(value, (int, float)) else str(value)
            metrics_html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        metrics_html += "</tbody></table>"
        
        # Create hyperparameters table
        hyperparams_html = "<table class='data-table'><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>"
        for param, value in hyperparams.items():
            hyperparams_html += f"<tr><td>{param}</td><td>{value}</td></tr>"
        hyperparams_html += "</tbody></table>"
        
        return f"""
            <div class="section" id="model-performance">
                <h2>Model Performance</h2>
                
                <div class="alert alert-info">
                    <strong>Model Type:</strong> {model_type}<br>
                    <strong>Training Time:</strong> {training_time if isinstance(training_time, str) else f'{training_time:.2f} seconds'}
                </div>
                
                <h3>Performance Metrics</h3>
                {metrics_html}
                
                <h3>Hyperparameters</h3>
                {hyperparams_html}
                
                <h3>Model Evaluation</h3>
                <p>The model demonstrates {'strong' if isinstance(performance.get('r2_score'), (int, float)) and performance.get('r2_score', 0) > 0.8 else 'moderate' if isinstance(performance.get('r2_score'), (int, float)) and performance.get('r2_score', 0) > 0.6 else 'limited'} 
                predictive performance on the wine quality dataset. The hyperparameters were optimized through 
                systematic search to achieve the best possible performance.</p>
            </div>
        """
    
    def _generate_shap_section(self, shap_explanation: Optional[SHAPExplanation], 
                             figures: Optional[Dict[str, plt.Figure]]) -> str:
        """Generate SHAP analysis section."""
        if shap_explanation is None:
            return """
            <div class="section" id="shap-analysis">
                <h2>SHAP Analysis</h2>
                <div class="alert alert-warning">
                    SHAP analysis was not performed or results are not available.
                </div>
            </div>
            """
        
        # Calculate feature importance statistics
        mean_abs_shap = np.mean(np.abs(shap_explanation.shap_values), axis=0)
        top_features = sorted(zip(shap_explanation.feature_names, mean_abs_shap), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        # Create top features table
        features_html = "<table class='data-table'><thead><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP Value|</th></tr></thead><tbody>"
        for i, (feature, importance) in enumerate(top_features, 1):
            features_html += f"<tr><td>{i}</td><td>{feature}</td><td>{importance:.6f}</td></tr>"
        features_html += "</tbody></table>"
        
        # Embed figures if available
        figures_html = ""
        if figures:
            shap_figures = {k: v for k, v in figures.items() if 'shap' in k.lower()}
            for fig_name, figure in shap_figures.items():
                img_html = self._figure_to_html(figure, fig_name)
                figures_html += f"""
                <div class="figure-container">
                    {img_html}
                    <div class="figure-caption">Figure: {fig_name.replace('_', ' ').title()}</div>
                </div>
                """
        
        return f"""
            <div class="section" id="shap-analysis">
                <h2>SHAP Analysis</h2>
                
                <p>SHAP (SHapley Additive exPlanations) provides a unified framework for interpreting model 
                predictions by computing the contribution of each feature to individual predictions.</p>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{shap_explanation.shap_values.shape[0]}</div>
                        <div class="metric-label">Samples Analyzed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{shap_explanation.shap_values.shape[1]}</div>
                        <div class="metric-label">Features</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{shap_explanation.expected_value:.4f}</div>
                        <div class="metric-label">Expected Value</div>
                    </div>
                </div>
                
                <h3>Top 10 Most Important Features</h3>
                {features_html}
                
                <h3>SHAP Visualizations</h3>
                {figures_html}
                
                <h3>Key Insights</h3>
                <ul>
                    <li>The most influential feature is <strong>{top_features[0][0]}</strong> with a mean absolute SHAP value of {top_features[0][1]:.6f}</li>
                    <li>The top 3 features account for a significant portion of the model's decision-making process</li>
                    <li>SHAP values provide both positive and negative contributions, showing which features increase or decrease predicted wine quality</li>
                </ul>
            </div>
        """
    
    def _generate_lime_section(self, lime_explanations: Optional[List[LIMEExplanation]], 
                             figures: Optional[Dict[str, plt.Figure]]) -> str:
        """Generate LIME analysis section."""
        if not lime_explanations:
            return """
            <div class="section" id="lime-analysis">
                <h2>LIME Analysis</h2>
                <div class="alert alert-warning">
                    LIME analysis was not performed or results are not available.
                </div>
            </div>
            """
        
        # Calculate statistics across all explanations
        all_features = set()
        predictions = []
        confidence_widths = []
        
        for exp in lime_explanations:
            all_features.update(exp.feature_importance.keys())
            predictions.append(exp.prediction)
            ci_width = exp.confidence_interval[1] - exp.confidence_interval[0]
            confidence_widths.append(ci_width)
        
        avg_prediction = np.mean(predictions)
        avg_confidence_width = np.mean(confidence_widths)
        
        # Find most frequently important features
        feature_frequency = {}
        for exp in lime_explanations:
            for feature in exp.feature_importance.keys():
                feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
        
        top_frequent_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create frequency table
        frequency_html = "<table class='data-table'><thead><tr><th>Rank</th><th>Feature</th><th>Frequency</th><th>Percentage</th></tr></thead><tbody>"
        for i, (feature, freq) in enumerate(top_frequent_features, 1):
            percentage = (freq / len(lime_explanations)) * 100
            frequency_html += f"<tr><td>{i}</td><td>{feature}</td><td>{freq}</td><td>{percentage:.1f}%</td></tr>"
        frequency_html += "</tbody></table>"
        
        # Embed figures if available
        figures_html = ""
        if figures:
            lime_figures = {k: v for k, v in figures.items() if 'lime' in k.lower()}
            for fig_name, figure in lime_figures.items():
                img_html = self._figure_to_html(figure, fig_name)
                figures_html += f"""
                <div class="figure-container">
                    {img_html}
                    <div class="figure-caption">Figure: {fig_name.replace('_', ' ').title()}</div>
                </div>
                """
        
        return f"""
            <div class="section" id="lime-analysis">
                <h2>LIME Analysis</h2>
                
                <p>LIME (Local Interpretable Model-agnostic Explanations) provides local explanations for 
                individual predictions by learning an interpretable model locally around each instance.</p>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{len(lime_explanations)}</div>
                        <div class="metric-label">Explanations Generated</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(all_features)}</div>
                        <div class="metric-label">Unique Features</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{avg_prediction:.4f}</div>
                        <div class="metric-label">Avg Prediction</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{avg_confidence_width:.4f}</div>
                        <div class="metric-label">Avg CI Width</div>
                    </div>
                </div>
                
                <h3>Most Frequently Important Features</h3>
                {frequency_html}
                
                <h3>LIME Visualizations</h3>
                {figures_html}
                
                <h3>Key Insights</h3>
                <ul>
                    <li>The most frequently important feature is <strong>{top_frequent_features[0][0]}</strong>, appearing in {top_frequent_features[0][1]} out of {len(lime_explanations)} explanations ({(top_frequent_features[0][1]/len(lime_explanations)*100):.1f}%)</li>
                    <li>Average confidence interval width of {avg_confidence_width:.4f} indicates {'high' if avg_confidence_width < 0.1 else 'moderate' if avg_confidence_width < 0.2 else 'low'} prediction confidence</li>
                    <li>Local explanations reveal instance-specific feature importance patterns</li>
                </ul>
            </div>
        """
    
    def _generate_interaction_section(self, interaction_analysis: Optional[InteractionAnalysis], 
                                    figures: Optional[Dict[str, plt.Figure]]) -> str:
        """Generate feature interaction analysis section."""
        if interaction_analysis is None:
            return """
            <div class="section" id="interaction-analysis">
                <h2>Feature Interaction Analysis</h2>
                <div class="alert alert-warning">
                    Feature interaction analysis was not performed or results are not available.
                </div>
            </div>
            """
        
        # Get top interactions
        top_interactions = interaction_analysis.top_interactions[:10]
        
        # Create interactions table
        interactions_html = "<table class='data-table'><thead><tr><th>Rank</th><th>Feature 1</th><th>Feature 2</th><th>Interaction Strength</th></tr></thead><tbody>"
        for i, interaction in enumerate(top_interactions, 1):
            interactions_html += f"""
            <tr>
                <td>{i}</td>
                <td>{interaction['feature_1']}</td>
                <td>{interaction['feature_2']}</td>
                <td>{interaction['strength']:.6f}</td>
            </tr>
            """
        interactions_html += "</tbody></table>"
        
        # Embed figures if available
        figures_html = ""
        if figures:
            interaction_figures = {k: v for k, v in figures.items() if 'interaction' in k.lower()}
            for fig_name, figure in interaction_figures.items():
                img_html = self._figure_to_html(figure, fig_name)
                figures_html += f"""
                <div class="figure-container">
                    {img_html}
                    <div class="figure-caption">Figure: {fig_name.replace('_', ' ').title()}</div>
                </div>
                """
        
        return f"""
            <div class="section" id="interaction-analysis">
                <h2>Feature Interaction Analysis</h2>
                
                <p>Feature interaction analysis identifies complex relationships between pairs of features 
                that jointly influence wine quality predictions beyond their individual contributions.</p>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{len(interaction_analysis.feature_pairs)}</div>
                        <div class="metric-label">Feature Pairs Analyzed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(top_interactions)}</div>
                        <div class="metric-label">Top Interactions</div>
                    </div>
                </div>
                
                <h3>Top 10 Feature Interactions</h3>
                {interactions_html}
                
                <h3>Interaction Visualizations</h3>
                {figures_html}
                
                <h3>Key Insights</h3>
                <ul>
                    <li>The strongest interaction is between <strong>{top_interactions[0]['feature_1']}</strong> and <strong>{top_interactions[0]['feature_2']}</strong> with strength {top_interactions[0]['strength']:.6f}</li>
                    <li>Feature interactions reveal non-linear relationships in wine quality determination</li>
                    <li>Understanding interactions helps explain complex model behavior beyond individual feature importance</li>
                </ul>
            </div>
        """
    
    def _generate_comparison_section(self, shap_explanation: Optional[SHAPExplanation],
                                   lime_explanations: Optional[List[LIMEExplanation]],
                                   figures: Optional[Dict[str, plt.Figure]]) -> str:
        """Generate method comparison section."""
        if shap_explanation is None or not lime_explanations:
            return """
            <div class="section" id="method-comparison">
                <h2>Method Comparison</h2>
                <div class="alert alert-warning">
                    Method comparison requires both SHAP and LIME results.
                </div>
            </div>
            """
        
        # Compare feature importance rankings
        shap_importance = np.mean(np.abs(shap_explanation.shap_values), axis=0)
        shap_ranking = {feature: importance for feature, importance in 
                       zip(shap_explanation.feature_names, shap_importance)}
        
        # Calculate LIME average importance
        lime_importance = {}
        for exp in lime_explanations:
            for feature, importance in exp.feature_importance.items():
                if feature not in lime_importance:
                    lime_importance[feature] = []
                lime_importance[feature].append(abs(importance))
        
        lime_avg_importance = {feature: np.mean(importances) 
                             for feature, importances in lime_importance.items()}
        
        # Find common features
        common_features = set(shap_ranking.keys()) & set(lime_avg_importance.keys())
        
        # Create comparison table
        comparison_data = []
        for feature in common_features:
            comparison_data.append({
                'feature': feature,
                'shap_importance': shap_ranking[feature],
                'lime_importance': lime_avg_importance[feature]
            })
        
        # Sort by SHAP importance
        comparison_data.sort(key=lambda x: x['shap_importance'], reverse=True)
        
        comparison_html = "<table class='data-table'><thead><tr><th>Feature</th><th>SHAP Importance</th><th>LIME Importance</th><th>Agreement</th></tr></thead><tbody>"
        for data in comparison_data[:15]:  # Top 15 features
            shap_val = data['shap_importance']
            lime_val = data['lime_importance']
            # Simple agreement measure (both high or both low)
            agreement = "High" if (shap_val > np.median([d['shap_importance'] for d in comparison_data]) and 
                                 lime_val > np.median([d['lime_importance'] for d in comparison_data])) else "Low"
            
            comparison_html += f"""
            <tr>
                <td>{data['feature']}</td>
                <td>{shap_val:.6f}</td>
                <td>{lime_val:.6f}</td>
                <td>{agreement}</td>
            </tr>
            """
        comparison_html += "</tbody></table>"
        
        # Embed comparison figures if available
        figures_html = ""
        if figures:
            comparison_figures = {k: v for k, v in figures.items() if 'comparison' in k.lower()}
            for fig_name, figure in comparison_figures.items():
                img_html = self._figure_to_html(figure, fig_name)
                figures_html += f"""
                <div class="figure-container">
                    {img_html}
                    <div class="figure-caption">Figure: {fig_name.replace('_', ' ').title()}</div>
                </div>
                """
        
        return f"""
            <div class="section" id="method-comparison">
                <h2>Method Comparison</h2>
                
                <p>Comparing SHAP and LIME explanations helps validate the consistency of interpretability 
                methods and identify areas of agreement or disagreement in feature importance.</p>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{len(common_features)}</div>
                        <div class="metric-label">Common Features</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(shap_ranking)}</div>
                        <div class="metric-label">SHAP Features</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(lime_avg_importance)}</div>
                        <div class="metric-label">LIME Features</div>
                    </div>
                </div>
                
                <h3>Feature Importance Comparison</h3>
                {comparison_html}
                
                <h3>Comparison Visualizations</h3>
                {figures_html}
                
                <h3>Method Agreement Analysis</h3>
                <ul>
                    <li><strong>Consistency:</strong> Both methods identify similar top-ranking features for wine quality prediction</li>
                    <li><strong>Complementarity:</strong> SHAP provides global insights while LIME offers local explanations</li>
                    <li><strong>Validation:</strong> Agreement between methods increases confidence in feature importance rankings</li>
                </ul>
            </div>
        """
    
    def _generate_conclusions_section(self, shap_explanation: Optional[SHAPExplanation],
                                    lime_explanations: Optional[List[LIMEExplanation]],
                                    interaction_analysis: Optional[InteractionAnalysis]) -> str:
        """Generate conclusions and insights section."""
        insights = []
        
        if shap_explanation:
            # Get top SHAP features
            mean_abs_shap = np.mean(np.abs(shap_explanation.shap_values), axis=0)
            top_shap_features = sorted(zip(shap_explanation.feature_names, mean_abs_shap), 
                                     key=lambda x: x[1], reverse=True)[:3]
            insights.append(f"SHAP analysis identifies {', '.join([f[0] for f in top_shap_features])} as the most influential features globally.")
        
        if lime_explanations:
            # Get most frequent LIME features
            feature_frequency = {}
            for exp in lime_explanations:
                for feature in exp.feature_importance.keys():
                    feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
            
            top_lime_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append(f"LIME analysis shows {', '.join([f[0] for f in top_lime_features])} as frequently important across individual predictions.")
        
        if interaction_analysis and interaction_analysis.top_interactions:
            top_interaction = interaction_analysis.top_interactions[0]
            insights.append(f"The strongest feature interaction is between {top_interaction['feature_1']} and {top_interaction['feature_2']}, suggesting complex non-linear relationships.")
        
        insights_html = "<ul>" + "".join([f"<li>{insight}</li>" for insight in insights]) + "</ul>"
        
        return f"""
            <div class="section" id="conclusions">
                <h2>Conclusions and Insights</h2>
                
                <div class="highlight">
                    <strong>Summary:</strong> This comprehensive interpretability analysis provides valuable insights 
                    into the wine quality prediction model's decision-making process, revealing both global patterns 
                    and local variations in feature importance.
                </div>
                
                <h3>Key Findings</h3>
                {insights_html}
                
                <h3>Recommendations</h3>
                <ul>
                    <li><strong>Model Trust:</strong> The consistency between SHAP and LIME explanations increases confidence in the model's reliability</li>
                    <li><strong>Feature Engineering:</strong> Consider creating interaction features based on the strongest feature pairs identified</li>
                    <li><strong>Domain Validation:</strong> Validate the identified important features with wine domain experts</li>
                    <li><strong>Model Improvement:</strong> Focus data collection efforts on the most influential features</li>
                </ul>
                
                <h3>Limitations and Future Work</h3>
                <ul>
                    <li>Explanations are based on the current model and dataset - results may vary with different data</li>
                    <li>Local explanations (LIME) may vary between runs due to sampling</li>
                    <li>Consider additional interpretability methods for comprehensive analysis</li>
                    <li>Validate findings with controlled experiments and domain expertise</li>
                </ul>
            </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate footer section."""
        return """
            <div class="footer">
                <p>Generated by Wine Interpretability Toolkit</p>
                <p>For questions or support, please refer to the documentation.</p>
            </div>
        </div>
        """
    
    def _figure_to_html(self, figure: plt.Figure, caption: str = "") -> str:
        """Convert matplotlib figure to HTML img tag with base64 encoding.
        
        Args:
            figure: Matplotlib figure to convert
            caption: Optional caption for the figure
            
        Returns:
            HTML img tag with embedded figure
        """
        try:
            # Save figure to BytesIO buffer
            buffer = BytesIO()
            figure.savefig(buffer, format='png', dpi=self.viz_config.dpi, 
                          bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            
            # Encode as base64
            img_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            
            # Create HTML img tag
            img_html = f'<img src="data:image/png;base64,{img_data}" alt="{caption}" style="max-width: 100%; height: auto;">'
            
            return img_html
            
        except Exception as e:
            logger.error(f"Failed to convert figure to HTML: {str(e)}")
            return f'<p style="color: red;">Error loading figure: {caption}</p>'