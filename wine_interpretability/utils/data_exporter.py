"""
Data export utilities for SHAP values, LIME explanations, and feature importance rankings.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging

import pandas as pd
import numpy as np

from ..explainers.base import SHAPExplanation, LIMEExplanation, InteractionAnalysis
from ..config import ExportConfig

logger = logging.getLogger(__name__)


class DataExporter:
    """Export SHAP values, LIME explanations, and feature importance data."""
    
    def __init__(self, export_config: ExportConfig):
        """Initialize data exporter with configuration.
        
        Args:
            export_config: Export configuration object
        """
        self.config = export_config
        self.output_dir = Path(export_config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataExporter with output directory: {self.output_dir}")
    
    def export_shap_values(self, explanation: SHAPExplanation,
                          filename: str = "shap_values",
                          formats: Optional[List[str]] = None) -> Dict[str, Path]:
        """Export SHAP values and related data.
        
        Args:
            explanation: SHAP explanation object
            filename: Base filename for export
            formats: List of formats to export ('csv', 'json', 'parquet')
            
        Returns:
            Dictionary mapping format to exported file path
        """
        if formats is None:
            formats = self.config.data_formats
        
        # Prepare data for export
        export_data = self._prepare_shap_data(explanation)
        
        # Add timestamp if configured
        if self.config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        # Create subdirectory if configured
        if self.config.create_subdirectories:
            output_path = self.output_dir / "shap_data"
        else:
            output_path = self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'csv':
                    filepath = self._export_shap_csv(export_data, output_path, filename)
                elif fmt == 'json':
                    filepath = self._export_shap_json(export_data, output_path, filename)
                elif fmt == 'parquet':
                    filepath = self._export_shap_parquet(export_data, output_path, filename)
                else:
                    logger.warning(f"Unsupported format for SHAP export: {fmt}")
                    continue
                
                exported_files[fmt] = filepath
                logger.info(f"Exported SHAP data as {fmt}: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export SHAP data as {fmt}: {str(e)}")
                continue
        
        if not exported_files:
            raise RuntimeError("Failed to export SHAP data in any format")
        
        return exported_files
    
    def export_lime_explanations(self, explanations: List[LIMEExplanation],
                                filename: str = "lime_explanations",
                                formats: Optional[List[str]] = None) -> Dict[str, Path]:
        """Export LIME explanations data.
        
        Args:
            explanations: List of LIME explanation objects
            filename: Base filename for export
            formats: List of formats to export
            
        Returns:
            Dictionary mapping format to exported file path
        """
        if not explanations:
            raise ValueError("No LIME explanations provided")
        
        if formats is None:
            formats = self.config.data_formats
        
        # Prepare data for export
        export_data = self._prepare_lime_data(explanations)
        
        # Add timestamp if configured
        if self.config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        # Create subdirectory if configured
        if self.config.create_subdirectories:
            output_path = self.output_dir / "lime_data"
        else:
            output_path = self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'csv':
                    filepath = self._export_lime_csv(export_data, output_path, filename)
                elif fmt == 'json':
                    filepath = self._export_lime_json(export_data, output_path, filename)
                elif fmt == 'parquet':
                    filepath = self._export_lime_parquet(export_data, output_path, filename)
                else:
                    logger.warning(f"Unsupported format for LIME export: {fmt}")
                    continue
                
                exported_files[fmt] = filepath
                logger.info(f"Exported LIME data as {fmt}: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export LIME data as {fmt}: {str(e)}")
                continue
        
        return exported_files
    
    def export_feature_importance_ranking(self, 
                                        shap_explanation: Optional[SHAPExplanation] = None,
                                        lime_explanations: Optional[List[LIMEExplanation]] = None,
                                        filename: str = "feature_importance_ranking",
                                        formats: Optional[List[str]] = None) -> Dict[str, Path]:
        """Export feature importance rankings with statistical measures.
        
        Args:
            shap_explanation: SHAP explanation for global importance
            lime_explanations: List of LIME explanations for local importance
            filename: Base filename for export
            formats: List of formats to export
            
        Returns:
            Dictionary mapping format to exported file path
        """
        if shap_explanation is None and lime_explanations is None:
            raise ValueError("At least one explanation type must be provided")
        
        if formats is None:
            formats = self.config.data_formats
        
        # Calculate feature importance rankings
        ranking_data = self._calculate_feature_rankings(shap_explanation, lime_explanations)
        
        # Add timestamp if configured
        if self.config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        # Create subdirectory if configured
        if self.config.create_subdirectories:
            output_path = self.output_dir / "feature_rankings"
        else:
            output_path = self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'csv':
                    filepath = self._export_ranking_csv(ranking_data, output_path, filename)
                elif fmt == 'json':
                    filepath = self._export_ranking_json(ranking_data, output_path, filename)
                else:
                    logger.warning(f"Unsupported format for ranking export: {fmt}")
                    continue
                
                exported_files[fmt] = filepath
                logger.info(f"Exported feature rankings as {fmt}: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export feature rankings as {fmt}: {str(e)}")
                continue
        
        return exported_files
    
    def _prepare_shap_data(self, explanation: SHAPExplanation) -> Dict[str, Any]:
        """Prepare SHAP data for export.
        
        Args:
            explanation: SHAP explanation object
            
        Returns:
            Dictionary with prepared data
        """
        return {
            'metadata': {
                'explanation_type': explanation.explanation_type,
                'expected_value': float(explanation.expected_value),
                'num_samples': explanation.shap_values.shape[0],
                'num_features': explanation.shap_values.shape[1],
                'feature_names': explanation.feature_names,
                'export_timestamp': datetime.now().isoformat()
            },
            'shap_values': explanation.shap_values.tolist(),
            'data_values': explanation.data.tolist() if explanation.data is not None else None,
            'feature_names': explanation.feature_names
        }
    
    def _prepare_lime_data(self, explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Prepare LIME data for export.
        
        Args:
            explanations: List of LIME explanations
            
        Returns:
            Dictionary with prepared data
        """
        # Collect all unique features
        all_features = set()
        for exp in explanations:
            all_features.update(exp.feature_importance.keys())
        all_features = sorted(list(all_features))
        
        # Prepare explanation data
        explanation_data = []
        for i, exp in enumerate(explanations):
            exp_data = {
                'sample_id': i,
                'prediction': float(exp.prediction),
                'confidence_interval': [float(exp.confidence_interval[0]), 
                                      float(exp.confidence_interval[1])],
                'local_prediction': float(exp.local_prediction),
                'intercept': float(exp.intercept),
                'feature_importance': {k: float(v) for k, v in exp.feature_importance.items()}
            }
            explanation_data.append(exp_data)
        
        return {
            'metadata': {
                'num_explanations': len(explanations),
                'all_features': all_features,
                'export_timestamp': datetime.now().isoformat()
            },
            'explanations': explanation_data
        }
    
    def _calculate_feature_rankings(self, 
                                  shap_explanation: Optional[SHAPExplanation],
                                  lime_explanations: Optional[List[LIMEExplanation]]) -> Dict[str, Any]:
        """Calculate comprehensive feature importance rankings.
        
        Args:
            shap_explanation: SHAP explanation for global importance
            lime_explanations: List of LIME explanations
            
        Returns:
            Dictionary with ranking data
        """
        rankings = {
            'metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'has_shap': shap_explanation is not None,
                'has_lime': lime_explanations is not None and len(lime_explanations) > 0
            },
            'rankings': {}
        }
        
        # SHAP-based rankings
        if shap_explanation is not None:
            shap_rankings = self._calculate_shap_rankings(shap_explanation)
            rankings['rankings']['shap'] = shap_rankings
        
        # LIME-based rankings
        if lime_explanations:
            lime_rankings = self._calculate_lime_rankings(lime_explanations)
            rankings['rankings']['lime'] = lime_rankings
        
        # Combined rankings if both are available
        if shap_explanation is not None and lime_explanations:
            combined_rankings = self._calculate_combined_rankings(
                rankings['rankings']['shap'], 
                rankings['rankings']['lime']
            )
            rankings['rankings']['combined'] = combined_rankings
        
        return rankings
    
    def _calculate_shap_rankings(self, explanation: SHAPExplanation) -> Dict[str, Any]:
        """Calculate SHAP-based feature rankings.
        
        Args:
            explanation: SHAP explanation object
            
        Returns:
            Dictionary with SHAP rankings
        """
        # Calculate various importance measures
        mean_abs_shap = np.mean(np.abs(explanation.shap_values), axis=0)
        mean_shap = np.mean(explanation.shap_values, axis=0)
        std_shap = np.std(explanation.shap_values, axis=0)
        max_abs_shap = np.max(np.abs(explanation.shap_values), axis=0)
        
        # Create ranking data
        feature_data = []
        for i, feature_name in enumerate(explanation.feature_names):
            feature_data.append({
                'feature_name': feature_name,
                'mean_abs_importance': float(mean_abs_shap[i]),
                'mean_importance': float(mean_shap[i]),
                'std_importance': float(std_shap[i]),
                'max_abs_importance': float(max_abs_shap[i]),
                'rank_by_mean_abs': 0,  # Will be filled after sorting
                'rank_by_mean': 0,
                'rank_by_std': 0,
                'rank_by_max_abs': 0
            })
        
        # Sort and assign ranks
        for rank_type in ['mean_abs', 'mean', 'std', 'max_abs']:
            sorted_features = sorted(feature_data, 
                                   key=lambda x: abs(x[f'{rank_type}_importance']), 
                                   reverse=True)
            for rank, feature in enumerate(sorted_features, 1):
                feature[f'rank_by_{rank_type}'] = rank
        
        return {
            'method': 'SHAP',
            'num_features': len(explanation.feature_names),
            'num_samples': explanation.shap_values.shape[0],
            'features': feature_data
        }
    
    def _calculate_lime_rankings(self, explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Calculate LIME-based feature rankings.
        
        Args:
            explanations: List of LIME explanations
            
        Returns:
            Dictionary with LIME rankings
        """
        # Collect all features and their importance values
        feature_importances = {}
        for exp in explanations:
            for feature, importance in exp.feature_importance.items():
                if feature not in feature_importances:
                    feature_importances[feature] = []
                feature_importances[feature].append(importance)
        
        # Calculate statistics for each feature
        feature_data = []
        for feature_name, importances in feature_importances.items():
            importances = np.array(importances)
            feature_data.append({
                'feature_name': feature_name,
                'mean_abs_importance': float(np.mean(np.abs(importances))),
                'mean_importance': float(np.mean(importances)),
                'std_importance': float(np.std(importances)),
                'max_abs_importance': float(np.max(np.abs(importances))),
                'frequency': len(importances),
                'rank_by_mean_abs': 0,
                'rank_by_frequency': 0
            })
        
        # Sort and assign ranks
        sorted_by_importance = sorted(feature_data, 
                                    key=lambda x: x['mean_abs_importance'], 
                                    reverse=True)
        for rank, feature in enumerate(sorted_by_importance, 1):
            feature['rank_by_mean_abs'] = rank
        
        sorted_by_frequency = sorted(feature_data, 
                                   key=lambda x: x['frequency'], 
                                   reverse=True)
        for rank, feature in enumerate(sorted_by_frequency, 1):
            feature['rank_by_frequency'] = rank
        
        return {
            'method': 'LIME',
            'num_features': len(feature_importances),
            'num_explanations': len(explanations),
            'features': feature_data
        }
    
    def _calculate_combined_rankings(self, shap_rankings: Dict[str, Any], 
                                   lime_rankings: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined rankings from SHAP and LIME.
        
        Args:
            shap_rankings: SHAP ranking data
            lime_rankings: LIME ranking data
            
        Returns:
            Dictionary with combined rankings
        """
        # Create feature mapping
        shap_features = {f['feature_name']: f for f in shap_rankings['features']}
        lime_features = {f['feature_name']: f for f in lime_rankings['features']}
        
        # Find common features
        common_features = set(shap_features.keys()) & set(lime_features.keys())
        
        combined_data = []
        for feature_name in common_features:
            shap_data = shap_features[feature_name]
            lime_data = lime_features[feature_name]
            
            # Calculate combined scores (simple average of normalized ranks)
            shap_rank_norm = 1.0 - (shap_data['rank_by_mean_abs'] - 1) / len(shap_features)
            lime_rank_norm = 1.0 - (lime_data['rank_by_mean_abs'] - 1) / len(lime_features)
            combined_score = (shap_rank_norm + lime_rank_norm) / 2
            
            combined_data.append({
                'feature_name': feature_name,
                'shap_importance': shap_data['mean_abs_importance'],
                'lime_importance': lime_data['mean_abs_importance'],
                'shap_rank': shap_data['rank_by_mean_abs'],
                'lime_rank': lime_data['rank_by_mean_abs'],
                'combined_score': float(combined_score),
                'combined_rank': 0  # Will be filled after sorting
            })
        
        # Sort by combined score and assign ranks
        combined_data.sort(key=lambda x: x['combined_score'], reverse=True)
        for rank, feature in enumerate(combined_data, 1):
            feature['combined_rank'] = rank
        
        return {
            'method': 'Combined SHAP+LIME',
            'num_common_features': len(common_features),
            'features': combined_data
        }
    
    def _export_shap_csv(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export SHAP data as CSV files."""
        # Export SHAP values
        shap_df = pd.DataFrame(data['shap_values'], columns=data['feature_names'])
        shap_path = output_path / f"{filename}_values.csv"
        shap_df.to_csv(shap_path, index=False)
        
        # Export data values if available
        if data['data_values'] is not None:
            data_df = pd.DataFrame(data['data_values'], columns=data['feature_names'])
            data_path = output_path / f"{filename}_data.csv"
            data_df.to_csv(data_path, index=False)
        
        # Export metadata
        metadata_path = output_path / f"{filename}_metadata.csv"
        metadata_df = pd.DataFrame([data['metadata']])
        metadata_df.to_csv(metadata_path, index=False)
        
        return shap_path
    
    def _export_shap_json(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export SHAP data as JSON."""
        json_path = output_path / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return json_path
    
    def _export_shap_parquet(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export SHAP data as Parquet."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Create combined dataframe
            shap_df = pd.DataFrame(data['shap_values'], columns=data['feature_names'])
            
            # Add metadata as columns
            for key, value in data['metadata'].items():
                if key not in ['feature_names']:
                    shap_df[f'_meta_{key}'] = value
            
            parquet_path = output_path / f"{filename}.parquet"
            shap_df.to_parquet(parquet_path, index=False)
            return parquet_path
            
        except ImportError:
            logger.warning("PyArrow not available, skipping Parquet export")
            raise RuntimeError("PyArrow required for Parquet export")
    
    def _export_lime_csv(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export LIME data as CSV."""
        # Create main explanations dataframe
        rows = []
        for exp in data['explanations']:
            base_row = {
                'sample_id': exp['sample_id'],
                'prediction': exp['prediction'],
                'confidence_lower': exp['confidence_interval'][0],
                'confidence_upper': exp['confidence_interval'][1],
                'local_prediction': exp['local_prediction'],
                'intercept': exp['intercept']
            }
            
            # Add feature importance columns
            for feature, importance in exp['feature_importance'].items():
                base_row[f'importance_{feature}'] = importance
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        csv_path = output_path / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def _export_lime_json(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export LIME data as JSON."""
        json_path = output_path / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return json_path
    
    def _export_lime_parquet(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export LIME data as Parquet."""
        try:
            # Convert to DataFrame format similar to CSV
            rows = []
            for exp in data['explanations']:
                base_row = {
                    'sample_id': exp['sample_id'],
                    'prediction': exp['prediction'],
                    'confidence_lower': exp['confidence_interval'][0],
                    'confidence_upper': exp['confidence_interval'][1],
                    'local_prediction': exp['local_prediction'],
                    'intercept': exp['intercept']
                }
                
                for feature, importance in exp['feature_importance'].items():
                    base_row[f'importance_{feature}'] = importance
                
                rows.append(base_row)
            
            df = pd.DataFrame(rows)
            parquet_path = output_path / f"{filename}.parquet"
            df.to_parquet(parquet_path, index=False)
            return parquet_path
            
        except ImportError:
            logger.warning("PyArrow not available, skipping Parquet export")
            raise RuntimeError("PyArrow required for Parquet export")
    
    def _export_ranking_csv(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export feature rankings as CSV."""
        csv_path = output_path / f"{filename}.csv"
        
        # Create separate CSV files for each ranking method
        for method, ranking_data in data['rankings'].items():
            method_path = output_path / f"{filename}_{method}.csv"
            df = pd.DataFrame(ranking_data['features'])
            df.to_csv(method_path, index=False)
        
        return csv_path
    
    def _export_ranking_json(self, data: Dict[str, Any], output_path: Path, filename: str) -> Path:
        """Export feature rankings as JSON."""
        json_path = output_path / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return json_path