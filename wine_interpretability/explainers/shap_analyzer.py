"""
SHAP analysis and interaction detection components.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from scipy.stats import pearsonr, spearmanr
try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    # Fallback if not available
    mutual_info_regression = None

from .base import BaseAnalyzer, SHAPExplanation, InteractionAnalysis

logger = logging.getLogger(__name__)


class SHAPAnalyzer(BaseAnalyzer):
    """SHAP analyzer for feature interaction analysis and insight generation."""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """Initialize SHAP analyzer.
        
        Args:
            feature_names: List of feature names (optional)
        """
        self.feature_names = feature_names
    
    def analyze_feature_importance(self, explanations: List[SHAPExplanation]) -> Dict[str, Any]:
        """Analyze feature importance patterns across multiple SHAP explanations.
        
        Args:
            explanations: List of SHAP explanation objects
            
        Returns:
            Analysis results dictionary containing importance statistics
        """
        try:
            if not explanations:
                raise ValueError("No explanations provided for analysis")
            
            # Combine all SHAP values
            all_shap_values = []
            all_feature_names = explanations[0].feature_names
            
            for explanation in explanations:
                if explanation.feature_names != all_feature_names:
                    logger.warning("Feature names inconsistent across explanations")
                all_shap_values.append(explanation.shap_values)
            
            # Stack all SHAP values
            combined_shap = np.vstack(all_shap_values)
            
            # Calculate various importance metrics
            mean_abs_importance = np.mean(np.abs(combined_shap), axis=0)
            mean_importance = np.mean(combined_shap, axis=0)
            std_importance = np.std(combined_shap, axis=0)
            max_importance = np.max(np.abs(combined_shap), axis=0)
            
            # Calculate stability (consistency across explanations)
            stability_scores = []
            for feat_idx in range(combined_shap.shape[1]):
                feat_importances = [np.mean(np.abs(exp.shap_values[:, feat_idx])) 
                                 for exp in explanations]
                stability = 1.0 - (np.std(feat_importances) / (np.mean(feat_importances) + 1e-8))
                stability_scores.append(max(0.0, stability))
            
            # Rank features by different criteria
            feature_rankings = {
                'mean_abs': self._rank_features(all_feature_names, mean_abs_importance),
                'mean': self._rank_features(all_feature_names, mean_importance),
                'std': self._rank_features(all_feature_names, std_importance),
                'max': self._rank_features(all_feature_names, max_importance),
                'stability': self._rank_features(all_feature_names, stability_scores)
            }
            
            # Identify consistently important features
            top_features_sets = [set(ranking[:5]) for ranking in feature_rankings.values()]
            consistent_features = list(set.intersection(*top_features_sets))
            
            analysis_results = {
                'n_explanations': len(explanations),
                'n_samples': combined_shap.shape[0],
                'n_features': combined_shap.shape[1],
                'importance_metrics': {
                    'mean_abs': dict(zip(all_feature_names, mean_abs_importance)),
                    'mean': dict(zip(all_feature_names, mean_importance)),
                    'std': dict(zip(all_feature_names, std_importance)),
                    'max': dict(zip(all_feature_names, max_importance)),
                    'stability': dict(zip(all_feature_names, stability_scores))
                },
                'feature_rankings': feature_rankings,
                'consistent_top_features': consistent_features,
                'summary_stats': {
                    'total_importance': np.sum(mean_abs_importance),
                    'mean_stability': np.mean(stability_scores),
                    'importance_concentration': self._calculate_concentration(mean_abs_importance)
                }
            }
            
            logger.info(f"Analyzed feature importance across {len(explanations)} explanations")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze feature importance: {str(e)}")
            raise RuntimeError(f"Feature importance analysis failed: {str(e)}")
    
    def detect_interactions(self, explanation: SHAPExplanation, 
                          interaction_values: Optional[np.ndarray] = None) -> InteractionAnalysis:
        """Detect and analyze feature interactions from SHAP values.
        
        Args:
            explanation: SHAP explanation object
            interaction_values: Pre-computed SHAP interaction values (optional)
            
        Returns:
            InteractionAnalysis object with interaction results
        """
        try:
            feature_names = explanation.feature_names
            n_features = len(feature_names)
            
            if interaction_values is not None:
                # Use provided interaction values
                if interaction_values.ndim != 3:
                    raise ValueError("Interaction values must be 3D array")
                shap_interactions = interaction_values
            else:
                # Estimate interactions from SHAP values using correlation
                shap_interactions = self._estimate_interactions_from_shap(explanation.shap_values)
            
            # Find top feature pairs by interaction strength
            feature_pairs = []
            interaction_strengths = {}
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if shap_interactions.ndim == 3:
                        # Use actual interaction values
                        interaction_strength = np.mean(np.abs(shap_interactions[:, i, j]))
                    else:
                        # Use correlation-based estimate
                        interaction_strength = abs(shap_interactions[i, j])
                    
                    pair = (feature_names[i], feature_names[j])
                    feature_pairs.append(pair)
                    interaction_strengths[pair] = interaction_strength
            
            # Sort pairs by interaction strength
            sorted_pairs = sorted(feature_pairs, key=lambda x: interaction_strengths[x], reverse=True)
            
            # Create top interactions list
            top_interactions = []
            for i, (feat1, feat2) in enumerate(sorted_pairs[:20]):  # Top 20 interactions
                interaction_info = {
                    'rank': i + 1,
                    'feature_1': feat1,
                    'feature_2': feat2,
                    'feature_1_idx': feature_names.index(feat1),
                    'feature_2_idx': feature_names.index(feat2),
                    'strength': interaction_strengths[(feat1, feat2)],
                    'interpretation': self._interpret_interaction(
                        feat1, feat2, interaction_strengths[(feat1, feat2)]
                    )
                }
                top_interactions.append(interaction_info)
            
            interaction_analysis = InteractionAnalysis(
                interaction_values=shap_interactions,
                feature_pairs=sorted_pairs,
                top_interactions=top_interactions,
                interaction_strength=interaction_strengths
            )
            
            logger.info(f"Detected {len(feature_pairs)} feature interactions")
            return interaction_analysis
            
        except Exception as e:
            logger.error(f"Failed to detect interactions: {str(e)}")
            raise RuntimeError(f"Interaction detection failed: {str(e)}")
    
    def generate_insights(self, explanations: List[SHAPExplanation], 
                         interaction_analysis: Optional[InteractionAnalysis] = None) -> List[str]:
        """Generate actionable insights from SHAP explanations.
        
        Args:
            explanations: List of SHAP explanation objects
            interaction_analysis: Optional interaction analysis results
            
        Returns:
            List of insight strings
        """
        try:
            insights = []
            
            # Analyze feature importance
            importance_analysis = self.analyze_feature_importance(explanations)
            
            # Generate importance insights
            top_features = importance_analysis['feature_rankings']['mean_abs'][:5]
            insights.append(f"Top 5 most important features: {', '.join(top_features)}")
            
            # Stability insights
            stability_scores = importance_analysis['importance_metrics']['stability']
            most_stable = max(stability_scores.items(), key=lambda x: x[1])
            least_stable = min(stability_scores.items(), key=lambda x: x[1])
            
            insights.append(f"Most stable feature: {most_stable[0]} (stability: {most_stable[1]:.3f})")
            insights.append(f"Least stable feature: {least_stable[0]} (stability: {least_stable[1]:.3f})")
            
            # Concentration insight
            concentration = importance_analysis['summary_stats']['importance_concentration']
            if concentration > 0.8:
                insights.append("Feature importance is highly concentrated in a few features")
            elif concentration < 0.3:
                insights.append("Feature importance is well distributed across many features")
            else:
                insights.append("Feature importance shows moderate concentration")
            
            # Directional insights
            mean_importance = importance_analysis['importance_metrics']['mean']
            positive_features = [feat for feat, imp in mean_importance.items() if imp > 0]
            negative_features = [feat for feat, imp in mean_importance.items() if imp < 0]
            
            if positive_features:
                insights.append(f"Features with positive average impact: {', '.join(positive_features[:3])}")
            if negative_features:
                insights.append(f"Features with negative average impact: {', '.join(negative_features[:3])}")
            
            # Interaction insights
            if interaction_analysis:
                top_interaction = interaction_analysis.top_interactions[0] if interaction_analysis.top_interactions else None
                if top_interaction:
                    insights.append(
                        f"Strongest feature interaction: {top_interaction['feature_1']} × "
                        f"{top_interaction['feature_2']} (strength: {top_interaction['strength']:.3f})"
                    )
                
                # Count significant interactions
                significant_interactions = [
                    inter for inter in interaction_analysis.top_interactions 
                    if inter['strength'] > 0.1  # Threshold for significance
                ]
                insights.append(f"Number of significant interactions: {len(significant_interactions)}")
            
            # Model behavior insights
            combined_shap = np.vstack([exp.shap_values for exp in explanations])
            
            # Check for non-linear patterns
            feature_ranges = {}
            for i, feat_name in enumerate(explanations[0].feature_names):
                feat_shap = combined_shap[:, i]
                feat_range = np.max(feat_shap) - np.min(feat_shap)
                feature_ranges[feat_name] = feat_range
            
            highest_range_feat = max(feature_ranges.items(), key=lambda x: x[1])
            insights.append(
                f"Feature with highest SHAP value range: {highest_range_feat[0]} "
                f"(range: {highest_range_feat[1]:.3f})"
            )
            
            # Outlier detection
            outlier_threshold = 3 * np.std(combined_shap)
            outlier_counts = np.sum(np.abs(combined_shap) > outlier_threshold, axis=0)
            if np.any(outlier_counts > 0):
                outlier_features = [
                    explanations[0].feature_names[i] for i, count in enumerate(outlier_counts) 
                    if count > 0
                ]
                insights.append(f"Features with outlier SHAP values: {', '.join(outlier_features[:3])}")
            
            logger.info(f"Generated {len(insights)} insights from SHAP analysis")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            raise RuntimeError(f"Insight generation failed: {str(e)}")
    
    def identify_important_feature_pairs(self, interaction_analysis: InteractionAnalysis,
                                       top_k: int = 10) -> List[Dict[str, Any]]:
        """Identify the most important feature pairs from interaction analysis.
        
        Args:
            interaction_analysis: Interaction analysis results
            top_k: Number of top pairs to return
            
        Returns:
            List of dictionaries containing pair information
        """
        try:
            top_pairs = interaction_analysis.top_interactions[:top_k]
            
            important_pairs = []
            for pair_info in top_pairs:
                pair_data = {
                    'feature_1': pair_info['feature_1'],
                    'feature_2': pair_info['feature_2'],
                    'interaction_strength': pair_info['strength'],
                    'rank': pair_info['rank'],
                    'interpretation': pair_info['interpretation'],
                    'significance': self._assess_interaction_significance(pair_info['strength'])
                }
                important_pairs.append(pair_data)
            
            logger.info(f"Identified {len(important_pairs)} important feature pairs")
            return important_pairs
            
        except Exception as e:
            logger.error(f"Failed to identify important feature pairs: {str(e)}")
            raise RuntimeError(f"Feature pair identification failed: {str(e)}")
    
    def _rank_features(self, feature_names: List[str], scores: np.ndarray) -> List[str]:
        """Rank features by scores in descending order.
        
        Args:
            feature_names: List of feature names
            scores: Array of scores for each feature
            
        Returns:
            List of feature names ranked by score
        """
        feature_scores = list(zip(feature_names, scores))
        feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return [name for name, _ in feature_scores]
    
    def _calculate_concentration(self, importance_scores: np.ndarray) -> float:
        """Calculate concentration of feature importance using Gini coefficient.
        
        Args:
            importance_scores: Array of importance scores
            
        Returns:
            Concentration score between 0 and 1
        """
        # Sort scores in ascending order
        sorted_scores = np.sort(importance_scores)
        n = len(sorted_scores)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_scores)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_scores))) / (n * cumsum[-1]) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))
    
    def _estimate_interactions_from_shap(self, shap_values: np.ndarray) -> np.ndarray:
        """Estimate feature interactions from SHAP values using correlation.
        
        Args:
            shap_values: SHAP values array
            
        Returns:
            Estimated interaction matrix
        """
        n_features = shap_values.shape[1]
        interaction_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Calculate correlation between SHAP values
                corr, _ = pearsonr(shap_values[:, i], shap_values[:, j])
                interaction_matrix[i, j] = corr
                interaction_matrix[j, i] = corr
        
        return interaction_matrix
    
    def _interpret_interaction(self, feat1: str, feat2: str, strength: float) -> str:
        """Generate interpretation text for feature interaction.
        
        Args:
            feat1: First feature name
            feat2: Second feature name
            strength: Interaction strength
            
        Returns:
            Interpretation string
        """
        if strength > 0.5:
            level = "very strong"
        elif strength > 0.3:
            level = "strong"
        elif strength > 0.1:
            level = "moderate"
        else:
            level = "weak"
        
        return f"{level.title()} interaction between {feat1} and {feat2}"
    
    def _assess_interaction_significance(self, strength: float) -> str:
        """Assess the significance level of an interaction.
        
        Args:
            strength: Interaction strength value
            
        Returns:
            Significance level string
        """
        if strength > 0.5:
            return "highly_significant"
        elif strength > 0.3:
            return "significant"
        elif strength > 0.1:
            return "moderately_significant"
        else:
            return "low_significance"