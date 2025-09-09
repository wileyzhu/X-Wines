"""
LIME analysis and comparison utilities.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

from .base import BaseAnalyzer, LIMEExplanation, ComparisonResult

logger = logging.getLogger(__name__)


class LIMEAnalyzer(BaseAnalyzer):
    """LIME analyzer for explanation analysis and comparison utilities."""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """Initialize LIME analyzer.
        
        Args:
            feature_names: List of feature names (optional)
        """
        self.feature_names = feature_names
    
    def analyze_feature_importance(self, explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Analyze feature importance patterns across multiple LIME explanations.
        
        Args:
            explanations: List of LIME explanation objects
            
        Returns:
            Analysis results dictionary containing importance statistics
        """
        try:
            if not explanations:
                raise ValueError("No explanations provided for analysis")
            
            # Collect all feature importance values
            feature_importance_matrix = defaultdict(list)
            all_features = set()
            
            for explanation in explanations:
                all_features.update(explanation.feature_importance.keys())
                for feature, importance in explanation.feature_importance.items():
                    feature_importance_matrix[feature].append(importance)
            
            # Ensure all features have values for all explanations (fill with 0 if missing)
            for feature in all_features:
                while len(feature_importance_matrix[feature]) < len(explanations):
                    feature_importance_matrix[feature].append(0.0)
            
            # Calculate statistics for each feature
            feature_stats = {}
            for feature, importances in feature_importance_matrix.items():
                importances = np.array(importances)
                feature_stats[feature] = {
                    'mean': np.mean(importances),
                    'std': np.std(importances),
                    'mean_abs': np.mean(np.abs(importances)),
                    'min': np.min(importances),
                    'max': np.max(importances),
                    'median': np.median(importances),
                    'consistency': self._calculate_consistency(importances)
                }
            
            # Rank features by different criteria
            feature_rankings = {
                'mean_abs': self._rank_features_by_stat(feature_stats, 'mean_abs'),
                'mean': self._rank_features_by_stat(feature_stats, 'mean'),
                'std': self._rank_features_by_stat(feature_stats, 'std'),
                'consistency': self._rank_features_by_stat(feature_stats, 'consistency')
            }
            
            # Calculate prediction statistics
            predictions = [exp.prediction for exp in explanations]
            prediction_stats = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'range': np.max(predictions) - np.min(predictions)
            }
            
            # Calculate confidence interval statistics
            ci_widths = [exp.confidence_interval[1] - exp.confidence_interval[0] for exp in explanations]
            confidence_stats = {
                'mean_width': np.mean(ci_widths),
                'std_width': np.std(ci_widths),
                'min_width': np.min(ci_widths),
                'max_width': np.max(ci_widths)
            }
            
            analysis_results = {
                'n_explanations': len(explanations),
                'n_features': len(all_features),
                'feature_stats': feature_stats,
                'feature_rankings': feature_rankings,
                'prediction_stats': prediction_stats,
                'confidence_stats': confidence_stats,
                'summary': {
                    'most_important_feature': feature_rankings['mean_abs'][0],
                    'most_consistent_feature': feature_rankings['consistency'][0],
                    'total_features_analyzed': len(all_features),
                    'average_confidence_width': confidence_stats['mean_width']
                }
            }
            
            logger.info(f"Analyzed feature importance across {len(explanations)} LIME explanations")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze LIME feature importance: {str(e)}")
            raise RuntimeError(f"LIME feature importance analysis failed: {str(e)}")
    
    def detect_interactions(self, explanation: LIMEExplanation) -> Dict[str, Any]:
        """Detect potential feature interactions from LIME explanation.
        
        Note: LIME doesn't directly provide interaction values, so this method
        provides basic interaction detection based on feature importance patterns.
        
        Args:
            explanation: LIME explanation object
            
        Returns:
            Dictionary with interaction analysis results
        """
        try:
            feature_importance = explanation.feature_importance
            
            # Sort features by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Identify potential interactions based on importance patterns
            top_features = [feat for feat, _ in sorted_features[:5]]
            potential_interactions = []
            
            # Generate all pairs of top features
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    imp1 = abs(feature_importance[feat1])
                    imp2 = abs(feature_importance[feat2])
                    
                    # Simple heuristic: features with similar importance might interact
                    similarity = 1.0 - abs(imp1 - imp2) / (imp1 + imp2 + 1e-8)
                    
                    potential_interactions.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'importance_1': feature_importance[feat1],
                        'importance_2': feature_importance[feat2],
                        'similarity_score': similarity,
                        'combined_importance': imp1 + imp2
                    })
            
            # Sort by similarity score
            potential_interactions.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            interaction_results = {
                'potential_interactions': potential_interactions[:10],  # Top 10
                'top_features': top_features,
                'interaction_method': 'importance_similarity',
                'note': 'LIME interactions are estimated based on feature importance patterns'
            }
            
            logger.info(f"Detected {len(potential_interactions)} potential interactions")
            return interaction_results
            
        except Exception as e:
            logger.error(f"Failed to detect LIME interactions: {str(e)}")
            raise RuntimeError(f"LIME interaction detection failed: {str(e)}")
    
    def generate_insights(self, explanations: List[LIMEExplanation]) -> List[str]:
        """Generate actionable insights from LIME explanations.
        
        Args:
            explanations: List of LIME explanation objects
            
        Returns:
            List of insight strings
        """
        try:
            insights = []
            
            if not explanations:
                return ["No explanations available for insight generation"]
            
            # Analyze feature importance
            importance_analysis = self.analyze_feature_importance(explanations)
            
            # Most important feature insight
            top_feature = importance_analysis['summary']['most_important_feature']
            top_importance = importance_analysis['feature_stats'][top_feature]['mean_abs']
            insights.append(f"Most important feature: {top_feature} (avg importance: {top_importance:.3f})")
            
            # Consistency insights
            most_consistent = importance_analysis['summary']['most_consistent_feature']
            consistency_score = importance_analysis['feature_stats'][most_consistent]['consistency']
            insights.append(f"Most consistent feature: {most_consistent} (consistency: {consistency_score:.3f})")
            
            # Prediction variability
            pred_stats = importance_analysis['prediction_stats']
            if pred_stats['std'] > 0.5:
                insights.append(f"High prediction variability detected (std: {pred_stats['std']:.3f})")
            else:
                insights.append(f"Low prediction variability (std: {pred_stats['std']:.3f})")
            
            # Confidence interval insights
            conf_stats = importance_analysis['confidence_stats']
            avg_width = conf_stats['mean_width']
            if avg_width > 1.0:
                insights.append(f"Wide confidence intervals suggest high uncertainty (avg width: {avg_width:.3f})")
            else:
                insights.append(f"Narrow confidence intervals suggest good certainty (avg width: {avg_width:.3f})")
            
            # Feature direction insights
            positive_features = []
            negative_features = []
            
            for feature, stats in importance_analysis['feature_stats'].items():
                if stats['mean'] > 0.1:
                    positive_features.append((feature, stats['mean']))
                elif stats['mean'] < -0.1:
                    negative_features.append((feature, stats['mean']))
            
            if positive_features:
                top_positive = max(positive_features, key=lambda x: x[1])
                insights.append(f"Strongest positive influence: {top_positive[0]} ({top_positive[1]:.3f})")
            
            if negative_features:
                top_negative = min(negative_features, key=lambda x: x[1])
                insights.append(f"Strongest negative influence: {top_negative[0]} ({top_negative[1]:.3f})")
            
            # Feature stability insights
            unstable_features = []
            for feature, stats in importance_analysis['feature_stats'].items():
                if stats['consistency'] < 0.5 and stats['mean_abs'] > 0.1:
                    unstable_features.append((feature, stats['consistency']))
            
            if unstable_features:
                most_unstable = min(unstable_features, key=lambda x: x[1])
                insights.append(f"Most unstable important feature: {most_unstable[0]} (consistency: {most_unstable[1]:.3f})")
            
            # Overall model behavior
            n_significant_features = sum(1 for stats in importance_analysis['feature_stats'].values() 
                                       if stats['mean_abs'] > 0.1)
            insights.append(f"Number of significantly important features: {n_significant_features}")
            
            # Prediction range insight
            pred_range = pred_stats['range']
            insights.append(f"Prediction range across samples: {pred_range:.3f}")
            
            logger.info(f"Generated {len(insights)} insights from LIME analysis")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate LIME insights: {str(e)}")
            raise RuntimeError(f"LIME insight generation failed: {str(e)}")
    
    def compare_explanations(self, explanations1: List[LIMEExplanation], 
                           explanations2: List[LIMEExplanation],
                           method: str = 'correlation') -> ComparisonResult:
        """Compare two sets of LIME explanations.
        
        Args:
            explanations1: First set of LIME explanations
            explanations2: Second set of LIME explanations
            method: Comparison method ('correlation', 'ranking', 'agreement')
            
        Returns:
            ComparisonResult object with comparison metrics
        """
        try:
            if len(explanations1) != len(explanations2):
                raise ValueError("Explanation sets must have the same length")
            
            # Get feature importance for both sets
            analysis1 = self.analyze_feature_importance(explanations1)
            analysis2 = self.analyze_feature_importance(explanations2)
            
            # Extract mean importance for comparison
            importance1 = {feat: stats['mean'] for feat, stats in analysis1['feature_stats'].items()}
            importance2 = {feat: stats['mean'] for feat, stats in analysis2['feature_stats'].items()}
            
            # Find common features
            common_features = set(importance1.keys()) & set(importance2.keys())
            
            if not common_features:
                raise ValueError("No common features found between explanation sets")
            
            # Calculate correlation
            values1 = [importance1[feat] for feat in common_features]
            values2 = [importance2[feat] for feat in common_features]
            
            correlation_score, _ = pearsonr(values1, values2)
            
            # Identify agreement and disagreement
            agreement_features = []
            disagreement_features = []
            
            for feat in common_features:
                imp1, imp2 = importance1[feat], importance2[feat]
                
                # Check if both have same sign and similar magnitude
                if (imp1 * imp2 > 0) and (abs(imp1 - imp2) < 0.5 * (abs(imp1) + abs(imp2))):
                    agreement_features.append(feat)
                else:
                    disagreement_features.append(feat)
            
            # Generate comparison insights
            insights = []
            insights.append(f"Correlation between explanation sets: {correlation_score:.3f}")
            insights.append(f"Features in agreement: {len(agreement_features)}/{len(common_features)}")
            insights.append(f"Features in disagreement: {len(disagreement_features)}/{len(common_features)}")
            
            if correlation_score > 0.7:
                insights.append("High agreement between explanation methods")
            elif correlation_score > 0.3:
                insights.append("Moderate agreement between explanation methods")
            else:
                insights.append("Low agreement between explanation methods")
            
            # Calculate consistency score
            consistency_score = len(agreement_features) / len(common_features) if common_features else 0.0
            
            comparison_result = ComparisonResult(
                shap_importance=importance1,  # Using as first set
                lime_importance=importance2,  # Using as second set
                correlation_score=correlation_score,
                agreement_features=agreement_features,
                disagreement_features=disagreement_features,
                insights=insights,
                consistency_score=consistency_score
            )
            
            logger.info(f"Compared {len(explanations1)} explanation pairs")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Failed to compare LIME explanations: {str(e)}")
            raise RuntimeError(f"LIME explanation comparison failed: {str(e)}")
    
    def _calculate_consistency(self, values: np.ndarray) -> float:
        """Calculate consistency score for a feature across explanations.
        
        Args:
            values: Array of importance values for a feature
            
        Returns:
            Consistency score between 0 and 1
        """
        if len(values) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (inverted for consistency)
        mean_val = np.mean(np.abs(values))
        if mean_val == 0:
            return 1.0
        
        cv = np.std(values) / mean_val
        consistency = 1.0 / (1.0 + cv)  # Higher consistency for lower variation
        
        return max(0.0, min(1.0, consistency))
    
    def _rank_features_by_stat(self, feature_stats: Dict[str, Dict[str, float]], 
                              stat_name: str) -> List[str]:
        """Rank features by a specific statistic.
        
        Args:
            feature_stats: Dictionary of feature statistics
            stat_name: Name of statistic to rank by
            
        Returns:
            List of feature names ranked by the statistic
        """
        feature_scores = [(feat, stats[stat_name]) for feat, stats in feature_stats.items()]
        feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return [feat for feat, _ in feature_scores]
    
    def calculate_explanation_stability(self, explanations: List[LIMEExplanation]) -> Dict[str, float]:
        """Calculate stability metrics for LIME explanations.
        
        Args:
            explanations: List of LIME explanations
            
        Returns:
            Dictionary with stability metrics
        """
        try:
            if len(explanations) < 2:
                return {'overall_stability': 1.0}
            
            # Calculate pairwise correlations
            correlations = []
            
            for i in range(len(explanations)):
                for j in range(i + 1, len(explanations)):
                    exp1, exp2 = explanations[i], explanations[j]
                    
                    # Get common features
                    common_features = set(exp1.feature_importance.keys()) & set(exp2.feature_importance.keys())
                    
                    if len(common_features) > 1:
                        values1 = [exp1.feature_importance[feat] for feat in common_features]
                        values2 = [exp2.feature_importance[feat] for feat in common_features]
                        
                        corr, _ = pearsonr(values1, values2)
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            # Calculate stability metrics
            stability_metrics = {
                'overall_stability': np.mean(correlations) if correlations else 0.0,
                'stability_std': np.std(correlations) if correlations else 0.0,
                'min_stability': np.min(correlations) if correlations else 0.0,
                'max_stability': np.max(correlations) if correlations else 0.0,
                'n_comparisons': len(correlations)
            }
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate explanation stability: {str(e)}")
            return {'overall_stability': 0.0, 'error': str(e)}