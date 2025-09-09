"""
Explanation comparison utilities for comparing SHAP and LIME results.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr
import logging

from .base import SHAPExplanation, LIMEExplanation, ComparisonResult

logger = logging.getLogger(__name__)


class ExplanationComparator:
    """Compare SHAP and LIME explanation results."""
    
    def __init__(self, agreement_threshold: float = 0.7, 
                 correlation_threshold: float = 0.5):
        """Initialize explanation comparator.
        
        Args:
            agreement_threshold: Threshold for determining feature agreement (0-1)
            correlation_threshold: Minimum correlation for consistency (0-1)
        """
        self.agreement_threshold = agreement_threshold
        self.correlation_threshold = correlation_threshold
        
    def compare_explanations(self, shap_explanation: SHAPExplanation,
                           lime_explanations: List[LIMEExplanation]) -> ComparisonResult:
        """Compare SHAP and LIME explanations for consistency analysis.
        
        Args:
            shap_explanation: Global or local SHAP explanation
            lime_explanations: List of LIME explanations (local explanations)
            
        Returns:
            ComparisonResult with detailed comparison analysis
        """
        if not lime_explanations:
            raise ValueError("LIME explanations list cannot be empty")
            
        logger.info(f"Comparing SHAP explanation with {len(lime_explanations)} LIME explanations")
        
        # Extract feature importance from SHAP
        shap_importance = self._extract_shap_importance(shap_explanation)
        
        # Aggregate LIME feature importance
        lime_importance = self._aggregate_lime_importance(lime_explanations)
        
        # Ensure both have the same features
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        if not common_features:
            raise ValueError("No common features found between SHAP and LIME explanations")
            
        # Filter to common features
        shap_filtered = {f: shap_importance[f] for f in common_features}
        lime_filtered = {f: lime_importance[f] for f in common_features}
        
        # Calculate correlation
        correlation_score = self._calculate_correlation(shap_filtered, lime_filtered)
        
        # Identify agreement and disagreement
        agreement_features, disagreement_features = self._identify_agreement(
            shap_filtered, lime_filtered
        )
        
        # Calculate consistency score
        consistency_score = len(agreement_features) / len(common_features)
        
        # Generate insights
        insights = self._generate_comparison_insights(
            shap_filtered, lime_filtered, correlation_score, 
            agreement_features, disagreement_features
        )
        
        return ComparisonResult(
            shap_importance=shap_filtered,
            lime_importance=lime_filtered,
            correlation_score=correlation_score,
            agreement_features=agreement_features,
            disagreement_features=disagreement_features,
            insights=insights,
            consistency_score=consistency_score
        )
    
    def compare_feature_rankings(self, shap_explanation: SHAPExplanation,
                               lime_explanations: List[LIMEExplanation],
                               top_k: int = 10) -> Dict[str, Any]:
        """Compare top-k feature rankings between SHAP and LIME.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            top_k: Number of top features to compare
            
        Returns:
            Dictionary with ranking comparison results
        """
        # Extract importance scores
        shap_importance = self._extract_shap_importance(shap_explanation)
        lime_importance = self._aggregate_lime_importance(lime_explanations)
        
        # Get top-k features for each method
        shap_top_k = self._get_top_k_features(shap_importance, top_k)
        lime_top_k = self._get_top_k_features(lime_importance, top_k)
        
        # Calculate ranking metrics
        overlap = len(set(shap_top_k) & set(lime_top_k))
        overlap_ratio = overlap / top_k
        
        # Calculate rank correlation for overlapping features
        overlapping_features = set(shap_top_k) & set(lime_top_k)
        if overlapping_features:
            shap_ranks = {f: i for i, f in enumerate(shap_top_k)}
            lime_ranks = {f: i for i, f in enumerate(lime_top_k)}
            
            shap_rank_values = [shap_ranks[f] for f in overlapping_features]
            lime_rank_values = [lime_ranks[f] for f in overlapping_features]
            
            rank_correlation, _ = spearmanr(shap_rank_values, lime_rank_values)
        else:
            rank_correlation = 0.0
        
        return {
            'shap_top_k': shap_top_k,
            'lime_top_k': lime_top_k,
            'overlap_count': overlap,
            'overlap_ratio': overlap_ratio,
            'rank_correlation': rank_correlation,
            'unique_to_shap': list(set(shap_top_k) - set(lime_top_k)),
            'unique_to_lime': list(set(lime_top_k) - set(shap_top_k))
        }
    
    def analyze_directional_agreement(self, shap_explanation: SHAPExplanation,
                                    lime_explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Analyze whether SHAP and LIME agree on feature effect directions.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            
        Returns:
            Dictionary with directional agreement analysis
        """
        shap_importance = self._extract_shap_importance(shap_explanation, method='mean')
        lime_importance = self._aggregate_lime_importance(lime_explanations)
        
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        
        directional_agreement = {}
        agreement_count = 0
        
        for feature in common_features:
            shap_sign = np.sign(shap_importance[feature])
            lime_sign = np.sign(lime_importance[feature])
            
            agrees = shap_sign == lime_sign
            directional_agreement[feature] = {
                'shap_direction': 'positive' if shap_sign > 0 else 'negative' if shap_sign < 0 else 'neutral',
                'lime_direction': 'positive' if lime_sign > 0 else 'negative' if lime_sign < 0 else 'neutral',
                'agrees': agrees
            }
            
            if agrees:
                agreement_count += 1
        
        agreement_ratio = agreement_count / len(common_features) if common_features else 0
        
        return {
            'directional_agreement': directional_agreement,
            'agreement_ratio': agreement_ratio,
            'total_features': len(common_features),
            'agreeing_features': agreement_count
        }
    
    def _extract_shap_importance(self, shap_explanation: SHAPExplanation, 
                               method: str = 'mean_abs') -> Dict[str, float]:
        """Extract feature importance from SHAP explanation.
        
        Args:
            shap_explanation: SHAP explanation object
            method: Method to compute importance ('mean_abs', 'mean', 'std')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        shap_values = shap_explanation.shap_values
        feature_names = shap_explanation.feature_names
        
        if method == 'mean_abs':
            importance_scores = np.mean(np.abs(shap_values), axis=0)
        elif method == 'mean':
            importance_scores = np.mean(shap_values, axis=0)
        elif method == 'std':
            importance_scores = np.std(shap_values, axis=0)
        else:
            raise ValueError(f"Unsupported importance method: {method}")
        
        return dict(zip(feature_names, importance_scores))
    
    def _aggregate_lime_importance(self, lime_explanations: List[LIMEExplanation]) -> Dict[str, float]:
        """Aggregate feature importance across multiple LIME explanations.
        
        Args:
            lime_explanations: List of LIME explanation objects
            
        Returns:
            Dictionary with aggregated feature importance
        """
        if not lime_explanations:
            return {}
        
        # Collect all feature importance values
        feature_values = {}
        for explanation in lime_explanations:
            for feature, importance in explanation.feature_importance.items():
                if feature not in feature_values:
                    feature_values[feature] = []
                feature_values[feature].append(importance)
        
        # Calculate mean importance for each feature
        aggregated_importance = {}
        for feature, values in feature_values.items():
            aggregated_importance[feature] = np.mean(values)
        
        return aggregated_importance
    
    def _calculate_correlation(self, shap_importance: Dict[str, float],
                             lime_importance: Dict[str, float]) -> float:
        """Calculate correlation between SHAP and LIME importance scores.
        
        Args:
            shap_importance: SHAP feature importance dictionary
            lime_importance: LIME feature importance dictionary
            
        Returns:
            Pearson correlation coefficient
        """
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        
        if len(common_features) < 2:
            return 0.0
        
        shap_values = [shap_importance[f] for f in common_features]
        lime_values = [lime_importance[f] for f in common_features]
        
        try:
            correlation, _ = pearsonr(shap_values, lime_values)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    def _identify_agreement(self, shap_importance: Dict[str, float],
                          lime_importance: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identify features with agreement and disagreement between methods.
        
        Args:
            shap_importance: SHAP feature importance dictionary
            lime_importance: LIME feature importance dictionary
            
        Returns:
            Tuple of (agreement_features, disagreement_features)
        """
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        
        # Normalize importance scores to [0, 1] for comparison
        shap_normalized = self._normalize_importance(shap_importance)
        lime_normalized = self._normalize_importance(lime_importance)
        
        agreement_features = []
        disagreement_features = []
        
        for feature in common_features:
            shap_score = shap_normalized[feature]
            lime_score = lime_normalized[feature]
            
            # Calculate relative difference
            max_score = max(shap_score, lime_score)
            if max_score > 0:
                relative_diff = abs(shap_score - lime_score) / max_score
                if relative_diff <= (1 - self.agreement_threshold):
                    agreement_features.append(feature)
                else:
                    disagreement_features.append(feature)
            else:
                # Both scores are zero - consider as agreement
                agreement_features.append(feature)
        
        return agreement_features, disagreement_features
    
    def _normalize_importance(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize importance scores to [0, 1] range.
        
        Args:
            importance_dict: Dictionary of feature importance scores
            
        Returns:
            Dictionary with normalized scores
        """
        values = list(importance_dict.values())
        if not values:
            return {}
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in importance_dict.keys()}
        
        return {
            k: (v - min_val) / (max_val - min_val) 
            for k, v in importance_dict.items()
        }
    
    def _get_top_k_features(self, importance_dict: Dict[str, float], k: int) -> List[str]:
        """Get top-k features by importance score.
        
        Args:
            importance_dict: Dictionary of feature importance scores
            k: Number of top features to return
            
        Returns:
            List of top-k feature names
        """
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        return [feature for feature, _ in sorted_features[:k]]
    
    def _generate_comparison_insights(self, shap_importance: Dict[str, float],
                                    lime_importance: Dict[str, float],
                                    correlation_score: float,
                                    agreement_features: List[str],
                                    disagreement_features: List[str]) -> List[str]:
        """Generate insights from explanation comparison.
        
        Args:
            shap_importance: SHAP feature importance dictionary
            lime_importance: LIME feature importance dictionary
            correlation_score: Correlation between methods
            agreement_features: Features with agreement
            disagreement_features: Features with disagreement
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Overall correlation insight
        if correlation_score >= self.correlation_threshold:
            insights.append(f"Strong correlation ({correlation_score:.3f}) between SHAP and LIME explanations indicates consistent feature importance rankings.")
        elif correlation_score >= 0.3:
            insights.append(f"Moderate correlation ({correlation_score:.3f}) between SHAP and LIME explanations suggests some consistency in feature importance.")
        else:
            insights.append(f"Low correlation ({correlation_score:.3f}) between SHAP and LIME explanations indicates significant disagreement in feature importance.")
        
        # Agreement insights
        total_features = len(agreement_features) + len(disagreement_features)
        if total_features > 0:
            agreement_ratio = len(agreement_features) / total_features
            if agreement_ratio >= 0.8:
                insights.append(f"High agreement ({agreement_ratio:.1%}) on feature importance suggests robust explanations.")
            elif agreement_ratio >= 0.6:
                insights.append(f"Moderate agreement ({agreement_ratio:.1%}) on feature importance indicates reasonable consistency.")
            else:
                insights.append(f"Low agreement ({agreement_ratio:.1%}) on feature importance suggests method-specific biases.")
        
        # Top disagreement features
        if disagreement_features:
            # Find features with largest disagreement
            disagreement_scores = {}
            for feature in disagreement_features:
                if feature in shap_importance and feature in lime_importance:
                    disagreement_scores[feature] = abs(shap_importance[feature] - lime_importance[feature])
            
            if disagreement_scores:
                top_disagreement = max(disagreement_scores.items(), key=lambda x: x[1])
                insights.append(f"Largest disagreement found for feature '{top_disagreement[0]}' - investigate model behavior for this feature.")
        
        # Method-specific insights
        shap_top_feature = max(shap_importance.items(), key=lambda x: abs(x[1]))[0]
        lime_top_feature = max(lime_importance.items(), key=lambda x: abs(x[1]))[0]
        
        if shap_top_feature == lime_top_feature:
            insights.append(f"Both methods identify '{shap_top_feature}' as the most important feature.")
        else:
            insights.append(f"Methods disagree on top feature: SHAP identifies '{shap_top_feature}', LIME identifies '{lime_top_feature}'.")
        
        return insights