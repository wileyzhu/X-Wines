"""
Insight generation and analysis utilities for extracting actionable insights from explanations.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter
import logging

from .base import SHAPExplanation, LIMEExplanation, ComparisonResult, InteractionAnalysis

logger = logging.getLogger(__name__)


class InsightGenerator:
    """Generate actionable insights from model explanations."""
    
    def __init__(self, significance_threshold: float = 0.1,
                 consistency_threshold: float = 0.7,
                 interaction_threshold: float = 0.05):
        """Initialize insight generator.
        
        Args:
            significance_threshold: Minimum importance score to consider significant
            consistency_threshold: Minimum consistency score for reliable insights
            interaction_threshold: Minimum interaction strength to report
        """
        self.significance_threshold = significance_threshold
        self.consistency_threshold = consistency_threshold
        self.interaction_threshold = interaction_threshold
        
    def generate_comprehensive_insights(self, 
                                      shap_explanation: SHAPExplanation,
                                      lime_explanations: List[LIMEExplanation],
                                      comparison_result: ComparisonResult,
                                      interaction_analysis: Optional[InteractionAnalysis] = None) -> Dict[str, Any]:
        """Generate comprehensive insights from all explanation types.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            comparison_result: Comparison between SHAP and LIME
            interaction_analysis: Optional feature interaction analysis
            
        Returns:
            Dictionary with comprehensive insights
        """
        logger.info("Generating comprehensive insights from explanations")
        
        insights = {
            'feature_insights': self._generate_feature_insights(shap_explanation, lime_explanations),
            'consistency_insights': self._generate_consistency_insights(comparison_result),
            'model_behavior_insights': self._generate_model_behavior_insights(shap_explanation, lime_explanations),
            'actionable_recommendations': self._generate_actionable_recommendations(
                shap_explanation, lime_explanations, comparison_result
            ),
            'quality_assessment': self._assess_explanation_quality(comparison_result),
            'summary': []
        }
        
        # Add interaction insights if available
        if interaction_analysis:
            insights['interaction_insights'] = self._generate_interaction_insights(interaction_analysis)
        
        # Generate overall summary
        insights['summary'] = self._generate_summary_insights(insights)
        
        return insights
    
    def detect_explanation_inconsistencies(self, 
                                         shap_explanation: SHAPExplanation,
                                         lime_explanations: List[LIMEExplanation],
                                         comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Detect and analyze inconsistencies in explanations.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            comparison_result: Comparison result between methods
            
        Returns:
            Dictionary with inconsistency analysis
        """
        logger.info("Detecting explanation inconsistencies")
        
        inconsistencies = {
            'method_disagreements': self._analyze_method_disagreements(comparison_result),
            'feature_stability': self._analyze_feature_stability(lime_explanations),
            'prediction_variance': self._analyze_prediction_variance(lime_explanations),
            'directional_conflicts': self._detect_directional_conflicts(shap_explanation, lime_explanations),
            'magnitude_discrepancies': self._detect_magnitude_discrepancies(comparison_result),
            'reliability_scores': self._calculate_reliability_scores(comparison_result, lime_explanations)
        }
        
        return inconsistencies
    
    def extract_wine_quality_insights(self, 
                                    shap_explanation: SHAPExplanation,
                                    lime_explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Extract domain-specific insights for wine quality prediction.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            
        Returns:
            Dictionary with wine quality specific insights
        """
        logger.info("Extracting wine quality specific insights")
        
        # Extract feature importance
        shap_importance = self._extract_feature_importance(shap_explanation)
        lime_importance = self._aggregate_lime_importance(lime_explanations)
        
        # Wine quality specific analysis
        wine_insights = {
            'chemical_drivers': self._identify_chemical_drivers(shap_importance, lime_importance),
            'quality_predictors': self._identify_quality_predictors(shap_importance, lime_importance),
            'feature_relationships': self._analyze_wine_feature_relationships(shap_explanation),
            'quality_improvement_suggestions': self._generate_quality_improvement_suggestions(
                shap_importance, lime_importance
            ),
            'critical_thresholds': self._identify_critical_thresholds(shap_explanation, lime_explanations)
        }
        
        return wine_insights
    
    def _generate_feature_insights(self, shap_explanation: SHAPExplanation,
                                 lime_explanations: List[LIMEExplanation]) -> List[str]:
        """Generate insights about individual features.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            
        Returns:
            List of feature-specific insights
        """
        insights = []
        
        # Extract importance scores
        shap_importance = self._extract_feature_importance(shap_explanation)
        lime_importance = self._aggregate_lime_importance(lime_explanations)
        
        # Find most important features
        top_shap_features = sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_lime_features = sorted(lime_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Generate insights for top features
        for feature, importance in top_shap_features:
            if abs(importance) > self.significance_threshold:
                direction = "positively" if importance > 0 else "negatively"
                insights.append(f"Feature '{feature}' {direction} influences wine quality with high importance (SHAP: {importance:.3f})")
        
        # Compare top features between methods
        shap_top_names = [f[0] for f in top_shap_features]
        lime_top_names = [f[0] for f in top_lime_features]
        common_top = set(shap_top_names) & set(lime_top_names)
        
        if common_top:
            insights.append(f"Features consistently ranked as important by both methods: {', '.join(common_top)}")
        
        return insights
    
    def _generate_consistency_insights(self, comparison_result: ComparisonResult) -> List[str]:
        """Generate insights about explanation consistency.
        
        Args:
            comparison_result: Comparison result between SHAP and LIME
            
        Returns:
            List of consistency insights
        """
        insights = []
        
        # Overall consistency
        if comparison_result.consistency_score >= self.consistency_threshold:
            insights.append(f"High explanation consistency ({comparison_result.consistency_score:.1%}) indicates reliable model interpretations")
        else:
            insights.append(f"Low explanation consistency ({comparison_result.consistency_score:.1%}) suggests caution in interpretation")
        
        # Correlation insights
        if comparison_result.correlation_score >= 0.7:
            insights.append("Strong correlation between SHAP and LIME indicates robust feature importance rankings")
        elif comparison_result.correlation_score <= 0.3:
            insights.append("Weak correlation between methods suggests model complexity or explanation method limitations")
        
        # Agreement/disagreement insights
        if comparison_result.disagreement_features:
            insights.append(f"Features with significant disagreement: {', '.join(comparison_result.disagreement_features[:3])}")
        
        return insights
    
    def _generate_model_behavior_insights(self, shap_explanation: SHAPExplanation,
                                        lime_explanations: List[LIMEExplanation]) -> List[str]:
        """Generate insights about model behavior patterns.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            
        Returns:
            List of model behavior insights
        """
        insights = []
        
        # Analyze prediction variance
        predictions = [exp.prediction for exp in lime_explanations]
        if predictions:
            pred_std = np.std(predictions)
            pred_mean = np.mean(predictions)
            
            if pred_std / pred_mean > 0.2:  # High coefficient of variation
                insights.append("High prediction variance suggests model sensitivity to input variations")
            else:
                insights.append("Low prediction variance indicates stable model behavior")
        
        # Analyze feature contribution patterns
        shap_values = shap_explanation.shap_values
        if len(shap_values.shape) > 1:
            feature_std = np.std(shap_values, axis=0)
            feature_names = shap_explanation.feature_names
            
            # Find features with high variance in contributions
            high_variance_features = []
            for i, std in enumerate(feature_std):
                if std > np.mean(feature_std) * 1.5:  # 1.5x above average
                    high_variance_features.append(feature_names[i])
            
            if high_variance_features:
                insights.append(f"Features with variable contributions across samples: {', '.join(high_variance_features[:3])}")
        
        return insights
    
    def _generate_actionable_recommendations(self, shap_explanation: SHAPExplanation,
                                           lime_explanations: List[LIMEExplanation],
                                           comparison_result: ComparisonResult) -> List[str]:
        """Generate actionable recommendations based on explanations.
        
        Args:
            shap_explanation: SHAP explanation object
            lime_explanations: List of LIME explanations
            comparison_result: Comparison result
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Model improvement recommendations
        if comparison_result.consistency_score < self.consistency_threshold:
            recommendations.append("Consider ensemble methods or model regularization to improve explanation consistency")
        
        # Feature engineering recommendations
        shap_importance = self._extract_feature_importance(shap_explanation)
        low_importance_features = [f for f, imp in shap_importance.items() if abs(imp) < self.significance_threshold]
        
        if len(low_importance_features) > len(shap_importance) * 0.3:  # More than 30% low importance
            recommendations.append("Consider feature selection to remove low-importance features and improve model interpretability")
        
        # Data quality recommendations
        if comparison_result.disagreement_features:
            recommendations.append(f"Investigate data quality for features with explanation disagreements: {', '.join(comparison_result.disagreement_features[:2])}")
        
        # Validation recommendations
        predictions = [exp.prediction for exp in lime_explanations]
        if predictions and len(set(np.round(predictions, 1))) < len(predictions) * 0.5:
            recommendations.append("High prediction similarity suggests potential overfitting - validate on diverse test sets")
        
        return recommendations
    
    def _assess_explanation_quality(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Assess the quality of explanations.
        
        Args:
            comparison_result: Comparison result between methods
            
        Returns:
            Dictionary with quality assessment
        """
        quality_score = 0.0
        quality_factors = []
        
        # Consistency factor
        consistency_weight = 0.4
        consistency_score = comparison_result.consistency_score
        quality_score += consistency_weight * consistency_score
        quality_factors.append(f"Consistency: {consistency_score:.2f}")
        
        # Correlation factor
        correlation_weight = 0.3
        correlation_score = max(0, comparison_result.correlation_score)  # Ensure non-negative
        quality_score += correlation_weight * correlation_score
        quality_factors.append(f"Correlation: {correlation_score:.2f}")
        
        # Agreement factor
        agreement_weight = 0.3
        total_features = len(comparison_result.agreement_features) + len(comparison_result.disagreement_features)
        agreement_score = len(comparison_result.agreement_features) / total_features if total_features > 0 else 0
        quality_score += agreement_weight * agreement_score
        quality_factors.append(f"Agreement: {agreement_score:.2f}")
        
        # Quality rating
        if quality_score >= 0.8:
            quality_rating = "Excellent"
        elif quality_score >= 0.6:
            quality_rating = "Good"
        elif quality_score >= 0.4:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        return {
            'overall_score': quality_score,
            'rating': quality_rating,
            'factors': quality_factors,
            'recommendations': self._get_quality_recommendations(quality_score, comparison_result)
        }
    
    def _generate_interaction_insights(self, interaction_analysis: InteractionAnalysis) -> List[str]:
        """Generate insights from feature interaction analysis.
        
        Args:
            interaction_analysis: Feature interaction analysis results
            
        Returns:
            List of interaction insights
        """
        insights = []
        
        # Top interactions
        if interaction_analysis.top_interactions:
            top_interaction = interaction_analysis.top_interactions[0]
            insights.append(f"Strongest feature interaction: {top_interaction['feature_pair']} (strength: {top_interaction['strength']:.3f})")
        
        # Interaction patterns
        strong_interactions = [
            interaction for interaction in interaction_analysis.top_interactions
            if interaction['strength'] > self.interaction_threshold
        ]
        
        if len(strong_interactions) > 3:
            insights.append(f"Model exhibits complex feature interactions ({len(strong_interactions)} significant pairs)")
        elif len(strong_interactions) == 0:
            insights.append("Model shows minimal feature interactions - features contribute independently")
        
        return insights
    
    def _generate_summary_insights(self, insights: Dict[str, Any]) -> List[str]:
        """Generate overall summary insights.
        
        Args:
            insights: Dictionary with all generated insights
            
        Returns:
            List of summary insights
        """
        summary = []
        
        # Quality summary
        quality = insights.get('quality_assessment', {})
        if quality:
            summary.append(f"Explanation quality: {quality['rating']} (score: {quality['overall_score']:.2f})")
        
        # Feature summary
        feature_insights = insights.get('feature_insights', [])
        if feature_insights:
            summary.append(f"Identified {len(feature_insights)} key feature insights")
        
        # Consistency summary
        consistency_insights = insights.get('consistency_insights', [])
        if consistency_insights:
            summary.append("Explanation consistency analysis completed")
        
        # Recommendations summary
        recommendations = insights.get('actionable_recommendations', [])
        if recommendations:
            summary.append(f"Generated {len(recommendations)} actionable recommendations")
        
        return summary
    
    def _extract_feature_importance(self, shap_explanation: SHAPExplanation) -> Dict[str, float]:
        """Extract feature importance from SHAP explanation."""
        shap_values = shap_explanation.shap_values
        feature_names = shap_explanation.feature_names
        importance_scores = np.mean(np.abs(shap_values), axis=0)
        return dict(zip(feature_names, importance_scores))
    
    def _aggregate_lime_importance(self, lime_explanations: List[LIMEExplanation]) -> Dict[str, float]:
        """Aggregate LIME feature importance across explanations."""
        if not lime_explanations:
            return {}
        
        feature_values = defaultdict(list)
        for explanation in lime_explanations:
            for feature, importance in explanation.feature_importance.items():
                feature_values[feature].append(importance)
        
        return {feature: np.mean(values) for feature, values in feature_values.items()}
    
    def _analyze_method_disagreements(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Analyze disagreements between explanation methods."""
        disagreements = {}
        
        for feature in comparison_result.disagreement_features:
            shap_score = comparison_result.shap_importance.get(feature, 0)
            lime_score = comparison_result.lime_importance.get(feature, 0)
            
            disagreements[feature] = {
                'shap_importance': shap_score,
                'lime_importance': lime_score,
                'difference': abs(shap_score - lime_score),
                'relative_difference': abs(shap_score - lime_score) / max(abs(shap_score), abs(lime_score), 1e-8)
            }
        
        return disagreements
    
    def _analyze_feature_stability(self, lime_explanations: List[LIMEExplanation]) -> Dict[str, float]:
        """Analyze stability of feature importance across LIME explanations."""
        feature_values = defaultdict(list)
        
        for explanation in lime_explanations:
            for feature, importance in explanation.feature_importance.items():
                feature_values[feature].append(importance)
        
        stability_scores = {}
        for feature, values in feature_values.items():
            if len(values) > 1:
                stability_scores[feature] = 1.0 - (np.std(values) / (np.mean(np.abs(values)) + 1e-8))
            else:
                stability_scores[feature] = 1.0
        
        return stability_scores
    
    def _analyze_prediction_variance(self, lime_explanations: List[LIMEExplanation]) -> Dict[str, float]:
        """Analyze variance in predictions across LIME explanations."""
        predictions = [exp.prediction for exp in lime_explanations]
        
        if not predictions:
            return {}
        
        return {
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'coefficient_of_variation': np.std(predictions) / (np.mean(predictions) + 1e-8),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions)
        }
    
    def _detect_directional_conflicts(self, shap_explanation: SHAPExplanation,
                                    lime_explanations: List[LIMEExplanation]) -> List[str]:
        """Detect features with directional conflicts between methods."""
        shap_importance = self._extract_feature_importance(shap_explanation)
        lime_importance = self._aggregate_lime_importance(lime_explanations)
        
        conflicts = []
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        
        for feature in common_features:
            shap_sign = np.sign(shap_importance[feature])
            lime_sign = np.sign(lime_importance[feature])
            
            if shap_sign != lime_sign and abs(shap_importance[feature]) > self.significance_threshold:
                conflicts.append(feature)
        
        return conflicts
    
    def _detect_magnitude_discrepancies(self, comparison_result: ComparisonResult) -> List[Tuple[str, float]]:
        """Detect features with large magnitude discrepancies."""
        discrepancies = []
        
        for feature in comparison_result.shap_importance.keys():
            if feature in comparison_result.lime_importance:
                shap_score = abs(comparison_result.shap_importance[feature])
                lime_score = abs(comparison_result.lime_importance[feature])
                
                if max(shap_score, lime_score) > 0:
                    ratio = max(shap_score, lime_score) / (min(shap_score, lime_score) + 1e-8)
                    if ratio > 3.0:  # 3x difference threshold
                        discrepancies.append((feature, ratio))
        
        return sorted(discrepancies, key=lambda x: x[1], reverse=True)
    
    def _calculate_reliability_scores(self, comparison_result: ComparisonResult,
                                    lime_explanations: List[LIMEExplanation]) -> Dict[str, float]:
        """Calculate reliability scores for features."""
        stability_scores = self._analyze_feature_stability(lime_explanations)
        
        reliability_scores = {}
        for feature in comparison_result.shap_importance.keys():
            # Base reliability on consistency and stability
            consistency_score = 1.0 if feature in comparison_result.agreement_features else 0.0
            stability_score = stability_scores.get(feature, 0.5)
            
            reliability_scores[feature] = (consistency_score + stability_score) / 2.0
        
        return reliability_scores
    
    def _identify_chemical_drivers(self, shap_importance: Dict[str, float],
                                 lime_importance: Dict[str, float]) -> List[str]:
        """Identify key chemical drivers of wine quality."""
        # Wine chemistry features (common wine dataset features)
        chemical_features = [
            'alcohol', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'fixed_acidity'
        ]
        
        # Find chemical features with high importance
        important_chemicals = []
        all_importance = {**shap_importance, **lime_importance}
        
        for feature in chemical_features:
            # Check for partial matches in feature names
            matching_features = [f for f in all_importance.keys() if feature.lower() in f.lower()]
            for match in matching_features:
                if abs(all_importance[match]) > self.significance_threshold:
                    important_chemicals.append(match)
        
        return important_chemicals
    
    def _identify_quality_predictors(self, shap_importance: Dict[str, float],
                                   lime_importance: Dict[str, float]) -> Dict[str, str]:
        """Identify features that predict higher vs lower quality."""
        quality_predictors = {}
        
        # Combine importance scores
        combined_importance = {}
        all_features = set(shap_importance.keys()) | set(lime_importance.keys())
        
        for feature in all_features:
            shap_score = shap_importance.get(feature, 0)
            lime_score = lime_importance.get(feature, 0)
            combined_importance[feature] = (shap_score + lime_score) / 2
        
        # Categorize features
        for feature, importance in combined_importance.items():
            if abs(importance) > self.significance_threshold:
                if importance > 0:
                    quality_predictors[feature] = "increases_quality"
                else:
                    quality_predictors[feature] = "decreases_quality"
        
        return quality_predictors
    
    def _analyze_wine_feature_relationships(self, shap_explanation: SHAPExplanation) -> List[str]:
        """Analyze relationships between wine features."""
        relationships = []
        
        # This is a simplified analysis - in practice, you'd use domain knowledge
        feature_names = shap_explanation.feature_names
        shap_values = shap_explanation.shap_values
        
        if len(shap_values.shape) > 1:
            # Calculate feature correlations in SHAP space
            feature_correlations = np.corrcoef(shap_values.T)
            
            # Find strong correlations
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    correlation = feature_correlations[i, j]
                    if abs(correlation) > 0.5:
                        relationship_type = "positively related" if correlation > 0 else "negatively related"
                        relationships.append(f"{feature_names[i]} and {feature_names[j]} are {relationship_type} in their impact on quality")
        
        return relationships[:5]  # Return top 5 relationships
    
    def _generate_quality_improvement_suggestions(self, shap_importance: Dict[str, float],
                                                lime_importance: Dict[str, float]) -> List[str]:
        """Generate suggestions for improving wine quality."""
        suggestions = []
        
        # Find features that increase quality
        quality_increasers = []
        quality_decreasers = []
        
        combined_importance = {}
        all_features = set(shap_importance.keys()) | set(lime_importance.keys())
        
        for feature in all_features:
            shap_score = shap_importance.get(feature, 0)
            lime_score = lime_importance.get(feature, 0)
            avg_score = (shap_score + lime_score) / 2
            
            if abs(avg_score) > self.significance_threshold:
                if avg_score > 0:
                    quality_increasers.append((feature, avg_score))
                else:
                    quality_decreasers.append((feature, abs(avg_score)))
        
        # Generate suggestions
        if quality_increasers:
            top_increaser = max(quality_increasers, key=lambda x: x[1])
            suggestions.append(f"To improve wine quality, focus on optimizing {top_increaser[0]}")
        
        if quality_decreasers:
            top_decreaser = max(quality_decreasers, key=lambda x: x[1])
            suggestions.append(f"To improve wine quality, minimize {top_decreaser[0]}")
        
        return suggestions
    
    def _identify_critical_thresholds(self, shap_explanation: SHAPExplanation,
                                    lime_explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Identify critical thresholds for important features."""
        # This is a simplified implementation
        # In practice, you'd analyze the relationship between feature values and SHAP values
        
        thresholds = {}
        feature_names = shap_explanation.feature_names
        shap_values = shap_explanation.shap_values
        data = shap_explanation.data
        
        if len(shap_values.shape) > 1 and data is not None:
            for i, feature in enumerate(feature_names):
                feature_values = data[:, i]
                feature_shap = shap_values[:, i]
                
                # Find where SHAP values change sign (simplified threshold detection)
                positive_mask = feature_shap > 0
                negative_mask = feature_shap < 0
                
                if np.any(positive_mask) and np.any(negative_mask):
                    pos_values = feature_values[positive_mask]
                    neg_values = feature_values[negative_mask]
                    
                    if len(pos_values) > 0 and len(neg_values) > 0:
                        threshold = (np.min(pos_values) + np.max(neg_values)) / 2
                        thresholds[feature] = {
                            'threshold': threshold,
                            'interpretation': f"Values above {threshold:.3f} tend to increase quality"
                        }
        
        return thresholds
    
    def _get_quality_recommendations(self, quality_score: float,
                                   comparison_result: ComparisonResult) -> List[str]:
        """Get recommendations based on explanation quality."""
        recommendations = []
        
        if quality_score < 0.4:
            recommendations.append("Consider using different explanation methods or model types")
            recommendations.append("Validate model performance and check for overfitting")
        elif quality_score < 0.6:
            recommendations.append("Increase background sample size for more stable explanations")
            recommendations.append("Consider feature engineering to improve model interpretability")
        else:
            recommendations.append("Explanations are reliable and can be used for decision making")
        
        if len(comparison_result.disagreement_features) > len(comparison_result.agreement_features):
            recommendations.append("High disagreement suggests model complexity - consider simpler models")
        
        return recommendations