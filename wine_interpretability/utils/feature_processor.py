"""
Feature engineering and validation utilities for wine interpretability analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Feature engineering and processing utilities."""
    
    def __init__(self):
        """Initialize FeatureProcessor."""
        self.feature_selectors = {}
        self.polynomial_features = None
        self.feature_statistics = {}
        self.label_encoders = {}
        self.onehot_encoder = None
        self.encoded_feature_names = []
        
    def validate_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Validate and report missing values in the dataset.
        
        Args:
            df: Input dataframe
            threshold: Maximum allowed proportion of missing values per column
            
        Returns:
            Dictionary with missing value analysis
            
        Raises:
            ValueError: If any column exceeds missing value threshold
        """
        missing_info = {}
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Identify problematic columns
        problematic_cols = missing_percentages[missing_percentages > threshold * 100].index.tolist()
        
        missing_info = {
            'total_rows': len(df),
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'problematic_columns': problematic_cols,
            'threshold_exceeded': len(problematic_cols) > 0
        }
        
        if problematic_cols:
            logger.warning(f"Columns exceeding {threshold*100}% missing threshold: {problematic_cols}")
            raise ValueError(f"Columns with excessive missing values: {problematic_cols}")
            
        logger.info(f"Missing value validation passed. Max missing: {missing_percentages.max():.2f}%")
        return missing_info
        
    def detect_outliers(self, df: pd.DataFrame, numerical_cols: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers in numerical columns.
        
        Args:
            df: Input dataframe
            numerical_cols: List of numerical column names
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier analysis
        """
        outlier_info = {}
        
        for col in numerical_cols:
            if col not in df.columns:
                continue
                
            data = df[col].dropna()
            outliers = []
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > threshold].index.tolist()
                
            elif method == 'modified_zscore':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = data[np.abs(modified_z_scores) > threshold].index.tolist()
                
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'indices': outliers[:100]  # Limit to first 100 for memory
            }
            
        total_outliers = sum([info['count'] for info in outlier_info.values()])
        logger.info(f"Detected {total_outliers} outliers across {len(numerical_cols)} numerical columns")
        
        return outlier_info
        
    def validate_feature_consistency(self, train_features: List[str], 
                                   test_features: List[str]) -> Dict[str, Any]:
        """
        Validate feature name consistency between training and test sets.
        
        Args:
            train_features: List of training feature names
            test_features: List of test feature names
            
        Returns:
            Dictionary with consistency analysis
            
        Raises:
            ValueError: If feature sets are inconsistent
        """
        train_set = set(train_features)
        test_set = set(test_features)
        
        missing_in_test = train_set - test_set
        extra_in_test = test_set - train_set
        common_features = train_set.intersection(test_set)
        
        consistency_info = {
            'train_feature_count': len(train_features),
            'test_feature_count': len(test_features),
            'common_feature_count': len(common_features),
            'missing_in_test': list(missing_in_test),
            'extra_in_test': list(extra_in_test),
            'is_consistent': len(missing_in_test) == 0 and len(extra_in_test) == 0
        }
        
        if not consistency_info['is_consistent']:
            error_msg = f"Feature inconsistency detected. Missing in test: {missing_in_test}, Extra in test: {extra_in_test}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Feature consistency validated. {len(common_features)} common features")
        return consistency_info
        
    def create_interaction_features(self, X: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            X: Input feature dataframe
            feature_pairs: List of tuples specifying feature pairs for interaction
            
        Returns:
            Dataframe with original features plus interaction features
        """
        X_with_interactions = X.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                X_with_interactions[interaction_name] = X[feat1] * X[feat2]
                logger.debug(f"Created interaction feature: {interaction_name}")
                
        logger.info(f"Created {len(feature_pairs)} interaction features")
        return X_with_interactions
        
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2,
                                 include_bias: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Create polynomial features up to specified degree.
        
        Args:
            X: Input feature dataframe
            degree: Maximum degree of polynomial features
            include_bias: Whether to include bias column
            
        Returns:
            Tuple of (polynomial_features_array, feature_names)
        """
        self.polynomial_features = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=False
        )
        
        X_poly = self.polynomial_features.fit_transform(X)
        feature_names = self.polynomial_features.get_feature_names_out(X.columns)
        
        logger.info(f"Created polynomial features: {X.shape[1]} -> {X_poly.shape[1]} features")
        return X_poly, feature_names.tolist()
        
    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, 
                                 k: int = 10, score_func: str = 'f_regression') -> Dict[str, Any]:
        """
        Select top k features using univariate statistical tests.
        
        Args:
            X: Feature dataframe
            y: Target series
            k: Number of features to select
            score_func: Scoring function ('f_regression', 'mutual_info_regression')
            
        Returns:
            Dictionary with selected features and scores
        """
        if score_func == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif score_func == 'mutual_info_regression':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unsupported score function: {score_func}")
            
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature information
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        feature_scores = selector.scores_[selected_indices]
        
        self.feature_selectors[score_func] = selector
        
        selection_info = {
            'selected_features': selected_features,
            'feature_scores': dict(zip(selected_features, feature_scores)),
            'selected_indices': selected_indices.tolist(),
            'X_selected': X_selected
        }
        
        logger.info(f"Selected {k} features using {score_func}")
        return selection_info
        
    def calculate_feature_statistics(self, X: pd.DataFrame, 
                                   categorical_cols: List[str],
                                   numerical_cols: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for all features.
        
        Args:
            X: Feature dataframe
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
            
        Returns:
            Dictionary with feature statistics
        """
        stats_info = {
            'numerical_stats': {},
            'categorical_stats': {},
            'overall_stats': {
                'total_features': len(X.columns),
                'numerical_count': len(numerical_cols),
                'categorical_count': len(categorical_cols),
                'total_samples': len(X)
            }
        }
        
        # Numerical feature statistics
        for col in numerical_cols:
            if col in X.columns:
                data = X[col].dropna()
                stats_info['numerical_stats'][col] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'missing_count': int(X[col].isnull().sum()),
                    'unique_count': int(data.nunique())
                }
                
        # Categorical feature statistics
        for col in categorical_cols:
            if col in X.columns:
                data = X[col].dropna()
                value_counts = data.value_counts()
                stats_info['categorical_stats'][col] = {
                    'unique_count': int(data.nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'missing_count': int(X[col].isnull().sum()),
                    'value_distribution': value_counts.head(10).to_dict()
                }
                
        self.feature_statistics = stats_info
        logger.info("Calculated comprehensive feature statistics")
        return stats_info
        
    def validate_data_quality(self, X: pd.DataFrame, y: pd.Series,
                            categorical_cols: List[str],
                            numerical_cols: List[str],
                            missing_threshold: float = 0.5,
                            outlier_threshold: float = 1.5) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Args:
            X: Feature dataframe
            y: Target series
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
            missing_threshold: Maximum allowed proportion of missing values
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with comprehensive data quality report
        """
        quality_report = {
            'validation_passed': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Validate missing values
            missing_info = self.validate_missing_values(X, missing_threshold)
            quality_report['missing_values'] = missing_info
            
        except ValueError as e:
            quality_report['validation_passed'] = False
            quality_report['issues'].append(f"Missing values: {str(e)}")
            
        # Detect outliers (non-blocking)
        try:
            outlier_info = self.detect_outliers(X, numerical_cols, threshold=outlier_threshold)
            quality_report['outliers'] = outlier_info
            
            # Add warnings for high outlier percentages
            for col, info in outlier_info.items():
                if info['percentage'] > 10:  # More than 10% outliers
                    quality_report['warnings'].append(
                        f"High outlier percentage in {col}: {info['percentage']:.2f}%"
                    )
                    
        except Exception as e:
            quality_report['warnings'].append(f"Outlier detection failed: {str(e)}")
            
        # Calculate feature statistics
        try:
            stats_info = self.calculate_feature_statistics(X, categorical_cols, numerical_cols)
            quality_report['feature_statistics'] = stats_info
            
        except Exception as e:
            quality_report['warnings'].append(f"Feature statistics calculation failed: {str(e)}")
            
        # Validate target variable
        try:
            y_stats = {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
                'missing_count': int(y.isnull().sum()),
                'unique_count': int(y.nunique())
            }
            quality_report['target_statistics'] = y_stats
            
            if y.isnull().sum() > 0:
                quality_report['issues'].append(f"Target variable has {y.isnull().sum()} missing values")
                quality_report['validation_passed'] = False
                
        except Exception as e:
            quality_report['issues'].append(f"Target validation failed: {str(e)}")
            quality_report['validation_passed'] = False
            
        # Final validation status
        if quality_report['issues']:
            logger.error(f"Data quality validation failed with {len(quality_report['issues'])} issues")
        else:
            logger.info("Data quality validation passed")
            
        if quality_report['warnings']:
            logger.warning(f"Data quality validation completed with {len(quality_report['warnings'])} warnings")
            
        return quality_report
        
    def engineer_wine_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create wine-specific engineered features.
        
        Args:
            df: Input dataframe with wine features
            
        Returns:
            Dataframe with additional engineered features
        """
        engineered_df = df.copy()
        
        # ABV categories
        if 'ABV' in engineered_df.columns:
            engineered_df['ABV_Category'] = pd.cut(
                engineered_df['ABV'], 
                bins=[0, 10, 12, 14, 16, 100], 
                labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
            )
            
        # Body-Acidity interaction
        if 'Body' in engineered_df.columns and 'Acidity' in engineered_df.columns:
            engineered_df['Body_Acidity_Combo'] = (
                engineered_df['Body'].astype(str) + '_' + 
                engineered_df['Acidity'].astype(str)
            )
            
        # Grape diversity score
        if 'GrapeCount' in engineered_df.columns:
            engineered_df['Grape_Diversity'] = pd.cut(
                engineered_df['GrapeCount'],
                bins=[0, 1, 2, 3, 100],
                labels=['Single', 'Blend_2', 'Blend_3', 'Complex_Blend']
            )
            
        # Rating reliability score
        if 'RatingCount' in engineered_df.columns:
            engineered_df['Rating_Reliability'] = pd.cut(
                engineered_df['RatingCount'],
                bins=[0, 5, 20, 50, 1000],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
            
        logger.info(f"Engineered wine-specific features: {df.shape[1]} -> {engineered_df.shape[1]} features")
        return engineered_df
        
    def encode_categorical_features(self, X: pd.DataFrame, 
                                  categorical_cols: List[str],
                                  encoding_method: str = 'label',
                                  handle_unknown: str = 'ignore') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical features using specified method.
        
        Args:
            X: Input dataframe
            categorical_cols: List of categorical column names to encode
            encoding_method: 'label' for label encoding, 'onehot' for one-hot encoding
            handle_unknown: How to handle unknown categories ('ignore', 'error')
            
        Returns:
            Tuple of (encoded_dataframe, encoding_info)
        """
        X_encoded = X.copy()
        encoding_info = {
            'method': encoding_method,
            'encoded_columns': [],
            'original_columns': categorical_cols.copy(),
            'feature_mapping': {}
        }
        
        if encoding_method == 'label':
            # Label encoding - each category gets a unique integer
            for col in categorical_cols:
                if col in X_encoded.columns:
                    # Handle missing values by filling with 'Unknown'
                    X_encoded[col] = X_encoded[col].fillna('Unknown')
                    
                    # Initialize and fit label encoder
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        
                    # Fit and transform
                    try:
                        X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
                        encoding_info['encoded_columns'].append(col)
                        encoding_info['feature_mapping'][col] = {
                            'classes': self.label_encoders[col].classes_.tolist(),
                            'n_classes': len(self.label_encoders[col].classes_)
                        }
                        logger.debug(f"Label encoded {col}: {len(self.label_encoders[col].classes_)} unique values")
                    except Exception as e:
                        logger.warning(f"Failed to encode column {col}: {str(e)}")
                        
        elif encoding_method == 'onehot':
            # One-hot encoding - creates binary columns for each category
            categorical_data = X_encoded[categorical_cols].fillna('Unknown')
            
            # Initialize one-hot encoder
            if self.onehot_encoder is None:
                self.onehot_encoder = OneHotEncoder(
                    sparse_output=False, 
                    handle_unknown=handle_unknown,
                    drop='first'  # Drop first category to avoid multicollinearity
                )
                
            try:
                # Fit and transform
                encoded_array = self.onehot_encoder.fit_transform(categorical_data)
                
                # Get feature names
                feature_names = self.onehot_encoder.get_feature_names_out(categorical_cols)
                self.encoded_feature_names = feature_names.tolist()
                
                # Create dataframe with encoded features
                encoded_df = pd.DataFrame(
                    encoded_array, 
                    columns=feature_names, 
                    index=X_encoded.index
                )
                
                # Drop original categorical columns and add encoded ones
                X_encoded = X_encoded.drop(columns=categorical_cols)
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
                
                encoding_info['encoded_columns'] = feature_names.tolist()
                encoding_info['feature_mapping']['onehot_features'] = {
                    'original_columns': categorical_cols,
                    'encoded_columns': feature_names.tolist(),
                    'n_features': len(feature_names)
                }
                
                logger.info(f"One-hot encoded {len(categorical_cols)} columns into {len(feature_names)} features")
                
            except Exception as e:
                logger.error(f"One-hot encoding failed: {str(e)}")
                raise
                
        else:
            raise ValueError(f"Unsupported encoding method: {encoding_method}")
            
        logger.info(f"Categorical encoding completed using {encoding_method} method")
        return X_encoded, encoding_info
        
    def transform_categorical_features(self, X: pd.DataFrame, 
                                     categorical_cols: List[str],
                                     encoding_method: str = 'label') -> pd.DataFrame:
        """
        Transform categorical features using previously fitted encoders.
        
        Args:
            X: Input dataframe
            categorical_cols: List of categorical column names to transform
            encoding_method: 'label' or 'onehot' (must match training)
            
        Returns:
            Transformed dataframe
        """
        X_transformed = X.copy()
        
        if encoding_method == 'label':
            for col in categorical_cols:
                if col in X_transformed.columns and col in self.label_encoders:
                    # Handle missing values
                    X_transformed[col] = X_transformed[col].fillna('Unknown')
                    
                    # Transform using fitted encoder
                    try:
                        # Handle unknown categories by mapping them to a default value
                        col_values = X_transformed[col].astype(str)
                        known_classes = set(self.label_encoders[col].classes_)
                        
                        # Map unknown values to the first class (or add 'Unknown' if it exists)
                        if 'Unknown' in known_classes:
                            col_values = col_values.apply(
                                lambda x: x if x in known_classes else 'Unknown'
                            )
                        else:
                            col_values = col_values.apply(
                                lambda x: x if x in known_classes else self.label_encoders[col].classes_[0]
                            )
                            
                        X_transformed[col] = self.label_encoders[col].transform(col_values)
                        logger.debug(f"Transformed column {col} using fitted label encoder")
                        
                    except Exception as e:
                        logger.warning(f"Failed to transform column {col}: {str(e)}")
                        
        elif encoding_method == 'onehot':
            if self.onehot_encoder is not None:
                categorical_data = X_transformed[categorical_cols].fillna('Unknown')
                
                try:
                    # Transform using fitted encoder
                    encoded_array = self.onehot_encoder.transform(categorical_data)
                    
                    # Create dataframe with encoded features
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=self.encoded_feature_names,
                        index=X_transformed.index
                    )
                    
                    # Drop original categorical columns and add encoded ones
                    X_transformed = X_transformed.drop(columns=categorical_cols)
                    X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                    
                    logger.info(f"Transformed categorical features using fitted one-hot encoder")
                    
                except Exception as e:
                    logger.error(f"One-hot transformation failed: {str(e)}")
                    raise
            else:
                raise ValueError("One-hot encoder not fitted. Call encode_categorical_features first.")
                
        return X_transformed
        
    def prepare_features_for_modeling(self, X: pd.DataFrame, 
                                    categorical_cols: List[str],
                                    numerical_cols: List[str],
                                    encoding_method: str = 'label',
                                    handle_missing: str = 'median') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete feature preparation pipeline for modeling.
        
        Args:
            X: Input dataframe
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names  
            encoding_method: 'label' or 'onehot' for categorical encoding
            handle_missing: 'median', 'mean', 'mode', or 'drop' for missing values
            
        Returns:
            Tuple of (prepared_dataframe, preparation_info)
        """
        preparation_info = {
            'original_shape': X.shape,
            'categorical_columns': categorical_cols.copy(),
            'numerical_columns': numerical_cols.copy(),
            'encoding_method': encoding_method,
            'missing_strategy': handle_missing
        }
        
        X_prepared = X.copy()
        
        # Handle missing values in numerical columns
        if handle_missing != 'drop':
            for col in numerical_cols:
                if col in X_prepared.columns:
                    if handle_missing == 'median':
                        fill_value = X_prepared[col].median()
                    elif handle_missing == 'mean':
                        fill_value = X_prepared[col].mean()
                    else:  # mode
                        fill_value = X_prepared[col].mode().iloc[0] if not X_prepared[col].mode().empty else 0
                        
                    X_prepared[col] = X_prepared[col].fillna(fill_value)
                    logger.debug(f"Filled missing values in {col} with {handle_missing}: {fill_value}")
        else:
            # Drop rows with missing values
            initial_rows = len(X_prepared)
            X_prepared = X_prepared.dropna()
            dropped_rows = initial_rows - len(X_prepared)
            logger.info(f"Dropped {dropped_rows} rows with missing values")
            
        # Encode categorical features
        if categorical_cols:
            X_prepared, encoding_info = self.encode_categorical_features(
                X_prepared, categorical_cols, encoding_method
            )
            preparation_info['encoding_info'] = encoding_info
            
        preparation_info['final_shape'] = X_prepared.shape
        preparation_info['features_created'] = X_prepared.shape[1] - len(numerical_cols)
        
        logger.info(f"Feature preparation completed: {X.shape} -> {X_prepared.shape}")
        return X_prepared, preparation_info
        
    def detect_target_leakage(self, X: pd.DataFrame, y: pd.Series, 
                            correlation_threshold: float = 0.95,
                            suspicious_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Detect potential target leakage features.
        
        Args:
            X: Feature dataframe
            y: Target series
            correlation_threshold: Correlation threshold above which features are flagged
            suspicious_patterns: List of suspicious column name patterns
            
        Returns:
            Dictionary with leakage detection results
        """
        if suspicious_patterns is None:
            suspicious_patterns = [
                'rating', 'score', 'quality', 'avg', 'mean', 'std', 'target'
            ]
        
        leakage_info = {
            'high_correlation_features': [],
            'suspicious_name_features': [],
            'recommended_exclusions': [],
            'correlation_analysis': {}
        }
        
        # Analyze correlations for numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_features:
            try:
                correlation = X[col].corr(y)
                leakage_info['correlation_analysis'][col] = float(correlation)
                
                if abs(correlation) > correlation_threshold:
                    leakage_info['high_correlation_features'].append({
                        'feature': col,
                        'correlation': float(correlation)
                    })
                    leakage_info['recommended_exclusions'].append(col)
                    
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {col}: {str(e)}")
        
        # Check for suspicious column names
        for col in X.columns:
            col_lower = col.lower()
            if any(pattern.lower() in col_lower for pattern in suspicious_patterns):
                leakage_info['suspicious_name_features'].append(col)
                if col not in leakage_info['recommended_exclusions']:
                    leakage_info['recommended_exclusions'].append(col)
        
        # Summary
        leakage_info['summary'] = {
            'total_features_analyzed': len(X.columns),
            'high_correlation_count': len(leakage_info['high_correlation_features']),
            'suspicious_names_count': len(leakage_info['suspicious_name_features']),
            'recommended_exclusions_count': len(leakage_info['recommended_exclusions'])
        }
        
        if leakage_info['recommended_exclusions']:
            logger.warning(f"Potential target leakage detected in features: {leakage_info['recommended_exclusions']}")
        else:
            logger.info("No obvious target leakage detected")
            
        return leakage_info