"""
Data validation utilities for wine interpretability analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation and quality checks."""
    
    def __init__(self):
        """Initialize DataValidator."""
        self.validation_history = []
        
    def validate_dataframe_structure(self, df: pd.DataFrame, 
                                   required_columns: List[str],
                                   name: str = "dataframe") -> Dict[str, Any]:
        """
        Validate basic dataframe structure and required columns.
        
        Args:
            df: Dataframe to validate
            required_columns: List of required column names
            name: Name of dataframe for logging
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValueError: If validation fails
        """
        validation_result = {
            'name': name,
            'passed': True,
            'issues': [],
            'warnings': [],
            'info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
        }
        
        # Check if dataframe is empty
        if df.empty:
            validation_result['passed'] = False
            validation_result['issues'].append(f"{name} is empty")
            
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['passed'] = False
            validation_result['issues'].append(f"Missing required columns: {missing_columns}")
            
        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            validation_result['warnings'].append(f"Duplicate columns found: {duplicate_columns}")
            
        # Log results
        if validation_result['passed']:
            logger.info(f"{name} structure validation passed")
        else:
            logger.error(f"{name} structure validation failed: {validation_result['issues']}")
            raise ValueError(f"Dataframe validation failed for {name}: {validation_result['issues']}")
            
        self.validation_history.append(validation_result)
        return validation_result
        
    def validate_data_types(self, df: pd.DataFrame, 
                          expected_types: Dict[str, str],
                          name: str = "dataframe") -> Dict[str, Any]:
        """
        Validate column data types.
        
        Args:
            df: Dataframe to validate
            expected_types: Dictionary mapping column names to expected types
            name: Name of dataframe for logging
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'name': name,
            'passed': True,
            'issues': [],
            'warnings': [],
            'type_mismatches': {}
        }
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                
                # Check for type compatibility
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[column]):
                    validation_result['type_mismatches'][column] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
                elif expected_type == 'categorical' and not pd.api.types.is_object_dtype(df[column]):
                    validation_result['type_mismatches'][column] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
                elif expected_type not in ['numeric', 'categorical'] and expected_type not in actual_type:
                    validation_result['type_mismatches'][column] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
                    
        if validation_result['type_mismatches']:
            validation_result['warnings'].append(f"Type mismatches found: {validation_result['type_mismatches']}")
            
        logger.info(f"{name} data type validation completed")
        self.validation_history.append(validation_result)
        return validation_result
        
    def validate_value_ranges(self, df: pd.DataFrame, 
                            range_constraints: Dict[str, Tuple[float, float]],
                            name: str = "dataframe") -> Dict[str, Any]:
        """
        Validate that numerical columns fall within expected ranges.
        
        Args:
            df: Dataframe to validate
            range_constraints: Dictionary mapping column names to (min, max) tuples
            name: Name of dataframe for logging
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'name': name,
            'passed': True,
            'issues': [],
            'warnings': [],
            'range_violations': {}
        }
        
        for column, (min_val, max_val) in range_constraints.items():
            if column in df.columns:
                col_data = df[column].dropna()
                
                if len(col_data) == 0:
                    continue
                    
                actual_min = col_data.min()
                actual_max = col_data.max()
                
                violations = []
                if actual_min < min_val:
                    violations.append(f"minimum {actual_min} < expected {min_val}")
                if actual_max > max_val:
                    violations.append(f"maximum {actual_max} > expected {max_val}")
                    
                if violations:
                    validation_result['range_violations'][column] = {
                        'violations': violations,
                        'actual_range': (actual_min, actual_max),
                        'expected_range': (min_val, max_val),
                        'violation_count': len(col_data[(col_data < min_val) | (col_data > max_val)])
                    }
                    
        if validation_result['range_violations']:
            validation_result['warnings'].append(f"Range violations found: {list(validation_result['range_violations'].keys())}")
            
        logger.info(f"{name} value range validation completed")
        self.validation_history.append(validation_result)
        return validation_result
        
    def validate_categorical_values(self, df: pd.DataFrame,
                                  allowed_values: Dict[str, List[str]],
                                  name: str = "dataframe") -> Dict[str, Any]:
        """
        Validate that categorical columns contain only allowed values.
        
        Args:
            df: Dataframe to validate
            allowed_values: Dictionary mapping column names to lists of allowed values
            name: Name of dataframe for logging
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'name': name,
            'passed': True,
            'issues': [],
            'warnings': [],
            'invalid_values': {}
        }
        
        for column, allowed in allowed_values.items():
            if column in df.columns:
                col_data = df[column].dropna().astype(str)
                unique_values = set(col_data.unique())
                allowed_set = set(str(val) for val in allowed)
                
                invalid_values = unique_values - allowed_set
                if invalid_values:
                    validation_result['invalid_values'][column] = {
                        'invalid_values': list(invalid_values),
                        'allowed_values': allowed,
                        'invalid_count': len(col_data[col_data.isin(invalid_values)])
                    }
                    
        if validation_result['invalid_values']:
            validation_result['warnings'].append(f"Invalid categorical values found: {list(validation_result['invalid_values'].keys())}")
            
        logger.info(f"{name} categorical value validation completed")
        self.validation_history.append(validation_result)
        return validation_result
        
    def validate_feature_target_alignment(self, X: pd.DataFrame, y: pd.Series,
                                        name: str = "dataset") -> Dict[str, Any]:
        """
        Validate alignment between features and target variable.
        
        Args:
            X: Feature dataframe
            y: Target series
            name: Name of dataset for logging
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValueError: If alignment validation fails
        """
        validation_result = {
            'name': name,
            'passed': True,
            'issues': [],
            'warnings': [],
            'alignment_info': {
                'X_shape': X.shape,
                'y_shape': y.shape,
                'X_index_type': type(X.index).__name__,
                'y_index_type': type(y.index).__name__
            }
        }
        
        # Check shape alignment
        if len(X) != len(y):
            validation_result['passed'] = False
            validation_result['issues'].append(f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples")
            
        # Check index alignment
        if not X.index.equals(y.index):
            validation_result['warnings'].append("Index mismatch between X and y")
            
        # Check for empty datasets
        if X.empty:
            validation_result['passed'] = False
            validation_result['issues'].append("Feature dataframe X is empty")
            
        if y.empty:
            validation_result['passed'] = False
            validation_result['issues'].append("Target series y is empty")
            
        if validation_result['passed']:
            logger.info(f"{name} feature-target alignment validation passed")
        else:
            logger.error(f"{name} alignment validation failed: {validation_result['issues']}")
            raise ValueError(f"Feature-target alignment failed for {name}: {validation_result['issues']}")
            
        self.validation_history.append(validation_result)
        return validation_result
        
    def validate_train_test_consistency(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                      name: str = "train-test") -> Dict[str, Any]:
        """
        Validate consistency between training and test sets.
        
        Args:
            X_train: Training feature dataframe
            X_test: Test feature dataframe
            name: Name for logging
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValueError: If consistency validation fails
        """
        validation_result = {
            'name': name,
            'passed': True,
            'issues': [],
            'warnings': [],
            'consistency_info': {
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'common_features': 0,
                'train_only_features': [],
                'test_only_features': []
            }
        }
        
        train_features = set(X_train.columns)
        test_features = set(X_test.columns)
        
        common_features = train_features.intersection(test_features)
        train_only = train_features - test_features
        test_only = test_features - train_features
        
        validation_result['consistency_info']['common_features'] = len(common_features)
        validation_result['consistency_info']['train_only_features'] = list(train_only)
        validation_result['consistency_info']['test_only_features'] = list(test_only)
        
        # Check for feature mismatches
        if train_only:
            validation_result['passed'] = False
            validation_result['issues'].append(f"Features in train but not test: {list(train_only)}")
            
        if test_only:
            validation_result['passed'] = False
            validation_result['issues'].append(f"Features in test but not train: {list(test_only)}")
            
        # Check data types for common features
        dtype_mismatches = []
        for feature in common_features:
            if str(X_train[feature].dtype) != str(X_test[feature].dtype):
                dtype_mismatches.append({
                    'feature': feature,
                    'train_dtype': str(X_train[feature].dtype),
                    'test_dtype': str(X_test[feature].dtype)
                })
                
        if dtype_mismatches:
            validation_result['warnings'].append(f"Data type mismatches: {dtype_mismatches}")
            
        if validation_result['passed']:
            logger.info(f"{name} consistency validation passed")
        else:
            logger.error(f"{name} consistency validation failed: {validation_result['issues']}")
            raise ValueError(f"Train-test consistency failed for {name}: {validation_result['issues']}")
            
        self.validation_history.append(validation_result)
        return validation_result
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.
        
        Returns:
            Dictionary with validation summary
        """
        summary = {
            'total_validations': len(self.validation_history),
            'passed_validations': len([v for v in self.validation_history if v['passed']]),
            'failed_validations': len([v for v in self.validation_history if not v['passed']]),
            'total_issues': sum(len(v['issues']) for v in self.validation_history),
            'total_warnings': sum(len(v['warnings']) for v in self.validation_history),
            'validation_details': self.validation_history
        }
        
        logger.info(f"Validation summary: {summary['passed_validations']}/{summary['total_validations']} passed")
        return summary