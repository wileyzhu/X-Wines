"""
Data loading utilities for wine interpretability analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import ast
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess wine datasets."""
    
    def __init__(self, wine_data_path: str, ratings_data_path: str):
        """
        Initialize DataLoader with dataset paths.
        
        Args:
            wine_data_path: Path to wine features CSV file
            ratings_data_path: Path to wine ratings CSV file
        """
        self.wine_data_path = wine_data_path
        self.ratings_data_path = ratings_data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw wine and ratings data from CSV files.
        
        Returns:
            Tuple of (wine_df, ratings_df)
            
        Raises:
            FileNotFoundError: If data files don't exist
            pd.errors.EmptyDataError: If files are empty
        """
        try:
            wine_df = pd.read_csv(self.wine_data_path)
            ratings_df = pd.read_csv(self.ratings_data_path)
            
            logger.info(f"Loaded wine data: {wine_df.shape[0]} wines, {wine_df.shape[1]} features")
            logger.info(f"Loaded ratings data: {ratings_df.shape[0]} ratings")
            
            return wine_df, ratings_df
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data file: {e}")
            raise
            
    def validate_data(self, wine_df: pd.DataFrame, ratings_df: pd.DataFrame) -> bool:
        """
        Validate loaded data for basic consistency.
        
        Args:
            wine_df: Wine features dataframe
            ratings_df: Wine ratings dataframe
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data validation fails
        """
        # Check required columns
        required_wine_cols = ['WineID', 'Type', 'ABV', 'Body', 'Acidity']
        required_rating_cols = ['WineID', 'Rating']
        
        missing_wine_cols = [col for col in required_wine_cols if col not in wine_df.columns]
        missing_rating_cols = [col for col in required_rating_cols if col not in ratings_df.columns]
        
        if missing_wine_cols:
            raise ValueError(f"Missing required wine columns: {missing_wine_cols}")
        if missing_rating_cols:
            raise ValueError(f"Missing required rating columns: {missing_rating_cols}")
            
        # Check for empty dataframes
        if wine_df.empty or ratings_df.empty:
            raise ValueError("One or both dataframes are empty")
            
        # Check for overlapping wine IDs
        wine_ids = set(wine_df['WineID'])
        rating_wine_ids = set(ratings_df['WineID'])
        common_ids = wine_ids.intersection(rating_wine_ids)
        
        if len(common_ids) == 0:
            raise ValueError("No common wine IDs between wine and rating datasets")
            
        logger.info(f"Data validation passed. {len(common_ids)} wines have ratings")
        return True
        
    def merge_data(self, wine_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge wine features with ratings data.
        
        Args:
            wine_df: Wine features dataframe
            ratings_df: Wine ratings dataframe
            
        Returns:
            Merged dataframe with wine features and average ratings
        """
        # Calculate average rating per wine
        avg_ratings = ratings_df.groupby('WineID')['Rating'].agg(['mean', 'count']).reset_index()
        avg_ratings.columns = ['WineID', 'AvgRating', 'RatingCount']
        
        # Merge with wine features
        merged_df = wine_df.merge(avg_ratings, on='WineID', how='inner')
        
        logger.info(f"Merged data: {merged_df.shape[0]} wines with ratings")
        return merged_df
        
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess wine features for model training.
        
        Args:
            df: Merged dataframe with wine features and ratings
            
        Returns:
            Preprocessed dataframe
        """
        processed_df = df.copy()
        
        # Parse list-like string columns
        list_columns = ['Grapes', 'Harmonize', 'Vintages']
        for col in list_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(self._parse_list_string)
                
        # Extract numerical features from list columns
        if 'Grapes' in processed_df.columns:
            processed_df['GrapeCount'] = processed_df['Grapes'].apply(len)
            
        if 'Harmonize' in processed_df.columns:
            processed_df['HarmonizeCount'] = processed_df['Harmonize'].apply(len)
            
        if 'Vintages' in processed_df.columns:
            processed_df['VintageCount'] = processed_df['Vintages'].apply(len)
            
        # Handle missing values in ABV
        if 'ABV' in processed_df.columns:
            processed_df['ABV'] = processed_df['ABV'].fillna(processed_df['ABV'].median())
            
        # Identify categorical and numerical features
        self.categorical_features = ['Type', 'Elaborate', 'Body', 'Acidity', 'Country']
        self.numerical_features = ['ABV', 'GrapeCount', 'HarmonizeCount', 'VintageCount', 'RatingCount']
        
        # Filter features that actually exist in the dataframe
        self.categorical_features = [f for f in self.categorical_features if f in processed_df.columns]
        self.numerical_features = [f for f in self.numerical_features if f in processed_df.columns]
        
        logger.info(f"Identified {len(self.categorical_features)} categorical and {len(self.numerical_features)} numerical features")
        
        return processed_df
        
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Dataframe with categorical features
            fit: Whether to fit encoders (True for training, False for test)
            
        Returns:
            Dataframe with encoded categorical features
        """
        encoded_df = df.copy()
        
        for feature in self.categorical_features:
            if feature in encoded_df.columns:
                if fit:
                    # Fit and transform for training data
                    self.label_encoders[feature] = LabelEncoder()
                    encoded_df[feature] = self.label_encoders[feature].fit_transform(
                        encoded_df[feature].astype(str)
                    )
                else:
                    # Transform only for test data
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(self.label_encoders[feature].classes_)
                        encoded_df[feature] = encoded_df[feature].astype(str)
                        encoded_df[feature] = encoded_df[feature].apply(
                            lambda x: x if x in unique_values else 'unknown'
                        )
                        
                        # Add 'unknown' to encoder if not present
                        if 'unknown' not in unique_values:
                            self.label_encoders[feature].classes_ = np.append(
                                self.label_encoders[feature].classes_, 'unknown'
                            )
                            
                        encoded_df[feature] = self.label_encoders[feature].transform(encoded_df[feature])
                        
        return encoded_df
        
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X: Feature dataframe
            fit: Whether to fit scaler (True for training, False for test)
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled
        
    def create_train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        stratify_bins: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            X: Feature dataframe
            y: Target series
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility
            stratify_bins: Number of bins for stratifying continuous target
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Create stratification bins for continuous target
        y_binned = pd.cut(y, bins=stratify_bins, labels=False)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y_binned
        )
        
        # Create new bins for remaining data
        y_temp_binned = pd.cut(y_temp, bins=stratify_bins, labels=False)
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state,
            stratify=y_temp_binned
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def load_and_preprocess(
        self, 
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Complete data loading and preprocessing pipeline.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set  
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing processed data splits and metadata
        """
        # Load raw data
        wine_df, ratings_df = self.load_raw_data()
        
        # Validate data
        self.validate_data(wine_df, ratings_df)
        
        # Merge datasets
        merged_df = self.merge_data(wine_df, ratings_df)
        
        # Preprocess features
        processed_df = self.preprocess_features(merged_df)
        
        # Encode categorical features
        encoded_df = self.encode_categorical_features(processed_df, fit=True)
        
        # Prepare features and target
        feature_cols = self.categorical_features + self.numerical_features
        X = encoded_df[feature_cols]
        y = encoded_df['AvgRating']
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Create train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_test_split(
            X, y, test_size, val_size, random_state
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_val_scaled = self.scale_features(X_val, fit=False)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'raw_data': {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test
            }
        }
        
    def _parse_list_string(self, list_str: str) -> List[str]:
        """
        Parse string representation of list into actual list.
        
        Args:
            list_str: String representation of list
            
        Returns:
            Parsed list or empty list if parsing fails
        """
        if pd.isna(list_str) or list_str == '':
            return []
            
        try:
            # Try to parse as literal list
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                return parsed
            else:
                return [str(parsed)]
        except (ValueError, SyntaxError):
            # If parsing fails, return as single item list
            return [str(list_str)]