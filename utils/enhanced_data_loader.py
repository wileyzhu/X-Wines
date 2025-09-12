#!/usr/bin/env python3
"""
Simple enhanced data loader for wine interpretability analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import ast
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_list_string(list_str):
    """Parse string representation of list into actual list."""
    if pd.isna(list_str) or list_str == '':
        return []
    try:
        parsed = ast.literal_eval(list_str)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        else:
            return [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(list_str)]


from pathlib import Path
from typing import Dict, Any

def create_enhanced_wine_features(
    data_path1: str = str(Path(__file__).parent / "Dataset/last/XWines_Test_100_wines.csv"),
    data_path2: str = str(Path(__file__).parent / "Dataset/last/XWines_Test_1k_ratings.csv"),
    standardize: bool = True,
    imputation_strategy: str = 'median'
) -> Dict[str, Any]:
    """
    Create enhanced wine dataset with additional categorical features.
    
    Args:
        data_path1: Path to wine data CSV (or merged data if data_path2 is None)
        data_path2: Path to ratings CSV (optional)
        standardize: Whether to standardize numerical features
        imputation_strategy: Strategy for numerical imputation ('median', 'mean', 'knn')
        
    Returns:
        Enhanced dataset dictionary with proper preprocessing
    """
    logger.info("Loading wine data...")
    wine_df = pd.read_csv(data_path1)
    ratings_df = pd.read_csv(data_path2)
    ratings_df = ratings_df.rename(columns={'Rating': 'quality'})
    print(ratings_df)
    df = pd.merge(wine_df, ratings_df, on='WineID', how='inner')
    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Create enhanced features dataframe
    enhanced_df = df.copy()
    
    # === CATEGORICAL FEATURES ===
    categorical_features = []
    
    # 1. Basic wine characteristics (already encoded)
    if 'wine_type_encoded' in df.columns:
        categorical_features.append('wine_type_encoded')
    if 'body_encoded' in df.columns:
        categorical_features.append('body_encoded')
    if 'acidity_encoded' in df.columns:
        categorical_features.append('acidity_encoded')
    
    # 2. Original categorical columns
    original_categorical = ['Type', 'Elaborate', 'Body', 'Acidity', 'Country', 'RegionName']
    for col in original_categorical:
        if col in df.columns:
            categorical_features.append(col)
    
    # 3. Derived categorical features from complex columns
    derived_categorical = []
    
    # Extract primary grape variety
    if 'Grapes' in df.columns:
        logger.info("Processing Grapes column...")
        try:
            enhanced_df['grape_varieties'] = enhanced_df['Grapes'].apply(parse_list_string)
            enhanced_df['primary_grape'] = enhanced_df['grape_varieties'].apply(
                lambda x: x[0] if len(x) > 0 else 'Unknown'
            )
            enhanced_df['is_blend'] = enhanced_df['grape_varieties'].apply(lambda x: len(x) > 1)
            derived_categorical.extend(['primary_grape', 'is_blend'])
        except Exception as e:
            logger.warning(f"Could not process Grapes column: {e}")
    
    # Extract primary food pairing
    if 'Harmonize' in df.columns:
        logger.info("Processing Harmonize column...")
        try:
            enhanced_df['food_pairings'] = enhanced_df['Harmonize'].apply(parse_list_string)
            enhanced_df['primary_pairing'] = enhanced_df['food_pairings'].apply(
                lambda x: x[0] if len(x) > 0 else 'Unknown'
            )
            derived_categorical.append('primary_pairing')
        except Exception as e:
            logger.warning(f"Could not process Harmonize column: {e}")
    
    # Create ABV categories
    if 'ABV' in df.columns:
        logger.info("Creating ABV categories...")
        enhanced_df['abv_category'] = pd.cut(
            enhanced_df['ABV'], 
            bins=[0, 12, 14, 16, float('inf')], 
            labels=['Low_ABV', 'Medium_ABV', 'High_ABV', 'Very_High_ABV']
        ).astype(str)
        derived_categorical.append('abv_category')
    
    # Create region popularity categories
    if 'RegionName' in df.columns:
        logger.info("Creating region categories...")
        region_counts = enhanced_df['RegionName'].value_counts()
        enhanced_df['region_popularity'] = enhanced_df['RegionName'].map(
            lambda x: 'Popular_Region' if region_counts.get(x, 0) > 2 else 'Rare_Region'
        )
        derived_categorical.append('region_popularity')
    
    # Create country categories
    if 'Country' in df.columns:
        logger.info("Creating country categories...")
        country_counts = enhanced_df['Country'].value_counts()
        enhanced_df['country_category'] = enhanced_df['Country'].map(
            lambda x: 'Major_Producer' if country_counts.get(x, 0) > 5 else 'Minor_Producer'
        )
        derived_categorical.append('country_category')
    
    # Add derived categorical features
    categorical_features.extend(derived_categorical)
    
    # === NUMERICAL FEATURES ===
    numerical_features = []
    
    # 1. Basic numerical features
    basic_numerical = ['ABV', 'rating_count']
    for col in basic_numerical:
        if col in df.columns:
            numerical_features.append(col)
    
    # 2. Derived numerical features
    derived_numerical = []
    
    # Grape variety count
    if 'grape_varieties' in enhanced_df.columns:
        enhanced_df['grape_count'] = enhanced_df['grape_varieties'].apply(len)
        derived_numerical.append('grape_count')
    
    # Food pairing count
    if 'food_pairings' in enhanced_df.columns:
        enhanced_df['pairing_count'] = enhanced_df['food_pairings'].apply(len)
        derived_numerical.append('pairing_count')
    
    # Vintage information
    if 'Vintages' in df.columns:
        logger.info("Processing Vintages column...")
        enhanced_df['vintages_list'] = enhanced_df['Vintages'].apply(parse_list_string)
        enhanced_df['vintage_count'] = enhanced_df['vintages_list'].apply(len)
        
        def get_vintage_range(vintages):
            try:
                numeric_vintages = [int(v) for v in vintages if str(v).isdigit()]
                if len(numeric_vintages) > 1:
                    return max(numeric_vintages) - min(numeric_vintages)
                return 0
            except:
                return 0
        
        enhanced_df['vintage_range'] = enhanced_df['vintages_list'].apply(get_vintage_range)
        derived_numerical.extend(['vintage_count', 'vintage_range'])
    
    # Add derived numerical features
    numerical_features.extend(derived_numerical)
    
    # === PREPARE FINAL FEATURE SET ===
    
    # Filter features that actually exist
    available_categorical = [f for f in categorical_features if f in enhanced_df.columns]
    available_numerical = [f for f in numerical_features if f in enhanced_df.columns]
    
    logger.info(f"Available categorical features ({len(available_categorical)}): {available_categorical}")
    logger.info(f"Available numerical features ({len(available_numerical)}): {available_numerical}")
    
    # Create feature dataframe
    all_features = available_categorical + available_numerical
    feature_df = enhanced_df[all_features].copy()
    
    # Handle missing values with proper imputation
    logger.info("Applying imputation and preprocessing...")
    
    # Categorical imputation
    for col in available_categorical:
        feature_df[col] = feature_df[col].fillna('Unknown').astype(str)
        logger.info(f"Categorical imputation for {col}: filled {feature_df[col].isna().sum()} missing values")
    
    # Numerical imputation with configurable strategy
    if imputation_strategy == 'knn':
        numerical_imputer = KNNImputer(n_neighbors=5)
        logger.info("Using KNN imputation for numerical features")
    else:
        numerical_imputer = SimpleImputer(strategy=imputation_strategy)
        logger.info(f"Using {imputation_strategy} imputation for numerical features")
    
    for col in available_numerical:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        missing_count = feature_df[col].isna().sum()
        if missing_count > 0:
            feature_df[col] = numerical_imputer.fit_transform(feature_df[[col]]).ravel()
            logger.info(f"Numerical imputation for {col}: filled {missing_count} missing values with {imputation_strategy}")
    
    # Encode categorical features
    label_encoders = {}
    encoded_df = feature_df.copy()
    
    for col in available_categorical:
        logger.info(f"Encoding {col}...")
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(feature_df[col])
        label_encoders[col] = le
        logger.info(f"  {col}: {len(le.classes_)} categories")
    
    # Prepare target
    target_column = 'quality'
    if target_column not in enhanced_df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    X = encoded_df
    y = enhanced_df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature standardization - conditional and only for numerical features
    scaler = None
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if standardize and available_numerical:
        logger.info("Applying feature standardization...")
        scaler = StandardScaler()
        
        # Get indices of numerical features
        numerical_indices = [X_train.columns.get_loc(col) for col in available_numerical]
        
        # Fit scaler on training data and transform both train and test
        X_train_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X_train.iloc[:, numerical_indices])
        X_test_scaled.iloc[:, numerical_indices] = scaler.transform(X_test.iloc[:, numerical_indices])
        
        logger.info(f"Standardized {len(available_numerical)} numerical features")
        
        # Log standardization info
        for i, col in enumerate(available_numerical):
            mean_val = scaler.mean_[i]
            std_val = scaler.scale_[i]
            logger.info(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}")
    elif standardize:
        logger.info("No numerical features to standardize")
    else:
        logger.info("Standardization disabled")
    
    # Create feature mapping for interpretability
    feature_mapping = {}
    for col in available_categorical:
        feature_mapping[col] = {
            'type': 'categorical',
            'classes': list(label_encoders[col].classes_),
            'n_classes': len(label_encoders[col].classes_)
        }
    
    for col in available_numerical:
        feature_mapping[col] = {
            'type': 'numerical',
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean())
        }
    
    result = {
        # Standardized data for ML models
        'X_train': X_train_scaled.values,
        'X_test': X_test_scaled.values,
        'y_train': y_train.values,
        'y_test': y_test.values,
        
        # Feature information
        'feature_names': all_features,
        'categorical_features': available_categorical,
        'numerical_features': available_numerical,
        'derived_categorical': derived_categorical,
        'derived_numerical': derived_numerical,
        'feature_mapping': feature_mapping,
        
        # Preprocessing objects for future use
        'label_encoders': label_encoders,
        'scaler': scaler,
        'numerical_imputer': numerical_imputer if available_numerical else None,
        
        # Raw data (before standardization) for interpretability
        'raw_data': {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        },
        
        # Dataset information
        'data_info': {
            'n_samples': len(enhanced_df),
            'n_features': len(all_features),
            'n_categorical': len(available_categorical),
            'n_numerical': len(available_numerical),
            'n_derived_categorical': len(derived_categorical),
            'n_derived_numerical': len(derived_numerical),
            'preprocessing': {
                'categorical_imputation': 'mode (Unknown)',
                'numerical_imputation': imputation_strategy,
                'standardization': 'StandardScaler on numerical features' if standardize else 'None',
                'categorical_encoding': 'LabelEncoder'
            },
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
    }
    
    return result


def create_enhanced_wine_features_from_merged(
    data_path1: str = str(Path(__file__).parent / "Dataset/last/XWines_Test_100_wines.csv"),
    data_path2: str = str(Path(__file__).parent / "Dataset/last/XWines_Test_1k_ratings.csv"),
    standardize: bool = True,
    imputation_strategy: str = 'median'
) -> Dict[str, Any]:
    """
    Create enhanced wine dataset with additional categorical features.
    
    Args:
        data_path1: Path to wine data CSV (or merged data if data_path2 is None)
        data_path2: Path to ratings CSV (optional)
        standardize: Whether to standardize numerical features
        imputation_strategy: Strategy for numerical imputation ('median', 'mean', 'knn')
        
    Returns:
        Enhanced dataset dictionary with proper preprocessing
    """
    logger.info("Loading wine data...")
    wine_df = pd.read_csv(data_path1)
    ratings_df = pd.read_csv(data_path2)
    ratings_df = ratings_df.rename(columns={'Rating': 'quality'})
    print(ratings_df)
    df = pd.merge(wine_df, ratings_df, on='WineID', how='inner')
    """
    Create enhanced wine dataset from already merged data file.
    
    Args:
        data_path: Path to merged wine data CSV
        standardize: Whether to standardize numerical features
        imputation_strategy: Strategy for numerical imputation
        
    Returns:
        Enhanced dataset dictionary with proper preprocessing
    """
    
    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Create enhanced features dataframe
    enhanced_df = df.copy()
    
    # === CATEGORICAL FEATURES ===
    categorical_features = []
    
    # 1. Basic wine characteristics (already encoded)
    if 'wine_type_encoded' in df.columns:
        categorical_features.append('wine_type_encoded')
    if 'body_encoded' in df.columns:
        categorical_features.append('body_encoded')
    if 'acidity_encoded' in df.columns:
        categorical_features.append('acidity_encoded')
    
    # 2. Original categorical columns
    original_categorical = ['Type', 'Elaborate', 'Body', 'Acidity', 'Country', 'RegionName']
    for col in original_categorical:
        if col in df.columns:
            categorical_features.append(col)
    
    # 3. Derived categorical features from complex columns
    derived_categorical = []
    
    # Extract primary grape variety
    if 'Grapes' in df.columns:
        logger.info("Processing Grapes column...")
        try:
            enhanced_df['grape_varieties'] = enhanced_df['Grapes'].apply(parse_list_string)
            enhanced_df['primary_grape'] = enhanced_df['grape_varieties'].apply(
                lambda x: x[0] if len(x) > 0 else 'Unknown'
            )
            enhanced_df['is_blend'] = enhanced_df['grape_varieties'].apply(lambda x: len(x) > 1)
            derived_categorical.extend(['primary_grape', 'is_blend'])
        except Exception as e:
            logger.warning(f"Could not process Grapes column: {e}")
    
    # Extract primary food pairing
    if 'Harmonize' in df.columns:
        logger.info("Processing Harmonize column...")
        try:
            enhanced_df['food_pairings'] = enhanced_df['Harmonize'].apply(parse_list_string)
            enhanced_df['primary_pairing'] = enhanced_df['food_pairings'].apply(
                lambda x: x[0] if len(x) > 0 else 'Unknown'
            )
            derived_categorical.append('primary_pairing')
        except Exception as e:
            logger.warning(f"Could not process Harmonize column: {e}")
    
    # Create ABV categories
    if 'ABV' in df.columns:
        logger.info("Creating ABV categories...")
        enhanced_df['abv_category'] = pd.cut(
            enhanced_df['ABV'], 
            bins=[0, 12, 14, 16, float('inf')], 
            labels=['Low_ABV', 'Medium_ABV', 'High_ABV', 'Very_High_ABV']
        ).astype(str)
        derived_categorical.append('abv_category')
    
    # Create region popularity categories
    if 'RegionName' in df.columns:
        logger.info("Creating region categories...")
        region_counts = enhanced_df['RegionName'].value_counts()
        enhanced_df['region_popularity'] = enhanced_df['RegionName'].map(
            lambda x: 'Popular_Region' if region_counts.get(x, 0) > 2 else 'Rare_Region'
        )
        derived_categorical.append('region_popularity')
    
    # Create country categories
    if 'Country' in df.columns:
        logger.info("Creating country categories...")
        country_counts = enhanced_df['Country'].value_counts()
        enhanced_df['country_category'] = enhanced_df['Country'].map(
            lambda x: 'Major_Producer' if country_counts.get(x, 0) > 5 else 'Minor_Producer'
        )
        derived_categorical.append('country_category')
    
    # Add derived categorical features
    categorical_features.extend(derived_categorical)
    
    # === NUMERICAL FEATURES ===
    numerical_features = []
    
    # 1. Basic numerical features
    basic_numerical = ['ABV', 'rating_count']
    for col in basic_numerical:
        if col in df.columns:
            numerical_features.append(col)
    
    # 2. Derived numerical features
    derived_numerical = []
    
    # Grape variety count
    if 'grape_varieties' in enhanced_df.columns:
        enhanced_df['grape_count'] = enhanced_df['grape_varieties'].apply(len)
        derived_numerical.append('grape_count')
    
    # Food pairing count
    if 'food_pairings' in enhanced_df.columns:
        enhanced_df['pairing_count'] = enhanced_df['food_pairings'].apply(len)
        derived_numerical.append('pairing_count')
    
    # Vintage information
    if 'Vintages' in df.columns:
        logger.info("Processing Vintages column...")
        enhanced_df['vintages_list'] = enhanced_df['Vintages'].apply(parse_list_string)
        enhanced_df['vintage_count'] = enhanced_df['vintages_list'].apply(len)
        
        def get_vintage_range(vintages):
            try:
                numeric_vintages = [int(v) for v in vintages if str(v).isdigit()]
                if len(numeric_vintages) > 1:
                    return max(numeric_vintages) - min(numeric_vintages)
                return 0
            except:
                return 0
        
        enhanced_df['vintage_range'] = enhanced_df['vintages_list'].apply(get_vintage_range)
        derived_numerical.extend(['vintage_count', 'vintage_range'])
    
    # Add derived numerical features
    numerical_features.extend(derived_numerical)
    
    # === PREPARE FINAL FEATURE SET ===
    
    # Filter features that actually exist
    available_categorical = [f for f in categorical_features if f in enhanced_df.columns]
    available_numerical = [f for f in numerical_features if f in enhanced_df.columns]
    
    logger.info(f"Available categorical features ({len(available_categorical)}): {available_categorical}")
    logger.info(f"Available numerical features ({len(available_numerical)}): {available_numerical}")
    
    # Create feature dataframe
    all_features = available_categorical + available_numerical
    feature_df = enhanced_df[all_features].copy()
    
    # Handle missing values with proper imputation
    logger.info("Applying imputation and preprocessing...")
    
    # Categorical imputation
    for col in available_categorical:
        feature_df[col] = feature_df[col].fillna('Unknown').astype(str)
        logger.info(f"Categorical imputation for {col}: filled {feature_df[col].isna().sum()} missing values")
    
    # Numerical imputation with configurable strategy
    if imputation_strategy == 'knn':
        numerical_imputer = KNNImputer(n_neighbors=5)
        logger.info("Using KNN imputation for numerical features")
    else:
        numerical_imputer = SimpleImputer(strategy=imputation_strategy)
        logger.info(f"Using {imputation_strategy} imputation for numerical features")
    
    for col in available_numerical:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        missing_count = feature_df[col].isna().sum()
        if missing_count > 0:
            feature_df[col] = numerical_imputer.fit_transform(feature_df[[col]]).ravel()
            logger.info(f"Numerical imputation for {col}: filled {missing_count} missing values with {imputation_strategy}")
    
    # Encode categorical features
    label_encoders = {}
    encoded_df = feature_df.copy()
    
    for col in available_categorical:
        logger.info(f"Encoding {col}...")
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(feature_df[col])
        label_encoders[col] = le
        logger.info(f"  {col}: {len(le.classes_)} categories")
    
    # Prepare target
    target_column = 'quality'
    if target_column not in enhanced_df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    X = encoded_df
    y = enhanced_df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature standardization - conditional and only for numerical features
    scaler = None
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if standardize and available_numerical:
        logger.info("Applying feature standardization...")
        scaler = StandardScaler()
        
        # Get indices of numerical features
        numerical_indices = [X_train.columns.get_loc(col) for col in available_numerical]
        
        # Fit scaler on training data and transform both train and test
        X_train_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X_train.iloc[:, numerical_indices])
        X_test_scaled.iloc[:, numerical_indices] = scaler.transform(X_test.iloc[:, numerical_indices])
        
        logger.info(f"Standardized {len(available_numerical)} numerical features")
        
        # Log standardization info
        for i, col in enumerate(available_numerical):
            mean_val = scaler.mean_[i]
            std_val = scaler.scale_[i]
            logger.info(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}")
    elif standardize:
        logger.info("No numerical features to standardize")
    else:
        logger.info("Standardization disabled")
    
    # Create feature mapping for interpretability
    feature_mapping = {}
    for col in available_categorical:
        feature_mapping[col] = {
            'type': 'categorical',
            'classes': list(label_encoders[col].classes_),
            'n_classes': len(label_encoders[col].classes_)
        }
    
    for col in available_numerical:
        feature_mapping[col] = {
            'type': 'numerical',
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean())
        }
    
    result = {
        # Standardized data for ML models
        'X_train': X_train_scaled.values,
        'X_test': X_test_scaled.values,
        'y_train': y_train.values,
        'y_test': y_test.values,
        
        # Feature information
        'feature_names': all_features,
        'categorical_features': available_categorical,
        'numerical_features': available_numerical,
        'derived_categorical': derived_categorical,
        'derived_numerical': derived_numerical,
        'feature_mapping': feature_mapping,
        
        # Preprocessing objects for future use
        'label_encoders': label_encoders,
        'scaler': scaler,
        'numerical_imputer': numerical_imputer if available_numerical else None,
        
        # Raw data (before standardization) for interpretability
        'raw_data': {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        },
        
        # Dataset information
        'data_info': {
            'n_samples': len(enhanced_df),
            'n_features': len(all_features),
            'n_categorical': len(available_categorical),
            'n_numerical': len(available_numerical),
            'n_derived_categorical': len(derived_categorical),
            'n_derived_numerical': len(derived_numerical),
            'preprocessing': {
                'categorical_imputation': 'mode (Unknown)',
                'numerical_imputation': imputation_strategy,
                'standardization': 'StandardScaler on numerical features' if standardize else 'None',
                'categorical_encoding': 'LabelEncoder'
            },
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
    }
    
    return result


if __name__ == "__main__":
    print("🚀 Testing Enhanced Wine Data Loader with Preprocessing")
    print("=" * 60)
    
    try:
        # Test with standardization
        print("\n📊 Test 1: With Standardization and Median Imputation")
        print("-" * 50)
        data = create_enhanced_wine_features_from_merged(
            standardize=True, 
            imputation_strategy='median'
        )
        
        print(f"✅ Enhanced dataset created successfully!")
        print(f"\n📈 Dataset Summary:")
        info = data['data_info']
        print(f"  - Total samples: {info['n_samples']}")
        print(f"  - Total features: {info['n_features']}")
        print(f"  - Categorical features: {info['n_categorical']}")
        print(f"  - Numerical features: {info['n_numerical']}")
        
        print(f"\n🔧 Preprocessing Applied:")
        preprocessing = info['preprocessing']
        for key, value in preprocessing.items():
            print(f"  - {key}: {value}")
        
        print(f"\n📊 Standardization Check (first 3 numerical features):")
        for i, col in enumerate(data['numerical_features'][:3]):
            train_data = data['X_train'][:, data['feature_names'].index(col)]
            print(f"  - {col}: mean={train_data.mean():.3f}, std={train_data.std():.3f}")
        
        # Test without standardization
        print(f"\n📊 Test 2: Without Standardization")
        print("-" * 40)
        data_no_std = create_enhanced_wine_features_from_merged(
            standardize=False, 
            imputation_strategy='mean'
        )
        
        print(f"✅ Dataset without standardization created!")
        print(f"🔧 Preprocessing: {data_no_std['data_info']['preprocessing']['standardization']}")
        
        # Save both versions
        enhanced_df = pd.DataFrame(data['X_train'], columns=data['feature_names'])
        enhanced_df['quality'] = data['y_train']
        enhanced_df.to_csv('enhanced_wine_dataset_standardized.csv', index=False)
        
        enhanced_df_no_std = pd.DataFrame(data_no_std['X_train'], columns=data_no_std['feature_names'])
        enhanced_df_no_std['quality'] = data_no_std['y_train']
        enhanced_df_no_std.to_csv('enhanced_wine_dataset_raw.csv', index=False)
        
        print(f"\n💾 Datasets saved:")
        print(f"  - Standardized: enhanced_wine_dataset_standardized.csv")
        print(f"  - Raw: enhanced_wine_dataset_raw.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()