#!/usr/bin/env python3
"""
Feature Engineering Module for Bank Customer Churn Analysis

This module handles:
- Creation of new features
- Feature interactions
- Domain-specific transformations
- Feature selection and validation

Author: Data Science Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, List
from datetime import datetime
import joblib

# Statistical and ML imports
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles feature engineering for the churn analysis project.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.processed_data_dir = self.project_root / 'data' / 'processed'
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.models_dir = self.project_root / 'models'
        self.reports_dir = self.project_root / 'reports'
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering configuration
        self.target_column = 'Exited'
        self.id_columns = ['RowNumber', 'CustomerId']
        
        # Store feature importance and selection results
        self.feature_importance = {}
        self.selected_features = []
        
    def load_processed_data(self, filename: str = "churn_cleaned.parquet") -> pd.DataFrame:
        """
        Load processed data from the processed directory.
        
        Args:
            filename: Name of the parquet file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.processed_data_dir / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        try:
            logger.info(f"Loading processed data from: {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading parquet file: {str(e)}")
            raise
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio and interaction features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new ratio features
        """
        df_features = df.copy()
        
        logger.info("Creating ratio and interaction features")
        
        # Tenure-Balance Ratio (key feature from requirements)
        if 'Tenure' in df.columns and 'Balance' in df.columns:
            df_features['TenureBalanceRatio'] = df['Balance'] / (df['Tenure'] + 1)  # +1 to avoid division by zero
            logger.info("Created TenureBalanceRatio feature")
        
        # Balance-to-Salary Ratio
        if 'EstimatedSalary' in df.columns and 'Balance' in df.columns:
            df_features['BalanceToSalary'] = df['Balance'] / (df['EstimatedSalary'] + 1)
            logger.info("Created BalanceToSalary feature")
        
        # Credit Score to Age ratio (financial maturity indicator)
        if 'CreditScore' in df.columns and 'Age' in df.columns:
            df_features['CreditScoreToAge'] = df['CreditScore'] / df['Age']
            logger.info("Created CreditScoreToAge feature")
        
        # Tenure to Age ratio (relationship longevity indicator)
        if 'Tenure' in df.columns and 'Age' in df.columns:
            df_features['TenureToAge'] = df['Tenure'] / df['Age']
            logger.info("Created TenureToAge feature")
        
        # Products per Tenure (product adoption rate)
        if 'NumOfProducts' in df.columns and 'Tenure' in df.columns:
            df_features['ProductsPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
            logger.info("Created ProductsPerTenure feature")
        
        # Balance per Product (average product value)
        if 'Balance' in df.columns and 'NumOfProducts' in df.columns:
            df_features['BalancePerProduct'] = df['Balance'] / df['NumOfProducts']
            logger.info("Created BalancePerProduct feature")
        
        # Age-Tenure Difference (how long before joining)
        if 'Age' in df.columns and 'Tenure' in df.columns:
            df_features['AgeWhenJoined'] = df['Age'] - df['Tenure']
            logger.info("Created AgeWhenJoined feature")
        
        return df_features
    
    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned/categorical versions of continuous features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with binned features
        """
        df_features = df.copy()
        
        logger.info("Creating binned features")
        
        # Age groups
        if 'Age' in df.columns:
            age_bins = pd.cut(
                df['Age'], 
                bins=[0, 30, 40, 50, 60, 100], 
                labels=['Young', 'Adult', 'MiddleAge', 'Senior', 'Elder']
            )
            # Convert to numeric for modeling
            df_features['AgeGroup'] = age_bins.cat.codes
            logger.info("Created AgeGroup features")
        
        # Credit Score bins
        if 'CreditScore' in df.columns:
            credit_bins = pd.cut(
                df['CreditScore'],
                bins=[0, 580, 670, 740, 850],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
            df_features['CreditScoreGroup'] = credit_bins.cat.codes
            logger.info("Created CreditScoreGroup features")
        
        # Balance bins
        if 'Balance' in df.columns:
            # Use quantile-based binning for balance
            try:
                balance_bins = pd.qcut(
                    df['Balance'],
                    q=5,
                    labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'],
                    duplicates='drop'
                )
                df_features['BalanceGroup'] = balance_bins.cat.codes
                logger.info("Created BalanceGroup features")
            except ValueError as e:
                # Handle case where qcut can't create 5 bins due to duplicate values
                logger.warning(f"Could not create 5 balance bins: {e}. Using 3 bins instead.")
                balance_bins = pd.qcut(
                    df['Balance'],
                    q=3,
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
                df_features['BalanceGroup'] = balance_bins.cat.codes
        
        # Salary bins
        if 'EstimatedSalary' in df.columns:
            try:
                df_features['SalaryGroup'] = pd.qcut(
                    df['EstimatedSalary'],
                    q=4,
                    labels=['Low', 'Medium', 'High', 'VeryHigh'],
                    duplicates='drop'
                )
                df_features['SalaryGroup_num'] = df_features['SalaryGroup'].cat.codes
                logger.info("Created SalaryGroup features")
            except ValueError as e:
                # Handle case where qcut can't create 4 bins due to duplicate values
                logger.warning(f"Could not create 4 salary bins: {e}. Using 3 bins instead.")
                df_features['SalaryGroup'] = pd.qcut(
                    df['EstimatedSalary'],
                    q=3,
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
                df_features['SalaryGroup_num'] = df_features['SalaryGroup'].cat.codes
        
        return df_features
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral and engagement features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with behavioral features
        """
        df_features = df.copy()
        
        logger.info("Creating behavioral features")
        
        # Customer engagement score (composite feature)
        engagement_features = []
        
        if 'IsActiveMember' in df.columns:
            engagement_features.append('IsActiveMember')
        
        if 'HasCrCard' in df.columns:
            engagement_features.append('HasCrCard')
        
        if 'NumOfProducts' in df.columns:
            # Normalize number of products (0-1 scale)
            df_features['NumOfProducts_norm'] = (df['NumOfProducts'] - 1) / 3  # Assuming 1-4 products
            engagement_features.append('NumOfProducts_norm')
        
        if engagement_features:
            df_features['EngagementScore'] = df_features[engagement_features].mean(axis=1)
            logger.info("Created EngagementScore feature")
        
        # Zero balance indicator
        if 'Balance' in df.columns:
            df_features['IsZeroBalance'] = (df['Balance'] == 0)
            logger.info("Created IsZeroBalance feature")
        
        # High-value customer indicator
        if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
            balance_threshold = df['Balance'].quantile(0.75)
            salary_threshold = df['EstimatedSalary'].quantile(0.75)
            
            df_features['IsHighValue'] = (
                (df['Balance'] >= balance_threshold) | 
                (df['EstimatedSalary'] >= salary_threshold)
            )
            logger.info("Created IsHighValue feature")
        
        # Risk profile (based on age and credit score)
        if 'Age' in df.columns and 'CreditScore' in df.columns:
            # Younger customers with low credit scores are higher risk
            df_features['RiskProfile'] = np.where(
                (df['Age'] < 35) & (df['CreditScore'] < 650), 1,
                np.where(
                    (df['Age'] > 60) | (df['CreditScore'] > 750), -1, 0
                )
            )
            logger.info("Created RiskProfile feature")
        
        # Product diversity (whether customer has multiple products)
        if 'NumOfProducts' in df.columns:
            df_features['HasMultipleProducts'] = (df['NumOfProducts'] > 1).astype(int)
            df_features['ProductDiversity'] = df['NumOfProducts']  # Same as NumOfProducts for diversity measure
            logger.info("Created HasMultipleProducts and ProductDiversity features")
        
        # Tenure-based features
        if 'Tenure' in df.columns:
            df_features['IsNewCustomer'] = (df['Tenure'] <= 1).astype(int)
            df_features['IsLoyalCustomer'] = (df['Tenure'] >= 7).astype(int)
            logger.info("Created tenure-based features")
        
        return df_features
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with age group features
        """
        df_features = df.copy()
        
        if 'Age' in df.columns:
            # Create age groups
            age_groups = pd.cut(
                df['Age'], 
                bins=[0, 30, 40, 50, 60, 100], 
                labels=['Young', 'Adult', 'MiddleAge', 'Senior', 'Elder']
            )
            
            # Create one-hot encoded age group features
            age_group_dummies = pd.get_dummies(age_groups, prefix='AgeGroup', dummy_na=False)
            df_features = pd.concat([df_features, age_group_dummies], axis=1)
            
            # Also keep the original categorical and numeric versions
            df_features['AgeGroup'] = age_groups
            df_features['AgeGroup_num'] = age_groups.cat.codes
            logger.info("Created age group features with one-hot encoding")
        
        return df_features
    
    def create_balance_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create balance category features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with balance category features
        """
        df_features = df.copy()
        
        if 'Balance' in df.columns:
            # Create balance categories with special handling for zero balances
            balance_categories = pd.Series(index=df.index, dtype='object')
            
            # Handle zero balances first
            zero_balance_mask = df['Balance'] == 0
            balance_categories[zero_balance_mask] = 'Zero'
            
            # For non-zero balances, use quantile-based binning
            non_zero_balances = df['Balance'][~zero_balance_mask]
            if len(non_zero_balances) > 0:
                try:
                    non_zero_categories = pd.qcut(
                        non_zero_balances,
                        q=4,
                        labels=['Low', 'Medium', 'High', 'VeryHigh'],
                        duplicates='drop'
                    )
                    balance_categories[~zero_balance_mask] = non_zero_categories.astype(str)
                except ValueError as e:
                    # Handle case where qcut can't create 4 bins due to duplicate values
                    logger.warning(f"Could not create 4 balance categories: {e}. Using 3 categories instead.")
                    non_zero_categories = pd.qcut(
                        non_zero_balances,
                        q=3,
                        labels=['Low', 'Medium', 'High'],
                        duplicates='drop'
                    )
                    balance_categories[~zero_balance_mask] = non_zero_categories.astype(str)
            
            # Convert to categorical after all assignments
            balance_categories = balance_categories.astype('category')
            
            # Create one-hot encoded balance category features
            balance_cat_dummies = pd.get_dummies(balance_categories, prefix='BalanceCategory', dummy_na=False)
            df_features = pd.concat([df_features, balance_cat_dummies], axis=1)
            
            # Also keep the original categorical and numeric versions
            df_features['BalanceCategory'] = balance_categories
            df_features['BalanceCategory_num'] = balance_categories.cat.codes
            logger.info("Created balance category features with one-hot encoding")
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df_features = df.copy()
        
        logger.info("Creating interaction features")
        
        # Age * Tenure interaction
        if 'Age' in df.columns and 'Tenure' in df.columns:
            df_features['Age_x_Tenure'] = df['Age'] * df['Tenure']
        
        # CreditScore * Balance interaction
        if 'CreditScore' in df.columns and 'Balance' in df.columns:
            df_features['CreditScore_x_Balance'] = df['CreditScore'] * df['Balance']
        
        # IsActiveMember * NumOfProducts interaction
        if 'IsActiveMember' in df.columns and 'NumOfProducts' in df.columns:
            df_features['IsActiveMember_x_NumOfProducts'] = df['IsActiveMember'] * df['NumOfProducts']
        
        # Additional interactions
        if 'Age' in df.columns and 'Balance' in df.columns:
            df_features['Age_Balance_interaction'] = df['Age'] * df['Balance']
        
        if 'EstimatedSalary' in df.columns and 'HasCrCard' in df.columns:
            df_features['Salary_CrCard_interaction'] = df['EstimatedSalary'] * df['HasCrCard']
        
        logger.info("Created interaction features")
        return df_features
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 3) -> pd.DataFrame:
        """
        Create polynomial features for key numeric variables.
        
        Args:
            df: Input dataframe
            degree: Polynomial degree
            
        Returns:
            pd.DataFrame: Dataframe with polynomial features
        """
        df_features = df.copy()
        
        logger.info(f"Creating polynomial features (degree={degree})")
        
        # Key numeric features for polynomial transformation
        poly_features = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure']
        
        # Degree name mapping
        degree_names = {2: 'squared', 3: 'cubed'}
        
        for feature in poly_features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    if d in degree_names:
                        df_features[f'{feature}_{degree_names[d]}'] = df[feature] ** d
                    else:
                        df_features[f'{feature}_poly_{d}'] = df[feature] ** d
        
        logger.info("Created polynomial features")
        return df_features
    
    def select_features_univariate(self, df: pd.DataFrame, k: int = 10) -> Tuple[pd.DataFrame, Any]:
        """
        Select features using univariate statistical tests.
        
        Args:
            df: Input dataframe
            k: Number of features to select
            
        Returns:
            Tuple of (selected_dataframe, selector_object)
        """
        if self.target_column not in df.columns:
            logger.error("Target column not found in dataframe")
            return df, None
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col not in [self.target_column] + self.id_columns]
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[self.target_column]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Create and fit selector
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create result dataframe
        result_df = df.copy()
        
        # Keep only selected features plus ID and target columns
        keep_columns = selected_features + self.id_columns + [self.target_column]
        keep_columns = [col for col in keep_columns if col in result_df.columns]
        result_df = result_df[keep_columns]
        
        logger.info(f"Selected {len(selected_features)} features using univariate selection")
        return result_df, selector
    
    def select_features_importance(self, df: pd.DataFrame, n_features: int = 15) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Select features based on importance scores.
        
        Args:
            df: Input dataframe
            n_features: Number of features to select
            
        Returns:
            Tuple of (selected_dataframe, importance_dict)
        """
        # Calculate feature importance if not already done
        if not self.feature_importance:
            self.calculate_feature_importance(df)
        
        # Get top N features
        top_features = list(self.feature_importance.keys())[:n_features]
        
        # Create result dataframe with selected features
        keep_columns = top_features + self.id_columns + [self.target_column]
        keep_columns = [col for col in keep_columns if col in df.columns]
        result_df = df[keep_columns]
        
        # Create importance dict for selected features
        importance_dict = {feature: self.feature_importance[feature] 
                          for feature in top_features if feature in self.feature_importance}
        
        logger.info(f"Selected {len(top_features)} features based on importance scores")
        return result_df, importance_dict
    
    def engineer_features(self) -> bool:
        """
        Run the complete feature engineering pipeline and save results.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Starting feature engineering pipeline")
            
            # Step 1: Load processed data
            df_original = self.load_processed_data()
            
            # Step 2: Create all feature types
            df_features = self.create_ratio_features(df_original)
            df_features = self.create_age_groups(df_features)
            df_features = self.create_balance_categories(df_features)
            df_features = self.create_interaction_features(df_features)
            df_features = self.create_polynomial_features(df_features)
            df_features = self.create_binned_features(df_features)
            df_features = self.create_behavioral_features(df_features)
            df_features = self.create_statistical_features(df_features)
            
            # Step 3: Calculate feature importance
            self.calculate_feature_importance(df_features)
            
            # Step 4: Select features using different methods
            df_univariate, selector = self.select_features_univariate(df_features, k=20)
            df_importance, importance_dict = self.select_features_importance(df_features, n_features=25)
            
            # Step 5: Create feature summary
            summary = self.create_feature_summary(df_original, df_features)
            
            # Step 6: Save all results
            import json
            
            # Save feature importance
            with open(self.reports_dir / 'feature_importance.json', 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            # Save feature summary
            with open(self.reports_dir / 'feature_engineering_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save selected features list
            with open(self.models_dir / 'selected_features.json', 'w') as f:
                json.dump(list(df_importance.columns), f, indent=2)
            
            # Save engineered datasets
            if not self.save_engineered_data(df_features, "churn_features_all.parquet"):
                logger.error("Failed to save all engineered features")
                return False
            
            if not self.save_engineered_data(df_importance, "churn_features_selected.parquet"):
                logger.error("Failed to save selected features")
                return False
            
            logger.info("Feature engineering pipeline completed successfully")
            logger.info(f"Original features: {len(df_original.columns)}")
            logger.info(f"Total features after engineering: {len(df_features.columns)}")
            logger.info(f"Selected features for modeling: {len(df_importance.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {str(e)}")
            return False
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical and mathematical features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with statistical features
        """
        df_features = df.copy()
        
        logger.info("Creating statistical features")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in self.id_columns + [self.target_column]]
        
        # Z-scores for outlier detection
        for col in numeric_columns[:5]:  # Limit to avoid too many features
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_features[f'{col}_zscore'] = (df[col] - mean_val) / std_val
        
        # Log transformations for skewed features
        skewed_features = []
        for col in numeric_columns:
            if col in df.columns and df[col].min() > 0:  # Only for positive values
                skewness = stats.skew(df[col])
                if abs(skewness) > 1:  # Highly skewed
                    df_features[f'{col}_log'] = np.log1p(df[col])
                    skewed_features.append(col)
        
        if skewed_features:
            logger.info(f"Created log features for skewed columns: {skewed_features}")
        
        # Polynomial features for key relationships
        if 'Age' in df.columns:
            df_features['Age_squared'] = df['Age'] ** 2
        
        if 'CreditScore' in df.columns:
            df_features['CreditScore_squared'] = df['CreditScore'] ** 2
        
        return df_features
    
    def calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance using multiple methods.
        
        Args:
            df: Input dataframe with features and target
            
        Returns:
            Dict with feature importance scores
        """
        if self.target_column not in df.columns:
            logger.error("Target column not found in dataframe")
            return {}
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col not in [self.target_column] + self.id_columns]
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[self.target_column]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        importance_scores = {}
        
        logger.info("Calculating feature importance")
        
        # Method 1: Mutual Information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            for i, col in enumerate(X.columns):
                importance_scores[f'{col}_mi'] = mi_scores[i]
            logger.info("Calculated mutual information scores")
        except Exception as e:
            logger.warning(f"Failed to calculate mutual information: {e}")
        
        # Method 2: F-statistic
        try:
            f_scores, _ = f_classif(X, y)
            for i, col in enumerate(X.columns):
                importance_scores[f'{col}_f_stat'] = f_scores[i]
            logger.info("Calculated F-statistic scores")
        except Exception as e:
            logger.warning(f"Failed to calculate F-statistics: {e}")
        
        # Method 3: Random Forest Feature Importance
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            for i, col in enumerate(X.columns):
                importance_scores[f'{col}_rf'] = rf.feature_importances_[i]
            logger.info("Calculated Random Forest importance scores")
        except Exception as e:
            logger.warning(f"Failed to calculate Random Forest importance: {e}")
        
        # Aggregate scores
        feature_importance = {}
        for col in X.columns:
            scores = []
            if f'{col}_mi' in importance_scores:
                scores.append(importance_scores[f'{col}_mi'])
            if f'{col}_f_stat' in importance_scores:
                # Normalize F-statistic
                max_f = max([v for k, v in importance_scores.items() if '_f_stat' in k])
                scores.append(importance_scores[f'{col}_f_stat'] / max_f)
            if f'{col}_rf' in importance_scores:
                scores.append(importance_scores[f'{col}_rf'])
            
            if scores:
                feature_importance[col] = np.mean(scores)
        
        # Sort by importance
        self.feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Feature importance calculated for {len(self.feature_importance)} features")
        return self.feature_importance
    
    def select_top_features(self, df: pd.DataFrame, n_features: int = 20) -> List[str]:
        """
        Select top N features based on importance scores.
        
        Args:
            df: Input dataframe
            n_features: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        if not self.feature_importance:
            self.calculate_feature_importance(df)
        
        # Get top N features (excluding ID and target columns from the count)
        feature_cols = [col for col in self.feature_importance.keys() 
                       if col not in self.id_columns + [self.target_column]]
        top_features = feature_cols[:n_features]
        
        # Always include ID columns and target if present
        selected_features = []
        for col in self.id_columns + [self.target_column]:
            if col in df.columns:
                selected_features.append(col)
        
        selected_features.extend(top_features)
        
        # Remove duplicates while preserving order
        self.selected_features = list(dict.fromkeys(selected_features))
        
        logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features[:10]}...")
        return self.selected_features
    
    def create_feature_summary(self, df_original: pd.DataFrame, df_engineered: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a summary of feature engineering results.
        
        Args:
            df_original: Original dataframe
            df_engineered: Dataframe with engineered features
            
        Returns:
            Dict with feature engineering summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'original_features': len(df_original.columns),
            'engineered_features': len(df_engineered.columns),
            'new_features_created': len(df_engineered.columns) - len(df_original.columns),
            'feature_categories': {
                'ratio_features': [],
                'binned_features': [],
                'behavioral_features': [],
                'statistical_features': []
            },
            'top_features': [],
            'feature_importance_available': bool(self.feature_importance)
        }
        
        # Identify new features by category
        new_columns = set(df_engineered.columns) - set(df_original.columns)
        
        for col in new_columns:
            if 'Ratio' in col or 'Per' in col:
                summary['feature_categories']['ratio_features'].append(col)
            elif 'Group' in col or 'Bin' in col:
                summary['feature_categories']['binned_features'].append(col)
            elif any(keyword in col for keyword in ['Engagement', 'Risk', 'Customer', 'Multiple']):
                summary['feature_categories']['behavioral_features'].append(col)
            elif any(keyword in col for keyword in ['zscore', 'log', 'squared']):
                summary['feature_categories']['statistical_features'].append(col)
        
        # Add top features if available
        if self.feature_importance:
            summary['top_features'] = list(self.feature_importance.keys())[:10]
        
        logger.info(f"Feature engineering summary: {summary['new_features_created']} new features created")
        return summary
    
    def save_engineered_data(self, df: pd.DataFrame, filename: str = "churn_features.parquet") -> bool:
        """
        Save engineered features to the processed directory.
        
        Args:
            df: Dataframe with engineered features
            filename: Output filename
            
        Returns:
            bool: Success status
        """
        try:
            # Cast float16 columns to float32 before saving
            df_to_save = df.copy()
            float16_cols = df_to_save.select_dtypes(include=['float16']).columns
            if len(float16_cols) > 0:
                df_to_save[float16_cols] = df_to_save[float16_cols].astype(np.float32)
                logger.info(f"Cast {len(float16_cols)} float16 columns to float32 before saving")
            
            output_path = self.processed_data_dir / filename
            df_to_save.to_parquet(output_path, index=False)
            logger.info(f"Engineered features saved to: {output_path}")
            logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"Error saving engineered features: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        Run the complete feature engineering pipeline.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Starting feature engineering pipeline")
            
            # Step 1: Load processed data
            df_original = self.load_processed_data()
            
            # Step 2: Create ratio features
            df_features = self.create_ratio_features(df_original)
            
            # Step 3: Create binned features
            df_features = self.create_binned_features(df_features)
            
            # Step 4: Create behavioral features
            df_features = self.create_behavioral_features(df_features)
            
            # Step 5: Create statistical features
            df_features = self.create_statistical_features(df_features)
            
            # Cast float16 columns to float32 before saving
            float16_cols = df_features.select_dtypes(include=['float16']).columns
            if len(float16_cols) > 0:
                df_features[float16_cols] = df_features[float16_cols].astype('float32')
                logger.info(f"Cast {len(float16_cols)} float16 columns to float32")
            
            # Step 6: Calculate feature importance
            self.calculate_feature_importance(df_features)
            
            # Step 7: Select top features
            selected_features = self.select_top_features(df_features, n_features=25)
            df_selected = df_features[selected_features]
            
            # Step 8: Create feature summary
            summary = self.create_feature_summary(df_original, df_features)
            
            # Save feature engineering artifacts
            import json
            
            # Save feature importance
            with open(self.reports_dir / 'feature_importance.json', 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            # Save feature summary
            with open(self.reports_dir / 'feature_engineering_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save selected features list
            with open(self.models_dir / 'selected_features.json', 'w') as f:
                json.dump(selected_features, f, indent=2)
            
            # Step 9: Save engineered data
            if not self.save_engineered_data(df_features, "churn_features_all.parquet"):
                logger.error("Failed to save all engineered features")
                return False
            
            if not self.save_engineered_data(df_selected, "churn_features_selected.parquet"):
                logger.error("Failed to save selected features")
                return False
            
            logger.info("Feature engineering pipeline completed successfully")
            logger.info(f"Original features: {len(df_original.columns)}")
            logger.info(f"Total features after engineering: {len(df_features.columns)}")
            logger.info(f"Selected features for modeling: {len(df_selected.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {str(e)}")
            return False

def main():
    """
    Main function to run the feature engineering pipeline.
    """
    # Initialize and run feature engineer
    engineer = FeatureEngineer()
    
    success = engineer.run_full_pipeline()
    
    if success:
        print("\n✅ Feature engineering completed successfully!")
        print(f"📁 Engineered features: {engineer.processed_data_dir}")
        print(f"📊 Reports: {engineer.reports_dir}")
        print(f"🎯 Selected features: {engineer.models_dir}")
        
        # Display top features
        if engineer.feature_importance:
            print("\n🏆 Top 10 Features:")
            for i, (feature, importance) in enumerate(list(engineer.feature_importance.items())[:10], 1):
                print(f"  {i:2d}. {feature:<25} ({importance:.4f})")
    else:
        print("\n❌ Feature engineering failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()