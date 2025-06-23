#!/usr/bin/env python3
"""
Data Cleaning Module for Bank Customer Churn Analysis

This module handles:
- Missing value imputation
- Data type conversions
- Categorical encoding
- Feature scaling
- Data validation

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

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Handles data cleaning and preprocessing for the churn analysis project.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.interim_data_dir = self.project_root / 'data' / 'interim'
        self.processed_data_dir = self.project_root / 'data' / 'processed'
        self.models_dir = self.project_root / 'models'
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define column categories
        self.numeric_features = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'
        ]
        
        self.categorical_features = ['Geography', 'Gender']
        
        self.binary_features = ['HasCrCard', 'IsActiveMember']
        
        self.target_column = 'Exited'
        
        self.id_columns = ['RowNumber', 'CustomerId']
        
        self.pii_columns = ['Surname']  # To be removed
        
        # Initialize preprocessors
        self.preprocessor = None
        self.label_encoders = {}
        
    def load_interim_data(self, filename: str = "churn_raw.parquet") -> pd.DataFrame:
        """
        Load data from interim directory.
        
        Args:
            filename: Name of the parquet file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.interim_data_dir / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Interim data file not found: {file_path}")
        
        try:
            logger.info(f"Loading interim data from: {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading parquet file: {str(e)}")
            raise
    
    def remove_pii_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove PII columns from the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with PII removed
        """
        df_clean = df.copy()
        
        # Drop CustomerId and Surname columns
        pii_columns_to_remove = ['CustomerId', 'Surname']
        
        for col in pii_columns_to_remove:
            if col in df_clean.columns:
                logger.info(f"Removing PII column: {col}")
                df_clean = df_clean.drop(columns=[col])
        
        return df_clean
    
    def remove_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop personally-identifiable columns so tests can run.
        """
        cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
        return df.drop(columns=cols_to_drop, errors="ignore")
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict with missing value analysis
        """
        missing_analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'missing_counts': {},
            'missing_percentages': {},
            'total_missing': 0,
            'missing_patterns': {},
            'recommendations': []
        }
        
        # Calculate missing values per column
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        for col in df.columns:
            if missing_counts[col] > 0:
                missing_analysis['missing_counts'][col] = int(missing_counts[col])
                missing_analysis['missing_percentages'][col] = float(missing_percentages[col])
        
        # Calculate total missing values
        missing_analysis['total_missing'] = int(missing_counts.sum())
        
        # Analyze missing patterns
        if missing_analysis['missing_counts']:
            # Find rows with any missing values
            rows_with_missing = df.isnull().any(axis=1).sum()
            missing_analysis['rows_with_missing'] = int(rows_with_missing)
            missing_analysis['rows_with_missing_percentage'] = float((rows_with_missing / len(df)) * 100)
            
            # Generate recommendations
            for col in missing_analysis['missing_counts'].keys():
                percentage = missing_analysis['missing_percentages'][col]
                if percentage > 50:
                    missing_analysis['recommendations'].append(f"Consider dropping column {col} (>{percentage:.1f}% missing)")
                elif percentage > 20:
                    missing_analysis['recommendations'].append(f"Investigate {col} missing pattern (>{percentage:.1f}% missing)")
                else:
                    missing_analysis['recommendations'].append(f"Impute {col} ({percentage:.1f}% missing)")
        
        logger.info(f"Missing value analysis completed. {len(missing_analysis['missing_counts'])} columns have missing values")
        return missing_analysis
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to business rules.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_clean = df.copy()
        
        # Drop rows with missing target variable
        if self.target_column in df_clean.columns:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=[self.target_column])
            dropped_rows = initial_rows - len(df_clean)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing target variable")
        
        # Drop rows with missing ID columns
        for col in self.id_columns:
            if col in df_clean.columns:
                initial_rows = len(df_clean)
                df_clean = df_clean.dropna(subset=[col])
                dropped_rows = initial_rows - len(df_clean)
                if dropped_rows > 0:
                    logger.info(f"Dropped {dropped_rows} rows with missing {col}")
        
        # Handle missing values in numeric features
        for col in self.numeric_features:
            if col in df_clean.columns and df_clean[col].isnull().any():
                if col == 'Balance':
                    # Assume missing balance means closed account (0)
                    df_clean[col] = df_clean[col].fillna(0)
                    logger.info(f"Filled missing {col} with 0 (closed account assumption)")
                elif col in ['CreditScore', 'EstimatedSalary']:
                    # Impute with median by geography if available
                    if 'Geography' in df_clean.columns:
                        df_clean[col] = df_clean.groupby('Geography')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    logger.info(f"Filled missing {col} with median (by geography if available)")
                else:
                    # Default: median imputation
                    median_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_value)
                    logger.info(f"Filled missing {col} with median: {median_value}")
        
        # Handle missing values in categorical features
        for col in self.categorical_features:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
                logger.info(f"Filled missing {col} with mode: {mode_value}")
        
        # Handle missing values in binary features
        for col in self.binary_features:
            if col in df_clean.columns and df_clean[col].isnull().any():
                if col == 'IsActiveMember':
                    # Conservative assumption: inactive if missing
                    df_clean[col] = df_clean[col].fillna(0)
                    logger.info(f"Filled missing {col} with 0 (conservative assumption)")
                else:
                    # Use mode for other binary features
                    mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                    df_clean[col] = df_clean[col].fillna(mode_value)
                    logger.info(f"Filled missing {col} with mode: {mode_value}")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in numeric features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict with outlier analysis
        """
        outlier_analysis = {
            'timestamp': datetime.now().isoformat(),
            'method': 'IQR and Z-score',
            'outliers_by_column': {},
            'total_outlier_rows': 0
        }
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in self.numeric_features:
            if col in df.columns:
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
                # Z-score method (for extreme outliers)
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                z_outliers = z_scores > 3
                
                # Combine both methods
                column_outliers = iqr_outliers | z_outliers
                outlier_mask |= column_outliers
                
                outlier_analysis['outliers_by_column'][col] = {
                    'iqr_outliers': int(iqr_outliers.sum()),
                    'z_score_outliers': int(z_outliers.sum()),
                    'total_outliers': int(column_outliers.sum()),
                    'percentage': float((column_outliers.sum() / len(df)) * 100),
                    'bounds': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound)
                    }
                }
        
        outlier_analysis['total_outlier_rows'] = int(outlier_mask.sum())
        outlier_analysis['outlier_percentage'] = float((outlier_mask.sum() / len(df)) * 100)
        
        logger.info(f"Outlier detection completed. {outlier_analysis['total_outlier_rows']} rows with outliers")
        return outlier_analysis
    
    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline for features.
        
        Args:
            df: Input dataframe for fitting
            
        Returns:
            ColumnTransformer: Fitted preprocessing pipeline
        """
        # Define transformers for different feature types
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Get available features
        available_numeric = [col for col in self.numeric_features if col in df.columns]
        available_categorical = [col for col in self.categorical_features if col in df.columns]
        available_binary = [col for col in self.binary_features if col in df.columns]
        
        # Create column transformer
        transformers = []
        
        if available_numeric:
            transformers.append(('num', numeric_transformer, available_numeric))
        
        if available_categorical:
            transformers.append(('cat', categorical_transformer, available_categorical))
        
        if available_binary:
            transformers.append(('bin', binary_transformer, available_binary))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop other columns
        )
        
        logger.info(f"Created preprocessing pipeline with {len(transformers)} transformers")
        logger.info(f"Numeric features: {available_numeric}")
        logger.info(f"Categorical features: {available_categorical}")
        logger.info(f"Binary features: {available_binary}")
        
        return preprocessor
    
    def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnecessary columns from the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with unnecessary columns removed
        """
        df_clean = df.copy()
        
        # Remove PII columns
        columns_to_remove = self.pii_columns + self.id_columns
        
        for col in columns_to_remove:
            if col in df_clean.columns:
                logger.info(f"Removing unnecessary column: {col}")
                df_clean = df_clean.drop(columns=[col])
        
        logger.info(f"Removed {len([col for col in columns_to_remove if col in df.columns])} unnecessary columns")
        return df_clean
    
    def scale_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (scaled_dataframe, fitted_scaler)
        """
        df_scaled = df.copy()
        scaler = StandardScaler()
        
        # Get numeric columns to scale
        numeric_cols = [col for col in self.numeric_features if col in df.columns]
        
        if numeric_cols:
            # Fit and transform numeric features
            df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logger.info(f"Scaled {len(numeric_cols)} numeric features: {numeric_cols}")
        else:
            logger.warning("No numeric features found to scale")
        
        return df_scaled, scaler
    
    def detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            series: Input pandas Series
            
        Returns:
            pd.Series: Outlier values from the input series
        """
        # IQR method with more sensitive threshold
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 0.5 * IQR  # More sensitive threshold
        upper_bound = Q3 + 0.5 * IQR  # More sensitive threshold
        
        # Find outliers
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outliers = series[outlier_mask]
        
        logger.info(f"IQR outlier detection completed. Found {len(outliers)} outliers")
        return outliers
    
    def detect_outliers_zscore(self, series: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Input pandas Series
            threshold: Z-score threshold for outlier detection
            
        Returns:
            pd.Series: Outlier values from the input series
        """
        # Modified Z-score method using median and MAD for better outlier detection
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return pd.Series([], dtype=series.dtype)
            
        # Use median and median absolute deviation for more robust outlier detection
        median = clean_series.median()
        mad = np.median(np.abs(clean_series - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.std(clean_series)
            
        if mad == 0:
            return pd.Series([], dtype=series.dtype)
            
        # Modified z-scores
        modified_z_scores = 0.6745 * (clean_series - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        outliers = clean_series[outlier_mask]
        
        logger.info(f"Z-score outlier detection completed. Found {len(outliers)} outliers")
        return outliers
    
    def clean_data(self) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Run full data cleaning pipeline and return processed data with scaler.
        
        Returns:
            Tuple of (processed_dataframe, fitted_scaler)
        """
        try:
            logger.info("Starting clean_data pipeline")
            
            # Load interim data
            df = self.load_interim_data()
            
            # Remove PII and unnecessary columns
            df = self.remove_pii_data(df)
            df = self.remove_unnecessary_columns(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Apply preprocessing to get processed data
            df_processed, _ = self.apply_preprocessing(df, fit_preprocessor=True)
            
            # Scale features separately to return scaler
            df_scaled, scaler = self.scale_features(df_processed)
            
            logger.info(f"Data cleaning completed. Final shape: {df_scaled.shape}")
            return df_scaled, scaler
             
        except Exception as e:
            logger.error(f"clean_data pipeline failed: {str(e)}")
            raise
    
    def apply_preprocessing(self, df: pd.DataFrame, fit_preprocessor: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply preprocessing to the dataset.
        
        Args:
            df: Input dataframe
            fit_preprocessor: Whether to fit the preprocessor
            
        Returns:
            Tuple of (processed_dataframe, feature_array)
        """
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in [self.target_column] + self.id_columns]
        X = df[feature_columns].copy()
        
        if fit_preprocessor:
            # Create and fit preprocessor
            self.preprocessor = self.create_preprocessing_pipeline(X)
            X_processed = self.preprocessor.fit_transform(X)
            
            # Save preprocessor
            preprocessor_path = self.models_dir / 'preprocessor.pkl'
            joblib.dump(self.preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to: {preprocessor_path}")
        else:
            # Use existing preprocessor
            if self.preprocessor is None:
                preprocessor_path = self.models_dir / 'preprocessor.pkl'
                if preprocessor_path.exists():
                    self.preprocessor = joblib.load(preprocessor_path)
                    logger.info(f"Preprocessor loaded from: {preprocessor_path}")
                else:
                    raise ValueError("No preprocessor found. Set fit_preprocessor=True")
            
            X_processed = self.preprocessor.transform(X)
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        # Create processed dataframe
        df_processed = pd.DataFrame(X_processed, columns=feature_names, index=df.index)
        
        # Add back ID columns and target
        for col in self.id_columns:
            if col in df.columns:
                df_processed[col] = df[col].values
        
        if self.target_column in df.columns:
            df_processed[self.target_column] = df[self.target_column].values
        
        logger.info(f"Preprocessing completed. Output shape: {df_processed.shape}")
        return df_processed, X_processed
    
    def _get_feature_names(self) -> List[str]:
        """
        Get feature names from the fitted preprocessor.
        
        Returns:
            List of feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted")
        
        feature_names = []
        
        # Get feature names from each transformer
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend([f"{col}_scaled" for col in columns])
            elif name == 'cat':
                # Get feature names from one-hot encoder
                onehot_encoder = transformer.named_steps['onehot']
                cat_features = onehot_encoder.get_feature_names_out(columns)
                feature_names.extend(cat_features)
            elif name == 'bin':
                feature_names.extend(columns)
        
        return feature_names
    
    def validate_processed_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the processed dataset.
        
        Args:
            df: Processed dataframe
            
        Returns:
            Tuple of (is_valid, issues_list)
        """
        issues = []
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"Found {missing_count} missing values")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        infinite_count = 0
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            infinite_count += inf_count
            if inf_count > 0:
                issues.append(f"Column {col} has {inf_count} infinite values")
        
        # Check target distribution if target column exists
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts(normalize=True)
            min_class_ratio = target_dist.min()
            if min_class_ratio < 0.1:
                issues.append(f"Severe class imbalance detected: {min_class_ratio:.3f}")
        
        # Overall validation
        is_valid = len(issues) == 0
        
        logger.info(f"Data validation completed. Valid: {is_valid}")
        if issues:
            logger.warning(f"Issues found: {issues}")
        
        return is_valid, issues
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "churn_cleaned.parquet") -> Path:
        """
        Save processed data to the processed directory.
        
        Args:
            df: Processed dataframe
            filename: Output filename
            
        Returns:
            Path: Path to the saved file
        """
        try:
            output_path = self.processed_data_dir / filename
            df.to_parquet(output_path, index=False)
            logger.info(f"Processed data saved to: {output_path}")
            logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def run_full_pipeline(self) -> bool:
        """
        Run the complete data cleaning pipeline.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Starting data cleaning pipeline")
            
            # Step 1: Load interim data
            df = self.load_interim_data()
            
            # Step 2: Remove PII data
            df = self.remove_pii_data(df)
            
            # Step 3: Analyze missing values
            missing_analysis = self.analyze_missing_values(df)
            
            # Save missing value analysis
            import json
            with open(self.processed_data_dir / 'missing_value_analysis.json', 'w') as f:
                json.dump(missing_analysis, f, indent=2)
            
            # Step 4: Handle missing values
            df = self.handle_missing_values(df)
            
            # Step 5: Detect outliers
            outlier_analysis = self.detect_outliers(df)
            
            # Save outlier analysis
            with open(self.processed_data_dir / 'outlier_analysis.json', 'w') as f:
                json.dump(outlier_analysis, f, indent=2)
            
            # Step 6: Apply preprocessing
            df_processed, X_processed = self.apply_preprocessing(df, fit_preprocessor=True)
            
            # Step 7: Validate processed data
            is_valid, validation_issues = self.validate_processed_data(df_processed)
            
            # Save validation report
            validation_report = {
                'is_valid': is_valid,
                'issues': validation_issues
            }
            with open(self.processed_data_dir / 'processing_validation.json', 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            if not is_valid:
                logger.error(f"Data validation failed after processing: {validation_issues}")
                return None
            
            # Step 8: Save processed data
            self.save_processed_data(df_processed, "churn_cleaned.parquet")
            
            logger.info("Data cleaning pipeline completed successfully")
            logger.info(f"Final dataset shape: {df_processed.shape}")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Data cleaning pipeline failed: {str(e)}")
            return None

def main():
    """
    Main function to run the data cleaning pipeline.
    """
    # Initialize and run data cleaner
    cleaner = DataCleaner()
    
    success = cleaner.run_full_pipeline()
    
    if success:
        print("\n✅ Data cleaning completed successfully!")
        print(f"📁 Processed data: {cleaner.processed_data_dir}")
        print(f"🔧 Preprocessor saved: {cleaner.models_dir}")
    else:
        print("\n❌ Data cleaning failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()