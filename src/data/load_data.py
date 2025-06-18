#!/usr/bin/env python3
"""
Data Loading Module for Bank Customer Churn Analysis

This module handles:
- Kaggle API data acquisition
- Basic schema validation
- Data quality checks
- Raw data storage

Author: Data Science Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
import zipfile
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'data_loading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data acquisition and initial validation for the churn analysis project.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.interim_data_dir = self.data_dir / 'interim'
        self.interim_dir = self.interim_data_dir  # Alias for test compatibility
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected schema
        self.expected_schema = {
            'RowNumber': 'int64',
            'CustomerId': 'int64',
            'Surname': 'object',
            'CreditScore': 'int64',
            'Geography': 'object',
            'Gender': 'object',
            'Age': 'int64',
            'Tenure': 'int64',
            'Balance': 'float64',
            'NumOfProducts': 'int64',
            'HasCrCard': 'int64',
            'IsActiveMember': 'int64',
            'EstimatedSalary': 'float64',
            'Exited': 'int64'
        }
        
        self.expected_columns = list(self.expected_schema.keys())
        
    def download_from_kaggle(self, dataset_name: str = "mashlyn/customer-churn-modeling", 
                           filename: str = "Churn_Modelling.csv", 
                           max_retries: int = 3) -> Path:
        """
        Download dataset from Kaggle using the API with retry logic and credential handling.
        
        Args:
            dataset_name: Kaggle dataset identifier (default: mashlyn/customer-churn-modeling)
            filename: Expected filename to download (default: Churn_Modelling.csv)
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Path: Path to the downloaded file (even if download fails, returns expected path)
        """
        import time
        import json
        from pathlib import Path
        
        expected_file_path = self.raw_data_dir / filename
        
        # If file already exists, return its path
        if expected_file_path.exists():
            logger.info(f"File {filename} already exists at {expected_file_path}")
            return expected_file_path
        
        # Setup Kaggle credentials
        if not self._setup_kaggle_credentials():
            logger.warning("Failed to setup Kaggle credentials. Returning expected file path.")
            return expected_file_path
        
        # Attempt download with retry logic
        for attempt in range(max_retries):
            try:
                import kaggle
                
                logger.info(f"Downloading dataset: {dataset_name} (attempt {attempt + 1}/{max_retries})")
                
                # Download to raw data directory
                kaggle.api.dataset_download_files(
                    dataset_name, 
                    path=str(self.raw_data_dir),
                    unzip=False  # We'll handle unzipping manually
                )
                
                # Check for downloaded files and handle ZIP extraction
                downloaded_files = list(self.raw_data_dir.glob("*"))
                zip_files = [f for f in downloaded_files if f.suffix.lower() == '.zip']
                
                if zip_files:
                    logger.info(f"Found ZIP file(s): {[f.name for f in zip_files]}")
                    for zip_file in zip_files:
                        self._extract_and_cleanup_zip(zip_file)
                
                # Verify the expected file exists
                if expected_file_path.exists():
                    logger.info(f"Dataset downloaded successfully: {expected_file_path}")
                    return expected_file_path
                else:
                    # Check if file exists with different name
                    csv_files = list(self.raw_data_dir.glob("*.csv"))
                    if csv_files:
                        actual_file = csv_files[0]
                        logger.info(f"Found CSV file with different name: {actual_file.name}")
                        # Rename to expected filename
                        actual_file.rename(expected_file_path)
                        logger.info(f"Renamed to: {expected_file_path.name}")
                        return expected_file_path
                    else:
                        raise FileNotFoundError(f"Expected file {filename} not found after download")
                        
            except ImportError:
                logger.error("Kaggle package not installed. Install with: pip install kaggle")
                break
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} download attempts failed")
        
        # If all attempts failed, log warning and return expected path
        logger.warning(f"Failed to download {filename} from Kaggle. Returning expected path for pipeline continuity.")
        return expected_file_path
    
    def _setup_kaggle_credentials(self) -> bool:
        """
        Setup Kaggle API credentials from project kaggle.json or environment variables.
        
        Returns:
            bool: True if credentials were successfully configured
        """
        import json
        
        # Check if credentials are already configured
        try:
            import kaggle
            # Try to authenticate - this will raise an exception if not configured
            kaggle.api.authenticate()
            logger.info("Kaggle credentials already configured")
            return True
        except:
            pass
        
        # Method 1: Try to read from project root kaggle.json
        project_kaggle_json = self.project_root / "kaggle.json"
        if project_kaggle_json.exists():
            try:
                with open(project_kaggle_json, 'r') as f:
                    creds = json.load(f)
                
                # Set environment variables
                os.environ['KAGGLE_USERNAME'] = creds.get('username', '')
                os.environ['KAGGLE_KEY'] = creds.get('key', '')
                
                logger.info(f"Loaded Kaggle credentials from {project_kaggle_json}")
                return True
            except Exception as e:
                logger.warning(f"Failed to read kaggle.json from project root: {e}")
        
        # Method 2: Check environment variables
        kaggle_username = os.environ.get('KAGGLE_USERNAME', 'kathanparagshah')
        kaggle_key = os.environ.get('KAGGLE_KEY', '')
        
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
            logger.info("Using Kaggle credentials from environment variables")
            return True
        
        # Method 3: Try default ~/.kaggle/kaggle.json location
        home_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if home_kaggle_json.exists():
            logger.info("Using Kaggle credentials from ~/.kaggle/kaggle.json")
            return True
        
        logger.error("No Kaggle credentials found. Please provide credentials via:")
        logger.error("1. kaggle.json in project root")
        logger.error("2. Environment variables KAGGLE_USERNAME and KAGGLE_KEY")
        logger.error("3. ~/.kaggle/kaggle.json")
        return False
    
    def _extract_and_cleanup_zip(self, zip_path: Path) -> None:
        """
        Extract ZIP file and clean up the archive.
        
        Args:
            zip_path: Path to the ZIP file
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)
            
            logger.info(f"Extracted ZIP file: {zip_path.name}")
            
            # Delete the ZIP file
            zip_path.unlink()
            logger.info(f"Cleaned up ZIP file: {zip_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to extract ZIP file {zip_path}: {e}")
    
    def load_csv_data(self, filename: str = "Churn_Modelling.csv") -> pd.DataFrame:
        """
        Load CSV data from raw directory or direct path.
        
        Args:
            filename: Name of the CSV file or absolute/relative path
            
        Returns:
            pd.DataFrame: Loaded data
        """
        # Check if filename is an existing absolute or relative path
        file_path = Path(filename)
        if not file_path.exists():
            # Fall back to raw_data_dir/filename
            file_path = self.raw_data_dir / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            logger.info(f"Loading data from: {file_path}")
            # Try UTF-8 first, then fall back to latin-1
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning("UTF-8 encoding failed, trying latin-1")
                df = pd.read_csv(file_path, encoding='latin-1')
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data using the default churn file name.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        return self.load_csv_data("Churn_Modelling.csv")
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate dataframe schema against expected structure.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, issues)
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_columns': [],
            'extra_columns': [],
            'dtype_mismatches': [],
            'missing_values': {},
            'duplicate_rows': 0,
            'is_valid': True
        }
        
        # Check for missing columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            validation_report['missing_columns'] = list(missing_cols)
            validation_report['is_valid'] = False
            logger.error(f"Missing columns: {missing_cols}")
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(self.expected_columns)
        if extra_cols:
            validation_report['extra_columns'] = list(extra_cols)
            logger.warning(f"Extra columns found: {extra_cols}")
        
        # Check data types for existing columns
        for col in self.expected_columns:
            if col in df.columns:
                expected_dtype = self.expected_schema[col]
                actual_dtype = str(df[col].dtype)
                
                # Allow some flexibility in numeric types
                if not self._dtypes_compatible(expected_dtype, actual_dtype):
                    validation_report['dtype_mismatches'].append({
                        'column': col,
                        'expected': expected_dtype,
                        'actual': actual_dtype
                    })
                    validation_report['is_valid'] = False
                    logger.warning(f"Data type mismatch in {col}: expected {expected_dtype}, got {actual_dtype}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        validation_report['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Check for duplicates
        if 'CustomerId' in df.columns:
            duplicate_customers = df['CustomerId'].duplicated().sum()
            validation_report['duplicate_rows'] = duplicate_customers
            if duplicate_customers > 0:
                validation_report['is_valid'] = False
                logger.error(f"Found {duplicate_customers} duplicate customer IDs")
        
        # Collect issues for return
        issues = []
        if validation_report['missing_columns']:
            issues.extend([f"Missing column: {col}" for col in validation_report['missing_columns']])
        if validation_report['dtype_mismatches']:
            issues.extend([f"Type mismatch in {item['column']}: expected {item['expected']}, got {item['actual']}" for item in validation_report['dtype_mismatches']])
        if validation_report['duplicate_rows'] > 0:
            issues.append(f"Found {validation_report['duplicate_rows']} duplicate rows")
        
        # Log validation summary
        if validation_report['is_valid']:
            logger.info("Schema validation passed")
        else:
            logger.error("Schema validation failed")
        
        return validation_report['is_valid'], issues
    
    def _dtypes_compatible(self, expected: str, actual: str) -> bool:
        """
        Check if data types are compatible (allowing for some flexibility).
        """
        # Exact match
        if expected == actual:
            return True
        
        # Integer compatibility (allow float64 for int64 due to missing values)
        if expected == 'int64' and actual in ['int32', 'int64', 'Int64', 'float64']:
            return True
        
        # Float compatibility
        if expected == 'float64' and actual in ['float32', 'float64', 'Float64']:
            return True
        
        # Object/string compatibility
        if expected == 'object' and actual in ['object', 'string']:
            return True
        
        return False
    
    def perform_sanity_checks(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Perform basic sanity checks on the data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, issues)
        """
        checks = {
            'timestamp': datetime.now().isoformat(),
            'row_count_check': True,
            'value_range_checks': {},
            'categorical_checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Row count sanity check - only flag completely empty DataFrames
        if len(df) == 0:
            checks['errors'].append("DataFrame is completely empty")
        elif len(df) > 50000:
            checks['warnings'].append(f"High row count: {len(df)} rows")
        
        # Value range checks - properly detect out-of-range values
        if 'CreditScore' in df.columns and len(df) > 0:
            credit_score_range = (df['CreditScore'].min(), df['CreditScore'].max())
            checks['value_range_checks']['CreditScore'] = credit_score_range
            out_of_range_credit = ((df['CreditScore'] < 300) | (df['CreditScore'] > 900)).sum()
            if out_of_range_credit > 0:
                checks['errors'].append(f"Found {out_of_range_credit} credit scores out of valid range (300-900)")
        
        if 'Age' in df.columns and len(df) > 0:
            age_range = (df['Age'].min(), df['Age'].max())
            checks['value_range_checks']['Age'] = age_range
            out_of_range_age = ((df['Age'] < 18) | (df['Age'] > 100)).sum()
            if out_of_range_age > 0:
                checks['errors'].append(f"Found {out_of_range_age} ages out of valid range (18-100)")
        
        if 'Balance' in df.columns and len(df) > 0:
            negative_balances = (df['Balance'] < 0).sum()
            if negative_balances > 0:
                checks['warnings'].append(f"Found {negative_balances} negative balances")
        
        # Categorical checks
        if 'Geography' in df.columns and len(df) > 0:
            unique_geographies = df['Geography'].dropna().unique()
            checks['categorical_checks']['Geography'] = list(unique_geographies)
            expected_geographies = {'France', 'Spain', 'Germany'}
            unexpected = set(unique_geographies) - expected_geographies
            if unexpected:
                checks['errors'].append(f"Unexpected geography values: {unexpected}")
        
        if 'Gender' in df.columns and len(df) > 0:
            unique_genders = df['Gender'].dropna().unique()
            checks['categorical_checks']['Gender'] = list(unique_genders)
            expected_genders = {'Male', 'Female'}
            unexpected = set(unique_genders) - expected_genders
            if unexpected:
                checks['errors'].append(f"Unexpected gender values: {unexpected}")
        
        # Target variable check
        if 'Exited' in df.columns and len(df) > 0:
            churn_rate = df['Exited'].mean()
            checks['churn_rate'] = churn_rate
            if churn_rate < 0.05 or churn_rate > 0.5:
                checks['warnings'].append(f"Unusual churn rate: {churn_rate:.3f}")
        
        # Determine if checks passed
        is_valid = len(checks['errors']) == 0
        issues = checks['errors'] + checks['warnings']
        
        logger.info(f"Sanity checks completed. Warnings: {len(checks['warnings'])}, Errors: {len(checks['errors'])}")
        return is_valid, issues
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data using schema and sanity checks, then save interim data.
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            bool: True if validation passes and data is saved
        """
        try:
            logger.info("Starting data validation")
            
            # Step 1: Schema validation
            is_schema_valid, schema_issues = self.validate_schema(df)
            if not is_schema_valid:
                logger.error(f"Schema validation failed: {schema_issues}")
                return False
            
            # Step 2: Sanity checks
            is_sanity_valid, sanity_issues = self.perform_sanity_checks(df)
            if not is_sanity_valid:
                logger.error(f"Sanity checks failed: {sanity_issues}")
                return False
            
            logger.info("Data validation passed")
            
            # Step 3: Save interim data after successful validation
            self.save_interim_data(df, "churn_raw.parquet")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def save_interim_data(self, df: pd.DataFrame, filename: str = "churn_interim.parquet") -> Path:
        """
        Save interim data to the interim directory.
        
        Args:
            df: Dataframe to save
            filename: Output filename
            
        Returns:
            Path: Path to the saved file
        """
        try:
            # Ensure interim directory exists
            self.interim_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet for efficiency
            output_path = self.interim_dir / filename
            df.to_parquet(output_path, index=False)
            
            logger.info(f"Interim data saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save interim data: {e}")
            raise
    
    def run_full_pipeline(self, download_data: bool = True) -> bool:
        """
        Run the complete data loading pipeline.
        
        Args:
            download_data: Whether to download from Kaggle
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("Starting data loading pipeline")
            
            # Step 1: Download data (optional)
            if download_data:
                expected_file_path = self.download_from_kaggle()
                if not expected_file_path.exists():
                    logger.error("Failed to download data from Kaggle and file does not exist")
                    return False
            
            # Step 2: Load CSV data
            df = self.load_csv_data()
            
            # Step 3: Validate schema
            is_valid, validation_report = self.validate_schema(df)
            
            # Save validation report
            import json
            with open(self.raw_data_dir / 'validation_report.json', 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            if not is_valid:
                logger.error("Schema validation failed. Check validation report.")
                return False
            
            # Step 4: Perform sanity checks
            sanity_checks = self.perform_sanity_checks(df)
            
            # Save sanity check report
            with open(self.raw_data_dir / 'sanity_check_report.json', 'w') as f:
                json.dump(sanity_checks, f, indent=2)
            
            if sanity_checks['errors']:
                logger.error(f"Sanity checks failed with errors: {sanity_checks['errors']}")
                return False
            
            # Step 5: Save to interim directory
            if not self.save_interim_data(df):
                logger.error("Failed to save interim data")
                return False
            
            logger.info("Data loading pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False

    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate data quality and return issues.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple[bool, list]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if dataframe is empty
        if df.empty:
            issues.append("Dataset is empty")
            return False, issues
        
        # Check for high missing values (over 10%)
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percentages[missing_percentages > 10]
        if not high_missing_cols.empty:
            for col, pct in high_missing_cols.items():
                issues.append(f"High missing values in {col}: {pct:.1f}%")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for mixed data types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_types = df[col].dropna().apply(type).nunique()
            if unique_types > 1:
                issues.append(f"Mixed data types in column {col}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive data summary statistics.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Data summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {},
            'memory_usage': df.memory_usage(deep=True).to_dict()
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'count': df[col].count(),
                'unique': df[col].nunique(),
                'top': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'freq': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                'unique_values': df[col].unique().tolist()[:10]  # Limit to first 10
            }
        
        logger.info(f"Data summary generated for {summary['shape'][0]} rows and {summary['shape'][1]} columns")
        return summary

def main():
    """
    Main function to run the data loading pipeline.
    """
    # Create logs directory
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize and run data loader
    loader = DataLoader()
    
    # Check if we should download data
    download_data = True
    if (loader.raw_data_dir / "Churn_Modelling.csv").exists():
        response = input("Data file already exists. Download again? (y/N): ")
        download_data = response.lower().startswith('y')
    
    success = loader.run_full_pipeline(download_data=download_data)
    
    if success:
        print("\n✅ Data loading completed successfully!")
        print(f"📁 Raw data: {loader.raw_data_dir}")
        print(f"📁 Interim data: {loader.interim_data_dir}")
    else:
        print("\n❌ Data loading failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()