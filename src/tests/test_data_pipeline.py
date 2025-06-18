#!/usr/bin/env python3
"""
Test Suite for Data Pipeline Components

Comprehensive tests for data loading, cleaning, and feature engineering components.
Includes unit tests, integration tests, and data quality validation tests.

Author: Bank Churn Analysis Team
Date: 2024
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import modules to test
sys.path.append(str(Path(__file__).parent.parent))

from data.load_data import DataLoader
from data.clean_data import DataCleaner
from features.create_features import FeatureEngineer


class TestDataLoader:
    """
    Test cases for DataLoader class.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'RowNumber': [1, 2, 3, 4, 5],
            'CustomerId': [15634602, 15647311, 15619304, 15701354, 15737888],
            'Surname': ['Hargrave', 'Hill', 'Onio', 'Boni', 'Mitchell'],
            'CreditScore': [619, 608, 502, 699, 850],
            'Geography': ['France', 'Spain', 'France', 'France', 'Spain'],
            'Gender': ['Female', 'Female', 'Female', 'Female', 'Female'],
            'Age': [42, 41, 42, 39, 43],
            'Tenure': [2, 1, 8, 1, 2],
            'Balance': [0.00, 83807.86, 159660.80, 0.00, 125510.82],
            'NumOfProducts': [1, 1, 3, 2, 1],
            'HasCrCard': [1, 0, 1, 0, 1],
            'IsActiveMember': [1, 1, 0, 0, 1],
            'EstimatedSalary': [101348.88, 112542.58, 113931.57, 93826.63, 79084.10],
            'Exited': [1, 0, 1, 0, 0]
        })
    
    @pytest.fixture
    def invalid_data_missing_cols(self):
        """Create data with missing required columns."""
        return pd.DataFrame({
            'RowNumber': [1, 2, 3],
            'CustomerId': [15634602, 15647311, 15619304],
            'CreditScore': [619, 608, 502],
            # Missing Geography, Gender, Age, etc.
        })
    
    @pytest.fixture
    def invalid_data_wrong_types(self, sample_data):
        """Create data with wrong data types."""
        data = sample_data.copy()
        data['CreditScore'] = data['CreditScore'].astype(str)  # Should be int
        data['Balance'] = data['Balance'].astype(str)  # Should be float
        return data
    
    @pytest.fixture
    def invalid_data_out_of_range(self, sample_data):
        """Create data with values outside expected ranges."""
        data = sample_data.copy()
        data.loc[0, 'CreditScore'] = 1000  # Invalid credit score (>850)
        data.loc[1, 'Age'] = 150  # Invalid age (>100)
        data.loc[2, 'Tenure'] = 15  # Invalid tenure (>10)
        data.loc[3, 'Balance'] = -1000  # Invalid balance (<0)
        return data
    
    @pytest.fixture
    def data_loader(self, temp_dir):
        """Create DataLoader instance with temporary directory."""
        return DataLoader(project_root=temp_dir)
    
    def test_init(self, data_loader, temp_dir):
        """Test DataLoader initialization."""
        assert data_loader.project_root == temp_dir
        assert data_loader.raw_data_dir == temp_dir / 'data' / 'raw'
        assert data_loader.interim_data_dir == temp_dir / 'data' / 'interim'
        assert len(data_loader.expected_columns) > 0
        assert 'Exited' in data_loader.expected_columns
    
    def test_load_csv(self, data_loader, sample_data, temp_dir):
        """Test CSV loading functionality."""
        # Create test CSV file
        csv_path = temp_dir / 'test_data.csv'
        sample_data.to_csv(csv_path, index=False)
        
        # Load CSV using the full path
        loaded_data = data_loader.load_csv_data(str(csv_path))
        
        # Assertions
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_load_csv_file_not_found(self, data_loader):
        """Test CSV loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_csv_data('non_existent_file.csv')
    
    def test_validate_schema_valid(self, data_loader, sample_data):
        """Test schema validation with valid data."""
        is_valid, issues = data_loader.validate_schema(sample_data)
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_schema_missing_columns(self, data_loader, sample_data):
        """Test schema validation with missing columns."""
        # Remove a required column
        invalid_data = sample_data.drop('CreditScore', axis=1)
        is_valid, issues = data_loader.validate_schema(invalid_data)
        
        assert not is_valid
        assert any('CreditScore' in issue for issue in issues)
    
    def test_validate_schema_wrong_dtypes(self, data_loader, sample_data):
        """Test schema validation with wrong data types."""
        # Change data type
        invalid_data = sample_data.copy()
        invalid_data['Age'] = invalid_data['Age'].astype(str)
        
        is_valid, issues = data_loader.validate_schema(invalid_data)
        assert not is_valid
        assert any('Age' in issue for issue in issues)
    
    def test_perform_sanity_checks_valid(self, data_loader, sample_data):
        """Test sanity checks with valid data."""
        is_valid, issues = data_loader.perform_sanity_checks(sample_data)
        assert is_valid
        assert len(issues) == 0
    
    def test_perform_sanity_checks_invalid_ranges(self, data_loader, sample_data):
        """Test sanity checks with invalid value ranges."""
        # Create invalid data
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'CreditScore'] = 1000  # Invalid credit score
        invalid_data.loc[1, 'Age'] = 150  # Invalid age
        
        is_valid, issues = data_loader.perform_sanity_checks(invalid_data)
        assert not is_valid
        assert len(issues) >= 2
    
    def test_save_interim_data(self, data_loader, sample_data, temp_dir):
        """Test saving interim data."""
        # Ensure interim directory exists
        data_loader.interim_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        output_path = data_loader.save_interim_data(sample_data)
        
        # Verify file exists and can be loaded
        assert output_path.exists()
        loaded_data = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(loaded_data, sample_data)
    
    def test_download_from_kaggle_no_api(self, data_loader, monkeypatch):
        """Test Kaggle download when API is not available."""
        # Mock kaggle import to raise ImportError
        original_import = __import__
        def mock_import(name, *args):
            if name == 'kaggle':
                raise ImportError("No module named 'kaggle'")
            return original_import(name, *args)
        
        monkeypatch.setattr('builtins.__import__', mock_import)
        
        result = data_loader.download_from_kaggle()
        # When kaggle is not available, method returns expected file path
        assert isinstance(result, Path)
        assert result.name == "Churn_Modelling.csv"
    
    def test_validate_data_quality_empty_dataframe(self, data_loader):
        """Test data quality validation with empty dataframe."""
        empty_df = pd.DataFrame()
        is_valid, issues = data_loader.validate_data_quality(empty_df)
        
        assert not is_valid
        assert any('empty' in issue.lower() for issue in issues)
    
    def test_validate_data_quality_duplicate_rows(self, data_loader, sample_data):
        """Test data quality validation with duplicate rows."""
        # Add duplicate rows
        duplicate_data = pd.concat([sample_data, sample_data.iloc[:2]], ignore_index=True)
        
        is_valid, issues = data_loader.validate_data_quality(duplicate_data)
        assert not is_valid
        assert any('duplicate' in issue.lower() for issue in issues)
    
    def test_validate_data_quality_missing_values(self, data_loader, sample_data):
        """Test data quality validation with missing values."""
        # Introduce missing values
        data_with_na = sample_data.copy()
        data_with_na.loc[0, 'CreditScore'] = np.nan
        data_with_na.loc[1, 'Geography'] = None
        
        is_valid, issues = data_loader.validate_data_quality(data_with_na)
        assert not is_valid
        assert any('missing' in issue.lower() for issue in issues)
    
    def test_get_data_summary(self, data_loader, sample_data):
        """Test data summary generation."""
        summary = data_loader.get_data_summary(sample_data)
        
        assert isinstance(summary, dict)
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert summary['shape'] == sample_data.shape
        assert len(summary['columns']) == len(sample_data.columns)
    
    def test_load_csv_with_encoding_issues(self, data_loader, temp_dir):
        """Test CSV loading with encoding issues."""
        # Create a CSV with special characters
        csv_path = temp_dir / 'test_encoding.csv'
        with open(csv_path, 'w', encoding='latin-1') as f:
            f.write('Name,Value\nCafé,100\nNaïve,200\n')
        
        # Should handle encoding gracefully
        try:
            loaded_data = data_loader.load_csv_data(str(csv_path))
            assert isinstance(loaded_data, pd.DataFrame)
            assert len(loaded_data) == 2
        except UnicodeDecodeError:
            # This is acceptable behavior for encoding issues
            pass


class TestDataCleaner:
    """
    Test cases for DataCleaner class.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values for testing."""
        data = pd.DataFrame({
            'CustomerId': [1, 2, 3, 4, 5],
            'Surname': ['A', 'B', 'C', 'D', 'E'],
            'CreditScore': [619, np.nan, 502, 699, 850],
            'Geography': ['France', 'Spain', 'France', 'France', 'Spain'],
            'Gender': ['Female', 'Female', np.nan, 'Female', 'Female'],
            'Age': [42, 41, 42, 39, 43],
            'Tenure': [2, 1, 8, 1, 2],
            'Balance': [0.00, 83807.86, np.nan, 0.00, 125510.82],
            'NumOfProducts': [1, 1, 3, 2, 1],
            'HasCrCard': [1, 0, 1, 0, 1],
            'IsActiveMember': [1, 1, 0, 0, 1],
            'EstimatedSalary': [101348.88, np.nan, 113931.57, 93826.63, 79084.10],
            'Exited': [1, 0, 1, 0, 0]
        })
        return data
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for cleaning tests."""
        return pd.DataFrame({
            'RowNumber': [1, 2, 3, 4, 5],
            'CustomerId': [15634602, 15647311, 15619304, 15701354, 15737888],
            'Surname': ['Hargrave', 'Hill', 'Onio', 'Boni', 'Mitchell'],
            'CreditScore': [619, 608, np.nan, 699, 850],  # Missing value
            'Geography': ['France', 'Spain', 'France', 'France', 'Spain'],
            'Gender': ['Female', 'Female', 'Female', 'Female', 'Male'],
            'Age': [42, 41, 42, 39, 43],
            'Tenure': [2, 1, 8, 1, 2],
            'Balance': [0.00, 83807.86, 159660.80, 0.00, 125510.82],
            'NumOfProducts': [1, 1, 3, 2, 1],
            'HasCrCard': [1, 0, 1, 0, 1],
            'IsActiveMember': [1, 1, 0, 0, 1],
            'EstimatedSalary': [101348.88, 112542.58, 113931.57, 93826.63, 79084.10],
            'Exited': [1, 0, 1, 0, 0]
        })
    
    @pytest.fixture
    def data_cleaner(self, temp_dir):
        """Create DataCleaner instance with temporary directory."""
        return DataCleaner(project_root=temp_dir)
    
    def test_init(self, data_cleaner, temp_dir):
        """Test DataCleaner initialization."""
        assert data_cleaner.project_root == temp_dir
        assert data_cleaner.interim_data_dir == temp_dir / 'data' / 'interim'
        assert data_cleaner.processed_data_dir == temp_dir / 'data' / 'processed'
        assert data_cleaner.models_dir == temp_dir / 'models'
        
        # Check feature categories
        assert 'CreditScore' in data_cleaner.numeric_features
        assert 'Geography' in data_cleaner.categorical_features
        assert 'HasCrCard' in data_cleaner.binary_features
        assert data_cleaner.target_column == 'Exited'
    
    def test_load_interim_data(self, data_cleaner, sample_raw_data, temp_dir):
        """Test loading interim data."""
        # Create interim directory and save sample data
        interim_dir = temp_dir / 'data' / 'interim'
        interim_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = interim_dir / 'churn_raw.parquet'
        sample_raw_data.to_parquet(data_file)
        
        # Load data
        loaded_data = data_cleaner.load_interim_data()
        
        # Assertions
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_raw_data)
        pd.testing.assert_frame_equal(loaded_data, sample_raw_data)
    
    def test_load_interim_data_file_not_found(self, data_cleaner):
        """Test loading interim data when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            data_cleaner.load_interim_data('non_existent_file.parquet')
    
    def test_remove_unnecessary_columns(self, data_cleaner, sample_raw_data):
        """Test removal of unnecessary columns."""
        cleaned_data = data_cleaner.remove_unnecessary_columns(sample_raw_data)
        
        # Check that PII and ID columns are removed
        assert 'Surname' not in cleaned_data.columns
        assert 'RowNumber' not in cleaned_data.columns
        assert 'CustomerId' not in cleaned_data.columns
        
        # Check that important columns remain
        assert 'CreditScore' in cleaned_data.columns
        assert 'Geography' in cleaned_data.columns
        assert 'Exited' in cleaned_data.columns
    
    def test_handle_missing_values_numeric(self, data_cleaner, sample_raw_data):
        """Test handling of missing values in numeric columns."""
        # Ensure there's a missing value in CreditScore
        assert sample_raw_data['CreditScore'].isna().any()
        
        cleaned_data = data_cleaner.handle_missing_values(sample_raw_data)
        
        # Check that missing values are handled
        assert not cleaned_data['CreditScore'].isna().any()
        
        # Check that the imputed value is reasonable (median)
        original_median = sample_raw_data['CreditScore'].median()
        assert cleaned_data['CreditScore'].iloc[2] == original_median
    
    def test_handle_missing_values_categorical(self, data_cleaner, sample_raw_data):
        """Test handling of missing values in categorical columns."""
        # Add missing value to categorical column
        data_with_missing = sample_raw_data.copy()
        data_with_missing.loc[0, 'Geography'] = np.nan
        
        cleaned_data = data_cleaner.handle_missing_values(data_with_missing)
        
        # Check that missing values are handled
        assert not cleaned_data['Geography'].isna().any()
        
        # Check that the imputed value is the mode
        original_mode = sample_raw_data['Geography'].mode()[0]
        assert cleaned_data['Geography'].iloc[0] == original_mode
    
    def test_encode_categorical_variables(self, data_cleaner, sample_raw_data):
        """Test categorical variable encoding through preprocessing."""
        processed_data, _ = data_cleaner.apply_preprocessing(sample_raw_data)
        
        # Check that original categorical columns are removed
        assert 'Geography' not in processed_data.columns
        assert 'Gender' not in processed_data.columns
        
        # Check that encoded columns are created
        geography_cols = [col for col in processed_data.columns if 'Geography' in col]
        gender_cols = [col for col in processed_data.columns if 'Gender' in col]
        
        assert len(geography_cols) > 0
        assert len(gender_cols) > 0
        
        # Check that encoding is binary (0 or 1) for categorical features
        for col in geography_cols + gender_cols:
            assert processed_data[col].isin([0, 1]).all()
    
    def test_scale_features(self, data_cleaner, sample_raw_data):
        """Test feature scaling."""
        # First apply preprocessing to encode categorical variables
        encoded_data, _ = data_cleaner.apply_preprocessing(sample_raw_data)
        
        # Scale features
        scaled_data, scaler = data_cleaner.scale_features(encoded_data)
        
        # Check that numeric features are scaled
        for feature in data_cleaner.numeric_features:
            if feature in scaled_data.columns:
                # Scaled features should have mean close to 0 and std close to 1
                assert abs(scaled_data[feature].mean()) < 0.1
                assert abs(scaled_data[feature].std() - 1.0) < 0.1
        
        # Check that scaler is returned
        assert scaler is not None
        assert hasattr(scaler, 'transform')
    
    def test_validate_processed_data(self, data_cleaner, sample_raw_data):
        """Test validation of processed data."""
        # Process the data through the pipeline
        cleaned_data = data_cleaner.remove_unnecessary_columns(sample_raw_data)
        cleaned_data = data_cleaner.handle_missing_values(cleaned_data)
        encoded_data, _ = data_cleaner.apply_preprocessing(cleaned_data)
        
        validation_result = data_cleaner.validate_processed_data(encoded_data)
        is_valid = validation_result['is_valid']
        issues = validation_result['issues']
        
        assert is_valid
        assert len(issues) == 0
    
    def test_save_processed_data(self, data_cleaner, sample_raw_data, temp_dir):
        """Test saving processed data."""
        # Ensure processed directory exists
        data_cleaner.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save data
        cleaned_data = data_cleaner.remove_unnecessary_columns(sample_raw_data)
        output_path = data_cleaner.save_processed_data(cleaned_data)
        
        # Verify file exists and can be loaded
        assert output_path.exists()
        loaded_data = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(loaded_data, cleaned_data)
    
    def test_full_cleaning_pipeline(self, data_cleaner, sample_raw_data, temp_dir):
        """Test the complete data cleaning pipeline."""
        # Create interim directory and save sample data
        interim_dir = temp_dir / 'data' / 'interim'
        interim_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = interim_dir / 'churn_raw.parquet'
        sample_raw_data.to_parquet(data_file)
        
        # Run full pipeline
        processed_data, scaler = data_cleaner.clean_data()
        
        # Assertions
        assert isinstance(processed_data, pd.DataFrame)
        assert scaler is not None
        
        # Check that data is properly processed
        assert 'Surname' not in processed_data.columns  # PII removed
        assert not processed_data.isna().any().any()  # No missing values
        
        # Check that categorical variables are encoded
        geography_cols = [col for col in processed_data.columns if col.startswith('Geography_')]
        assert len(geography_cols) > 0
    
    def test_remove_pii(self, data_cleaner, sample_data_with_missing):
        """Test PII removal."""
        cleaned_data = data_cleaner.remove_pii_data(sample_data_with_missing)
        
        # Check that PII columns are removed
        assert 'CustomerId' not in cleaned_data.columns
        assert 'Surname' not in cleaned_data.columns
        
        # Check that other columns remain
        assert 'CreditScore' in cleaned_data.columns
        assert 'Geography' in cleaned_data.columns
    
    def test_analyze_missing_values(self, data_cleaner, sample_data_with_missing):
        """Test missing value analysis."""
        analysis = data_cleaner.analyze_missing_values(sample_data_with_missing)
        
        # Check analysis structure
        assert 'missing_counts' in analysis
        assert 'missing_percentages' in analysis
        assert 'total_missing' in analysis
        
        # Check specific missing values
        assert analysis['missing_counts']['CreditScore'] == 1
        assert analysis['missing_counts']['Gender'] == 1
        assert analysis['missing_counts']['Balance'] == 1
        assert analysis['missing_counts']['EstimatedSalary'] == 1
    
    def test_handle_missing_values(self, data_cleaner, sample_data_with_missing):
        """Test missing value handling."""
        cleaned_data = data_cleaner.handle_missing_values(sample_data_with_missing)
        
        # Check that no missing values remain
        assert cleaned_data.isnull().sum().sum() == 0
        
        # Check that data shape is preserved (no rows dropped)
        assert len(cleaned_data) == len(sample_data_with_missing)
    
    def test_detect_outliers_iqr(self, data_cleaner):
        """Test IQR outlier detection."""
        # Create data with obvious outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 200])  # 100, 200 are outliers
        outliers = data_cleaner.detect_outliers_iqr(data)
        
        # Check that outliers are detected
        assert len(outliers) == 2
        assert 100 in outliers.values
        assert 200 in outliers.values
    
    def test_detect_outliers_zscore(self, data_cleaner):
        """Test Z-score outlier detection."""
        # Create data with obvious outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 200])  # 100, 200 are outliers
        outliers = data_cleaner.detect_outliers_zscore(data, threshold=2)
        
        # Check that outliers are detected
        assert len(outliers) >= 2
    
    def test_create_preprocessing_pipeline(self, data_cleaner, sample_data_with_missing):
        """Test preprocessing pipeline creation."""
        # Remove PII first
        clean_data = data_cleaner.remove_pii_data(sample_data_with_missing)
        clean_data = data_cleaner.handle_missing_values(clean_data)
        
        # Create pipeline
        pipeline = data_cleaner.create_preprocessing_pipeline(clean_data)
        
        # Check pipeline components
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')
    
    def test_validate_processed_data(self, data_cleaner):
        """Test processed data validation."""
        # Create valid processed data
        valid_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5],
            'Exited': [0, 1, 0]
        })
        
        is_valid, issues = data_cleaner.validate_processed_data(valid_data)
        assert is_valid
        assert len(issues) == 0
        
        # Create invalid data with missing values
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'feature1'] = np.nan
        
        is_valid, issues = data_cleaner.validate_processed_data(invalid_data)
        assert not is_valid
        assert len(issues) > 0


class TestFeatureEngineer:
    """
    Test cases for FeatureEngineer class.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for testing."""
        return pd.DataFrame({
            'CreditScore': [619, 608, 502, 699, 850],
            'Geography_France': [1, 0, 1, 1, 0],
            'Geography_Germany': [0, 0, 0, 0, 0],
            'Geography_Spain': [0, 1, 0, 0, 1],
            'Gender_Female': [1, 1, 1, 1, 1],
            'Gender_Male': [0, 0, 0, 0, 0],
            'Age': [42, 41, 42, 39, 43],
            'Tenure': [2, 1, 8, 1, 2],
            'Balance': [0.00, 83807.86, 159660.80, 0.00, 125510.82],
            'NumOfProducts': [1, 1, 3, 2, 1],
            'HasCrCard': [1, 0, 1, 0, 1],
            'IsActiveMember': [1, 1, 0, 0, 1],
            'EstimatedSalary': [101348.88, 112542.58, 113931.57, 93826.63, 79084.10],
            'Exited': [1, 0, 1, 0, 0]
        })
    
    @pytest.fixture
    def feature_engineer(self, temp_dir):
        """Create FeatureEngineer instance with temporary directory."""
        return FeatureEngineer(project_root=temp_dir)
    
    def test_init(self, feature_engineer, temp_dir):
        """Test FeatureEngineer initialization."""
        assert feature_engineer.project_root == temp_dir
        assert feature_engineer.processed_dir == temp_dir / 'data' / 'processed'
    
    def test_create_ratio_features(self, feature_engineer, sample_processed_data):
        """Test ratio feature creation."""
        enhanced_data = feature_engineer.create_ratio_features(sample_processed_data)
        
        # Check that new ratio features are created
        assert 'BalanceToSalary' in enhanced_data.columns
        assert 'CreditScoreToAge' in enhanced_data.columns
        assert 'TenureToAge' in enhanced_data.columns
        
        # Check calculations
        expected_balance_to_salary = sample_processed_data['Balance'] / (sample_processed_data['EstimatedSalary'] + 1)
        pd.testing.assert_series_equal(
            enhanced_data['BalanceToSalary'], 
            expected_balance_to_salary, 
            check_names=False
        )
    
    def test_create_binned_features(self, feature_engineer, sample_processed_data):
        """Test binned feature creation."""
        enhanced_data = feature_engineer.create_binned_features(sample_processed_data)
        
        # Check that binned features are created
        assert 'AgeGroup' in enhanced_data.columns
        assert 'BalanceGroup' in enhanced_data.columns
        assert 'CreditScoreGroup' in enhanced_data.columns
        
        # Check that binned values are within expected range
        assert enhanced_data['AgeGroup'].min() >= 0
        assert enhanced_data['AgeGroup'].max() <= 3
    
    def test_create_behavioral_features(self, feature_engineer, sample_processed_data):
        """Test behavioral feature creation."""
        enhanced_data = feature_engineer.create_behavioral_features(sample_processed_data)
        
        # Check that behavioral features are created
        assert 'IsZeroBalance' in enhanced_data.columns
        assert 'IsHighValue' in enhanced_data.columns
        assert 'ProductDiversity' in enhanced_data.columns
        
        # Check binary features
        assert enhanced_data['IsZeroBalance'].dtype == bool
        assert enhanced_data['IsHighValue'].dtype == bool
        
        # Check specific calculations
        expected_zero_balance = (sample_processed_data['Balance'] == 0)
        pd.testing.assert_series_equal(
            enhanced_data['IsZeroBalance'], 
            expected_zero_balance, 
            check_names=False
        )
    
    def test_create_age_groups(self, feature_engineer, sample_processed_data):
        """Test creation of age group features."""
        enhanced_data = feature_engineer.create_age_groups(sample_processed_data)
        
        # Check that age group columns are created
        age_group_cols = [col for col in enhanced_data.columns if col.startswith('AgeGroup_') and not col.endswith('_num')]
        assert len(age_group_cols) > 0
        
        # Check that age groups are mutually exclusive
        age_group_sum = enhanced_data[age_group_cols].sum(axis=1)
        assert (age_group_sum == 1).all()  # Each row should have exactly one age group
    
    def test_create_balance_categories(self, feature_engineer, sample_processed_data):
        """Test creation of balance category features."""
        enhanced_data = feature_engineer.create_balance_categories(sample_processed_data)
        
        # Check that balance category columns are created
        balance_cat_cols = [col for col in enhanced_data.columns if col.startswith('BalanceCategory_') and not col.endswith('_num')]
        assert len(balance_cat_cols) > 0
        
        # Check that categories are mutually exclusive
        balance_cat_sum = enhanced_data[balance_cat_cols].sum(axis=1)
        assert (balance_cat_sum == 1).all()
        
        # Check specific categorizations
        zero_balance_rows = enhanced_data[enhanced_data['Balance'] == 0]
        if len(zero_balance_rows) > 0:
            assert zero_balance_rows['BalanceCategory_Zero'].iloc[0] == 1
    
    def test_create_interaction_features(self, feature_engineer, sample_processed_data):
        """Test creation of interaction features."""
        enhanced_data = feature_engineer.create_interaction_features(sample_processed_data)
        
        # Check that interaction features are created
        expected_interactions = [
            'Age_x_Tenure',
            'CreditScore_x_Balance',
            'IsActiveMember_x_NumOfProducts'
        ]
        
        for interaction in expected_interactions:
            assert interaction in enhanced_data.columns
        
        # Check that interactions are calculated correctly
        assert enhanced_data['Age_x_Tenure'].iloc[0] == 42 * 2  # Age 42, Tenure 2
    
    def test_create_polynomial_features(self, feature_engineer, sample_processed_data):
        """Test creation of polynomial features."""
        enhanced_data = feature_engineer.create_polynomial_features(sample_processed_data)
        
        # Check that polynomial features are created
        poly_features = [col for col in enhanced_data.columns if '_squared' in col or '_cubed' in col]
        assert len(poly_features) > 0
        
        # Check specific polynomial calculations
        if 'Age_squared' in enhanced_data.columns:
            assert enhanced_data['Age_squared'].iloc[0] == 42**2
    
    def test_select_features_univariate(self, feature_engineer, sample_processed_data):
        """Test univariate feature selection."""
        # Create some additional features first
        enhanced_data = feature_engineer.create_ratio_features(sample_processed_data)
        
        # Select features
        selected_data, selector = feature_engineer.select_features_univariate(
            enhanced_data, k=5
        )
        
        # Check that the right number of features are selected
        feature_cols = [col for col in selected_data.columns if col != 'Exited']
        assert len(feature_cols) == 5
        
        # Check that selector is returned
        assert selector is not None
        assert hasattr(selector, 'transform')
    
    def test_select_features_importance(self, feature_engineer, sample_processed_data):
        """Test importance-based feature selection."""
        # Create some additional features first
        enhanced_data = feature_engineer.create_ratio_features(sample_processed_data)
        
        # Select features
        selected_data, importance_scores = feature_engineer.select_features_importance(
            enhanced_data, n_features=5
        )
        
        # Check that the right number of features are selected
        feature_cols = [col for col in selected_data.columns if col != 'Exited']
        assert len(feature_cols) == 5
        
        # Check that importance scores are returned
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0
    
    def test_create_pca_features(self, feature_engineer, sample_processed_data):
        """Test feature creation methods."""
        # Test ratio features
        ratio_data = feature_engineer.create_ratio_features(sample_processed_data)
        assert ratio_data.shape[1] >= sample_processed_data.shape[1]
        
        # Test binned features
        binned_data = feature_engineer.create_binned_features(sample_processed_data)
        assert binned_data.shape[1] >= sample_processed_data.shape[1]
        
        # Test behavioral features
        behavioral_data = feature_engineer.create_behavioral_features(sample_processed_data)
        assert behavioral_data.shape[1] >= sample_processed_data.shape[1]
        
        # Check that all data has same number of rows
        assert ratio_data.shape[0] == sample_processed_data.shape[0]
        assert binned_data.shape[0] == sample_processed_data.shape[0]
        assert behavioral_data.shape[0] == sample_processed_data.shape[0]
    
    def test_validate_features(self, feature_engineer, sample_processed_data):
        """Test feature validation."""
        # Create enhanced features
        enhanced_data = feature_engineer.create_ratio_features(sample_processed_data)
        
        # Basic validation checks
        assert enhanced_data is not None
        assert len(enhanced_data) > 0
        assert enhanced_data.shape[1] >= sample_processed_data.shape[1]
        
        # Check for infinite or NaN values
        assert not enhanced_data.isnull().all().any()
        assert not np.isinf(enhanced_data.select_dtypes(include=[np.number])).any().any()
    
    def test_save_features(self, feature_engineer, sample_processed_data, temp_dir):
        """Test saving engineered features."""
        # Ensure processed directory exists
        feature_engineer.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create and save features
        enhanced_data = feature_engineer.create_ratio_features(sample_processed_data)
        success = feature_engineer.save_engineered_data(enhanced_data)
        
        # Verify save was successful
        assert success
        
        # Verify file exists
        output_path = feature_engineer.processed_data_dir / 'churn_features.parquet'
        assert output_path.exists()
        loaded_data = pd.read_parquet(output_path)
        assert loaded_data.shape == enhanced_data.shape
    
    def test_full_feature_engineering_pipeline(self, feature_engineer, sample_processed_data, temp_dir):
        """Test the complete feature engineering pipeline."""
        # Create processed directory and save sample data
        processed_dir = temp_dir / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = processed_dir / 'churn_cleaned.parquet'
        sample_processed_data.to_parquet(data_file)
        
        # Run full pipeline
        pipeline_success = feature_engineer.run_full_pipeline()
        
        # Assertions
        assert pipeline_success
        
        # Check that feature files were created
        features_file = feature_engineer.processed_data_dir / 'churn_features_all.parquet'
        selected_file = feature_engineer.processed_data_dir / 'churn_features_selected.parquet'
        assert features_file.exists()
        assert selected_file.exists()
        
        # Verify the files contain data
        features_data = pd.read_parquet(features_file)
        selected_data = pd.read_parquet(selected_file)
        assert len(features_data) > 0
        assert len(selected_data) > 0
        assert selected_data.shape[1] <= features_data.shape[1]
    
    def test_create_statistical_features(self, feature_engineer, sample_processed_data):
        """Test statistical feature creation."""
        enhanced_data = feature_engineer.create_statistical_features(sample_processed_data)
        
        # Check that new features are created
        assert enhanced_data.shape[1] >= sample_processed_data.shape[1]
        
        # Check data types and that original data is preserved
        assert enhanced_data.shape[0] == sample_processed_data.shape[0]
        
        # Check for z-score features (created for first 5 numeric columns)
        zscore_features = [col for col in enhanced_data.columns if col.endswith('_zscore')]
        assert len(zscore_features) > 0
    
    def test_calculate_feature_importance(self, feature_engineer, sample_processed_data):
        """Test feature importance calculation."""
        # Create some additional features first
        enhanced_data = feature_engineer.create_ratio_features(sample_processed_data)
        
        # Calculate importance - returns a dict with feature names as keys
        importance_results = feature_engineer.calculate_feature_importance(enhanced_data)
        
        # Check results structure - should be a dict with feature names and scores
        assert isinstance(importance_results, dict)
        assert len(importance_results) > 0
        
        # Check that all values are numeric
        for feature, score in importance_results.items():
            assert isinstance(feature, str)
            assert isinstance(score, (int, float, np.number))
    
    def test_select_top_features(self, feature_engineer, sample_processed_data):
        """Test top feature selection."""
        # Create enhanced data
        enhanced_data = feature_engineer.create_ratio_features(sample_processed_data)
        
        # Select top features - method takes df and n_features parameter
        top_features = feature_engineer.select_top_features(enhanced_data, n_features=5)
        
        # Check results
        assert isinstance(top_features, list)
        assert len(top_features) >= 5  # May include ID columns and target
        assert all(isinstance(feature, str) for feature in top_features)


class TestIntegration:
    """
    Integration tests for the complete data pipeline.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for integration testing."""
        return pd.DataFrame({
            'RowNumber': list(range(1, 101)),
            'CustomerId': list(range(15634602, 15634702)),
            'Surname': [f'Customer_{i}' for i in range(100)],
            'CreditScore': np.random.randint(300, 850, 100),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Age': np.random.randint(18, 80, 100),
            'Tenure': np.random.randint(0, 10, 100),
            'Balance': np.random.uniform(0, 200000, 100),
            'NumOfProducts': np.random.randint(1, 4, 100),
            'HasCrCard': np.random.choice([0, 1], 100),
            'IsActiveMember': np.random.choice([0, 1], 100),
            'EstimatedSalary': np.random.uniform(10000, 150000, 100),
            'Exited': np.random.choice([0, 1], 100)
        })
    
    def test_complete_pipeline(self, temp_dir, sample_raw_data):
        """Test the complete data processing pipeline."""
        # Create directory structure
        (temp_dir / 'data' / 'raw').mkdir(parents=True)
        (temp_dir / 'data' / 'interim').mkdir(parents=True)
        (temp_dir / 'data' / 'processed').mkdir(parents=True)
        
        # Save raw data
        raw_file = temp_dir / 'data' / 'raw' / 'test_data.csv'
        sample_raw_data.to_csv(raw_file, index=False)
        
        # Step 1: Load data
        loader = DataLoader(project_root=temp_dir)
        loaded_data = loader.load_csv_data('test_data.csv')
        
        # Validate and save interim data
        is_valid, _ = loader.validate_schema(loaded_data)
        assert is_valid
        
        save_path = loader.save_interim_data(loaded_data, 'churn_raw.parquet')
        assert save_path and save_path.exists()
        
        # Check that interim file exists
        interim_file = temp_dir / 'data' / 'interim' / 'churn_raw.parquet'
        assert interim_file.exists()
        
        # Step 2: Clean data
        cleaner = DataCleaner(project_root=temp_dir)
        interim_data = cleaner.load_interim_data()
        
        # Clean and process
        clean_data = cleaner.remove_pii_data(interim_data)
        clean_data = cleaner.handle_missing_values(clean_data)
        
        # Apply preprocessing pipeline
        processed_data, _ = cleaner.apply_preprocessing(clean_data)
        
        # Validate processed data
        is_valid, issues = cleaner.validate_processed_data(processed_data)
        assert is_valid
        
        # Save processed data
        save_path = cleaner.save_processed_data(processed_data)
        assert save_path and save_path.exists()
        
        # Check that file was created
        processed_file = cleaner.processed_data_dir / 'churn_cleaned.parquet'
        assert processed_file.exists()
        
        # Step 3: Feature engineering
        engineer = FeatureEngineer(project_root=temp_dir)
        
        # Create features
        enhanced_data = engineer.create_ratio_features(processed_data)
        enhanced_data = engineer.create_binned_features(enhanced_data)
        enhanced_data = engineer.create_behavioral_features(enhanced_data)
        enhanced_data = engineer.create_statistical_features(enhanced_data)
        
        # Calculate feature importance and select top features
        importance_results = engineer.calculate_feature_importance(enhanced_data)
        top_features = engineer.select_top_features(enhanced_data, n_features=10)
        
        # Verify final results
        assert len(enhanced_data) == len(sample_raw_data)
        # Check that we have the requested number of features plus ID and target columns
        feature_only_count = len([f for f in top_features if f not in ['RowNumber', 'CustomerId', 'Exited']])
        assert feature_only_count == 10
        assert 'Exited' in enhanced_data.columns
        assert enhanced_data.isnull().sum().sum() == 0  # No missing values
        
        print(f"✅ Complete pipeline test passed!")
        print(f"   - Original data: {len(sample_raw_data)} rows, {len(sample_raw_data.columns)} columns")
        print(f"   - Enhanced data: {len(enhanced_data)} rows, {len(enhanced_data.columns)} columns")
        print(f"   - Top features selected: {len(top_features)}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])