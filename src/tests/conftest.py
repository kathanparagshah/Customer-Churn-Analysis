#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures

Shared fixtures and configuration for all test modules in the bank churn analysis project.
Provides common test utilities, data fixtures, and test environment setup.

Author: Bank Churn Analysis Team
Date: 2024
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Test configuration
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API components"
    )
    config.addinivalue_line(
        "markers", "model: marks tests that require trained models"
    )
    config.addinivalue_line(
        "markers", "data: marks tests that require data files"
    )


# Session-scoped fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-scoped temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="churn_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_raw_data():
    """Create sample raw data for testing (session-scoped for performance)."""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'RowNumber': range(1, n_samples + 1),
        'CustomerId': range(15634602, 15634602 + n_samples),
        'Surname': [f'Customer_{i}' for i in range(n_samples)],
        'CreditScore': np.random.randint(350, 850, n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 92, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.randint(1, 4, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(11.58, 199992.48, n_samples),
        'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    data.loc[missing_indices[:25], 'CreditScore'] = np.nan
    data.loc[missing_indices[25:], 'EstimatedSalary'] = np.nan
    
    return data


# Function-scoped fixtures
@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory structure for testing."""
    temp_dir = tempfile.mkdtemp(prefix="churn_project_")
    project_root = Path(temp_dir)
    
    # Create standard project directory structure
    directories = [
        'data/raw',
        'data/interim',
        'data/processed',
        'models',
        'reports/figures',
        'notebooks',
        'src/data',
        'src/features',
        'src/models',
        'src/tests',
        'deployment',
        'deployment/monitoring'
    ]
    
    for directory in directories:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
    
    yield project_root
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_processed_data():
    """Create sample processed data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, n_samples),
        'Geography_France': np.random.choice([0, 1], n_samples),
        'Geography_Germany': np.random.choice([0, 1], n_samples),
        'Geography_Spain': np.random.choice([0, 1], n_samples),
        'Gender_Female': np.random.choice([0, 1], n_samples),
        'Gender_Male': np.random.choice([0, 1], n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 200000, n_samples),
        'NumOfProducts': np.random.randint(1, 4, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(10000, 150000, n_samples),
        'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    return data


@pytest.fixture
def sample_featured_data(sample_processed_data):
    """Create sample data with engineered features."""
    data = sample_processed_data.copy()
    
    # Add engineered features
    data['Balance_Salary_Ratio'] = data['Balance'] / (data['EstimatedSalary'] + 1)
    data['Age_Group_Young'] = (data['Age'] < 30).astype(int)
    data['Age_Group_Middle'] = ((data['Age'] >= 30) & (data['Age'] < 50)).astype(int)
    data['Age_Group_Senior'] = (data['Age'] >= 50).astype(int)
    data['High_Value_Customer'] = ((data['Balance'] > 100000) | (data['EstimatedSalary'] > 100000)).astype(int)
    data['Product_Diversity'] = data['NumOfProducts'] / 4.0
    
    return data


@pytest.fixture
def trained_model_package(sample_featured_data):
    """Create a trained model package for testing."""
    # Prepare data
    X = sample_featured_data.drop('Exited', axis=1)
    y = sample_featured_data['Exited']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    
    # Fit scaler and model
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'model_name': 'RandomForest',
        'version': '1.0.0',
        'feature_names': list(X.columns),
        'label_encoders': {},
        'training_date': '2024-01-01',
        'performance_metrics': {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'roc_auc': 0.88
        }
    }
    
    return model_package


@pytest.fixture
def customer_data_sample():
    """Create sample customer data for API testing."""
    return {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0
    }


@pytest.fixture
def batch_customer_data(customer_data_sample):
    """Create batch customer data for API testing."""
    # Create variations of the sample customer
    customers = []
    
    # Base customer
    customers.append(customer_data_sample.copy())
    
    # High-risk customer
    high_risk = customer_data_sample.copy()
    high_risk.update({
        "CreditScore": 400,
        "Age": 65,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "IsActiveMember": 0
    })
    customers.append(high_risk)
    
    # Low-risk customer
    low_risk = customer_data_sample.copy()
    low_risk.update({
        "CreditScore": 800,
        "Age": 25,
        "Balance": 150000.0,
        "NumOfProducts": 3,
        "IsActiveMember": 1
    })
    customers.append(low_risk)
    
    return customers


# Utility fixtures
@pytest.fixture
def mock_data_files(temp_project_dir, sample_raw_data, sample_processed_data, sample_featured_data):
    """Create mock data files in the project directory."""
    # Save raw data
    raw_path = temp_project_dir / 'data' / 'raw' / 'Churn_Modelling.csv'
    sample_raw_data.to_csv(raw_path, index=False)
    
    # Save interim data
    interim_path = temp_project_dir / 'data' / 'interim' / 'churn_raw.parquet'
    sample_raw_data.to_parquet(interim_path)
    
    # Save processed data
    processed_path = temp_project_dir / 'data' / 'processed' / 'churn_cleaned.parquet'
    sample_processed_data.to_parquet(processed_path)
    
    # Save featured data
    featured_path = temp_project_dir / 'data' / 'processed' / 'churn_features.parquet'
    sample_featured_data.to_parquet(featured_path)
    
    return {
        'raw': raw_path,
        'interim': interim_path,
        'processed': processed_path,
        'featured': featured_path
    }


@pytest.fixture
def mock_model_file(temp_project_dir, trained_model_package):
    """Create a mock model file in the project directory."""
    model_path = temp_project_dir / 'models' / 'churn_model.pkl'
    joblib.dump(trained_model_package, model_path)
    return model_path


# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_customer_data(n_customers=100, churn_rate=0.2, seed=42):
        """Create realistic customer data for testing."""
        np.random.seed(seed)
        
        data = pd.DataFrame({
            'CreditScore': np.random.randint(300, 850, n_customers),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_customers),
            'Gender': np.random.choice(['Male', 'Female'], n_customers),
            'Age': np.random.randint(18, 92, n_customers),
            'Tenure': np.random.randint(0, 10, n_customers),
            'Balance': np.random.uniform(0, 250000, n_customers),
            'NumOfProducts': np.random.randint(1, 4, n_customers),
            'HasCrCard': np.random.choice([0, 1], n_customers),
            'IsActiveMember': np.random.choice([0, 1], n_customers),
            'EstimatedSalary': np.random.uniform(11.58, 199992.48, n_customers),
            'Exited': np.random.choice([0, 1], n_customers, p=[1-churn_rate, churn_rate])
        })
        
        return data
    
    @staticmethod
    def create_edge_case_data():
        """Create edge case data for testing validation."""
        return pd.DataFrame({
            'CreditScore': [300, 850, 500, np.nan],  # Min, max, normal, missing
            'Geography': ['France', 'Germany', 'Spain', 'France'],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Age': [18, 92, 35, 50],  # Min, max, normal, normal
            'Tenure': [0, 10, 5, 3],  # Min, max, normal, normal
            'Balance': [0.0, 250000.0, 50000.0, np.nan],  # Min, max, normal, missing
            'NumOfProducts': [1, 4, 2, 3],  # Min, max, normal, normal
            'HasCrCard': [0, 1, 1, 0],
            'IsActiveMember': [0, 1, 1, 0],
            'EstimatedSalary': [11.58, 199992.48, 75000.0, 60000.0],
            'Exited': [0, 1, 0, 1]
        })


@pytest.fixture
def test_data_generator():
    """Provide test data generator utility."""
    return TestDataGenerator


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing test operations."""
    
    def __init__(self, max_time=None):
        self.max_time = max_time
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
        if self.max_time and self.elapsed_time > self.max_time:
            pytest.fail(f"Operation took {self.elapsed_time:.2f}s, expected < {self.max_time}s")


@pytest.fixture
def performance_timer():
    """Provide performance timer utility."""
    return PerformanceTimer


# Memory testing utilities
class MemoryMonitor:
    """Monitor memory usage during tests."""
    
    def __init__(self, max_memory_mb=None):
        self.max_memory_mb = max_memory_mb
        self.initial_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        try:
            import psutil
            import os
            self.process = psutil.Process(os.getpid())
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = self.peak_memory - self.initial_memory
        
        if self.max_memory_mb and memory_increase > self.max_memory_mb:
            pytest.fail(f"Memory usage increased by {memory_increase:.2f}MB, expected < {self.max_memory_mb}MB")


@pytest.fixture
def memory_monitor():
    """Provide memory monitor utility."""
    return MemoryMonitor


# Test markers and skip conditions
def pytest_runtest_setup(item):
    """Setup function called before each test."""
    # Skip slow tests if not explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--runslow", default=False):
        pytest.skip("need --runslow option to run")
    
    # Skip API tests if API module is not available
    if "api" in item.keywords:
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent / 'deployment'))
            import app
        except ImportError:
            pytest.skip("API module not available")


# Custom pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests"
    )


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


# Cleanup functions
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup code here if needed
    pass


@pytest.fixture(autouse=True)
def reset_model_state(request):
    """Reset model state before each test to ensure isolation."""
    # Temporarily disabled to fix API test issues
    # Only reset for API tests to avoid interfering with integration tests
    # if 'test_api.py' in str(request.fspath):
    #     try:
    #         # Reset global model state
    #         import sys
    #         if 'deployment.app' in sys.modules:
    #             from deployment.app import get_model_manager
    #             manager = get_model_manager()
    #             manager.unload_model()
    #     except (ImportError, AttributeError):
    #         pass
    yield


# Test reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom information to test summary."""
    if hasattr(terminalreporter, 'stats'):
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        
        terminalreporter.write_sep("=", "Bank Churn Analysis Test Summary")
        terminalreporter.write_line(f"Tests passed: {passed}")
        terminalreporter.write_line(f"Tests failed: {failed}")
        terminalreporter.write_line(f"Tests skipped: {skipped}")
        
        if failed == 0:
            terminalreporter.write_line("\n✅ All tests passed! The churn analysis pipeline is ready.")
        else:
            terminalreporter.write_line(f"\n❌ {failed} test(s) failed. Please review and fix issues.")