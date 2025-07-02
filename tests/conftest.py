"""Pytest configuration and shared fixtures for testing."""

import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import the FastAPI app
from app.main import app
from app.config import settings
from app.services.model_manager import ModelManager
from app.services.analytics import AnalyticsDB


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing.
    
    Returns:
        AppSettings: Test configuration
    """
    # TODO: Create test-specific settings
    # - Use in-memory databases
    # - Disable external services
    # - Set test-specific paths
    
    with patch.object(settings, 'ENVIRONMENT', 'test'):
        with patch.object(settings, 'DEBUG', True):
            with patch.object(settings, 'LOG_LEVEL', 'DEBUG'):
                yield settings


@pytest.fixture(scope="session")
def client(test_settings):
    """Create a test client for the FastAPI app.
    
    Args:
        test_settings: Test settings fixture
        
    Returns:
        TestClient: FastAPI test client
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager for testing.
    
    Returns:
        Mock: Mocked ModelManager instance
    """
    mock_manager = Mock(spec=ModelManager)
    mock_manager.is_loaded = True
    mock_manager.model_name = "test_model"
    mock_manager.version = "1.0.0"
    mock_manager.feature_names = ["feature1", "feature2", "feature3"]
    
    # Mock prediction methods
    mock_manager.predict_single.return_value = {
        'churn_probability': 0.75,
        'churn_prediction': True,
        'risk_level': 'High',
        'confidence': 0.85,
        'threshold_used': 0.5,
        'model_version': '1.0.0'
    }
    
    mock_manager.predict_batch.return_value = [
        {
            'churn_probability': 0.75,
            'churn_prediction': True,
            'risk_level': 'High',
            'confidence': 0.85,
            'threshold_used': 0.5,
            'model_version': '1.0.0'
        }
    ]
    
    # Mock info methods
    mock_manager.get_model_info.return_value = {
        "model_name": "test_model",
        "version": "1.0.0",
        "training_date": "2024-01-01",
        "model_type": "RandomForestClassifier",
        "features": ["feature1", "feature2", "feature3"],
        "feature_count": 3,
        "preprocessing_components": {
            "scaler": "StandardScaler",
            "label_encoders": ["Geography", "Gender"],
            "feature_count": 3
        },
        "performance_metrics": {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75
        },
        "model_path": "/test/model.pkl",
        "timestamp": "2024-01-01T00:00:00"
    }
    
    mock_manager.get_health_status.return_value = {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0",
        "uptime": "1:00:00",
        "timestamp": "2024-01-01T00:00:00",
        "model_status": {
            "loaded": True,
            "model_type": "RandomForestClassifier",
            "features_count": 3,
            "preprocessing_ready": True
        }
    }
    
    return mock_manager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.
    
    Returns:
        str: Path to temporary database file
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_analytics_db(temp_db):
    """Create a mock analytics database for testing.
    
    Args:
        temp_db: Temporary database path fixture
        
    Returns:
        AnalyticsDB: Test analytics database instance
    """
    return AnalyticsDB(temp_db)


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing.
    
    Returns:
        dict: Sample customer data
    """
    return {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0
    }


@pytest.fixture
def sample_batch_data(sample_customer_data):
    """Sample batch customer data for testing.
    
    Args:
        sample_customer_data: Sample customer data fixture
        
    Returns:
        dict: Sample batch data
    """
    return {
        "customers": [
            sample_customer_data,
            {**sample_customer_data, "Age": 45, "Balance": 75000.0},
            {**sample_customer_data, "Age": 25, "Balance": 25000.0}
        ]
    }


@pytest.fixture(autouse=True)
def mock_model_loading():
    """Mock model loading to avoid file system dependencies.
    
    This fixture automatically mocks model loading for all tests.
    """
    with patch('app.services.model_manager.is_model_loaded', return_value=True):
        yield


@pytest.fixture
def override_model_manager(mock_model_manager):
    """Override the model manager dependency for testing.
    
    Args:
        mock_model_manager: Mock model manager fixture
    """
    from app.main import app
    from app.services.model_manager import get_model_manager
    
    app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_metrics():
    """Mock Prometheus metrics for testing.
    
    Returns:
        Mock: Mocked metrics objects
    """
    with patch('app.routes.predict.PREDICTION_COUNTER') as mock_counter:
        with patch('app.routes.predict.BATCH_PREDICTION_COUNTER') as mock_batch_counter:
            with patch('app.routes.predict.PREDICTION_LATENCY') as mock_latency:
                with patch('app.routes.predict.ERROR_COUNTER') as mock_error_counter:
                    yield {
                        'prediction_counter': mock_counter,
                        'batch_prediction_counter': mock_batch_counter,
                        'prediction_latency': mock_latency,
                        'error_counter': mock_error_counter
                    }


# Test data constants
TEST_MODEL_VERSION = "1.0.0"
TEST_BATCH_ID = "test-batch-123"
TEST_CUSTOMER_ID = "test-customer-456"


# Helper functions for tests
def assert_prediction_response(response_data):
    """Assert that a prediction response has the correct structure.
    
    Args:
        response_data: Response data to validate
    """
    required_fields = [
        'churn_probability', 'churn_prediction', 'risk_level', 
        'confidence', 'timestamp', 'version'
    ]
    
    for field in required_fields:
        assert field in response_data, f"Missing field: {field}"
    
    assert isinstance(response_data['churn_probability'], float)
    assert isinstance(response_data['churn_prediction'], bool)
    assert response_data['risk_level'] in ['Low', 'Medium', 'High']
    assert isinstance(response_data['confidence'], float)
    assert 0 <= response_data['confidence'] <= 1


def assert_health_response(response_data):
    """Assert that a health response has the correct structure.
    
    Args:
        response_data: Response data to validate
    """
    required_fields = ['status', 'loaded', 'version', 'uptime', 'timestamp']
    
    for field in required_fields:
        assert field in response_data, f"Missing field: {field}"
    
    assert response_data['status'] in ['healthy', 'degraded', 'unhealthy']
    assert isinstance(response_data['loaded'], bool)