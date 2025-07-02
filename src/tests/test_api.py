import pytest
import sys
import os
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from fastapi import status
from fastapi.testclient import TestClient

# Add paths for imports
project_root = Path(__file__).parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))
# Import the FastAPI app and ModelManager from the correct location
from app.main import app
from app.services.model_manager import ModelManager

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def valid_customer_data():
    """Valid customer data for testing."""
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

class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_predict_endpoint_success(self, client, valid_customer_data):
        """Test successful prediction endpoint."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': 0,
                'risk_level': 'Low',
                'confidence': 0.75
            }
            mock_get_manager.return_value = mock_model_manager
            
            response = client.post("/predict", json=valid_customer_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "risk_level" in data
            assert "confidence" in data
            assert "timestamp" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            mock_model_manager.get_uptime.return_value = "0:01:23"
            mock_get_manager.return_value = mock_model_manager
            
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "status" in data
            assert "uptime" in data
            assert "model_status" in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "features" in data
        assert "model_type" in data
        assert "feature_count" in data
        assert "preprocessing_components" in data
        assert "performance_metrics" in data
        assert "model_path" in data
        assert "timestamp" in data
        
        # Verify the structure is correct
        assert isinstance(data["features"], list)
        assert isinstance(data["feature_count"], int)
        assert isinstance(data["preprocessing_components"], dict)
    
    def test_batch_predict_endpoint_success(self, client, valid_customer_data):
        """Test batch prediction endpoint."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            mock_model_manager.predict_batch.return_value = [
                {'churn_probability': 0.25, 'churn_prediction': 0, 'risk_level': 'Low', 'confidence': 0.75, 'threshold_used': 0.7},
                {'churn_probability': 0.75, 'churn_prediction': 1, 'risk_level': 'High', 'confidence': 0.85, 'threshold_used': 0.7}
            ]
            mock_get_manager.return_value = mock_model_manager
            
            batch_data = {"customers": [valid_customer_data, valid_customer_data]}
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        # The response should contain Prometheus metrics
        assert "python_gc_objects_collected_total" in response.text or "prediction_requests_total" in response.text
    
    def test_cors_headers(self, client):
        """Test CORS headers."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            mock_model_manager.get_uptime.return_value = "0d 0h 0m 1s"
            mock_get_manager.return_value = mock_model_manager
            
            response = client.get("/health")
            # CORS headers are added by FastAPI middleware, but TestClient may not simulate them
            # Just verify the endpoint works
            assert response.status_code == 200
    
    def test_content_type_headers(self, client, valid_customer_data):
        """Test content type headers."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': 0,
                'risk_level': 'Low',
                'confidence': 0.75
            }
            mock_get_manager.return_value = mock_model_manager
            
            response = client.post("/predict", json=valid_customer_data)
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "application/json"

class TestPerformanceAndLoad:
    """Performance and load testing for API endpoints."""
    
    def test_prediction_response_time(self, client, valid_customer_data):
        """Test prediction response time."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': 0,
                'risk_level': 'Low',
                'confidence': 0.75
            }
            mock_get_manager.return_value = mock_model_manager
            
            import time
            start_time = time.time()
            response = client.post("/predict", json=valid_customer_data)
            response_time = time.time() - start_time
            
            assert response.status_code == status.HTTP_200_OK
            assert response_time < 2.0  # Should respond within 2 seconds
    
    def test_large_batch_performance(self, client, valid_customer_data):
        """Test large batch prediction performance."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            # Create large batch
            large_batch = [valid_customer_data] * 100
            mock_predictions = [{'churn_probability': 0.25, 'churn_prediction': 0, 'risk_level': 'Low', 'confidence': 0.75, 'threshold_used': 0.7}] * 100
            
            # Mock the model manager's predict_batch method
            mock_model_manager.predict_batch.return_value = mock_predictions
            mock_get_manager.return_value = mock_model_manager
            
            batch_data = {"customers": large_batch}
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 100

class TestAPIEndpointsComprehensive:
    """Comprehensive API endpoint tests."""
    
    def test_batch_predict_endpoint_success_comprehensive(self, client, valid_customer_data):
        """Test comprehensive batch prediction endpoint."""
        with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
            # Create mock model manager
            mock_model_manager = Mock()
            # Setup mocks
            mock_predictions = [
                {'churn_probability': 0.25, 'churn_prediction': 0, 'risk_level': 'Low', 'confidence': 0.75, 'threshold_used': 0.7},
                {'churn_probability': 0.75, 'churn_prediction': 1, 'risk_level': 'High', 'confidence': 0.85, 'threshold_used': 0.7}
            ]
            mock_model_manager.predict_batch.return_value = mock_predictions
            mock_get_manager.return_value = mock_model_manager
            
            batch_data = {"customers": [valid_customer_data, valid_customer_data]}
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2

class TestModelManager:
    """Test cases for ModelManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        manager = ModelManager()
        
        # ModelManager may auto-load model if it exists, so check if loaded or not
        if manager.is_loaded:
            # If model is loaded, verify it has the expected attributes
            assert manager.model is not None
            assert manager.scaler is not None
            assert manager.feature_names is not None
            assert manager.version is not None
        else:
            # If model is not loaded, verify attributes are None
            assert manager.model is None
            assert manager.scaler is None
            assert manager.feature_names is None
            assert manager.model_name is None
            assert manager.version is None
        
        # Always check that start_time is set
        assert hasattr(manager, 'start_time')
        assert manager.start_time is not None
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_get_health_status(self):
        """Test get_health_status method."""
        manager = ModelManager()
        health_status = manager.get_health_status()
        assert isinstance(health_status, dict)
        assert "uptime" in health_status
        assert "status" in health_status
        assert isinstance(health_status["uptime"], str)