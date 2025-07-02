"""Test to verify 503 responses when model is not loaded."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.main import app


class Test503Verification:
    """Test class to verify 503 Service Unavailable responses."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def valid_customer_data(self):
        """Valid customer data for testing."""
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
            "EstimatedSalary": 60000.0
        }
    
    def test_predict_endpoint_503_when_model_not_loaded(self, client, valid_customer_data):
        """Test /predict endpoint returns 503 when model is not loaded."""
        with patch('app.services.model_manager.model_manager') as mock_model_manager:
            mock_model_manager.is_loaded = False
            
            response = client.post("/predict", json=valid_customer_data)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_predict_endpoint_200_when_model_loaded(self, client, valid_customer_data):
        """Test /predict endpoint returns 200 when model is loaded."""
        with patch('app.services.model_manager.model_manager') as mock_model_manager:
            mock_model_manager.is_loaded = True
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.75
            }
            
            response = client.post("/predict", json=valid_customer_data)
            assert response.status_code == 200
    
    def test_batch_predict_endpoint_503_when_model_not_loaded(self, client, valid_customer_data):
        """Test /predict/batch endpoint returns 503 when model is not loaded."""
        with patch('app.services.model_manager.model_manager') as mock_model_manager:
            mock_model_manager.is_loaded = False
            
            batch_data = {"customers": [valid_customer_data]}
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_batch_predict_endpoint_200_when_model_loaded(self, client, valid_customer_data):
        """Test /predict/batch endpoint returns 200 when model is loaded."""
        with patch('app.services.model_manager.model_manager') as mock_model_manager:
            mock_model_manager.is_loaded = True
            mock_model_manager.predict_batch.return_value = [
                {'churn_probability': 0.25, 'churn_prediction': False, 'risk_level': 'Low', 'confidence': 0.75}
            ]
            
            batch_data = {"customers": [valid_customer_data]}
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == 200
    
    def test_model_info_endpoint_503_when_model_not_loaded(self, client):
        """Test /model/info endpoint returns 503 when model is not loaded."""
        with patch('app.services.model_manager.model_manager') as mock_model_manager:
            mock_model_manager.is_loaded = False
            
            response = client.get("/model/info")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_model_info_endpoint_200_when_model_loaded(self, client):
        """Test /model/info endpoint returns 200 when model is loaded."""
        with patch('app.services.model_manager.model_manager') as mock_model_manager:
            mock_model_manager.is_loaded = True
            mock_model_manager.get_model_info.return_value = {
                "model_name": "churn_model",
                "version": "1.0.0",
                "features": ["CreditScore", "Geography"],
                "model_type": "RandomForest",
                "feature_count": 10,
                "preprocessing_components": {},
                "performance_metrics": {},
                "model_path": "/path/to/model",
                "timestamp": "2024-01-01T00:00:00"
            }
            
            response = client.get("/model/info")
            assert response.status_code == 200