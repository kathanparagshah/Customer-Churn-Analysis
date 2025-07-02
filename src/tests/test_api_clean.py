import pytest
import sys
import os
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient

# Add paths for imports
src_path = Path(__file__).parent.parent
deployment_path = src_path.parent / 'deployment'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(deployment_path))
sys.path.insert(0, str(src_path / 'tests' / 'deployment'))

try:
    from deployment.app import app
except ImportError:
    app = None

@pytest.fixture
def client():
    """Create test client."""
    if app is None:
        pytest.skip("API module not available")
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
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock(), 'Gender': MagicMock()}), \
             patch('deployment.app.feature_names', ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']), \
             patch('deployment.app.model_metadata') as mock_metadata, \
             patch('deployment.app.analytics_db') as mock_analytics_db, \
             patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
             patch('deployment.app.calculate_risk_level') as mock_risk, \
             patch('deployment.app.calculate_confidence') as mock_confidence, \
             patch('deployment.app.log_prediction') as mock_log, \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager:
            
            # Mock metadata with get method
            mock_metadata.get.return_value = '1.0.0'
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.get_uptime.return_value = "0:01:23"
            mock_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': 0,
                'risk_level': 'Low',
                'confidence': 0.75
            }
            mock_get_manager.return_value = mock_manager
            
            # Mock preprocessing and model predictions
            mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
            mock_model.predict.return_value = np.array([0])
            
            # Mock analytics database
            mock_analytics_db.log_prediction = MagicMock()
            
            # Mock risk and confidence calculations
            mock_risk.return_value = 'Low'
            mock_confidence.return_value = 0.75
            mock_log.return_value = None
            
            # Mock model manager
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': 0,
                'risk_level': 'Low',
                'confidence': 0.75
            }
            
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
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock()}), \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager:
            
            # Mock model class name
            mock_model.__class__.__name__ = 'RandomForestClassifier'
            
            # Mock ModelManager
            mock_manager = MagicMock()
            mock_manager.get_uptime.return_value = "0:01:23"
            mock_get_manager.return_value = mock_manager
            mock_model_manager.get_uptime.return_value = "0:01:23"
            
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "status" in data
            assert "uptime" in data
            assert "model_status" in data