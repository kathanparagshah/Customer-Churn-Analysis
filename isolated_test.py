#!/usr/bin/env python3
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi import status

# Add src and deployment to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'deployment'))

# Import API components
deployment_path = str(Path(__file__).parent / 'deployment')
if deployment_path not in sys.path:
    sys.path.insert(0, deployment_path)

from fastapi.testclient import TestClient
from deployment.app import app

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

def test_predict_endpoint_success_comprehensive(client, valid_customer_data):
    """Test successful prediction endpoint with comprehensive checks."""
    with patch('deployment.app.model_loaded', True), \
         patch('deployment.app.is_model_loaded', return_value=True), \
         patch('deployment.app.model') as mock_model, \
         patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
         patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
         patch('deployment.app.get_model_manager') as mock_get_manager, \
         patch('deployment.app.model_manager') as mock_model_manager, \
         patch('deployment.app.analytics_db') as mock_analytics_db:
        
        # Mock the ModelManager instance
        mock_manager = MagicMock()
        mock_manager.get_uptime.return_value = "0:01:23"
        mock_get_manager.return_value = mock_manager
        
        # Mock preprocessing to return 11 features
        mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        
        # Mock model predictions
        mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])
        mock_model.predict.return_value = np.array([0])
        
        # Mock the global model_manager predict_single method
        mock_model_manager.predict_single.return_value = {
            'churn_probability': 0.25,
            'churn_prediction': False,
            'risk_level': 'Low',
            'confidence': 0.75
        }
        
        response = client.post("/predict", json=valid_customer_data)
        
        # Debug: print response details if test fails
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.content}")
            print(f"Response text: {response.text}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "timestamp" in data
        
        # Check value ranges
        assert 0 <= data["churn_probability"] <= 1
        assert data["churn_prediction"] in [True, False]
        assert data["risk_level"] in ["Low", "Medium", "High"]
        assert 0 <= data["confidence"] <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])