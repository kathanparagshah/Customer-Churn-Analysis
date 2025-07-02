#!/usr/bin/env python3
"""
Isolated Test Script for Customer Churn Analysis API

This script provides comprehensive testing of the churn prediction API
with proper mocking and isolation from external dependencies.

Usage:
    python isolated_test.py
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import pytest

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from deployment.app_legacy import app
except ImportError as e:
    print(f"Import error: {e}")
    print("Available paths:")
    for path in sys.path:
        print(f"  {path}")
    raise

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
    with patch('app.services.model_manager.model_manager') as mock_manager, \
         patch('deployment.app_legacy.analytics_db') as mock_analytics_db:
        
        # Set up mock model manager
        mock_manager.is_loaded = True
        mock_manager.get_uptime.return_value = "0:01:23"
        mock_manager.predict_single.return_value = {
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