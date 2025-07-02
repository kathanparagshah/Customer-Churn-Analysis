#!/usr/bin/env python3
"""
Minimal Test Script for Customer Churn Analysis API

This script provides a simple test of the churn prediction API
with basic functionality verification.

Usage:
    python minimal_test.py
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from deployment.app_legacy import app
    print("Successfully imported app")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_predict():
    """Test the predict endpoint exactly like the test does."""
    client = TestClient(app)
    
    valid_customer_data = {
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
        
        print("Making request...")
        response = client.post("/predict", json=valid_customer_data)
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.content}")
        print(f"Response text: {response.text}")
        
        if response.status_code != 200:
            print("Test failed!")
            return False
        else:
            print("Test passed!")
            return True

if __name__ == "__main__":
    test_predict()