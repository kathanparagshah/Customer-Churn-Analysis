#!/usr/bin/env python3
"""
Debug Test Script for Customer Churn Analysis API

This script tests the basic functionality of the churn prediction API
to ensure the model loading and prediction endpoints work correctly.

Usage:
    python debug_test.py
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from deployment.app_legacy import app
    
    # Set up the test
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
    
    with patch('deployment.app_legacy.model_manager') as mock_manager, \
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
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.content}")
        print(f"Response text: {response.text}")
        
        if response.status_code != 200:
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail}")
            except:
                print("Could not parse error as JSON")
                
except Exception as e:
    print(f"Error running test: {e}")
    import traceback
    traceback.print_exc()