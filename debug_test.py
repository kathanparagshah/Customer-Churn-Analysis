#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add src and deployment to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'deployment'))

# Import API components
deployment_path = str(Path(__file__).parent / 'deployment')
if deployment_path not in sys.path:
    sys.path.insert(0, deployment_path)

try:
    from deployment.app import app
    
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
    
    with patch('deployment.app.model_loaded', True), \
         patch('deployment.app.is_model_loaded', return_value=True), \
         patch('deployment.app.model') as mock_model, \
         patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
         patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
         patch('deployment.app.get_model_manager') as mock_get_manager, \
         patch('deployment.app.model_manager') as mock_model_manager, \
         patch('deployment.app.analytics_db') as mock_analytics_db:
        
        # Set up mocks
        mock_model.predict_proba.return_value = [[0.75, 0.25]]
        mock_preprocess.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        
        # Mock the ModelManager instance
        mock_manager = MagicMock()
        mock_manager.get_uptime.return_value = "0:01:23"
        mock_manager.predict_single.return_value = {
            'churn_probability': 0.25,
            'churn_prediction': False,
            'risk_level': 'Low',
            'confidence': 0.75
        }
        mock_get_manager.return_value = mock_manager
        
        # Mock the global model_manager
        mock_model_manager.predict_single.return_value = {
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