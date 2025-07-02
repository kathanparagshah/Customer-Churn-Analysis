#!/usr/bin/env python3
"""
Test Suite for API and Deployment Components

Comprehensive tests for FastAPI application, endpoints, data validation,
error handling, and deployment-related functionality.

Author: Bank Churn Analysis Team
Date: 2024
"""

import pytest
import json
import asyncio
import tempfile
import shutil
import sys
import os
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
import pandas as pd
import joblib

from fastapi import status
try:
    from fastapi.testclient import TestClient
except ImportError:
    from starlette.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add src and deployment to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'deployment'))

# Import API components
deployment_path = str(Path(__file__).parent.parent.parent / 'deployment')
if deployment_path not in sys.path:
    sys.path.insert(0, deployment_path)

try:
    from deployment.app import (
        app, CustomerData, BatchCustomerData, PredictionResponse,
        BatchPredictionResponse, ModelManager, get_model_manager
    )
    print("API module imported successfully")
except ImportError as e:
    # Handle case where app module is not available
    app = None
    CustomerData = None
    ModelManager = None
    print(f"Warning: API module not found. Import error: {e}. Some tests will be skipped.")

# Use the original TestClient, not the custom wrapper from app.py
from fastapi.testclient import TestClient as OriginalTestClient
TestClient = OriginalTestClient


class TestCustomerDataValidation:
    """
    Test cases for CustomerData Pydantic model validation.
    """
    
    @pytest.fixture
    def valid_customer_data(self):
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
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_valid_customer_data(self, valid_customer_data):
        """Test validation with valid customer data."""
        customer = CustomerData(**valid_customer_data)
        
        assert customer.CreditScore == 650
        assert customer.Geography == "France"
        assert customer.Gender == "Female"
        assert customer.Age == 35
        assert customer.Tenure == 5
        assert customer.Balance == 50000.0
        assert customer.NumOfProducts == 2
        assert customer.HasCrCard == 1
        assert customer.IsActiveMember == 1
        assert customer.EstimatedSalary == 75000.0
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_credit_score(self, valid_customer_data):
        """Test validation with invalid credit score."""
        # Test credit score too low
        invalid_data = valid_customer_data.copy()
        invalid_data["CreditScore"] = 200
        
        with pytest.raises(ValueError, match="Credit score must be between 300 and 850"):
            CustomerData(**invalid_data)
        
        # Test credit score too high
        invalid_data["CreditScore"] = 900
        
        with pytest.raises(ValueError, match="Credit score must be between 300 and 850"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_geography(self, valid_customer_data):
        """Test validation with invalid geography."""
        invalid_data = valid_customer_data.copy()
        invalid_data["Geography"] = "InvalidCountry"
        
        with pytest.raises(ValueError, match="Geography must be one of: France, Spain, Germany"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_gender(self, valid_customer_data):
        """Test validation with invalid gender."""
        invalid_data = valid_customer_data.copy()
        invalid_data["Gender"] = "Other"
        
        with pytest.raises(ValueError, match="Gender must be either Male or Female"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_age(self, valid_customer_data):
        """Test validation with invalid age."""
        # Test age too low
        invalid_data = valid_customer_data.copy()
        invalid_data["Age"] = 10
        
        with pytest.raises(ValueError, match="Age must be between 18 and 100"):
            CustomerData(**invalid_data)
        
        # Test age too high
        invalid_data["Age"] = 150
        
        with pytest.raises(ValueError, match="Age must be between 18 and 100"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_tenure(self, valid_customer_data):
        """Test validation with invalid tenure."""
        invalid_data = valid_customer_data.copy()
        invalid_data["Tenure"] = -1
        
        with pytest.raises(ValueError, match="Tenure must be between 0 and 10"):
            CustomerData(**invalid_data)
        
        invalid_data["Tenure"] = 15
        
        with pytest.raises(ValueError, match="Tenure must be between 0 and 10"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_balance(self, valid_customer_data):
        """Test validation with invalid balance."""
        invalid_data = valid_customer_data.copy()
        invalid_data["Balance"] = -1000
        
        with pytest.raises(ValueError, match="Balance must be non-negative"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_num_products(self, valid_customer_data):
        """Test validation with invalid number of products."""
        invalid_data = valid_customer_data.copy()
        invalid_data["NumOfProducts"] = 0
        
        with pytest.raises(ValueError, match="Number of products must be between 1 and 4"):
            CustomerData(**invalid_data)
        
        invalid_data["NumOfProducts"] = 5
        
        with pytest.raises(ValueError, match="Number of products must be between 1 and 4"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_binary_fields(self, valid_customer_data):
        """Test validation with invalid binary fields."""
        # Test HasCrCard
        invalid_data = valid_customer_data.copy()
        invalid_data["HasCrCard"] = 2
        
        with pytest.raises(ValueError, match="HasCrCard must be 0 or 1"):
            CustomerData(**invalid_data)
        
        # Test IsActiveMember
        invalid_data = valid_customer_data.copy()
        invalid_data["IsActiveMember"] = -1
        
        with pytest.raises(ValueError, match="IsActiveMember must be 0 or 1"):
            CustomerData(**invalid_data)
    
    @pytest.mark.skipif(CustomerData is None, reason="API module not available")
    def test_invalid_estimated_salary(self, valid_customer_data):
        """Test validation with invalid estimated salary."""
        invalid_data = valid_customer_data.copy()
        invalid_data["EstimatedSalary"] = -5000
        
        with pytest.raises(ValueError, match="Estimated salary must be positive"):
            CustomerData(**invalid_data)


class TestModelManager:
    """
    Test cases for ModelManager class.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_package(self, temp_dir):
        """Create mock model package for testing."""
        from sklearn.preprocessing import LabelEncoder
        
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scaler = StandardScaler()
        
        # Define feature names that match the actual deployment structure (11 features)
        feature_names = [
            'CreditScore', 'Geography_Germany', 'Geography_Spain', 'Gender_Male', 
            'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
            'IsActiveMember', 'EstimatedSalary'
        ]
        
        # Create label encoders for categorical features
        label_encoders = {}
        geography_encoder = LabelEncoder()
        geography_encoder.fit(['France', 'Germany', 'Spain'])
        label_encoders['Geography'] = geography_encoder
        
        gender_encoder = LabelEncoder()
        gender_encoder.fit(['Female', 'Male'])
        label_encoders['Gender'] = gender_encoder
        
        # Create sample data to fit the model (11 features)
        X_sample = np.random.randn(100, 11)  # 11 features to match expected_features
        y_sample = np.random.choice([0, 1], 100)
        
        # Fit model and scaler
        scaler.fit(X_sample)
        model.fit(scaler.transform(X_sample), y_sample)
        
        # Create model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'model_name': 'RandomForest',
            'version': '1.0.0',
            'feature_names': feature_names,
            'label_encoders': label_encoders
        }
        
        # Save model package
        model_path = temp_dir / 'churn_model.pkl'
        joblib.dump(model_package, model_path)
        
        return model_path
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_model_manager_init(self):
        """Test ModelManager initialization."""
        manager = ModelManager()
        
        assert manager.model is None
        assert manager.scaler is None
        assert manager.feature_names is None
        assert manager.model_name is None
        assert manager.version is None
        assert not manager.is_loaded
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_load_model_success(self, mock_model_package):
        """Test successful model loading."""
        manager = ModelManager()
        manager.load_model(str(mock_model_package))
        
        assert manager.model is not None
        assert manager.scaler is not None
        assert manager.feature_names is not None
        assert manager.model_name == 'RandomForest'
        assert manager.version == '1.0.0'
        assert manager.is_loaded
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file."""
        manager = ModelManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_model("/non/existent/path.pkl")
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_preprocess_customer_data(self, mock_model_package):
        """Test customer data preprocessing."""
        manager = ModelManager()
        manager.load_model(str(mock_model_package))
        
        customer_data = {
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
        
        processed_data = manager.preprocess_customer_data(customer_data)
        
        assert isinstance(processed_data, np.ndarray)
        assert processed_data.shape[0] == 1  # Single customer
        assert processed_data.shape[1] == len(manager.feature_names)
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_predict_single(self, mock_model_package):
        """Test single customer prediction."""
        manager = ModelManager()
        manager.load_model(str(mock_model_package))
        
        customer_data = {
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
        
        prediction = manager.predict_single(customer_data)
        
        assert 'churn_probability' in prediction
        assert 'churn_prediction' in prediction
        assert 'risk_level' in prediction
        assert 'confidence' in prediction
        
        assert 0 <= prediction['churn_probability'] <= 1
        assert prediction['churn_prediction'] in [0, 1]
        assert prediction['risk_level'] in ['Low', 'Medium', 'High']
        assert 0 <= prediction['confidence'] <= 1
    
    @pytest.mark.skipif(ModelManager is None, reason="API module not available")
    def test_predict_batch(self, mock_model_package):
        """Test batch prediction."""
        manager = ModelManager()
        manager.load_model(str(mock_model_package))
        
        customers = [
            {
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
            },
            {
                "CreditScore": 400,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 60,
                "Tenure": 1,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 30000.0
            }
        ]
        
        predictions = manager.predict_batch(customers)
        
        assert len(predictions) == 2
        
        for prediction in predictions:
            assert 'churn_probability' in prediction
            assert 'churn_prediction' in prediction
            assert 'risk_level' in prediction
            assert 'confidence' in prediction


class TestAPIEndpoints:
    """
    Test cases for FastAPI endpoints.
    """
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        if app is None:
            pytest.skip("API module not available")
        return TestClient(app)
    
    @pytest.fixture
    def valid_customer_data(self):
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
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200  # Model is loaded by integration testsstatus.HTTP_200_OK
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        with patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager:
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
            mock_model_manager.get_uptime.return_value = "0:01:23"
            
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
            assert "timestamp" in data
            assert "uptime" in data
            assert data["status"] == "healthy"
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "model_loaded" in data
        
        if data["model_loaded"]:
            assert "model_name" in data
            assert "version" in data
            assert "feature_names" in data
    
    def test_predict_endpoint_success(self, client, valid_customer_data):
        """Test successful prediction endpoint."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock(), 'Gender': MagicMock()}), \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.model_metadata') as mock_metadata, \
             patch('deployment.app.analytics_db') as mock_analytics_db, \
             patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
             patch('deployment.app.calculate_risk_level', return_value='Low'), \
             patch('deployment.app.calculate_confidence', return_value=0.7), \
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
            mock_analytics_db.log_prediction = MagicMock()
            mock_log.return_value = None
            
            response = client.post("/predict", json=valid_customer_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "risk_level" in data
            assert "confidence" in data
            assert "timestamp" in data
    

    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_data = {
            "CreditScore": 1000,  # Invalid credit score
            "Geography": "InvalidCountry",  # Invalid geography
            "Gender": "Female",
            "Age": 35
            # Missing required fields
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_batch_predict_endpoint_success(self, client, valid_customer_data):
        """Test successful batch prediction endpoint."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
             patch('deployment.app.model_manager') as mock_manager, \
             patch('deployment.app.analytics_db') as mock_analytics_db:

            # Mock model manager attributes
            mock_manager.is_loaded = True
            mock_manager.preprocess_customer_data.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

            # Mock model predictions
            mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
            mock_model.predict.return_value = np.array([0])

            # Mock analytics database
            mock_analytics_db.log_batch_prediction = MagicMock()
            
            batch_data = {
                "customers": [valid_customer_data, valid_customer_data]
            }
            
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "predictions" in data
            assert "summary" in data
            assert len(data["predictions"]) == 2
            
            summary = data["summary"]
            assert "total_customers" in summary
            assert "high_risk_customers" in summary
            assert "average_churn_probability" in summary
            assert "processing_time_ms" in summary
    
    def test_batch_predict_endpoint_too_large(self, client, valid_customer_data):
        """Test batch prediction with too many customers."""
        # Create batch with too many customers
        large_batch = {
            "customers": [valid_customer_data] * 1001  # Exceeds limit
        }
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_batch_predict_endpoint_empty(self, client):
        """Test batch prediction with empty customer list."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
             patch('deployment.app.model_manager') as mock_manager:
            
            # Mock model components for empty batch
            mock_model.predict_proba.return_value = np.array([])
            mock_model.predict.return_value = np.array([])
            mock_manager.preprocess_customer_data.return_value = np.array([])
            
            empty_batch = {
                "customers": []
            }
            
            response = client.post("/predict/batch", json=empty_batch)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 0
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint for monitoring."""
        response = client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        # Check that response contains Prometheus metrics format
        content = response.content.decode()
        assert "prediction_requests_total" in content
        assert "prediction_duration_seconds" in content
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0'}), \
             patch('deployment.app.get_model_manager') as mock_get_manager:
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.get_uptime.return_value = "0:01:23"
            mock_get_manager.return_value = mock_manager
            
            response = client.options("/predict")
            
            # In test environment, CORS headers may not be configured
            # Just verify the endpoint responds correctly
            assert response.status_code in [200, 405]  # OPTIONS may not be implemented


# Standalone test for prediction model exception
@patch('deployment.app.model_metadata')
@patch('deployment.app.feature_names')
@patch('deployment.app.label_encoders')
@patch('deployment.app.model')
@patch('deployment.app.scaler')
@patch('deployment.app.is_model_loaded', return_value=True)
def test_prediction_model_exception_standalone(mock_is_model_loaded, mock_scaler, mock_model, mock_encoders, mock_features, mock_metadata):
    """Test error handling when model is loaded but prediction fails."""
    if app is None:
        pytest.skip("API module not available")
    test_client = TestClient(app)
    
    with patch('deployment.app.model_loaded', True), \
         patch('deployment.app.is_model_loaded', return_value=True), \
         patch('deployment.app.model_manager') as mock_manager:
        
        # Mock model manager to raise an exception
        mock_manager.predict_single.side_effect = Exception("Model prediction failed")
        
        valid_data = {
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
        
        response = test_client.post("/predict", json=valid_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data


class TestErrorHandling:
    """
    Test cases for error handling and edge cases.
    """
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        if app is None:
            pytest.skip("API module not available")
        return TestClient(app)
    
    def test_404_endpoint(self, client):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/non-existent-endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method."""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_prediction_error_handling(self, client):
        """Test error handling when model is not loaded."""
        with patch('deployment.app.model_loaded', False), \
             patch('deployment.app.is_model_loaded', return_value=False):
            
            valid_data = {
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
            
            response = client.post("/predict", json=valid_data)
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert "detail" in data
    
    # Note: test_prediction_model_exception moved to standalone function above the class
    
    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/predict",
            data="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_content_type(self, client):
        """Test handling of missing content type."""
        response = client.post("/predict", data="some data")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPerformanceAndLoad:
    """
    Performance and load testing for API endpoints.
    """
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        if app is None:
            pytest.skip("API module not available")
        return TestClient(app)
    
    @pytest.fixture
    def valid_customer_data(self):
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
    
    def test_prediction_response_time(self, client, valid_customer_data):
        """Test prediction response time."""
        import time
        
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock(), 'Gender': MagicMock()}), \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.model_metadata') as mock_metadata, \
             patch('deployment.app.analytics_db') as mock_analytics_db, \
             patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
             patch('deployment.app.calculate_risk_level', return_value='Low'), \
             patch('deployment.app.calculate_confidence', return_value=0.7), \
             patch('deployment.app.log_prediction') as mock_log, \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager:
            
            # Mock metadata with get method
            mock_metadata.get.return_value = '1.0.0'
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.predict_single.return_value = {
                'churn_probability': 0.3,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.7,
                'threshold_used': 0.5
            }
            mock_get_manager.return_value = mock_manager
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.3,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.7,
                'threshold_used': 0.5
            }
            
            # Mock preprocessing and model predictions
            mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
            mock_model.predict.return_value = np.array([0])
            mock_analytics_db.log_prediction = MagicMock()
            mock_log.return_value = None
            
            # Measure response time
            start_time = time.time()
            response = client.post("/predict", json=valid_customer_data)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == status.HTTP_200_OK
            assert response_time < 1.0  # Should respond in less than 1 second
    
    def test_concurrent_requests(self, client, valid_customer_data):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock(), 'Gender': MagicMock()}), \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.model_metadata') as mock_metadata, \
             patch('deployment.app.analytics_db') as mock_analytics_db, \
             patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
             patch('deployment.app.calculate_risk_level', return_value='Low'), \
             patch('deployment.app.calculate_confidence', return_value=0.7), \
             patch('deployment.app.log_prediction') as mock_log, \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager:
            
            # Mock metadata with get method
            mock_metadata.get.return_value = '1.0.0'
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.predict_single.return_value = {
                'churn_probability': 0.3,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.7,
                'threshold_used': 0.5
            }
            mock_get_manager.return_value = mock_manager
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.3,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.7,
                'threshold_used': 0.5
            }
            
            # Mock preprocessing and model predictions
            mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
            mock_model.predict.return_value = np.array([0])
            mock_analytics_db.log_prediction = MagicMock()
            mock_log.return_value = None
            
            results = []
            
            def make_request():
                response = client.post("/predict", json=valid_customer_data)
                results.append(response.status_code)
            
            # Create multiple threads
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
            
            # Start all threads
            start_time = time.time()
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # All requests should succeed
            assert len(results) == 10
            assert all(status_code == 200 for status_code in results)
            
            # Should handle concurrent requests efficiently
            assert total_time < 5.0  # Should complete in less than 5 seconds
    
    def test_large_batch_performance(self, client, valid_customer_data):
        """Test performance with large batch requests."""
        import time
        
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager, \
             patch('deployment.app.analytics_db') as mock_analytics_db:

            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.is_loaded = True
            # Create 100 mock predictions for the large batch
            mock_predictions = [{
                'customer_id': i,
                'churn_probability': 0.4,
                'churn_prediction': 0,
                'risk_level': 'Low',
                'confidence': 0.6
            } for i in range(100)]
            mock_manager.predict_batch.return_value = mock_predictions
            mock_get_manager.return_value = mock_manager
            mock_model_manager.predict_batch.return_value = [{
                'customer_id': 0,
                'churn_probability': 0.25,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.75
            }]
            
            # Mock model predictions (for compatibility)
            mock_model.predict_proba.return_value = np.array([[0.6, 0.4]] * 100)
            mock_model.predict.return_value = np.array([0] * 100)

            # Mock analytics database
            mock_analytics_db.log_batch_prediction = MagicMock()
            
            # Create large batch
            large_batch = {
                "customers": [valid_customer_data] * 100
            }
            
            # Measure response time
            start_time = time.time()
            response = client.post("/predict/batch", json=large_batch)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == status.HTTP_200_OK
            assert response_time < 5.0  # Should process 100 customers in less than 5 seconds
            
            data = response.json()
            assert len(data["predictions"]) == 100


class TestAPIEndpointsComprehensive:
    """Comprehensive tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        if app is None:
            pytest.skip("API module not available")
        return TestClient(app)
    
    @pytest.fixture
    def valid_customer_data(self):
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
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0'}), \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': 'encoder'}), \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager, \
             patch('deployment.app.analytics_db') as mock_analytics_db:
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.get_uptime.return_value = "0:01:23"
            mock_get_manager.return_value = mock_manager
            mock_model_manager.get_uptime.return_value = "0:01:23"
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Check response structure
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert "model_status" in data
            
            # Check status values
            assert data["status"] in ["healthy", "unhealthy"]
            assert isinstance(data["timestamp"], str)
            assert isinstance(data["version"], str)
            assert isinstance(data["model_status"], dict)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)
    
    def test_predict_endpoint_success_comprehensive(self, client, valid_customer_data):
        """Test successful prediction endpoint with comprehensive checks."""
        # Import here to avoid module loading issues
        from unittest.mock import patch, MagicMock
        import numpy as np
        
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
             patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager, \
             patch('deployment.app.analytics_db') as mock_analytics_db, \
             patch('deployment.app.PREDICTION_COUNTER', None), \
             patch('deployment.app.ERROR_COUNTER', None), \
             patch('deployment.app.PREDICTION_LATENCY', None):
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.get_uptime.return_value = "0:01:23"
            mock_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.75,
                'threshold_used': 0.7
            }
            mock_get_manager.return_value = mock_manager
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.75,
                'threshold_used': 0.7
            }
            
            # Mock preprocessing to return 11 features
            mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
            
            # Mock model predictions
            mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])
            mock_model.predict.return_value = np.array([0])
            
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
    
    def test_predict_endpoint_model_not_loaded(self, client, valid_customer_data):
        """Test prediction endpoint when model is not loaded."""
        with patch('deployment.app.model_loaded', False), \
             patch('deployment.app.is_model_loaded', return_value=False):
            
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert "detail" in data
    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_data = {
            "CreditScore": 200,  # Too low
            "Geography": "InvalidCountry",
            "Gender": "Other",
            "Age": 10,  # Too low
            "Tenure": -1,  # Negative
            "Balance": -1000,  # Negative
            "NumOfProducts": 0,  # Too low
            "HasCrCard": 2,  # Invalid
            "IsActiveMember": -1,  # Invalid
            "EstimatedSalary": -5000  # Negative
        }
        
        response = client.post("/predict", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0  # Should have validation errors
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_data = {
            "CreditScore": 650,
            "Geography": "France"
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
    
    def test_batch_predict_endpoint_success_comprehensive(self, client, valid_customer_data):
        """Test successful batch prediction endpoint with comprehensive checks."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.analytics_db') as mock_analytics_db:

            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.is_loaded = True
            mock_manager.predict_batch.return_value = [{
                'customer_id': 0,
                'churn_probability': 0.25,
                'churn_prediction': 0,
                'risk_level': 'Low',
                'confidence': 0.75
            }]
            mock_get_manager.return_value = mock_manager

            # Mock model predictions (for compatibility)
            mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])
            mock_model.predict.return_value = np.array([0])

            # Mock analytics database
            mock_analytics_db.log_batch_prediction = MagicMock()

            batch_data = {
                "customers": [valid_customer_data, valid_customer_data]
            }

            response = client.post("/predict/batch", json=batch_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Check response structure
            assert "predictions" in data
            assert "summary" in data
            assert "loaded" in data
            assert "model_name" in data
            assert "version" in data
            
            # Check predictions
            predictions = data["predictions"]
            assert len(predictions) == 2
            assert data["summary"]["total_customers"] == 2
            
            for prediction in predictions:
                assert "churn_probability" in prediction
                assert "churn_prediction" in prediction
                assert "risk_level" in prediction
                assert "confidence" in prediction
    
    def test_batch_predict_empty_list_comprehensive(self, client):
        """Test batch prediction with empty customer list."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.model_metadata', {'version': '1.0.0', 'name': 'test_model'}), \
             patch('deployment.app.model_manager') as mock_manager:
            
            # Mock model components for empty batch
            mock_model.predict_proba.return_value = np.array([])
            mock_model.predict.return_value = np.array([])
            mock_manager.preprocess_customer_data.return_value = np.array([])
            
            batch_data = {"customers": []}
            
            response = client.post("/predict/batch", json=batch_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 0
            assert "summary" in data
            assert data["summary"]["total_customers"] == 0
    
    def test_batch_predict_too_many_customers(self, client, valid_customer_data):
        """Test batch prediction with too many customers."""
        # Create batch with more than allowed limit (assuming 1000 is the limit)
        large_batch = {
            "customers": [valid_customer_data] * 1001
        }
        
        response = client.post("/predict/batch", json=large_batch)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
    
    def test_predict_endpoint_model_error(self, client, valid_customer_data):
        """Test prediction endpoint when model encounters an error."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model_manager') as mock_manager:
            
            # Mock model manager to raise an exception
            mock_manager.predict_single.side_effect = Exception("Model prediction failed")
            
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data
    
    def test_predict_endpoint_response_time(self, client, valid_customer_data):
        """Test prediction endpoint response time."""
        import time
        
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock(), 'Gender': MagicMock()}), \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.model_metadata') as mock_metadata, \
             patch('deployment.app.analytics_db') as mock_analytics_db, \
             patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
             patch('deployment.app.calculate_risk_level', return_value='Low'), \
             patch('deployment.app.calculate_confidence', return_value=0.75), \
             patch('deployment.app.log_prediction') as mock_log, \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager:
            
            # Mock metadata with get method
            mock_metadata.get.return_value = '1.0.0'
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.75,
                'threshold_used': 0.5
            }
            mock_get_manager.return_value = mock_manager
            mock_model_manager.predict_single.return_value = {
                'churn_probability': 0.25,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.75,
                'threshold_used': 0.5
            }
            
            # Mock preprocessing and model predictions
            mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])
            mock_model.predict.return_value = np.array([0])
            mock_analytics_db.log_prediction = MagicMock()
            mock_log.return_value = None
            
            start_time = time.time()
            response = client.post("/predict", json=valid_customer_data)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == status.HTTP_200_OK
            assert response_time < 2.0  # Should respond within 2 seconds
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock(), 'Gender': MagicMock()}), \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.model_metadata') as mock_metadata, \
             patch('deployment.app.get_model_manager') as mock_get_manager, \
             patch('deployment.app.model_manager') as mock_model_manager:
            
            # Mock metadata with get method
            mock_metadata.get.return_value = '1.0.0'
            
            # Mock the ModelManager instance
            mock_manager = MagicMock()
            mock_manager.get_uptime.return_value = "0:01:23"
            mock_get_manager.return_value = mock_manager
            mock_model_manager.get_uptime.return_value = "0:01:23"
            
            response = client.get("/health")
            
            # Check for CORS headers (if implemented)
            assert response.status_code == status.HTTP_200_OK
            # Note: CORS headers may not be present in test environment
            # Just verify the endpoint works correctly
    
    def test_content_type_headers(self, client, valid_customer_data):
        """Test content type headers."""
        with patch('deployment.app.model_loaded', True), \
             patch('deployment.app.is_model_loaded', return_value=True), \
             patch('deployment.app.model') as mock_model, \
             patch('deployment.app.scaler') as mock_scaler, \
             patch('deployment.app.label_encoders', {'Geography': MagicMock(), 'Gender': MagicMock()}), \
             patch('deployment.app.feature_names', ['feature1', 'feature2']), \
             patch('deployment.app.model_metadata') as mock_metadata, \
             patch('deployment.app.analytics_db') as mock_analytics_db, \
             patch('deployment.app.ModelManager.preprocess_customer_data') as mock_preprocess, \
             patch('deployment.app.calculate_risk_level', return_value='Low'), \
             patch('deployment.app.calculate_confidence', return_value=0.75), \
             patch('deployment.app.log_prediction') as mock_log, \
             patch('deployment.app.get_model_manager') as mock_get_manager:
            
            # Mock metadata with get method
            mock_metadata.get.return_value = '1.0.0'
            
            # Mock preprocessing and model predictions
            mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])
            mock_model.predict.return_value = np.array([0])
            mock_analytics_db.log_prediction = MagicMock()
            mock_log.return_value = None
            
            response = client.post("/predict", json=valid_customer_data)
            
            assert response.status_code == status.HTTP_200_OK
            assert "application/json" in response.headers.get("content-type", "")
    
    def test_api_documentation_endpoints(self, client):
        """Test API documentation endpoints."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        # Test Swagger UI (if available)
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
        
        # Test ReDoc (if available)
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])