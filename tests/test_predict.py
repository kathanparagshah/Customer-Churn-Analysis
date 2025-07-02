"""Tests for prediction endpoints."""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from tests.conftest import assert_prediction_response


class TestPredictEndpoint:
    """Test cases for the /predict endpoint."""
    
    def test_predict_success(self, client: TestClient, mock_model_manager, sample_customer_data, mock_metrics):
        """Test successful single prediction.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            sample_customer_data: Sample customer data
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        # Override the dependency
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict", json=sample_customer_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert_prediction_response(data)
            
            # Check specific prediction values
            assert data["churn_probability"] == 0.75
            assert data["churn_prediction"] is True
            assert data["risk_level"] == "High"
            assert data["confidence"] == 0.85
            assert data["version"] == "1.0.0"
            
            # Verify metrics were called
            mock_metrics['prediction_counter'].inc.assert_called_once()
            mock_metrics['prediction_latency'].observe.assert_called_once()
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_predict_model_not_loaded(self, client: TestClient, mock_model_manager, sample_customer_data, mock_metrics):
        """Test prediction when model is not loaded.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            sample_customer_data: Sample customer data
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=False):
                response = client.post("/predict", json=sample_customer_data)
            
            assert response.status_code == 503
            data = response.json()
            
            assert "detail" in data
            assert "Model not loaded" in data["detail"]
            
            # Verify error counter was incremented
            mock_metrics['error_counter'].inc.assert_called_once()
        finally:
            app.dependency_overrides.clear()
    
    def test_predict_invalid_data(self, client: TestClient, mock_model_manager, mock_metrics):
        """Test prediction with invalid data.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        invalid_data = {
            "CreditScore": "invalid",  # Should be numeric
            "Geography": "France",
            "Gender": "Male"
            # Missing required fields
        }
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict", json=invalid_data)
            
            assert response.status_code == 422  # Validation error
        finally:
            app.dependency_overrides.clear()
    
    def test_predict_exception_handling(self, client: TestClient, mock_model_manager, sample_customer_data, mock_metrics):
        """Test prediction exception handling.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            sample_customer_data: Sample customer data
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        # Configure mock to raise exception
        mock_model_manager.predict_single.side_effect = Exception("Prediction error")
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict", json=sample_customer_data)
            
            assert response.status_code == 500
            data = response.json()
            
            assert "detail" in data
            assert "Prediction failed" in data["detail"]
            
            # Verify error counter was incremented
            mock_metrics['error_counter'].inc.assert_called_once()
        finally:
            app.dependency_overrides.clear()


class TestPredictBatchEndpoint:
    """Test cases for the /predict/batch endpoint."""
    
    def test_predict_batch_success(self, client: TestClient, mock_model_manager, sample_batch_data, mock_metrics):
        """Test successful batch prediction.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            sample_batch_data: Sample batch data
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        # Configure mock for batch prediction
        batch_results = [
            {
                'churn_probability': 0.75,
                'churn_prediction': True,
                'risk_level': 'High',
                'confidence': 0.85,
                'threshold_used': 0.5,
                'model_version': '1.0.0'
            },
            {
                'churn_probability': 0.25,
                'churn_prediction': False,
                'risk_level': 'Low',
                'confidence': 0.75,
                'threshold_used': 0.5,
                'model_version': '1.0.0'
            },
            {
                'churn_probability': 0.55,
                'churn_prediction': True,
                'risk_level': 'Medium',
                'confidence': 0.65,
                'threshold_used': 0.5,
                'model_version': '1.0.0'
            }
        ]
        mock_model_manager.predict_batch.return_value = batch_results
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict/batch", json=sample_batch_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check batch response structure
            required_fields = ["batch_id", "predictions", "batch_size", "processing_time_ms", "timestamp"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Check batch details
            assert data["batch_size"] == 3
            assert len(data["predictions"]) == 3
            assert isinstance(data["batch_id"], str)
            assert isinstance(data["processing_time_ms"], (int, float))
            assert isinstance(data["timestamp"], (int, float))
            
            # Check individual predictions
            for prediction in data["predictions"]:
                assert_prediction_response(prediction)
            
            # Verify metrics were called
            mock_metrics['batch_prediction_counter'].inc.assert_called_once()
            mock_metrics['prediction_latency'].observe.assert_called_once()
        finally:
            app.dependency_overrides.clear()
    
    def test_predict_batch_model_not_loaded(self, client: TestClient, mock_model_manager, sample_batch_data, mock_metrics):
        """Test batch prediction when model is not loaded.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            sample_batch_data: Sample batch data
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=False):
                response = client.post("/predict/batch", json=sample_batch_data)
            
            assert response.status_code == 503
            data = response.json()
            
            assert "detail" in data
            assert "Model not loaded" in data["detail"]
            
            # Verify error counter was incremented
            mock_metrics['error_counter'].inc.assert_called_once()
        finally:
            app.dependency_overrides.clear()
    
    def test_predict_batch_empty_list(self, client: TestClient, mock_model_manager, mock_metrics):
        """Test batch prediction with empty customer list.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        empty_batch = {"customers": []}
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict/batch", json=empty_batch)
            
            assert response.status_code == 422  # Validation error for empty list
        finally:
            app.dependency_overrides.clear()
    
    def test_predict_batch_invalid_data(self, client: TestClient, mock_model_manager, mock_metrics):
        """Test batch prediction with invalid data.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        invalid_batch = {
            "customers": [
                {
                    "CreditScore": "invalid",  # Should be numeric
                    "Geography": "France"
                    # Missing required fields
                }
            ]
        }
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict/batch", json=invalid_batch)
            
            assert response.status_code == 422  # Validation error
        finally:
            app.dependency_overrides.clear()
    
    def test_predict_batch_exception_handling(self, client: TestClient, mock_model_manager, sample_batch_data, mock_metrics):
        """Test batch prediction exception handling.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            sample_batch_data: Sample batch data
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        # Configure mock to raise exception
        mock_model_manager.predict_batch.side_effect = Exception("Batch prediction error")
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict/batch", json=sample_batch_data)
            
            assert response.status_code == 500
            data = response.json()
            
            assert "detail" in data
            assert "Batch prediction failed" in data["detail"]
            
            # Verify error counter was incremented
            mock_metrics['error_counter'].inc.assert_called_once()
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.parametrize("endpoint", ["/predict", "/predict/batch"])
    @pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH"])
    def test_predict_endpoints_method_not_allowed(self, client: TestClient, endpoint: str, method: str):
        """Test that prediction endpoints only accept POST requests.
        
        Args:
            client: FastAPI test client
            endpoint: Prediction endpoint to test
            method: HTTP method to test
        """
        response = client.request(method, endpoint)
        assert response.status_code == 405  # Method Not Allowed
    
    def test_predict_content_type(self, client: TestClient, mock_model_manager, sample_customer_data, mock_metrics):
        """Test that prediction endpoints return correct content type.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
            sample_customer_data: Sample customer data
            mock_metrics: Mocked metrics
        """
        from app.routes.predict import get_model_manager
        from app.main import app
        
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        try:
            with patch('app.routes.predict.is_model_loaded', return_value=True):
                response = client.post("/predict", json=sample_customer_data)
            
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
        finally:
            app.dependency_overrides.clear()