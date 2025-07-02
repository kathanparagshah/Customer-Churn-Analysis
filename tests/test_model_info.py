"""Tests for model info endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


class TestModelInfoEndpoint:
    """Test cases for the /model/info endpoint."""
    
    def test_model_info_success(self, client: TestClient, override_model_manager):
        """Test successful model info retrieval.
        
        Args:
            client: FastAPI test client
            override_model_manager: Model manager dependency override
        """
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = [
            "model_name", "version", "training_date", "model_type",
            "features", "feature_count", "preprocessing_components",
            "performance_metrics", "model_path", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check specific values
        assert data["model_name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert data["model_type"] == "RandomForestClassifier"
        assert isinstance(data["features"], list)
        assert data["feature_count"] == len(data["features"])
        assert isinstance(data["preprocessing_components"], dict)
        assert isinstance(data["performance_metrics"], dict)
    
    def test_model_info_model_not_loaded(self, client: TestClient, mock_model_manager):
        """Test model info when model is not loaded.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
        """
        # Override the model manager to return not loaded
        mock_model_manager.is_loaded = False
        from app.main import app
        from app.services.model_manager import get_model_manager
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        response = client.get("/model/info")
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 503
        data = response.json()
        
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    def test_model_info_exception_handling(self, client: TestClient, mock_model_manager):
        """Test model info exception handling.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
        """
        # Configure mock to raise exception
        mock_model_manager.get_model_info.side_effect = Exception("Test error")
        from app.main import app
        from app.services.model_manager import get_model_manager
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        response = client.get("/model/info")
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 500
        data = response.json()
        
        assert "detail" in data
        assert "Failed to retrieve model information" in data["detail"]
    
    def test_model_info_response_structure(self, client: TestClient, override_model_manager):
        """Test model info response structure.
        
        Args:
            client: FastAPI test client
            override_model_manager: Model manager dependency override
        """
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check field types
        assert isinstance(data["model_name"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["training_date"], str)
        assert isinstance(data["model_type"], str)
        assert isinstance(data["features"], list)
        assert isinstance(data["feature_count"], int)
        assert isinstance(data["preprocessing_components"], dict)
        assert isinstance(data["performance_metrics"], dict)
        assert isinstance(data["model_path"], str)
        assert isinstance(data["timestamp"], str)
        
        # Check that feature_count matches features length
        assert data["feature_count"] == len(data["features"])
    
    def test_model_info_preprocessing_components(self, client: TestClient, override_model_manager):
        """Test model info preprocessing components.
        
        Args:
            client: FastAPI test client
            override_model_manager: Model manager dependency override
        """
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        preprocessing = data["preprocessing_components"]
        
        # Check expected preprocessing components
        expected_keys = ["scaler", "label_encoders", "feature_count"]
        for key in expected_keys:
            assert key in preprocessing, f"Missing preprocessing component: {key}"
        
        assert preprocessing["scaler"] == "StandardScaler"
        assert isinstance(preprocessing["label_encoders"], list)
        assert isinstance(preprocessing["feature_count"], int)
    
    def test_model_info_performance_metrics(self, client: TestClient, override_model_manager):
        """Test model info performance metrics.
        
        Args:
            client: FastAPI test client
            override_model_manager: Model manager dependency override
        """
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        metrics = data["performance_metrics"]
        
        # Check expected performance metrics
        expected_metrics = ["accuracy", "precision", "recall"]
        for metric in expected_metrics:
            assert metric in metrics, f"Missing performance metric: {metric}"
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1  # Assuming normalized metrics
    
    def test_model_info_content_type(self, client: TestClient, override_model_manager):
        """Test model info content type.
        
        Args:
            client: FastAPI test client
            override_model_manager: Model manager dependency override
        """
        response = client.get("/model/info")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
    
    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_model_info_method_not_allowed(self, client: TestClient, method: str):
        """Test that model info only accepts GET requests.
        
        Args:
            client: FastAPI test client
            method: HTTP method to test
        """
        response = client.request(method, "/model/info")
        assert response.status_code == 405  # Method Not Allowed