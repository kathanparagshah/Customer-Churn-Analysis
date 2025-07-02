"""Tests for health check endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from tests.conftest import assert_health_response


class TestHealthEndpoint:
    """Test cases for the /health endpoint."""
    
    def test_health_check_success(self, client: TestClient, mock_model_manager):
        """Test successful health check with loaded model.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
        """
        with patch('app.routes.health.get_model_manager', return_value=mock_model_manager):
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert_health_response(data)
        assert data["status"] == "healthy"
        assert data["loaded"] is True
        assert data["version"] == "1.0.0"
    
    def test_health_check_model_not_loaded(self, client: TestClient, mock_model_manager):
        """Test health check when model is not loaded.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
        """
        # Configure mock for unloaded model
        mock_model_manager.is_loaded = False
        mock_model_manager.version = None
        from app.main import app
        from app.services.model_manager import get_model_manager
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        response = client.get("/health")
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        assert_health_response(data)
        assert data["status"] == "degraded"
        assert data["loaded"] is False
        assert data["version"] == "Unknown"
    
    def test_health_check_exception_handling(self, client: TestClient, mock_model_manager):
        """Test health check exception handling.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
        """
        # Configure mock to raise exception
        mock_model_manager.get_health_status.side_effect = Exception("Test error")
        from app.main import app
        from app.services.model_manager import get_model_manager
        app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
        
        response = client.get("/health")
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        assert_health_response(data)
        assert data["status"] == "unhealthy"
        assert data["loaded"] is False
        assert data["version"] == "Unknown"
    
    def test_health_check_response_structure(self, client: TestClient, mock_model_manager):
        """Test that health check response has correct structure.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
        """
        with patch('app.routes.health.get_model_manager', return_value=mock_model_manager):
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = ["status", "loaded", "version", "uptime", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["status"], str)
        assert isinstance(data["loaded"], bool)
        assert isinstance(data["version"], str)
        assert isinstance(data["uptime"], str)
        assert isinstance(data["timestamp"], str)
    
    def test_health_check_content_type(self, client: TestClient, mock_model_manager):
        """Test that health check returns correct content type.
        
        Args:
            client: FastAPI test client
            mock_model_manager: Mocked model manager
        """
        with patch('app.routes.health.get_model_manager', return_value=mock_model_manager):
            response = client.get("/health")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
    
    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_health_check_method_not_allowed(self, client: TestClient, method: str):
        """Test that health check only accepts GET requests.
        
        Args:
            client: FastAPI test client
            method: HTTP method to test
        """
        response = client.request(method, "/health")
        assert response.status_code == 405  # Method Not Allowed