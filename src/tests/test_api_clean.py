import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Add paths for imports
src_path = Path(__file__).parent.parent
deployment_path = src_path.parent / "deployment"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(deployment_path))
sys.path.insert(0, str(src_path / "tests" / "deployment"))

try:
    from deployment.app import app  # noqa: E402
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
        "EstimatedSalary": 75000.0,
    }


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_predict_endpoint_success(self, client, valid_customer_data):
        """Test successful prediction endpoint."""
        with patch(
            "app.services.model_manager.model_manager"
        ) as mock_model_manager:

            # Mock model manager
            mock_model_manager.predict_single.return_value = {
                "churn_probability": 0.25,
                "churn_prediction": 0,
                "risk_level": "Low",
                "confidence": 0.75,
                "model_version": "1.0.0",
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
        with patch(
            "app.services.model_manager.model_manager"
        ) as mock_model_manager:

            # Mock ModelManager
            mock_model_manager.get_uptime.return_value = "0:01:23"

            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK

            data = response.json()
            assert "status" in data
            assert "uptime" in data
            assert "model_status" in data
