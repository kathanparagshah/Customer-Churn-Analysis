"""Tests for metrics endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


class TestMetricsEndpoint:
    """Test cases for the /metrics endpoint."""
    
    def test_metrics_success(self, client: TestClient):
        """Test successful metrics retrieval.
        
        Args:
            client: FastAPI test client
        """
        mock_metrics_content = (
            "# HELP prediction_counter_total Total number of predictions\n"
            "# TYPE prediction_counter_total counter\n"
            "prediction_counter_total 42.0\n"
            "# HELP batch_prediction_counter_total Total number of batch predictions\n"
            "# TYPE batch_prediction_counter_total counter\n"
            "batch_prediction_counter_total 5.0\n"
            "# HELP prediction_latency_seconds Prediction latency in seconds\n"
            "# TYPE prediction_latency_seconds histogram\n"
            "prediction_latency_seconds_bucket{le=\"0.1\"} 10.0\n"
            "prediction_latency_seconds_bucket{le=\"+Inf\"} 42.0\n"
            "prediction_latency_seconds_count 42.0\n"
            "prediction_latency_seconds_sum 4.2\n"
        )
        
        with patch('app.routes.metrics.get_metrics_content', return_value=mock_metrics_content):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        content = response.text
        assert "prediction_counter_total" in content
        assert "batch_prediction_counter_total" in content
        assert "prediction_latency_seconds" in content
        assert "42.0" in content
    
    def test_metrics_content_type(self, client: TestClient):
        """Test that metrics endpoint returns correct content type.
        
        Args:
            client: FastAPI test client
        """
        mock_metrics_content = "# Test metrics\nprediction_counter_total 1.0\n"
        
        with patch('app.routes.metrics.get_metrics_content', return_value=mock_metrics_content):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_exception_handling(self, client: TestClient):
        """Test metrics endpoint exception handling.
        
        Args:
            client: FastAPI test client
        """
        with patch('app.routes.metrics.get_metrics_content', side_effect=Exception("Metrics error")):
            response = client.get("/metrics")
        
        assert response.status_code == 200  # Should still return 200 with fallback
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        content = response.text
        assert "Metrics temporarily unavailable" in content
    
    def test_metrics_empty_content(self, client: TestClient):
        """Test metrics endpoint with empty content.
        
        Args:
            client: FastAPI test client
        """
        with patch('app.routes.metrics.get_metrics_content', return_value=""):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.text == ""
    
    def test_metrics_prometheus_format(self, client: TestClient):
        """Test that metrics are in Prometheus format.
        
        Args:
            client: FastAPI test client
        """
        mock_metrics_content = (
            "# HELP prediction_counter_total Total number of predictions\n"
            "# TYPE prediction_counter_total counter\n"
            "prediction_counter_total 10.0\n"
            "# HELP error_counter_total Total number of errors\n"
            "# TYPE error_counter_total counter\n"
            "error_counter_total 2.0\n"
        )
        
        with patch('app.routes.metrics.get_metrics_content', return_value=mock_metrics_content):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        content = response.text
        
        # Check Prometheus format elements
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "counter" in content
        assert "_total" in content
        
        # Check specific metrics
        lines = content.strip().split('\n')
        metric_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        
        for line in metric_lines:
            # Each metric line should have format: metric_name value
            parts = line.split()
            assert len(parts) >= 2, f"Invalid metric line format: {line}"
            
            metric_name = parts[0]
            metric_value = parts[1]
            
            # Metric name should be valid
            assert metric_name, "Metric name should not be empty"
            
            # Metric value should be numeric
            try:
                float(metric_value)
            except ValueError:
                pytest.fail(f"Metric value should be numeric: {metric_value}")
    
    def test_metrics_specific_counters(self, client: TestClient):
        """Test that specific expected counters are present.
        
        Args:
            client: FastAPI test client
        """
        mock_metrics_content = (
            "prediction_counter_total 15.0\n"
            "batch_prediction_counter_total 3.0\n"
            "error_counter_total 1.0\n"
            "prediction_latency_seconds_count 15.0\n"
            "prediction_latency_seconds_sum 1.5\n"
        )
        
        with patch('app.routes.metrics.get_metrics_content', return_value=mock_metrics_content):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        content = response.text
        
        # Check for expected metrics
        expected_metrics = [
            "prediction_counter_total",
            "batch_prediction_counter_total", 
            "error_counter_total",
            "prediction_latency_seconds"
        ]
        
        for metric in expected_metrics:
            assert metric in content, f"Expected metric not found: {metric}"
    
    def test_metrics_no_html_content(self, client: TestClient):
        """Test that metrics endpoint doesn't return HTML content.
        
        Args:
            client: FastAPI test client
        """
        mock_metrics_content = "prediction_counter_total 1.0\n"
        
        with patch('app.routes.metrics.get_metrics_content', return_value=mock_metrics_content):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        content = response.text
        
        # Should not contain HTML tags
        html_tags = ["<html>", "<body>", "<div>", "<p>", "<h1>", "<script>"]
        for tag in html_tags:
            assert tag not in content.lower(), f"Metrics should not contain HTML tag: {tag}"
    
    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_metrics_method_not_allowed(self, client: TestClient, method: str):
        """Test that metrics endpoint only accepts GET requests.
        
        Args:
            client: FastAPI test client
            method: HTTP method to test
        """
        response = client.request(method, "/metrics")
        assert response.status_code == 405  # Method Not Allowed
    
    def test_metrics_caching_headers(self, client: TestClient):
        """Test that metrics endpoint has appropriate caching headers.
        
        Args:
            client: FastAPI test client
        """
        mock_metrics_content = "prediction_counter_total 1.0\n"
        
        with patch('app.routes.metrics.get_metrics_content', return_value=mock_metrics_content):
            response = client.get("/metrics")
        
        assert response.status_code == 200
        
        # Metrics should typically not be cached for real-time monitoring
        # Check that no aggressive caching headers are set
        cache_control = response.headers.get("cache-control", "")
        if cache_control:
            assert "no-cache" in cache_control.lower() or "max-age=0" in cache_control.lower()