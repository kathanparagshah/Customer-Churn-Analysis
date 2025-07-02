"""Observability and monitoring setup with Prometheus metrics."""

import logging
from typing import Optional
from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from ..config import settings
from ..logging import get_logger

logger = get_logger("observability")

# Prometheus metrics
PREDICTION_COUNTER: Optional[Counter] = None
BATCH_PREDICTION_COUNTER: Optional[Counter] = None
PREDICTION_LATENCY: Optional[Histogram] = None
ERROR_COUNTER: Optional[Counter] = None


class DummyMetric:
    """Dummy metric class for fallback when Prometheus is not available."""
    
    def inc(self, *args, **kwargs):
        """No-op increment method."""
        pass
    
    def observe(self, *args, **kwargs):
        """No-op observe method."""
        pass


def setup_metrics() -> None:
    """Initialize Prometheus metrics with fallback to dummy metrics."""
    global PREDICTION_COUNTER, BATCH_PREDICTION_COUNTER, PREDICTION_LATENCY, ERROR_COUNTER
    
    if not settings.ENABLE_METRICS:
        logger.info("Metrics disabled by configuration")
        # Use dummy metrics
        dummy = DummyMetric()
        PREDICTION_COUNTER = dummy
        BATCH_PREDICTION_COUNTER = dummy
        PREDICTION_LATENCY = dummy
        ERROR_COUNTER = dummy
        return
    
    try:
        # Initialize Prometheus metrics
        PREDICTION_COUNTER = Counter(
            'prediction_requests_total', 
            'Total prediction requests'
        )
        BATCH_PREDICTION_COUNTER = Counter(
            'prediction_requests_batch_total', 
            'Total batch prediction requests'
        )
        PREDICTION_LATENCY = Histogram(
            'churn_prediction_duration_seconds', 
            'Prediction latency'
        )
        ERROR_COUNTER = Counter(
            'churn_prediction_errors_total', 
            'Total prediction errors'
        )
        
        logger.info("Prometheus metrics initialized successfully")
        
    except Exception as e:
        logger.warning(f"Prometheus setup failed: {e}")
        # Fallback to dummy metrics
        dummy = DummyMetric()
        PREDICTION_COUNTER = dummy
        BATCH_PREDICTION_COUNTER = dummy
        PREDICTION_LATENCY = dummy
        ERROR_COUNTER = dummy


def get_metrics_content() -> str:
    """Get Prometheus metrics content as string.
    
    Returns:
        str: Prometheus metrics in text format
    """
    try:
        if settings.ENABLE_METRICS:
            return generate_latest().decode('utf-8')
        else:
            return "# Prometheus metrics not available\n"
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return "# Error generating metrics\n"


def get_metrics_router() -> APIRouter:
    """Get metrics router with /metrics endpoint.
    
    Returns:
        APIRouter: Router with metrics endpoint
    """
    router = APIRouter(tags=["monitoring"])
    
    @router.get("/metrics")
    def metrics():
        """Prometheus metrics endpoint."""
        if not settings.ENABLE_METRICS:
            return Response(
                content="Metrics disabled",
                media_type="text/plain",
                status_code=503
            )
        
        metrics_content = get_metrics_content()
        return Response(
            content=metrics_content,
            media_type="text/plain; charset=utf-8"
        )
    
    return router


# Initialize metrics on module import
setup_metrics()

# Create metrics router
metrics_router = get_metrics_router()