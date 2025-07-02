"""Metrics endpoint for Prometheus monitoring."""

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from ..services.observability import get_metrics_content
from ..logging import get_logger

PLAIN_CONTENT_TYPE = "text/plain"

logger = get_logger("metrics")

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint.
    
    Returns:
        Response: Prometheus metrics in text format
    """
    try:
        # Get metrics content from observability service
        metrics_content = get_metrics_content()
        
        logger.debug("Metrics endpoint accessed")
        
        return Response(
            content=metrics_content,
            media_type=PLAIN_CONTENT_TYPE
        )
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        
        # Return empty metrics on error
        return Response(
            content="# Metrics temporarily unavailable\n",
            media_type=PLAIN_CONTENT_TYPE
        )