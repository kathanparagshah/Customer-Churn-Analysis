"""Health check endpoint for monitoring service status."""

from fastapi import APIRouter, Depends
from datetime import datetime

from ..models.schemas import HealthResponse
from ..services.model_manager import get_model_manager, ModelManager
from ..logging import get_logger

logger = get_logger("health")

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Health check endpoint.
    
    Returns:
        HealthResponse: Current health status of the service
    """
    try:
        # Get health status from model manager
        health_status = model_manager.get_health_status()
        
        # Determine overall status
        status = "healthy" if model_manager.is_loaded else "degraded"
        
        logger.debug(f"Health check: {status}")
        
        return HealthResponse(
            status=status,
            loaded=model_manager.is_loaded,
            version=model_manager.version or "Unknown",
            uptime=health_status.get("uptime", "Unknown"),
            timestamp=health_status.get("timestamp", datetime.now().isoformat()),
            model_status=health_status.get("model_status")
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        # Return unhealthy status on error
        return HealthResponse(
            status="unhealthy",
            loaded=False,
            version="Unknown",
            uptime="Unknown",
            timestamp=datetime.now().isoformat(),
            model_status=None
        )