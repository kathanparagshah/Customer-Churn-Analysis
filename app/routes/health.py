"""Health check endpoint for monitoring service status."""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any

from ..models.schemas import HealthResponse
from ..services.model_manager import get_model_manager, ModelManager
from ..logging import get_logger, log_performance
from ..exceptions import ChurnAPIException, to_http_exception

logger = get_logger("health")

router = APIRouter(tags=["health"])


def check_disk_space(path: str = ".", min_free_gb: float = 1.0) -> Dict[str, Any]:
    """Check available disk space.
    
    Args:
        path: Path to check disk space for
        min_free_gb: Minimum free space in GB required
        
    Returns:
        Dictionary with disk space information
    """
    try:
        statvfs = os.statvfs(path)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        total_bytes = statvfs.f_frsize * statvfs.f_blocks
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        
        return {
            "status": "healthy" if free_gb >= min_free_gb else "warning",
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "usage_percent": round((total_gb - free_gb) / total_gb * 100, 2)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def check_memory() -> Dict[str, Any]:
    """Check system memory usage.
    
    Returns:
        Dictionary with memory information
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "status": "healthy" if memory.percent < 90 else "warning",
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "usage_percent": memory.percent
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def check_model_files() -> Dict[str, Any]:
    """Check if required model files exist.
    
    Returns:
        Dictionary with model file status
    """
    try:
        model_paths = [
            "models/churn_model.pkl",
            "deployment/models/churn_model.pkl"
        ]
        
        existing_files = []
        missing_files = []
        
        for path in model_paths:
            if Path(path).exists():
                existing_files.append(path)
            else:
                missing_files.append(path)
        
        status = "healthy" if existing_files else "error"
        
        return {
            "status": status,
            "existing_files": existing_files,
            "missing_files": missing_files
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def check_dependencies() -> Dict[str, Any]:
    """Check critical dependencies and services.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        "disk_space": check_disk_space(),
        "memory": check_memory(),
        "model_files": check_model_files()
    }
    
    # Determine overall dependency health
    all_healthy = all(
        dep.get("status") == "healthy" 
        for dep in dependencies.values()
    )
    
    dependencies["overall_status"] = "healthy" if all_healthy else "degraded"
    
    return dependencies


@router.get("/health", response_model=HealthResponse)
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Comprehensive health check endpoint.
    
    Returns detailed health information including:
    - Service uptime
    - Model status
    - System resources
    - Dependencies
    
    Returns:
        HealthResponse: Current health status of the service
    """
    start_time = time.time()
    
    try:
        # Get basic health status
        health_status = model_manager.get_health_status()
        
        # Check dependencies
        dependencies = check_dependencies()
        
        # Determine overall service status
        model_healthy = model_manager.is_loaded
        deps_healthy = dependencies.get("overall_status") == "healthy"
        
        if model_healthy and deps_healthy:
            status = "healthy"
        elif model_healthy or deps_healthy:
            status = "degraded"
        else:
            status = "unhealthy"
        
        logger.debug(f"Health check: {status}")
        
        response = HealthResponse(
            status=status,
            loaded=model_manager.is_loaded,
            version=model_manager.version or "Unknown",
            uptime=health_status.get("uptime", "Unknown"),
            timestamp=health_status.get("timestamp", datetime.now().isoformat()),
            model_status=health_status.get("model_status"),
            dependencies=dependencies
        )
        
        # Log performance
        duration = time.time() - start_time
        log_performance(
            "health_check",
            duration,
            status=status,
            model_loaded=model_manager.is_loaded
        )
        
        return response
        
    except ChurnAPIException as e:
        logger.error(f"Health check failed: {e.message}", extra=e.details)
        raise to_http_exception(e)
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


@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint.
    
    Simple endpoint that returns 200 if the service is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe(model_manager: ModelManager = Depends(get_model_manager)):
    """Kubernetes readiness probe endpoint.
    
    Returns 200 only if the service is ready to handle requests.
    """
    try:
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "reason": "model_not_loaded"}
            )
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "error": str(e)}
        )