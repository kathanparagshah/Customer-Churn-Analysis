"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .config import settings
from .logging import include_logging, get_logger
from .services.model_manager import get_model_manager
# from .services.observability import get_metrics_router  # Removed - using routes/metrics.py instead
from .routes import health, model_info, predict, metrics, auth

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize model manager
    model_manager = get_model_manager()
    if model_manager.is_loaded:
        logger.info(f"Model loaded: {model_manager.model_name} v{model_manager.version}")
    else:
        logger.warning("Model not loaded - service will run in degraded mode")
    
    # Log configuration
    logger.info(f"API listening on {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Metrics enabled: {settings.ENABLE_METRICS}")
    logger.info(f"CORS enabled: {settings.CORS_ENABLED}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description="Customer Churn Prediction API - A machine learning service for predicting customer churn probability",
    version=settings.API_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None
)

# Add middleware
if settings.CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    logger.info(f"CORS middleware enabled with origins: {settings.CORS_ORIGINS}")

# Add trusted host middleware for security
if settings.TRUSTED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS
    )
    logger.info(f"Trusted host middleware enabled: {settings.TRUSTED_HOSTS}")

# Apply logging configuration and middleware
include_logging(app)

# Include routers
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(predict.router)
app.include_router(metrics.router)
app.include_router(auth.router)

# Include metrics router if enabled
if settings.ENABLE_METRICS:
    # Metrics router is included via routes/metrics.py
    logger.info("Metrics router included")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information.
    
    Returns:
        dict: API information and status
    """
    model_manager = get_model_manager()
    
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "model_status": {
            "loaded": model_manager.is_loaded,
            "name": model_manager.model_name or "Unknown",
            "version": model_manager.version or "Unknown"
        },
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics" if settings.ENABLE_METRICS else None,
            "auth": "/api/auth",
            "docs": "/docs" if settings.DEBUG else None
        },
        "features": {
            "cors_enabled": settings.CORS_ENABLED,
            "metrics_enabled": settings.ENABLE_METRICS,
            "debug_mode": settings.DEBUG,
            "authentication": "stub"
        }
    }


@app.get("/ping", tags=["health"])
async def ping():
    """Simple ping endpoint for basic health checks.
    
    Returns:
        dict: Simple pong response
    """
    return {"message": "pong"}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )