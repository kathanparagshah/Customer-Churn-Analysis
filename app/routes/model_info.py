"""Model information endpoint for retrieving model metadata."""

from fastapi import APIRouter, Depends, HTTPException

from ..models.schemas import ModelInfoResponse
from ..services.model_manager import get_model_manager, ModelManager
from ..logging import get_logger

logger = get_logger("model_info")

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(model_manager: ModelManager = Depends(get_model_manager)):
    """Get detailed model information and metadata.
    
    Returns:
        ModelInfoResponse: Comprehensive model information
        
    Raises:
        HTTPException: If model is not loaded
    """
    if not model_manager.is_loaded:
        logger.warning("Model info requested but model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is properly initialized."
        )
    
    try:
        # Get comprehensive model information
        model_info = model_manager.get_model_info()
        
        logger.debug(f"Model info requested: {model_info.get('model_name')}")
        
        return ModelInfoResponse(
            model_name=model_info.get("model_name", "Unknown"),
            version=model_info.get("version", "Unknown"),
            training_date=model_info.get("training_date", "Unknown"),
            model_type=model_info.get("model_type", "Unknown"),
            features=model_info.get("features", []),
            feature_count=len(model_info.get("features", [])),
            preprocessing_components=model_info.get("preprocessing_components", {}),
            performance_metrics=model_info.get("performance_metrics", {}),
            model_path=model_info.get("model_path", "Unknown"),
            timestamp=model_info.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model information: {str(e)}"
        )