"""Prediction endpoints for single and batch customer churn predictions."""

import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from ..models.schemas import (
    CustomerData, 
    BatchCustomerData, 
    PredictionResponse, 
    BatchPredictionResponse
)
from ..services.model_manager import get_model_manager, ModelManager
from ..services.analytics import log_prediction, log_batch_prediction
from ..services.observability import (
    PREDICTION_COUNTER, 
    BATCH_PREDICTION_COUNTER, 
    PREDICTION_LATENCY, 
    ERROR_COUNTER
)
from ..config import settings
from ..logging import get_logger

logger = get_logger("predict")

router = APIRouter(tags=["prediction"])


@router.post("/predict")
async def predict_churn(
    customer_data: CustomerData,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Predict churn probability for a single customer.
    
    Args:
        customer_data: Customer data for prediction
        background_tasks: FastAPI background tasks
        model_manager: Model manager dependency
        
    Returns:
        PredictionResponse: Prediction results
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not model_manager.is_loaded:
        ERROR_COUNTER.inc()
        logger.warning("Prediction requested but model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is properly initialized."
        )
    
    start_time = time.time()
    
    try:
        # Make prediction
        prediction_result = model_manager.predict_single(
            customer_data.model_dump(),
            threshold=settings.THRESHOLD
        )
        
        # Record metrics
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(processing_time / 1000)  # Prometheus expects seconds
        
        # Create response
        response = PredictionResponse(
            churn_probability=prediction_result['churn_probability'],
            churn_prediction=prediction_result['churn_prediction'],
            risk_level=prediction_result['risk_level'],
            confidence=prediction_result['confidence'],
            timestamp=time.time(),
            version=prediction_result.get('model_version', 'Unknown')
        )
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction,
            prediction_result,
            customer_id=getattr(customer_data, 'customer_id', None)
        )
        
        logger.info(
            f"Prediction completed: probability={prediction_result['churn_probability']:.3f}, "
            f"prediction={prediction_result['churn_prediction']}, "
            f"processing_time={processing_time:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        ERROR_COUNTER.inc()
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Prediction failed after {processing_time:.2f}ms: {e}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch")
async def predict_batch_churn(
    batch_data: BatchCustomerData,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Predict churn probability for multiple customers.
    
    Args:
        batch_data: Batch of customer data for prediction
        background_tasks: FastAPI background tasks
        model_manager: Model manager dependency
        
    Returns:
        BatchPredictionResponse: Batch prediction results
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not model_manager.is_loaded:
        ERROR_COUNTER.inc()
        logger.warning("Batch prediction requested but model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is properly initialized."
        )
    
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    batch_size = len(batch_data.customers)
    
    try:
        # Convert customers to dictionaries
        customer_dicts = [customer.model_dump() for customer in batch_data.customers]
        
        # Make batch prediction
        prediction_results = model_manager.predict_batch(
            customer_dicts,
            threshold=settings.THRESHOLD
        )
        
        # Record metrics
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        BATCH_PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(processing_time / 1000)  # Prometheus expects seconds
        
        # Create individual prediction responses
        predictions = [
            PredictionResponse(
                churn_probability=result['churn_probability'],
                churn_prediction=result['churn_prediction'],
                risk_level=result['risk_level'],
                confidence=result['confidence'],
                timestamp=time.time(),
                version=result.get('model_version', 'Unknown')
            )
            for result in prediction_results
        ]
        
        # Create batch response
        response = BatchPredictionResponse(
            batch_id=batch_id,
            predictions=predictions,
            batch_size=batch_size,
            processing_time_ms=processing_time,
            timestamp=time.time()
        )
        
        # Log batch prediction in background
        background_tasks.add_task(
            log_batch_prediction,
            batch_id,
            batch_size,
            prediction_results[0].get('model_version', 'Unknown') if prediction_results else 'Unknown',
            processing_time
        )
        
        # Log individual predictions in background
        for i, result in enumerate(prediction_results):
            background_tasks.add_task(
                log_prediction,
                result,
                customer_id=f"{batch_id}_{i}"
            )
        
        logger.info(
            f"Batch prediction completed: batch_id={batch_id}, "
            f"batch_size={batch_size}, processing_time={processing_time:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        ERROR_COUNTER.inc()
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            f"Batch prediction failed: batch_id={batch_id}, "
            f"batch_size={batch_size}, processing_time={processing_time:.2f}ms, error={e}"
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )