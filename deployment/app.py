#!/usr/bin/env python3
"""
Churn Prediction API

FastAPI application for serving churn prediction models in production.
Provides endpoints for individual predictions, batch predictions, and model health checks.

Author: Bank Churn Analysis Team
Date: 2024
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# TestClient compatibility handled in test files

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('churn_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics (with duplicate prevention)
try:
    from prometheus_client import CollectorRegistry, REGISTRY
    # Create new registry to avoid conflicts
    custom_registry = CollectorRegistry()
    PREDICTION_COUNTER = Counter('churn_predictions_total', 'Total number of predictions made', registry=custom_registry)
    PREDICTION_LATENCY = Histogram('churn_prediction_duration_seconds', 'Time spent on predictions', registry=custom_registry)
    ERROR_COUNTER = Counter('churn_prediction_errors_total', 'Total number of prediction errors', registry=custom_registry)
except Exception as e:
    logger.warning(f"Prometheus metrics setup failed: {e}. Using default metrics.")
    # Fallback to simple metrics without registry
    PREDICTION_COUNTER = None
    PREDICTION_LATENCY = None
    ERROR_COUNTER = None

# Initialize FastAPI app
app = FastAPI(
    title="Bank Churn Prediction API",
    description="API for predicting customer churn using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://customer-churn-analysis-kgz3.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing components
model = None
scaler = None
label_encoders = {}
feature_names = []
model_metadata = {}
model_loaded = False


class CustomerData(BaseModel):
    """
    Pydantic model for customer data input.
    """
    CreditScore: int = Field(..., ge=300, le=850, description="Customer credit score (300-850)")
    Geography: str = Field(..., description="Customer geography (France, Spain, Germany)")
    Gender: str = Field(..., description="Customer gender (Male, Female)")
    Age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    Tenure: int = Field(..., ge=0, le=10, description="Years with bank (0-10)")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of products (1-4)")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0 or 1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary")
    
    @validator('Geography', allow_reuse=True)
    def validate_geography(cls, v):
        allowed_geographies = ['France', 'Spain', 'Germany']
        if v not in allowed_geographies:
            raise ValueError(f'Geography must be one of {allowed_geographies}')
        return v
    
    @validator('Gender', allow_reuse=True)
    def validate_gender(cls, v):
        allowed_genders = ['Male', 'Female']
        if v not in allowed_genders:
            raise ValueError(f'Gender must be one of {allowed_genders}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female",
                "Age": 35,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0
            }
        }


class BatchCustomerData(BaseModel):
    """
    Pydantic model for batch prediction input.
    """
    customers: List[CustomerData] = Field(..., description="List of customer data")
    
    @validator('customers', allow_reuse=True)
    def validate_batch_size(cls, v):
        if len(v) > 1000:  # Limit batch size
            raise ValueError('Batch size cannot exceed 1000 customers')
        return v


class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response.
    """
    churn_probability: float = Field(..., description="Probability of customer churn (0-1)")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")
    confidence: float = Field(..., description="Model confidence score")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionResponse(BaseModel):
    """
    Pydantic model for batch prediction response.
    """
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")


class HealthResponse(BaseModel):
    """
    Pydantic model for health check response.
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Current model version")
    uptime: str = Field(..., description="Service uptime")
    timestamp: datetime = Field(..., description="Health check timestamp")


class ModelManager:
    """
    Manages model loading and preprocessing components.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        try:
            # existing load logic (opening pickle, setting model, scaler, etc.)
            self.model_path = Path(__file__).resolve().parent / 'models' / 'churn_model.pkl'
            model_package = joblib.load(self.model_path)
            self.model = model_package['model']
            self.scaler = model_package.get('scaler')
            self.feature_names = model_package['feature_names']
            self.label_encoders = model_package.get('label_encoders', {})
            self.model_name = model_package.get('model_name', 'unknown')
            self.version = model_package.get('version', '1.0.0')
            self.training_date = model_package.get('training_date', 'unknown')
            self.is_loaded = True
        except FileNotFoundError:
            logger.warning(f"Model file not found at {self.model_path}; running without a model for tests")
            self.model = None
            self.scaler = None
            self.feature_names = []
            self.label_encoders = {}
            self.model_name = 'unknown'
            self.version = '1.0.0'
            self.training_date = 'unknown'
            self.is_loaded = False
        self.performance_metrics = {}
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the trained model and preprocessing components.
        
        Args:
            model_path: Path to the model file
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        global model, scaler, label_encoders, feature_names, model_metadata, model_loaded
        
        try:
            model_file_path = Path(model_path)
            if not model_file_path.exists():
                logger.error(f"Model file not found: {model_file_path}")
                return False
            
            logger.info(f"Loading model from {model_file_path}")
            
            # Load model package
            model_package = joblib.load(model_file_path)
            
            # Set instance variables
            self.model = model_package['model']
            self.scaler = model_package.get('scaler')
            self.label_encoders = model_package.get('label_encoders', {})
            self.feature_names = model_package['feature_names']
            
            # Also set global variables for backward compatibility
            model = self.model
            scaler = self.scaler
            label_encoders = self.label_encoders
            feature_names = self.feature_names
            
            # Set instance metadata attributes
            self.model_name = model_package.get('model_name', 'Unknown')
            self.version = model_package.get('version', '1.0.0')
            self.training_date = model_package.get('training_date', 'Unknown')
            self.performance_metrics = model_package.get('performance_metrics', {})
            
            # Extract model metadata for global variable
            model_metadata = {
                'model_name': self.model_name,
                'version': self.version,
                'training_date': self.training_date,
                'performance_metrics': self.performance_metrics
            }
            
            # Set loaded flags
            self.is_loaded = True
            model_loaded = True
            
            logger.info(f"Model loaded successfully: {model_metadata['model_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_uptime(self) -> str:
        """
        Get service uptime.
        
        Returns:
            str: Formatted uptime string
        """
        uptime = datetime.now() - self.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{days}d {hours}h {minutes}m {seconds}s"


# Initialize model manager
model_manager = ModelManager()


def get_model_manager():
    """
    Dependency to get model manager instance.
    """
    return model_manager


def preprocess_customer_data(customer_data: CustomerData) -> np.ndarray:
    """
    Preprocess customer data for prediction.
    
    Args:
        customer_data (CustomerData): Customer data to preprocess
        
    Returns:
        np.ndarray: Preprocessed feature array
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer_data.dict()])
    
    # Apply label encoding for categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError as e:
                logger.warning(f"Unknown category in {col}: {df[col].iloc[0]}. Using default encoding.")
                # Handle unknown categories by using the most frequent class
                df[col] = encoder.transform([encoder.classes_[0]])
    
    # Ensure feature order matches training
    df = df[feature_names]
    
    # Apply scaling if available
    if scaler is not None:
        features = scaler.transform(df)
    else:
        features = df.values
    
    return features


def calculate_risk_level(probability: float) -> str:
    """
    Calculate risk level based on churn probability.
    
    Args:
        probability (float): Churn probability
        
    Returns:
        str: Risk level (Low, Medium, High)
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def calculate_confidence(probability: float) -> float:
    """
    Calculate model confidence based on probability.
    
    Args:
        probability (float): Churn probability
        
    Returns:
        float: Confidence score (0-1)
    """
    # Confidence is higher when probability is closer to 0 or 1
    return 2 * abs(probability - 0.5)


@app.on_event("startup")
async def startup_event():
    """
    Load model on startup.
    """
    global model, scaler, label_encoders, feature_names, model_loaded, model_metadata
    
    logger.info("Starting Churn Prediction API...")
    success = model_manager.load_model(str(model_manager.model_path))
    if not success:
        logger.error("Failed to load model on startup")
    else:
        # Ensure global variables are properly set
        model = model_manager.model
        scaler = model_manager.scaler
        label_encoders = model_manager.label_encoders
        feature_names = model_manager.feature_names
        model_loaded = True
        model_metadata = {
            'model_name': model_manager.model_name,
            'version': model_manager.version,
            'training_date': model_manager.training_date,
            'performance_metrics': model_manager.performance_metrics
        }
        logger.info(f"API started successfully with {len(feature_names)} features")
        logger.info(f"Feature names: {feature_names}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Bank Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=model_metadata.get('version', 'Unknown'),
        uptime=manager.get_uptime(),
        timestamp=datetime.now()
    )


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """
    Get model information and metadata.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_metadata": model_metadata,
        "feature_names": feature_names,
        "preprocessing_components": {
            "scaler": scaler.__class__.__name__ if scaler else None,
            "label_encoders": {k: v.__class__.__name__ for k, v in label_encoders.items()}
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(
    customer_data: CustomerData,
    background_tasks: BackgroundTasks
):
    """
    Predict churn for a single customer.
    
    Args:
        customer_data (CustomerData): Customer data for prediction
        background_tasks (BackgroundTasks): Background tasks for logging
        
    Returns:
        PredictionResponse: Prediction result
    """
    if not model_loaded:
        if ERROR_COUNTER:
            ERROR_COUNTER.inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Preprocess data
        features = preprocess_customer_data(customer_data)
        
        # Make prediction
        if PREDICTION_LATENCY:
            with PREDICTION_LATENCY.time():
                probability = float(model.predict_proba(features)[0, 1])
                prediction = bool(model.predict(features)[0])
        else:
            probability = float(model.predict_proba(features)[0, 1])
            prediction = bool(model.predict(features)[0])
        
        # Calculate additional metrics
        risk_level = calculate_risk_level(probability)
        confidence = calculate_confidence(probability)
        
        # Create response
        response = PredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.now(),
            model_version=model_metadata.get('version', '1.0.0')
        )
        
        # Update metrics
        if PREDICTION_COUNTER:
            PREDICTION_COUNTER.inc()
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            customer_data.dict(),
            response.dict()
        )
        
        return response
        
    except Exception as e:
        if ERROR_COUNTER:
            ERROR_COUNTER.inc()
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_churn_batch(
    batch_data: BatchCustomerData,
    background_tasks: BackgroundTasks
):
    """
    Predict churn for multiple customers.
    
    Args:
        batch_data (BatchCustomerData): Batch of customer data
        background_tasks (BackgroundTasks): Background tasks for logging
        
    Returns:
        dict: Batch prediction results
    """
    try:
        if not model_loaded:
            if ERROR_COUNTER:
                ERROR_COUNTER.inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        if PREDICTION_LATENCY:
            with PREDICTION_LATENCY.time():
                predictions = []
                
                for customer_data in batch_data.customers:
                    # Preprocess data
                    features = preprocess_customer_data(customer_data)
                    
                    # Make prediction
                    probability = float(model.predict_proba(features)[0, 1])
                    prediction = bool(model.predict(features)[0])
                    
                    # Calculate additional metrics
                    risk_level = calculate_risk_level(probability)
                    confidence = calculate_confidence(probability)
                    
                    # Create response
                    pred_response = {
                        "churn_probability": probability,
                        "churn_prediction": prediction,
                        "risk_level": risk_level,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat(),
                        "model_version": model_metadata.get('version', '1.0.0')
                    }
                    
                    predictions.append(pred_response)
        else:
            predictions = []
            
            for customer_data in batch_data.customers:
                # Preprocess data
                features = preprocess_customer_data(customer_data)
                
                # Make prediction
                probability = float(model.predict_proba(features)[0, 1])
                prediction = bool(model.predict(features)[0])
                
                # Calculate additional metrics
                risk_level = calculate_risk_level(probability)
                confidence = calculate_confidence(probability)
                
                # Create response
                pred_response = {
                    "churn_probability": probability,
                    "churn_prediction": prediction,
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "model_version": model_metadata.get('version', '1.0.0')
                }
                
                predictions.append(pred_response)
        
        # Calculate summary statistics
        probabilities = [p["churn_probability"] for p in predictions]
        churn_predictions = [p["churn_prediction"] for p in predictions]
        
        summary = {
            "total_customers": len(predictions),
            "predicted_churners": sum(churn_predictions),
            "churn_rate": sum(churn_predictions) / len(predictions) if predictions else 0,
            "avg_churn_probability": float(np.mean(probabilities)) if probabilities else 0,
            "high_risk_customers": sum(1 for p in predictions if p["risk_level"] == "High"),
            "medium_risk_customers": sum(1 for p in predictions if p["risk_level"] == "Medium"),
            "low_risk_customers": sum(1 for p in predictions if p["risk_level"] == "Low")
        }
        
        # Update metrics
        if PREDICTION_COUNTER:
            PREDICTION_COUNTER.inc(len(predictions))
        
        # Log batch prediction (background task)
        background_tasks.add_task(
            log_batch_prediction,
            len(batch_data.customers),
            summary
        )
        
        return {
            "predictions": predictions,
            "summary": summary,
            "model_loaded": model_loaded,
            "model_name": model_metadata.get('name', 'unknown'),
            "version": model_metadata.get('version', '1.0.0')
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(tb)
        return {"detail": tb}


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/model/reload")
async def reload_model(manager: ModelManager = Depends(get_model_manager)):
    """
    Reload the model (useful for model updates).
    """
    try:
        success = manager.load_model()
        if success:
            return {"message": "Model reloaded successfully", "timestamp": datetime.now()}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload model"
            )
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload error: {str(e)}"
        )


async def log_prediction(customer_data: Dict[str, Any], prediction: Dict[str, Any]):
    """
    Log individual prediction for monitoring and audit.
    
    Args:
        customer_data (Dict[str, Any]): Customer input data
        prediction (Dict[str, Any]): Prediction result
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "single_prediction",
        "customer_data": customer_data,
        "prediction": prediction
    }
    
    # Log to file (in production, consider using a proper logging service)
    logger.info(f"PREDICTION_LOG: {json.dumps(log_entry)}")


async def log_batch_prediction(batch_size: int, summary: Dict[str, Any]):
    """
    Log batch prediction for monitoring.
    
    Args:
        batch_size (int): Size of the batch
        summary (Dict[str, Any]): Batch prediction summary
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "batch_prediction",
        "batch_size": batch_size,
        "summary": summary
    }
    
    logger.info(f"BATCH_PREDICTION_LOG: {json.dumps(log_entry)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler.
    """
    if ERROR_COUNTER:
        ERROR_COUNTER.inc()
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )