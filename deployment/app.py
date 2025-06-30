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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import sklearn
from packaging import version

# TestClient is only needed for testing, not for the actual API

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app startup and shutdown.
    """
    global model, scaler, label_encoders, feature_names, model_loaded, model_metadata
    
    # Startup
    logger.info("Starting Churn Prediction API...")
    try:
        model_manager.load_model(str(model_manager.model_path))
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
    except FileNotFoundError as e:
        logger.warning(f"Model file not found at startup: {e}. Running without a model for tests.")
        model_loaded = False
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        model_loaded = False
    
    yield
    
    # Shutdown
    logger.info("Shutting down Churn Prediction API...")

# Initialize FastAPI app
app = FastAPI(
    title="Bank Churn Prediction API",
    description="API for predicting customer churn using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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

def is_model_loaded():
    """Check if the model is loaded."""
    return model_loaded


class CustomerData(BaseModel):
    """
    Pydantic model for customer data input with enhanced validation.
    """
    CreditScore: int = Field(
        ..., 
        description="Customer credit score (300-850)",
        example=650
    )
    Geography: str = Field(
        ..., 
        description="Customer geography (France, Spain, Germany)",
        example="France"
    )
    Gender: str = Field(
        ..., 
        description="Customer gender (Male, Female)",
        example="Female"
    )
    Age: int = Field(
        ..., 
        description="Customer age (18-100)",
        example=35
    )
    Tenure: int = Field(
        ..., 
        description="Years with bank (0-10)",
        example=5
    )
    Balance: float = Field(
        ..., 
        description="Account balance (non-negative)",
        example=50000.0
    )
    NumOfProducts: int = Field(
        ..., 
        description="Number of products (1-4)",
        example=2
    )
    HasCrCard: int = Field(
        ..., 
        description="Has credit card (0 or 1)",
        example=1
    )
    IsActiveMember: int = Field(
        ..., 
        description="Is active member (0 or 1)",
        example=1
    )
    EstimatedSalary: float = Field(
        ..., 
        description="Estimated salary (non-negative)",
        example=75000.0
    )
    
    @validator('Geography')
    def validate_geography(cls, v):
        if v not in ['France', 'Spain', 'Germany']:
            raise ValueError('Geography must be one of: France, Spain, Germany')
        return v

    @validator('Gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be either Male or Female')
        return v

    @validator('CreditScore')
    def validate_credit_score(cls, v):
        if not 300 <= v <= 850:
            raise ValueError('Credit score must be between 300 and 850')
        return v
    
    @validator('Age')
    def validate_age(cls, v):
        if not isinstance(v, int):
            raise ValueError('Age must be an integer')
        if v < 18 or v > 100:
            raise ValueError('Age must be between 18 and 100')
        return v
    
    @validator('Balance')
    def validate_balance(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Balance must be a number')
        if v < 0:
            raise ValueError('Balance must be non-negative')
        if v > 1000000:
            raise ValueError(f'Balance cannot exceed 1,000,000, got: {v}')
        return float(v)
    
    @validator('EstimatedSalary')
    def validate_estimated_salary(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('EstimatedSalary must be a number')
        if v < 0:
            raise ValueError('Estimated salary must be positive')
        if v > 500000:
            raise ValueError(f'EstimatedSalary cannot exceed 500,000, got: {v}')
        return float(v)
    
    @validator('Tenure')
    def validate_tenure(cls, v):
        if not isinstance(v, int):
            raise ValueError('Tenure must be an integer')
        if v < 0 or v > 10:
            raise ValueError('Tenure must be between 0 and 10')
        return v
    
    @validator('NumOfProducts')
    def validate_num_products(cls, v):
        if not isinstance(v, int):
            raise ValueError('NumOfProducts must be an integer')
        if v < 1 or v > 4:
            raise ValueError('Number of products must be between 1 and 4')
        return v
    
    @validator('HasCrCard')
    def validate_has_cr_card(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError('HasCrCard must be 0 or 1')
        return v
    
    @validator('IsActiveMember')
    def validate_is_active_member(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError('IsActiveMember must be 0 or 1')
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
    
    @validator('customers')
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
    version: str = Field(..., description="Model version used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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
    loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Current model version")
    uptime: str = Field(..., description="Service uptime")
    timestamp: datetime = Field(..., description="Health check timestamp")
    model_status: Dict[str, Any] = Field(..., description="Model status information")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelManager:
    """
    Manages model loading and preprocessing components.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        # Initialize all attributes to None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = None
        self.model_name = None
        self.version = None
        self.training_date = None
        self.is_loaded = False
        self.performance_metrics = {}
        
        # Set default model path for backward compatibility
        self.model_path = Path(__file__).resolve().parent / 'models' / 'churn_model.pkl'
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the trained model and preprocessing components.
        
        Args:
            model_path: Path to the model file
        
        Returns:
            bool: True if model loaded successfully
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        global model, scaler, label_encoders, feature_names, model_metadata, model_loaded
        
        model_file_path = Path(model_path)
        if not model_file_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        logger.info(f"Loading model from {model_file_path}")
        
        # Load model package
        pkg = joblib.load(model_file_path)
        
        # Validate scikit-learn version compatibility
        self._validate_sklearn_compatibility(pkg)
        
        # Populate instance variables from package
        self.model = pkg['model']
        self.scaler = pkg.get('scaler')
        self.feature_names = pkg['feature_names']
        self.label_encoders = pkg.get('label_encoders', {})
        self.model_name = pkg.get('model_name', 'Unknown')
        self.version = pkg.get('version', '1.0.0')
        self.training_date = pkg.get('training_date', 'Unknown')
        self.performance_metrics = pkg.get('performance_metrics', {})
        
        # Also set global variables for backward compatibility
        model = self.model
        scaler = self.scaler
        label_encoders = self.label_encoders
        feature_names = self.feature_names
        
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
    
    def unload_model(self) -> None:
        """
        Unload the current model and reset all state.
        """
        global model, scaler, label_encoders, feature_names, model_metadata, model_loaded
        
        # Reset global variables
        model = None
        scaler = None
        label_encoders = None
        feature_names = None
        model_metadata = None
        model_loaded = False
        
        # Reset instance attributes
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = None
        self.model_name = None
        self.version = None
        self.training_date = None
        self.is_loaded = False
        self.performance_metrics = {}
        
        logger.info("Model unloaded successfully")
    
    def _validate_sklearn_compatibility(self, model_package: Dict[str, Any]) -> None:
        """
        Validate scikit-learn version compatibility between training and runtime.
        
        Args:
            model_package: Loaded model package dictionary
        """
        current_sklearn_version = sklearn.__version__
        
        # Try to get training sklearn version from model metadata
        training_sklearn_version = model_package.get('sklearn_version')
        
        if training_sklearn_version:
            try:
                current_ver = version.parse(current_sklearn_version)
                training_ver = version.parse(training_sklearn_version)
                
                # Check for major version differences
                if current_ver.major != training_ver.major:
                    logger.error(
                        f"Major scikit-learn version mismatch! "
                        f"Model trained with {training_sklearn_version}, "
                        f"runtime using {current_sklearn_version}. "
                        f"This may cause prediction errors."
                    )
                    raise ValueError(
                        f"Incompatible scikit-learn versions: "
                        f"training={training_sklearn_version}, runtime={current_sklearn_version}"
                    )
                
                # Check for minor version differences (warning only)
                elif current_ver.minor != training_ver.minor:
                    logger.warning(
                        f"Minor scikit-learn version mismatch detected. "
                        f"Model trained with {training_sklearn_version}, "
                        f"runtime using {current_sklearn_version}. "
                        f"Consider retraining the model for optimal compatibility."
                    )
                
                else:
                    logger.info(
                        f"Scikit-learn version compatibility confirmed: {current_sklearn_version}"
                    )
                    
            except Exception as e:
                logger.warning(f"Could not parse version strings: {e}")
        else:
            logger.warning(
                f"No sklearn version metadata found in model. "
                f"Current runtime version: {current_sklearn_version}. "
                f"Consider retraining the model with version metadata."
            )
    
    def preprocess_customer_data(self, customer_data: dict) -> np.ndarray:
        """
        Preprocess customer data for prediction.
        
        Args:
            customer_data: Dictionary containing customer data
            
        Returns:
            np.ndarray: Preprocessed feature array
        """
        # Build a one-row pandas DataFrame from customer_data
        df = pd.DataFrame([customer_data])
        
        # Apply label encoding for categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError as e:
                    logger.warning(f"Unknown category in {col}: {df[col].iloc[0]}. Using default encoding.")
                    # Handle unknown categories by using the most frequent class
                    df[col] = encoder.transform([encoder.classes_[0]])
        
        # If feature_names is set, reorder columns to match
        if self.feature_names is not None:
            df = df[self.feature_names]
        
        # Return scaler.transform(df.values) as numpy array
        return self.scaler.transform(df.values)
    
    def predict_single(self, customer_data: dict) -> dict:
        """
        Predict churn for a single customer.
        
        Args:
            customer_data: Dictionary containing customer data
            
        Returns:
            dict: Prediction results with churn_probability, churn_prediction, risk_level, confidence
        """
        # Preprocess the data
        X = self.preprocess_customer_data(customer_data)
        
        # Compute churn_probability
        churn_probability = float(self.model.predict_proba(X)[0][1])
        
        # Compute churn_prediction
        churn_prediction = bool(self.model.predict(X)[0])
        
        # Define risk_level
        if churn_probability >= 0.66:
            risk_level = "High"
        elif churn_probability >= 0.33:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Define confidence
        confidence = max(churn_probability, 1 - churn_probability)
        
        return {
            'churn_probability': churn_probability,
            'churn_prediction': churn_prediction,
            'risk_level': risk_level,
            'confidence': confidence
        }
    
    def predict_batch(self, customers: list[dict]) -> list[dict]:
        """
        Predict churn for multiple customers.
        
        Args:
            customers: List of customer data dictionaries
            
        Returns:
            list[dict]: List of prediction results
        """
        return [self.predict_single(c) for c in customers]
    
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





@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Bank Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "healthy",
        "endpoints": [
            "/",
            "/health",
            "/predict",
            "/batch_predict",
            "/model/info",
            "/docs",
            "/redoc"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint.
    """
    model_status = {
        "loaded": model_loaded,
        "model_type": model.__class__.__name__ if model else None,
        "features_count": len(feature_names) if feature_names else 0,
        "preprocessing_ready": scaler is not None and len(label_encoders) > 0
    }
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        loaded=model_loaded,
        version=model_metadata.get('version', 'Unknown'),
        uptime=manager.get_uptime(),
        timestamp=datetime.now(),
        model_status=model_status
    )


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """
    Get model information and metadata.
    """
    if not is_model_loaded():
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
    if not is_model_loaded():
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
            version=model_metadata.get('version', '1.0.0')
        )
        
        # Update metrics
        if PREDICTION_COUNTER:
            PREDICTION_COUNTER.inc()
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            customer_data.dict(),
            json.loads(response.json())
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
        if not is_model_loaded():
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
                        "version": model_metadata.get('version', '1.0.0')
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
                    "version": model_metadata.get('version', '1.0.0')
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
            "loaded": model_loaded,
            "model_name": model_metadata.get('name', 'unknown'),
            "version": model_metadata.get('version', '1.0.0')
        }
        
    except Exception as e:
        if ERROR_COUNTER:
            ERROR_COUNTER.inc()
        logger.error(f"Error in batch prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


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
        manager.load_model(str(manager.model_path))
        return {"message": "Model reloaded successfully", "timestamp": datetime.now().isoformat()}
    except FileNotFoundError as e:
        logger.error(f"Model file not found during reload: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {str(e)}"
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )