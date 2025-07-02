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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
try:
    from analytics_db import analytics_db
except ImportError:
    # For testing environment, try different import paths
    try:
        from deployment.analytics_db import analytics_db
    except ImportError:
        # Create a mock analytics_db for testing
        class MockAnalyticsDB:
            def log_prediction(self, *args, **kwargs):
                pass
            def log_batch_prediction(self, *args, **kwargs):
                pass
        analytics_db = MockAnalyticsDB()

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
    global model, scaler, label_encoders, feature_names, model_loaded, model_metadata, model_manager
    
    # Startup
    logger.info("Starting Churn Prediction API...")
    try:
        # Initialize model manager
        model_manager = ModelManager()
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

# Initialize model manager early
model_manager = None

def is_model_loaded():
    """Check if the model is loaded."""
    return model_loaded


class CustomerData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
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
    )
    
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
    
    @field_validator('Geography')
    @classmethod
    def validate_geography(cls, v):
        if v not in ['France', 'Spain', 'Germany']:
            raise ValueError('Geography must be one of: France, Spain, Germany')
        return v

    @field_validator('Gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be either Male or Female')
        return v

    @field_validator('CreditScore')
    @classmethod
    def validate_credit_score(cls, v):
        if not 300 <= v <= 850:
            raise ValueError('Credit score must be between 300 and 850')
        return v
    
    @field_validator('Age')
    @classmethod
    def validate_age(cls, v):
        if not isinstance(v, int):
            raise ValueError('Age must be an integer')
        if v < 18 or v > 100:
            raise ValueError('Age must be between 18 and 100')
        return v
    
    @field_validator('Balance')
    @classmethod
    def validate_balance(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Balance must be a number')
        if v < 0:
            raise ValueError('Balance must be non-negative')
        if v > 1000000:
            raise ValueError(f'Balance cannot exceed 1,000,000, got: {v}')
        return float(v)
    
    @field_validator('EstimatedSalary')
    @classmethod
    def validate_estimated_salary(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('EstimatedSalary must be a number')
        if v < 0:
            raise ValueError('Estimated salary must be positive')
        if v > 500000:
            raise ValueError(f'EstimatedSalary cannot exceed 500,000, got: {v}')
        return float(v)
    
    @field_validator('Tenure')
    @classmethod
    def validate_tenure(cls, v):
        if not isinstance(v, int):
            raise ValueError('Tenure must be an integer')
        if v < 0 or v > 10:
            raise ValueError('Tenure must be between 0 and 10')
        return v
    
    @field_validator('NumOfProducts')
    @classmethod
    def validate_num_products(cls, v):
        if not isinstance(v, int):
            raise ValueError('NumOfProducts must be an integer')
        if v < 1 or v > 4:
            raise ValueError('Number of products must be between 1 and 4')
        return v
    
    @field_validator('HasCrCard')
    @classmethod
    def validate_has_cr_card(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError('HasCrCard must be 0 or 1')
        return v
    
    @field_validator('IsActiveMember')
    @classmethod
    def validate_is_active_member(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError('IsActiveMember must be 0 or 1')
        return v


class BatchCustomerData(BaseModel):
    """
    Pydantic model for batch prediction input.
    """
    customers: List[CustomerData] = Field(..., description="List of customer data")
    
    @field_validator('customers')
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > 1000:  # Limit batch size
            raise ValueError('Batch size cannot exceed 1000 customers')
        return v


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
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
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")


class HealthResponse(BaseModel):
    """
    Pydantic model for health check response.
    """
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    status: str = Field(..., description="Service status")
    loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Current model version")
    uptime: str = Field(..., description="Service uptime")
    timestamp: datetime = Field(..., description="Health check timestamp")
    model_status: Dict[str, Any] = Field(..., description="Model status information")


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
    
    def preprocess_customer_data(self, customer_data) -> np.ndarray:
        """
        Preprocess customer data for prediction using the model's scaler.
        
        Args:
            customer_data: Customer data to preprocess (CustomerData object or dict)
            
        Returns:
            np.ndarray: Preprocessed feature array
        """
        # Convert to DataFrame
        try:
            logger.info("preprocess_customer_data called")
            logger.info(str(type(customer_data)))
            is_dict = isinstance(customer_data, dict)
            logger.info(f"is_dict: {is_dict}")
            
            if is_dict:
                # Already a dictionary
                logger.info("Using dictionary directly")
                df = pd.DataFrame([customer_data])
            else:
                # CustomerData object
                logger.info("Converting CustomerData object to dict")
                if hasattr(customer_data, 'dict'):
                    df = pd.DataFrame([customer_data.dict()])
                else:
                    # Fallback: treat as dict anyway
                    logger.warning("Object doesn't have dict() method, treating as dict")
                    df = pd.DataFrame([customer_data])
        except Exception as e:
            logger.error(f"Error in preprocess_customer_data: {e}")
            logger.error(f"customer_data type: {type(customer_data)}")
            logger.error(f"customer_data value: {customer_data}")
            raise
        
        # Create one-hot encoded categorical features
        # Geography encoding
        df['Geography_Germany'] = 1 if df['Geography'].iloc[0] == 'Germany' else 0
        df['Geography_Spain'] = 1 if df['Geography'].iloc[0] == 'Spain' else 0
        
        # Gender encoding
        df['Gender_Male'] = 1 if df['Gender'].iloc[0] == 'Male' else 0
        
        # Create feature array with correct names (matching model's expected feature names)
        # The model expects features with '_scaled' suffix for numeric features
        feature_dict = {}
        
        # Add numeric features with '_scaled' suffix (values will be scaled later)
        numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        for feature in numeric_features:
            feature_dict[f'{feature}_scaled'] = df[feature].iloc[0]
        
        # Add categorical features
        feature_dict['Geography_Germany'] = df['Geography_Germany'].iloc[0]
        feature_dict['Geography_Spain'] = df['Geography_Spain'].iloc[0]
        feature_dict['Gender_Male'] = df['Gender_Male'].iloc[0]
        
        # Add binary features
        feature_dict['HasCrCard'] = df['HasCrCard'].iloc[0]
        feature_dict['IsActiveMember'] = df['IsActiveMember'].iloc[0]
        
        # Create DataFrame with correct feature names and order
        feature_df = pd.DataFrame([feature_dict])
        
        # Ensure we have all expected features in the correct order
        if self.feature_names:
            # Reorder columns to match the model's expected feature order
            feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0)
        
        # Apply scaling to the complete feature set (scaler was trained on this format)
        scaled_features = self.scaler.transform(feature_df)
        
        return scaled_features
    
    def predict_single(self, customer_data: dict, threshold: float = 0.7) -> dict:
        """
        Predict churn for a single customer with configurable threshold.
        
        Args:
            customer_data: Dictionary containing customer data
            threshold: Decision threshold for churn prediction (default: 0.7 to reduce false alarms)
            
        Returns:
            dict: Prediction results with churn_probability, churn_prediction, risk_level, confidence
        """
        # Preprocess the data
        X = self.preprocess_customer_data(customer_data)
        
        # Compute churn_probability
        churn_probability = float(self.model.predict_proba(X)[0][1])
        
        # Compute churn_prediction using custom threshold (higher threshold = fewer false alarms)
        churn_prediction = bool(churn_probability >= threshold)
        
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
            'confidence': confidence,
            'threshold_used': threshold
        }
    
    def predict_batch(self, customers: list[dict], threshold: float = 0.7) -> list[dict]:
        """
        Predict churn for multiple customers with configurable threshold.
        
        Args:
            customers: List of customer data dictionaries
            threshold: Decision threshold for churn prediction (default: 0.7 to reduce false alarms)
            
        Returns:
            list[dict]: List of prediction results
        """
        return [self.predict_single(c, threshold) for c in customers]
    
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


def get_model_manager():
    """
    Dependency to get model manager instance.
    """
    return model_manager


# Removed standalone preprocess_customer_data function - using ModelManager's method instead


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
    background_tasks: BackgroundTasks,
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Decision threshold for churn prediction (higher = fewer false alarms)"),
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict churn for a single customer with configurable threshold.
    
    Args:
        customer_data (CustomerData): Customer data for prediction
        background_tasks (BackgroundTasks): Background tasks for logging
        threshold (float): Decision threshold (default: 0.7 to reduce false alarms)
        manager (ModelManager): Model manager instance
        
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
        # Debug logging
        logger.info(f"API received customer_data type: {type(customer_data)}")
        if isinstance(customer_data, dict):
            customer_dict = customer_data
        else:
            customer_dict = customer_data.dict()
        logger.info(f"Using dict type: {type(customer_dict)}")
        logger.info(f"Dict content: {customer_dict}")
        
        # Make prediction using ModelManager with custom threshold
        if PREDICTION_LATENCY:
            with PREDICTION_LATENCY.time():
                prediction_result = manager.predict_single(customer_dict, threshold)
        else:
            prediction_result = manager.predict_single(customer_dict, threshold)
        
        # Create response
        response = PredictionResponse(
            churn_probability=prediction_result['churn_probability'],
            churn_prediction=prediction_result['churn_prediction'],
            risk_level=prediction_result['risk_level'],
            confidence=prediction_result['confidence'],
            timestamp=datetime.now(),
            version=model_metadata.get('version', '1.0.0')
        )
        
        # Update metrics
        if PREDICTION_COUNTER:
            PREDICTION_COUNTER.inc()
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            customer_dict,
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
    background_tasks: BackgroundTasks,
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Decision threshold for churn prediction (higher = fewer false alarms)"),
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict churn for multiple customers with configurable threshold.
    
    Args:
        batch_data (BatchCustomerData): Batch of customer data
        background_tasks (BackgroundTasks): Background tasks for logging
        threshold (float): Decision threshold (default: 0.7 to reduce false alarms)
        manager (ModelManager): Model manager instance
        
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
        
        # Convert customer data to list of dictionaries
        customers_list = [customer.dict() for customer in batch_data.customers]
        
        # Make batch prediction using ModelManager with custom threshold
        if PREDICTION_LATENCY:
            with PREDICTION_LATENCY.time():
                prediction_results = manager.predict_batch(customers_list, threshold)
        else:
            prediction_results = manager.predict_batch(customers_list, threshold)
        
        # Format predictions for API response
        predictions = []
        for result in prediction_results:
            pred_response = {
                "churn_probability": result['churn_probability'],
                "churn_prediction": result['churn_prediction'],
                "risk_level": result['risk_level'],
                "confidence": result['confidence'],
                "threshold_used": result['threshold_used'],
                "timestamp": datetime.now().isoformat(),
                "version": model_metadata.get('version', '1.0.0')
            }
            predictions.append(pred_response)
        
        # Calculate summary statistics
        probabilities = [p["churn_probability"] for p in predictions] if predictions else []
        churn_predictions = [p["churn_prediction"] for p in predictions] if predictions else []
        
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


@app.get("/analytics/daily-metrics")
async def get_daily_metrics(days: int = 30):
    """
    Get daily analytics metrics for the specified number of days.
    
    Args:
        days (int): Number of days to retrieve (default: 30)
        
    Returns:
        List[Dict]: Daily metrics data
    """
    try:
        metrics = analytics_db.get_daily_metrics(days)
        return {
            "metrics": metrics,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting daily metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics error: {str(e)}"
        )


@app.get("/analytics/prediction-trends")
async def get_prediction_trends(days: int = 30):
    """
    Get prediction trends and statistics.
    
    Args:
        days (int): Number of days to analyze (default: 30)
        
    Returns:
        Dict: Prediction trends and overall statistics
    """
    try:
        trends = analytics_db.get_prediction_trends(days)
        return {
            "trends": trends,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting prediction trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics error: {str(e)}"
        )


@app.get("/analytics/risk-distribution")
async def get_risk_distribution(days: int = 30):
    """
    Get risk level distribution for the specified period.
    
    Args:
        days (int): Number of days to analyze (default: 30)
        
    Returns:
        Dict: Risk level distribution
    """
    try:
        distribution = analytics_db.get_risk_distribution(days)
        return {
            "distribution": distribution,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting risk distribution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics error: {str(e)}"
        )


@app.get("/analytics/dashboard")
async def get_analytics_dashboard(days: int = 30):
    """
    Get comprehensive analytics data for dashboard.
    
    Args:
        days (int): Number of days to analyze (default: 30)
        
    Returns:
        Dict: Complete analytics dashboard data
    """
    try:
        # Get all analytics data
        daily_metrics = analytics_db.get_daily_metrics(days)
        trends = analytics_db.get_prediction_trends(days)
        risk_distribution = analytics_db.get_risk_distribution(days)
        
        return {
            "daily_metrics": daily_metrics,
            "prediction_trends": trends,
            "risk_distribution": risk_distribution,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics error: {str(e)}"
        )


async def log_prediction(customer_data: Dict[str, Any], prediction: Dict[str, Any]):
    """
    Log prediction for monitoring.
    
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
    
    # Log to analytics database
    analytics_db.log_prediction(log_entry)


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
    
    # Log to analytics database
    analytics_db.log_batch_prediction(log_entry)


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