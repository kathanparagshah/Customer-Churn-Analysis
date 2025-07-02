"""Pydantic models and schemas for the Customer Churn Analysis API."""

from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator


class CustomerData(BaseModel):
    """Pydantic model for customer data input."""
    
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
    
    CreditScore: int = Field(
        ..., 
        description="Customer credit score (300-850)",
        example=650
    )
    Geography: str = Field(
        ..., 
        description="Customer geography (France, Germany, Spain)",
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
        description="Number of years as customer (0-10)",
        example=5
    )
    Balance: float = Field(
        ..., 
        description="Account balance",
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
        description="Estimated salary",
        example=75000.0
    )
    
    @validator('CreditScore')
    def validate_credit_score(cls, v):
        if not 300 <= v <= 850:
            raise ValueError('Credit score must be between 300 and 850')
        return v
    
    @validator('Age')
    def validate_age(cls, v):
        if not 18 <= v <= 100:
            raise ValueError('Age must be between 18 and 100')
        return v
    
    @validator('Geography')
    def validate_geography(cls, v):
        if v not in ['France', 'Germany', 'Spain']:
            raise ValueError('Geography must be one of: France, Germany, Spain')
        return v
    
    @validator('Gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be either Male or Female')
        return v
    
    @validator('HasCrCard', 'IsActiveMember')
    def validate_binary_fields(cls, v):
        if v not in [0, 1]:
            raise ValueError('Value must be 0 or 1')
        return v
    
    @validator('NumOfProducts')
    def validate_num_products(cls, v):
        if not 1 <= v <= 4:
            raise ValueError('Number of products must be between 1 and 4')
        return v


class BatchCustomerData(BaseModel):
    """Pydantic model for batch prediction input."""
    
    customers: List[CustomerData] = Field(..., description="List of customer data")
    
    @validator('customers')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError('Batch must contain at least one customer')
        if len(v) > 1000:  # Limit batch size
            raise ValueError('Batch size cannot exceed 1000 customers')
        return v


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    churn_probability: float = Field(..., description="Probability of customer churn (0-1)")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")
    confidence: float = Field(..., description="Model confidence score")
    timestamp: float = Field(..., description="Prediction timestamp")
    version: str = Field(..., description="Model version used")


class BatchPredictionResponse(BaseModel):
    """Pydantic model for batch prediction response."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    batch_id: str = Field(..., description="Unique batch identifier")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of customers in batch")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: float = Field(..., description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Pydantic model for health check response."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    status: str = Field(..., description="Service status")
    loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Current model version")
    uptime: str = Field(..., description="Service uptime")
    timestamp: str = Field(..., description="Health check timestamp")
    model_status: Optional[Dict[str, Any]] = Field(None, description="Model status information")


class ModelInfoResponse(BaseModel):
    """Pydantic model for model info response."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    model_name: Optional[str] = Field(None, description="Model name")
    version: Optional[str] = Field(None, description="Model version")
    training_date: Optional[str] = Field(None, description="Model training date")
    features: Optional[List[str]] = Field(None, description="Model features")
    model_type: Optional[str] = Field(None, description="Model type")
    feature_count: Optional[int] = Field(None, description="Number of features")
    preprocessing_components: Optional[Dict[str, Any]] = Field(None, description="Preprocessing components")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Model performance metrics")
    model_path: Optional[str] = Field(None, description="Model file path")
    timestamp: Optional[str] = Field(None, description="Response timestamp")


class ErrorResponse(BaseModel):
    """Pydantic model for error responses."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")