"""Model management service for loading, preprocessing, and prediction."""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from ..config import settings
from ..logging import get_logger
from ..models.schemas import CustomerData

logger = get_logger("model_manager")

# Global model state for backward compatibility
model = None
scaler = None
label_encoders = None
feature_names = None
model_metadata = None
model_loaded = False


# Global is_model_loaded() function removed - use ModelManager.is_loaded instead


def calculate_risk_level(probability: float) -> str:
    """Calculate risk level based on churn probability.
    
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
    """Calculate model confidence based on probability.
    
    Args:
        probability (float): Churn probability
        
    Returns:
        float: Confidence score (0-1)
    """
    # Confidence is higher when probability is closer to 0 or 1
    return 2 * abs(probability - 0.5)


class ModelManager:
    """Manages model loading and preprocessing components."""
    
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
        
        # Set default model path from settings
        self.model_path = Path(settings.MODEL_PATH)
        
        # Try to load model on initialization
        if self.model_path.exists():
            try:
                self.load_model(str(self.model_path))
            except Exception as e:
                logger.warning(f"Failed to load model on initialization: {e}")
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained model and preprocessing components.
        
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
        
        try:
            # Load the model package
            model_package = joblib.load(model_file_path)
            
            # Extract components
            model = model_package['model']
            scaler = model_package.get('scaler')
            label_encoders = model_package.get('label_encoders', {})
            feature_names = model_package.get('feature_names', [])
            model_metadata = model_package.get('metadata', {})
            
            # Update instance attributes
            self.model = model
            self.scaler = scaler
            self.feature_names = feature_names
            self.label_encoders = label_encoders
            self.model_name = model_metadata.get('model_name', 'Unknown')
            self.version = model_metadata.get('version', '1.0.0')
            self.training_date = model_metadata.get('training_date', 'Unknown')
            self.performance_metrics = model_metadata.get('performance_metrics', {})
            self.model_path = model_file_path
            self.is_loaded = True
            
            # Update global state
            model_loaded = True
            
            logger.info(f"Model loaded successfully: {self.model_name} v{self.version}")
            logger.info(f"Features: {len(self.feature_names)} features")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.unload_model()
            raise
    
    def unload_model(self) -> None:
        """Unload the current model and reset all state."""
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
        
        logger.info("Model unloaded")
    
    def preprocess_customer_data(self, customer_data) -> np.ndarray:
        """Preprocess customer data for prediction using the model's scaler.
        
        Args:
            customer_data: Customer data to preprocess (CustomerData object or dict)
            
        Returns:
            np.ndarray: Preprocessed feature array
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Convert to dict if it's a Pydantic model
        if hasattr(customer_data, 'model_dump'):
            data_dict = customer_data.model_dump()
        elif hasattr(customer_data, 'dict'):
            data_dict = customer_data.dict()
        else:
            data_dict = customer_data
        
        # Create DataFrame
        df = pd.DataFrame([data_dict])
        
        # Apply label encoding for categorical variables
        if self.label_encoders:
            for column, encoder in self.label_encoders.items():
                if column in df.columns:
                    try:
                        df[column] = encoder.transform(df[column])
                    except ValueError as e:
                        logger.warning(f"Label encoding failed for {column}: {e}")
                        # Use the most frequent class as fallback
                        df[column] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all expected features are present
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in df.columns:
                    logger.warning(f"Missing feature: {feature}, setting to 0")
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
        
        # Apply scaling
        if self.scaler:
            features = self.scaler.transform(df)
        else:
            features = df.values
        
        return features
    
    def predict_single(self, customer_data: dict, threshold: float = None) -> dict:
        """Predict churn for a single customer with configurable threshold.
        
        Args:
            customer_data: Dictionary containing customer data
            threshold: Decision threshold for churn prediction
            
        Returns:
            dict: Prediction results
        """
        if threshold is None:
            threshold = settings.THRESHOLD
        
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess the data
            features = self.preprocess_customer_data(customer_data)
            
            # Make prediction
            probability = self.model.predict_proba(features)[0][1]  # Probability of churn (class 1)
            prediction = probability >= threshold
            
            # Calculate additional metrics
            risk_level = calculate_risk_level(probability)
            confidence = calculate_confidence(probability)
            
            return {
                'churn_probability': float(probability),
                'churn_prediction': bool(prediction),
                'risk_level': risk_level,
                'confidence': float(confidence),
                'threshold_used': float(threshold),
                'model_version': self.version or 'Unknown'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, customers: list[dict], threshold: float = None) -> list[dict]:
        """Predict churn for multiple customers with configurable threshold.
        
        Args:
            customers: List of customer data dictionaries
            threshold: Decision threshold for churn prediction
            
        Returns:
            list[dict]: List of prediction results
        """
        if threshold is None:
            threshold = settings.THRESHOLD
        
        return [self.predict_single(c, threshold) for c in customers]
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information.
        
        Returns:
            dict: Model information and metadata
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "training_date": self.training_date,
            "features": self.feature_names,
            "model_type": self.model.__class__.__name__ if self.model else None,
            "model_loaded": self.is_loaded,
            "preprocessing_components": {
                "scaler": self.scaler.__class__.__name__ if self.scaler else None,
                "label_encoders": list(self.label_encoders.keys()) if self.label_encoders else [],
                "feature_count": len(self.feature_names) if self.feature_names else 0
            },
            "performance_metrics": self.performance_metrics,
            "model_path": str(self.model_path),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_health_status(self) -> dict:
        """Get model health status for health checks.
        
        Returns:
            dict: Health status information
        """
        uptime = datetime.now() - self.start_time
        
        return {
            "status": "healthy" if self.is_loaded else "unhealthy",
            "model_loaded": self.is_loaded,
            "version": self.version or "Unknown",
            "uptime": str(uptime),
            "timestamp": datetime.now().isoformat(),
            "model_status": {
                "loaded": self.is_loaded,
                "model_type": self.model.__class__.__name__ if self.model else None,
                "features_count": len(self.feature_names) if self.feature_names else 0,
                "preprocessing_ready": bool(self.scaler and self.label_encoders)
            }
        }


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Dependency to get model manager instance.
    
    Returns:
        ModelManager: The global model manager instance
    """
    return model_manager