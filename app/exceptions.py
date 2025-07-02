"""Custom exception classes for the Customer Churn Analysis API.

This module defines custom exceptions to provide better error handling
and more informative error messages throughout the application.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class ChurnAPIException(Exception):
    """Base exception class for all custom exceptions in the Churn API.
    
    Attributes:
        message: Human-readable error message
        error_code: Application-specific error code
        details: Additional error details
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ModelNotLoadedException(ChurnAPIException):
    """Raised when attempting to use a model that hasn't been loaded."""
    
    def __init__(self, message: str = "Model is not loaded"):
        super().__init__(
            message=message,
            error_code="MODEL_NOT_LOADED",
            details={"suggestion": "Load a model before making predictions"}
        )


class ModelLoadException(ChurnAPIException):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, model_path: Optional[str] = None):
        details = {}
        if model_path:
            details["model_path"] = model_path
        
        super().__init__(
            message=f"Failed to load model: {message}",
            error_code="MODEL_LOAD_FAILED",
            details=details
        )


class DataValidationException(ChurnAPIException):
    """Raised when input data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)
        
        super().__init__(
            message=f"Data validation failed: {message}",
            error_code="DATA_VALIDATION_FAILED",
            details=details
        )


class PredictionException(ChurnAPIException):
    """Raised when prediction fails."""
    
    def __init__(self, message: str, customer_data: Optional[Dict[str, Any]] = None):
        details = {}
        if customer_data:
            details["customer_data_keys"] = list(customer_data.keys())
        
        super().__init__(
            message=f"Prediction failed: {message}",
            error_code="PREDICTION_FAILED",
            details=details
        )


class PreprocessingException(ChurnAPIException):
    """Raised when data preprocessing fails."""
    
    def __init__(self, message: str, step: Optional[str] = None):
        details = {}
        if step:
            details["preprocessing_step"] = step
        
        super().__init__(
            message=f"Preprocessing failed: {message}",
            error_code="PREPROCESSING_FAILED",
            details=details
        )


class ConfigurationException(ChurnAPIException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=f"Configuration error: {message}",
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class DatabaseException(ChurnAPIException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=f"Database error: {message}",
            error_code="DATABASE_ERROR",
            details=details
        )


class RateLimitException(ChurnAPIException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


# HTTP Exception mappings
def to_http_exception(exc: ChurnAPIException) -> HTTPException:
    """Convert custom exception to FastAPI HTTPException.
    
    Args:
        exc: Custom exception to convert
        
    Returns:
        HTTPException with appropriate status code and details
    """
    status_code_mapping = {
        "MODEL_NOT_LOADED": status.HTTP_503_SERVICE_UNAVAILABLE,
        "MODEL_LOAD_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "DATA_VALIDATION_FAILED": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "PREDICTION_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "PREPROCESSING_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "DATABASE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "RATE_LIMIT_EXCEEDED": status.HTTP_429_TOO_MANY_REQUESTS,
    }
    
    status_code = status_code_mapping.get(
        exc.error_code, 
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    
    return HTTPException(
        status_code=status_code,
        detail=exc.to_dict()
    )


# Decorator for exception handling
def handle_exceptions(func):
    """Decorator to automatically convert custom exceptions to HTTP exceptions.
    
    Usage:
        @handle_exceptions
        def my_endpoint():
            # Your endpoint logic here
            pass
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ChurnAPIException as e:
            raise to_http_exception(e)
        except Exception as e:
            # Convert unexpected exceptions to generic server error
            generic_exc = ChurnAPIException(
                message="An unexpected error occurred",
                error_code="INTERNAL_SERVER_ERROR",
                details={"original_error": str(e)}
            )
            raise to_http_exception(generic_exc)
    
    return wrapper