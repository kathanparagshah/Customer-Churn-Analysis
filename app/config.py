"""Application configuration using Pydantic BaseSettings.

This module provides centralized configuration management with support for:
- Environment-specific settings
- Type validation
- Default values
- Configuration validation
"""

import os
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from enum import Enum


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_TITLE: str = Field("Customer Churn Analysis API", description="API title")
    API_VERSION: str = Field("1.0.0", description="API version")
    API_HOST: str = Field("0.0.0.0", description="Host address to bind the FastAPI server")
    API_PORT: int = Field(8000, description="Port for the FastAPI server")
    
    # Environment
    ENVIRONMENT: Environment = Field(Environment.DEVELOPMENT, description="Environment (development, staging, production)")
    DEBUG: bool = Field(True, description="Enable debug mode")
    
    # Model Configuration
    MODEL_PATH: str = Field("models/customer_churn_model.pkl", description="Filesystem path to churn model pickle")
    MODEL_DIR: str = Field("models", description="Model directory")
    MODEL_BACKUP_DIR: str = Field("models/backup", description="Model backup directory")
    THRESHOLD: float = Field(0.7, ge=0.0, le=1.0, description="Default churn decision threshold")
    DEFAULT_THRESHOLD: float = Field(0.7, ge=0.0, le=1.0, description="Default churn decision threshold (alias)")
    MODEL_AUTO_LOAD: bool = Field(True, description="Auto-load model on startup")
    MODEL_LOAD_TIMEOUT: int = Field(60, description="Model loading timeout in seconds")
    BATCH_SIZE: int = Field(1000, description="Default batch prediction size")
    MAX_BATCH_SIZE: int = Field(10000, description="Maximum batch prediction size")
    
    # Logging
    LOG_LEVEL: LogLevel = Field(LogLevel.INFO, description="Logging level")
    LOG_FORMAT: str = Field("json", description="Log format (json or plain)")
    LOG_FILE: Optional[str] = Field(None, description="Log file path")
    LOG_ROTATION: str = Field("1 day", description="Log rotation interval")
    LOG_RETENTION: str = Field("30 days", description="Log retention period")
    
    # Database
    DATABASE_URL: str = Field("postgresql://user:password@localhost:5432/churn_db", description="PostgreSQL database URL")
    ANALYTICS_DB_PATH: str = Field("data/analytics.db", description="Path to SQLite analytics database")
    DB_MIN_CONNECTIONS: int = Field(1, description="Minimum database connections")
    DB_MAX_CONNECTIONS: int = Field(10, description="Maximum database connections")
    DB_CONNECTION_TIMEOUT: int = Field(30, description="Database connection timeout")
    
    # Redis Configuration
    REDIS_HOST: str = Field("localhost", description="Redis host")
    REDIS_PORT: int = Field(6379, description="Redis port")
    REDIS_PASSWORD: Optional[str] = Field(None, description="Redis password")
    REDIS_DB: int = Field(0, description="Redis database number")
    REDIS_MAX_CONNECTIONS: int = Field(20, description="Redis max connections")
    
    # CORS Configuration
    CORS_ENABLED: bool = Field(True, description="Enable CORS middleware")
    CORS_ORIGINS: List[str] = Field(["http://localhost:3000", "http://localhost:8080"], description="CORS allowed origins")
    ALLOWED_ORIGINS: List[str] = Field(["http://localhost:3000", "http://localhost:8080"], description="CORS allowed origins (alias)")
    CORS_METHODS: List[str] = Field(["GET", "POST", "PUT", "DELETE"], description="Allowed CORS methods")
    
    # Security
    TRUSTED_HOSTS: Optional[List[str]] = Field(None, description="Trusted hosts for security")
    API_KEY: Optional[str] = Field(None, description="API key for authentication")
    SECRET_KEY: str = Field("your-secret-key-change-in-production", description="Secret key for JWT")
    JWT_ALGORITHM: str = Field("HS256", description="JWT algorithm")
    JWT_EXPIRATION: int = Field(3600, description="JWT expiration in seconds")
    RATE_LIMIT_ENABLED: bool = Field(True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(100, description="Requests per minute")
    RATE_LIMIT_WINDOW: int = Field(60, description="Rate limit window in seconds")
    ENABLE_SECURITY_HEADERS: bool = Field(True, description="Enable security headers")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(9090, description="Metrics server port")
    HEALTH_CHECK_INTERVAL: int = Field(30, description="Health check interval in seconds")
    DEPENDENCY_TIMEOUT: int = Field(5, description="Dependency check timeout")
    SLOW_QUERY_THRESHOLD: float = Field(1.0, description="Slow query threshold in seconds")
    MEMORY_THRESHOLD: float = Field(90.0, description="Memory usage threshold percentage")
    DISK_THRESHOLD: float = Field(1.0, description="Minimum free disk space in GB")
    ALERT_WEBHOOK_URL: Optional[str] = Field(None, description="Webhook URL for alerts")
    ALERT_EMAIL: Optional[str] = Field(None, description="Email for alerts")
    
    # Feature Flags
    ENABLE_BATCH_PREDICTIONS: bool = Field(True, description="Enable batch predictions")
    ENABLE_MODEL_RETRAINING: bool = Field(False, description="Enable model retraining")
    ENABLE_DATA_VALIDATION: bool = Field(True, description="Enable input data validation")
    ENABLE_PREDICTION_LOGGING: bool = Field(True, description="Enable prediction logging")
    
    # Performance
    REQUEST_TIMEOUT: int = Field(30, description="Request timeout in seconds")
    MAX_REQUEST_SIZE: int = Field(10 * 1024 * 1024, description="Max request size in bytes")
    WORKERS: int = Field(1, description="Number of worker processes")
    
    @validator('ENVIRONMENT', pre=True)
    def validate_environment(cls, v):
        """Validate environment value."""
        if isinstance(v, str):
            # Handle common variations
            env_map = {
                'test': Environment.TESTING,
                'testing': Environment.TESTING,
                'dev': Environment.DEVELOPMENT,
                'development': Environment.DEVELOPMENT,
                'prod': Environment.PRODUCTION,
                'production': Environment.PRODUCTION,
                'staging': Environment.STAGING,
                'stage': Environment.STAGING
            }
            normalized = v.lower().strip()
            if normalized in env_map:
                return env_map[normalized]
            # Try direct enum conversion
            try:
                return Environment(normalized)
            except ValueError:
                # Default to development for unknown values
                return Environment.DEVELOPMENT
        return v
    
    @validator('LOG_LEVEL', pre=True)
    def validate_log_level(cls, v):
        """Validate log level value."""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @validator('DEBUG')
    def validate_debug_mode(cls, v, values):
        """Auto-enable debug in development."""
        if values.get('ENVIRONMENT') == Environment.DEVELOPMENT:
            return True
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.ENVIRONMENT == Environment.TESTING
    
    @property
    def model_path(self) -> Path:
        """Get model path as Path object."""
        return Path(self.MODEL_PATH)
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return {
            "allow_origins": self.CORS_ORIGINS,
            "allow_methods": self.CORS_METHODS,
            "allow_headers": ["*"],
            "allow_credentials": True
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


# Global settings instance
settings = AppSettings()


def get_settings() -> AppSettings:
    """Get application settings.
    
    Returns:
        AppSettings: Application settings instance
    """
    return settings


def reload_settings() -> AppSettings:
    """Reload settings from environment.
    
    Returns:
        AppSettings: Reloaded settings instance
    """
    global settings
    settings = AppSettings()
    return settings


def validate_settings() -> List[str]:
    """Validate current settings and return any issues.
    
    Returns:
        List[str]: List of validation issues
    """
    issues = []
    
    # Check production requirements
    if settings.is_production:
        if settings.SECRET_KEY == "your-secret-key-change-in-production":
            issues.append("Secret key must be changed in production")
        
        if not settings.API_KEY:
            issues.append("API key should be set in production")
        
        if settings.DEBUG:
            issues.append("Debug mode should be disabled in production")
        
        if "*" in settings.CORS_ORIGINS:
            issues.append("CORS origins should be restricted in production")
    
    # Check model configuration
    if not settings.model_path.exists() and settings.MODEL_AUTO_LOAD:
        issues.append(f"Model file not found: {settings.model_path}")
    
    # Check required directories
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    
    return issues