"""Application configuration using Pydantic BaseSettings."""

from pydantic import PostgresDsn, Field
from pydantic_settings import BaseSettings
from typing import List, Optional


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_TITLE: str = Field("Customer Churn Analysis API", description="API title")
    API_VERSION: str = Field("1.0.0", description="API version")
    API_HOST: str = Field("0.0.0.0", description="Host address to bind the FastAPI server")
    API_PORT: int = Field(8000, description="Port for the FastAPI server")
    
    # Environment
    ENVIRONMENT: str = Field("development", description="Environment (development, staging, production)")
    DEBUG: bool = Field(True, description="Enable debug mode")
    
    # Model Configuration
    MODEL_PATH: str = Field(..., description="Filesystem path to churn model pickle")
    THRESHOLD: float = Field(0.7, ge=0.0, le=1.0, description="Default churn decision threshold")
    DEFAULT_THRESHOLD: float = Field(0.7, ge=0.0, le=1.0, description="Default churn decision threshold (alias)")
    
    # Logging
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    
    # Database
    DATABASE_URL: PostgresDsn = Field(..., description="PostgreSQL database URL")
    ANALYTICS_DB_PATH: str = Field("data/analytics.db", description="Path to SQLite analytics database")
    
    # CORS Configuration
    CORS_ENABLED: bool = Field(True, description="Enable CORS middleware")
    CORS_ORIGINS: List[str] = Field(["*"], description="CORS allowed origins")
    ALLOWED_ORIGINS: List[str] = Field(["*"], description="CORS allowed origins (alias)")
    
    # Security
    TRUSTED_HOSTS: Optional[List[str]] = Field(None, description="Trusted hosts for security")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(True, description="Enable Prometheus metrics")
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = AppSettings()