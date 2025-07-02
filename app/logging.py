"""Structured logging configuration for the application."""

import time
import logging
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextvars import ContextVar
from fastapi import FastAPI, Request

from .config import settings

# Context variable for request correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        if correlation_id.get():
            log_entry["correlation_id"] = correlation_id.get()
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "getMessage"
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class PlainFormatter(logging.Formatter):
    """Plain text formatter for console output."""
    
    def __init__(self):
        super().__init__(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as plain text."""
        # First call parent format to populate asctime
        formatted = super().format(record)
        
        # Add correlation ID if available
        correlation = f"[{correlation_id.get()}] " if correlation_id.get() else ""
        if correlation:
            # Insert correlation ID after timestamp
            parts = formatted.split(' - ', 1)
            if len(parts) == 2:
                return f"{parts[0]} - {correlation}{parts[1]}"
        
        return formatted


def init_logging() -> None:
    """Initialize structured JSON logging configuration."""
    # Set log level from settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler with structured logging
    file_handler = logging.FileHandler(logs_dir / "app.log")
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(log_level)
    
    # Console handler with plain text
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(PlainFormatter())
    console_handler.setLevel(log_level)
    
    # Error file handler
    error_handler = logging.FileHandler(logs_dir / "error.log")
    error_handler.setFormatter(JSONFormatter())
    error_handler.setLevel(logging.ERROR)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(log_level)


def set_correlation_id(request_id: Optional[str] = None) -> str:
    """Set correlation ID for request tracing.
    
    Args:
        request_id: Optional request ID, generates UUID if not provided
        
    Returns:
        The correlation ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    correlation_id.set(request_id)
    return request_id


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID.
    
    Returns:
        Current correlation ID or None
    """
    return correlation_id.get()


def log_performance(func_name: str, duration: float, **kwargs) -> None:
    """Log performance metrics.
    
    Args:
        func_name: Name of the function being measured
        duration: Execution duration in seconds
        **kwargs: Additional metrics to log
    """
    logger = logging.getLogger("performance")
    logger.info(
        f"Performance metric: {func_name}",
        extra={
            "metric_type": "performance",
            "function": func_name,
            "duration_seconds": duration,
            **kwargs
        }
    )


def include_logging(app: FastAPI) -> None:
    """Apply logging middleware and configuration to FastAPI instance."""
    init_logging()
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log request start and completion with timing."""
        start = time.time()
        
        # Set correlation ID for this request
        request_id = set_correlation_id()
        
        # Log request start
        logger = logging.getLogger("app")
        logger.info(
            "request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "request_id": request_id,
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_ms = int((time.time() - start) * 1000)
        
        # Log request completion
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": process_ms,
                "client_ip": request.client.host if request.client else None,
                "request_id": request_id,
            }
        )
        
        return response


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(f"churn_api.{name}")