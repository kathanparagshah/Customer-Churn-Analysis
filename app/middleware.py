"""Custom middleware for the Customer Churn Analysis API.

This module provides middleware for:
- Security headers
- Rate limiting
- Request/response logging
- Performance monitoring
- Error handling
- CORS handling
"""

import time
import uuid
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta

from app.logging import get_logger, set_correlation_id, log_performance
from app.config import get_settings
from app.exceptions import RateLimitException, to_http_exception

logger = get_logger(__name__)
settings = get_settings()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    def __init__(self, app, enable_security_headers: bool = True):
        super().__init__(app)
        self.enable_security_headers = enable_security_headers
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        if self.enable_security_headers:
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            
            # Content Security Policy
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
            response.headers["Content-Security-Policy"] = csp
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window algorithm."""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        window_size: int = 60,
        enabled: bool = True
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self.enabled = enabled
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    async def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        async with self.lock:
            now = time.time()
            window_start = now - self.window_size
            
            # Clean old requests
            client_requests = self.request_counts[client_id]
            while client_requests and client_requests[0] < window_start:
                client_requests.popleft()
            
            # Check if limit exceeded
            if len(client_requests) >= self.requests_per_minute:
                return True
            
            # Add current request
            client_requests.append(now)
            return False
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting."""
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        if await self._is_rate_limited(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise RateLimitException(
                message="Rate limit exceeded",
                details={
                    "client_id": client_id,
                    "limit": self.requests_per_minute,
                    "window": self.window_size
                }
            )
        
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and performance monitoring."""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log requests and responses with performance metrics."""
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        
        # Start timing
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            logger.info(
                "Request started",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "headers": dict(request.headers),
                    "client_host": request.client.host if request.client else None
                }
            )
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            logger.error(
                "Request failed with exception",
                extra={
                    "correlation_id": correlation_id,
                    "error": error,
                    "exception_type": type(e).__name__
                }
            )
            # Re-raise to let FastAPI handle it
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Log response
        log_level = "error" if status_code >= 400 else "info"
        getattr(logger, log_level)(
            "Request completed",
            extra={
                "correlation_id": correlation_id,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2),
                "error": error
            }
        )
        
        # Log performance metrics
        log_performance(
            operation=f"{request.method} {request.url.path}",
            duration=duration,
            status_code=status_code,
            correlation_id=correlation_id
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Handle uncaught exceptions."""
        try:
            return await call_next(request)
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
        except Exception as e:
            logger.error(
                "Unhandled exception in request",
                extra={
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "path": request.url.path,
                    "method": request.method
                },
                exc_info=True
            )
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {
                        "timestamp": datetime.now().isoformat(),
                        "path": request.url.path
                    }
                }
            )


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check request size."""
        content_length = request.headers.get("content-length")
        
        if content_length and int(content_length) > self.max_size:
            logger.warning(
                "Request body too large",
                extra={
                    "content_length": content_length,
                    "max_size": self.max_size,
                    "path": request.url.path
                }
            )
            return JSONResponse(
                status_code=413,
                content={
                    "error": "REQUEST_TOO_LARGE",
                    "message": f"Request body too large. Maximum size: {self.max_size} bytes",
                    "details": {
                        "max_size_bytes": self.max_size,
                        "max_size_mb": round(self.max_size / (1024 * 1024), 2)
                    }
                }
            )
        
        return await call_next(request)


def setup_middleware(app):
    """Setup all middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    # Error handling (should be first)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Request size limiting
    app.add_middleware(
        RequestSizeMiddleware,
        max_size=settings.MAX_REQUEST_SIZE
    )
    
    # Rate limiting
    if settings.RATE_LIMIT_ENABLED:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.RATE_LIMIT_REQUESTS,
            window_size=settings.RATE_LIMIT_WINDOW,
            enabled=settings.RATE_LIMIT_ENABLED
        )
    
    # Security headers
    if settings.ENABLE_SECURITY_HEADERS:
        app.add_middleware(
            SecurityHeadersMiddleware,
            enable_security_headers=settings.ENABLE_SECURITY_HEADERS
        )
    
    # Request logging and performance monitoring
    app.add_middleware(
        RequestLoggingMiddleware,
        log_requests=True,
        log_responses=settings.DEBUG
    )
    
    # CORS (should be last)
    if settings.CORS_ENABLED:
        app.add_middleware(
            CORSMiddleware,
            **settings.get_cors_config()
        )
    
    logger.info("Middleware setup completed")


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check optimization."""
    
    def __init__(self, app, cache_duration: int = 10):
        super().__init__(app)
        self.cache_duration = cache_duration
        self._cached_response: Optional[Response] = None
        self._cache_time: Optional[datetime] = None
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Cache health check responses for better performance."""
        # Only cache GET requests to /health/live
        if request.method == "GET" and request.url.path == "/health/live":
            now = datetime.now()
            
            # Return cached response if still valid
            if (
                self._cached_response and 
                self._cache_time and 
                (now - self._cache_time).total_seconds() < self.cache_duration
            ):
                return self._cached_response
            
            # Get fresh response and cache it
            response = await call_next(request)
            self._cached_response = response
            self._cache_time = now
            return response
        
        return await call_next(request)