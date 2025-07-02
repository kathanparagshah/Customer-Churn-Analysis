"""Monitoring and metrics collection for the Customer Churn Analysis API.

This module provides:
- Prometheus metrics collection
- Performance monitoring
- Health metrics
- Business metrics
- Alert management
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, CollectorRegistry,
        generate_latest, CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return MockTimer()
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class CollectorRegistry:
        def __init__(self): pass
    
    class MockTimer:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def generate_latest(*args): return b""
    def start_http_server(*args, **kwargs): pass
    CONTENT_TYPE_LATEST = "text/plain"

from app.logging import get_logger
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.registry = CollectorRegistry() if self.enable_prometheus else None
        self._setup_metrics()
        self._alerts: List[Alert] = []
        self._alert_lock = threading.Lock()
        
        # Performance tracking
        self._request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        logger.info(f"Metrics collector initialized (Prometheus: {self.enable_prometheus})")
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        if not self.enable_prometheus:
            # Create mock metrics
            self.request_count = Counter()
            self.request_duration = Histogram()
            self.prediction_count = Counter()
            self.prediction_duration = Histogram()
            self.model_load_duration = Histogram()
            self.error_count = Counter()
            self.active_requests = Gauge()
            self.system_memory_usage = Gauge()
            self.system_cpu_usage = Gauge()
            self.system_disk_usage = Gauge()
            self.model_info = Info()
            self.prediction_accuracy = Gauge()
            self.batch_size = Histogram()
            return
        
        # HTTP Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_requests = Gauge(
            'http_requests_active',
            'Number of active HTTP requests',
            registry=self.registry
        )
        
        # Prediction metrics
        self.prediction_count = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['prediction_type', 'model_version'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'prediction_duration_seconds',
            'Time spent making predictions',
            ['prediction_type'],
            registry=self.registry
        )
        
        self.batch_size = Histogram(
            'prediction_batch_size',
            'Size of prediction batches',
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )
        
        # Model metrics
        self.model_load_duration = Histogram(
            'model_load_duration_seconds',
            'Time spent loading models',
            registry=self.registry
        )
        
        self.model_info = Info(
            'model_info',
            'Information about the loaded model',
            registry=self.registry
        )
        
        self.prediction_accuracy = Gauge(
            'model_prediction_accuracy',
            'Model prediction accuracy',
            ['model_version'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
        
        # System metrics
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        if self.enable_prometheus:
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        
        # Track request times for performance analysis
        key = f"{method}:{endpoint}"
        self._request_times[key].append(duration)
        
        # Track errors
        if status_code >= 400:
            self._error_counts[key] += 1
    
    def record_prediction(self, prediction_type: str, duration: float, 
                         batch_size: int = 1, model_version: str = "unknown"):
        """Record prediction metrics."""
        if self.enable_prometheus:
            self.prediction_count.labels(
                prediction_type=prediction_type,
                model_version=model_version
            ).inc()
            
            self.prediction_duration.labels(
                prediction_type=prediction_type
            ).observe(duration)
            
            self.batch_size.observe(batch_size)
    
    def record_model_load(self, duration: float, model_info: Dict[str, Any]):
        """Record model loading metrics."""
        if self.enable_prometheus:
            self.model_load_duration.observe(duration)
            
            # Update model info
            self.model_info.info(model_info)
    
    def record_error(self, error_type: str, endpoint: str):
        """Record error metrics."""
        if self.enable_prometheus:
            self.error_count.labels(
                error_type=error_type,
                endpoint=endpoint
            ).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            if self.enable_prometheus:
                self.system_memory_usage.set(memory.percent)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if self.enable_prometheus:
                self.system_cpu_usage.set(cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if self.enable_prometheus:
                self.system_disk_usage.set(disk_percent)
            
            # Check for alerts
            self._check_system_alerts(memory.percent, cpu_percent, disk_percent)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _check_system_alerts(self, memory_percent: float, cpu_percent: float, disk_percent: float):
        """Check system metrics for alert conditions."""
        # Memory alerts
        if memory_percent > settings.MEMORY_THRESHOLD:
            self.create_alert(
                AlertLevel.WARNING if memory_percent < 95 else AlertLevel.CRITICAL,
                "High Memory Usage",
                f"Memory usage is {memory_percent:.1f}%",
                {"metric": "memory", "value": str(memory_percent)}
            )
        
        # CPU alerts
        if cpu_percent > 90:
            self.create_alert(
                AlertLevel.WARNING if cpu_percent < 95 else AlertLevel.CRITICAL,
                "High CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                {"metric": "cpu", "value": str(cpu_percent)}
            )
        
        # Disk alerts
        if disk_percent > 90:
            self.create_alert(
                AlertLevel.WARNING if disk_percent < 95 else AlertLevel.CRITICAL,
                "High Disk Usage",
                f"Disk usage is {disk_percent:.1f}%",
                {"metric": "disk", "value": str(disk_percent)}
            )
    
    def create_alert(self, level: AlertLevel, title: str, message: str, tags: Dict[str, str] = None):
        """Create a new alert."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            tags=tags or {}
        )
        
        with self._alert_lock:
            # Check if similar alert already exists
            existing = next(
                (a for a in self._alerts 
                 if not a.resolved and a.title == title and a.level == level),
                None
            )
            
            if not existing:
                self._alerts.append(alert)
                logger.warning(f"Alert created: {title} - {message}", extra={"alert": alert})
    
    def resolve_alert(self, title: str, level: AlertLevel = None):
        """Resolve an alert."""
        with self._alert_lock:
            for alert in self._alerts:
                if (alert.title == title and not alert.resolved and 
                    (level is None or alert.level == level)):
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Alert resolved: {title}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._alert_lock:
            return [alert for alert in self._alerts if not alert.resolved]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            "total_endpoints": len(self._request_times),
            "total_errors": sum(self._error_counts.values()),
            "endpoints": {}
        }
        
        for endpoint, times in self._request_times.items():
            if times:
                summary["endpoints"][endpoint] = {
                    "request_count": len(times),
                    "avg_duration_ms": round(sum(times) / len(times) * 1000, 2),
                    "min_duration_ms": round(min(times) * 1000, 2),
                    "max_duration_ms": round(max(times) * 1000, 2),
                    "error_count": self._error_counts.get(endpoint, 0)
                }
        
        return summary
    
    def get_metrics_data(self) -> bytes:
        """Get Prometheus metrics data."""
        if self.enable_prometheus:
            return generate_latest(self.registry)
        return b"# Prometheus not available\n"
    
    def start_metrics_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        if self.enable_prometheus:
            try:
                start_http_server(port, registry=self.registry)
                logger.info(f"Metrics server started on port {port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
        else:
            logger.warning("Prometheus not available, metrics server not started")


class PerformanceMonitor:
    """Performance monitoring and alerting."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: int = 30):
        """Start performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self.metrics.update_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)


# Global metrics collector instance
metrics_collector = MetricsCollector(enable_prometheus=settings.ENABLE_METRICS)
performance_monitor = PerformanceMonitor(metrics_collector)


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return performance_monitor


def setup_monitoring():
    """Setup monitoring system."""
    # Start metrics server if enabled
    if settings.ENABLE_METRICS and PROMETHEUS_AVAILABLE:
        metrics_collector.start_metrics_server(settings.METRICS_PORT)
    
    # Start performance monitoring
    performance_monitor.start_monitoring(settings.HEALTH_CHECK_INTERVAL)
    
    logger.info("Monitoring system setup completed")


def shutdown_monitoring():
    """Shutdown monitoring system."""
    performance_monitor.stop_monitoring()
    logger.info("Monitoring system shutdown completed")