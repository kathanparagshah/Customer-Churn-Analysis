"""Analytics service for logging predictions and metrics to database."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..config import settings
from ..logging import get_logger

logger = get_logger("analytics")


class AnalyticsDB:
    """Analytics database manager for storing prediction logs and metrics."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize analytics database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path or settings.ANALYTICS_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        customer_id TEXT,
                        churn_probability REAL NOT NULL,
                        churn_prediction INTEGER NOT NULL,
                        risk_level TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        threshold_used REAL NOT NULL,
                        model_version TEXT,
                        prediction_type TEXT DEFAULT 'single',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create batch_predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS batch_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        batch_size INTEGER NOT NULL,
                        model_version TEXT,
                        processing_time_ms REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        tags TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info(f"Analytics database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize analytics database: {e}")
            raise
    
    def log_prediction(self, prediction_data: Dict[str, Any], customer_id: Optional[str] = None) -> None:
        """Log a single prediction to the database.
        
        Args:
            prediction_data: Prediction results dictionary
            customer_id: Optional customer identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO predictions (
                        timestamp, customer_id, churn_probability, churn_prediction,
                        risk_level, confidence, threshold_used, model_version, prediction_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    customer_id,
                    prediction_data.get('churn_probability'),
                    int(prediction_data.get('churn_prediction', False)),
                    prediction_data.get('risk_level'),
                    prediction_data.get('confidence'),
                    prediction_data.get('threshold_used'),
                    prediction_data.get('model_version'),
                    'single'
                ))
                
                conn.commit()
                logger.debug(f"Logged prediction for customer {customer_id}")
                
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def log_batch_prediction(self, batch_id: str, batch_size: int, 
                           model_version: str, processing_time_ms: float) -> None:
        """Log a batch prediction to the database.
        
        Args:
            batch_id: Unique identifier for the batch
            batch_size: Number of predictions in the batch
            model_version: Model version used
            processing_time_ms: Processing time in milliseconds
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO batch_predictions (
                        batch_id, timestamp, batch_size, model_version, processing_time_ms
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    batch_id,
                    datetime.now().isoformat(),
                    batch_size,
                    model_version,
                    processing_time_ms
                ))
                
                conn.commit()
                logger.debug(f"Logged batch prediction {batch_id} with {batch_size} predictions")
                
        except Exception as e:
            logger.error(f"Failed to log batch prediction: {e}")
    
    def log_metric(self, metric_name: str, metric_value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Log a custom metric to the database.
        
        Args:
            metric_name: Name of the metric
            metric_value: Metric value
            tags: Optional tags as key-value pairs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                tags_json = None
                if tags:
                    import json
                    tags_json = json.dumps(tags)
                
                cursor.execute("""
                    INSERT INTO metrics (metric_name, metric_value, timestamp, tags)
                    VALUES (?, ?, ?, ?)
                """, (
                    metric_name,
                    metric_value,
                    datetime.now().isoformat(),
                    tags_json
                ))
                
                conn.commit()
                logger.debug(f"Logged metric {metric_name}: {metric_value}")
                
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
    
    def get_daily_metrics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily prediction metrics.
        
        Args:
            date: Date in YYYY-MM-DD format, defaults to today
            
        Returns:
            Dict containing daily metrics
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get prediction counts
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) as churn_predictions,
                        AVG(churn_probability) as avg_churn_probability,
                        AVG(confidence) as avg_confidence
                    FROM predictions 
                    WHERE DATE(timestamp) = ?
                """, (date,))
                
                result = cursor.fetchone()
                
                return {
                    'date': date,
                    'total_predictions': result[0] or 0,
                    'churn_predictions': result[1] or 0,
                    'avg_churn_probability': result[2] or 0.0,
                    'avg_confidence': result[3] or 0.0,
                    'churn_rate': (result[1] or 0) / max(result[0] or 1, 1)
                }
                
        except Exception as e:
            logger.error(f"Failed to get daily metrics: {e}")
            return {
                'date': date,
                'total_predictions': 0,
                'churn_predictions': 0,
                'avg_churn_probability': 0.0,
                'avg_confidence': 0.0,
                'churn_rate': 0.0
            }
    
    def get_prediction_trends(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get prediction trends over the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of daily metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) as churn_predictions,
                        AVG(churn_probability) as avg_churn_probability
                    FROM predictions 
                    WHERE DATE(timestamp) >= DATE('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """.format(days))
                
                results = cursor.fetchall()
                
                return [
                    {
                        'date': row[0],
                        'total_predictions': row[1],
                        'churn_predictions': row[2],
                        'avg_churn_probability': row[3],
                        'churn_rate': row[2] / max(row[1], 1)
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Failed to get prediction trends: {e}")
            return []


# Global analytics database instance
analytics_db = AnalyticsDB()


def get_analytics_db() -> AnalyticsDB:
    """Dependency to get analytics database instance.
    
    Returns:
        AnalyticsDB: The global analytics database instance
    """
    return analytics_db


# Convenience functions for backward compatibility
def log_prediction(prediction_data: Dict[str, Any], customer_id: Optional[str] = None) -> None:
    """Log a single prediction (convenience function)."""
    analytics_db.log_prediction(prediction_data, customer_id)


def log_batch_prediction(batch_id: str, batch_size: int, 
                        model_version: str, processing_time_ms: float) -> None:
    """Log a batch prediction (convenience function)."""
    analytics_db.log_batch_prediction(batch_id, batch_size, model_version, processing_time_ms)


def get_daily_metrics(date: Optional[str] = None) -> Dict[str, Any]:
    """Get daily metrics (convenience function)."""
    return analytics_db.get_daily_metrics(date)


def get_prediction_trends(days: int = 7) -> List[Dict[str, Any]]:
    """Get prediction trends (convenience function)."""
    return analytics_db.get_prediction_trends(days)