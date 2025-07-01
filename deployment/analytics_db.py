import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AnalyticsDatabase:
    """
    SQLite database for storing and retrieving analytics data.
    """
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize the database with required tables.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    customer_data TEXT NOT NULL,
                    churn_probability REAL,
                    churn_prediction INTEGER,
                    risk_level TEXT,
                    confidence REAL,
                    model_version TEXT
                )
            """)
            
            # Create batch_predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    batch_size INTEGER NOT NULL,
                    total_customers INTEGER,
                    predicted_churners INTEGER,
                    churn_rate REAL,
                    avg_churn_probability REAL,
                    high_risk_customers INTEGER,
                    medium_risk_customers INTEGER,
                    low_risk_customers INTEGER
                )
            """)
            
            # Create daily_metrics table for aggregated data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    total_churners INTEGER DEFAULT 0,
                    avg_churn_rate REAL DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    high_risk_count INTEGER DEFAULT 0,
                    medium_risk_count INTEGER DEFAULT 0,
                    low_risk_count INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            logger.info("Analytics database initialized successfully")
    
    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        Log a single prediction to the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO predictions (
                        timestamp, type, customer_data, churn_probability,
                        churn_prediction, risk_level, confidence, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_data['timestamp'],
                    prediction_data['type'],
                    json.dumps(prediction_data['customer_data']),
                    prediction_data['prediction']['churn_probability'],
                    prediction_data['prediction']['churn_prediction'],
                    prediction_data['prediction']['risk_level'],
                    prediction_data['prediction']['confidence'],
                    prediction_data['prediction'].get('version', '1.0.0')
                ))
                
                conn.commit()
                
                # Update daily metrics
                self._update_daily_metrics(prediction_data['timestamp'])
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def log_batch_prediction(self, batch_data: Dict[str, Any]):
        """
        Log a batch prediction to the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO batch_predictions (
                        timestamp, batch_size, total_customers, predicted_churners,
                        churn_rate, avg_churn_probability, high_risk_customers,
                        medium_risk_customers, low_risk_customers
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch_data['timestamp'],
                    batch_data['batch_size'],
                    batch_data['summary']['total_customers'],
                    batch_data['summary']['predicted_churners'],
                    batch_data['summary']['churn_rate'],
                    batch_data['summary']['avg_churn_probability'],
                    batch_data['summary']['high_risk_customers'],
                    batch_data['summary']['medium_risk_customers'],
                    batch_data['summary']['low_risk_customers']
                ))
                
                conn.commit()
                
                # Update daily metrics
                self._update_daily_metrics(batch_data['timestamp'])
                
        except Exception as e:
            logger.error(f"Error logging batch prediction: {e}")
    
    def _update_daily_metrics(self, timestamp: str):
        """
        Update daily aggregated metrics.
        """
        try:
            date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate daily metrics from predictions
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(churn_prediction) as total_churners,
                        AVG(churn_probability) as avg_churn_rate,
                        AVG(confidence) as avg_confidence,
                        SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_risk,
                        SUM(CASE WHEN risk_level = 'Medium' THEN 1 ELSE 0 END) as medium_risk,
                        SUM(CASE WHEN risk_level = 'Low' THEN 1 ELSE 0 END) as low_risk
                    FROM predictions 
                    WHERE DATE(timestamp) = ?
                """, (date,))
                
                result = cursor.fetchone()
                
                if result and result[0] > 0:
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_metrics (
                            date, total_predictions, total_churners, avg_churn_rate,
                            avg_confidence, high_risk_count, medium_risk_count, low_risk_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        date, result[0], result[1] or 0, result[2] or 0,
                        result[3] or 0, result[4] or 0, result[5] or 0, result[6] or 0
                    ))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating daily metrics: {e}")
    
    def get_daily_metrics(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily metrics for the specified number of days.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM daily_metrics 
                    WHERE date >= date('now', '-{} days')
                    ORDER BY date DESC
                """.format(days))
                
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting daily metrics: {e}")
            return []
    
    def get_prediction_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Get prediction trends and statistics.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent predictions
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as predictions,
                        AVG(churn_probability) as avg_probability,
                        SUM(churn_prediction) as churners
                    FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """.format(days))
                
                daily_data = cursor.fetchall()
                
                # Get overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        AVG(churn_probability) as avg_churn_rate,
                        SUM(churn_prediction) as total_churners,
                        AVG(confidence) as avg_confidence
                    FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} days')
                """.format(days))
                
                stats = cursor.fetchone()
                
                return {
                    'daily_data': [
                        {
                            'date': row[0],
                            'predictions': row[1],
                            'avg_probability': row[2],
                            'churners': row[3]
                        } for row in daily_data
                    ],
                    'overall_stats': {
                        'total_predictions': stats[0] or 0,
                        'avg_churn_rate': stats[1] or 0,
                        'total_churners': stats[2] or 0,
                        'avg_confidence': stats[3] or 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting prediction trends: {e}")
            return {'daily_data': [], 'overall_stats': {}}
    
    def get_risk_distribution(self, days: int = 30) -> Dict[str, int]:
        """
        Get risk level distribution for the specified period.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        risk_level,
                        COUNT(*) as count
                    FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY risk_level
                """.format(days))
                
                results = dict(cursor.fetchall())
                
                return {
                    'High': results.get('High', 0),
                    'Medium': results.get('Medium', 0),
                    'Low': results.get('Low', 0)
                }
                
        except Exception as e:
            logger.error(f"Error getting risk distribution: {e}")
            return {'High': 0, 'Medium': 0, 'Low': 0}

# Global analytics database instance
analytics_db = AnalyticsDatabase()