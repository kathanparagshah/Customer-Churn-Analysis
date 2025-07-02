#!/usr/bin/env python3
"""
Legacy FastAPI application stub for backward compatibility.

This file maintains backward compatibility for existing CI/CD pipelines
while redirecting to the new modular FastAPI application structure.

For the new modular application, use: python -m app.main
"""

import sys
import warnings
from pathlib import Path

# Add the parent directory to the path to import the new app structure
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.warn(
    "deployment/app.py is deprecated. Use 'python -m app.main' or 'uvicorn app.main:app' instead.",
    DeprecationWarning,
    stacklevel=2
)

try:
    # Import the new modular FastAPI application
    from app.main import app
    from app.services.model_manager import (
        ModelManager, get_model_manager, model_manager
    )
    from app.models.schemas import (
        CustomerData, BatchCustomerData, PredictionResponse, BatchPredictionResponse
    )
    
    # Import analytics database for backward compatibility
    from deployment.analytics_db import analytics_db
    
    # Create compatibility variables for legacy code
    # These will be None initially but can be accessed through model_manager
    model = None
    scaler = None
    label_encoders = None
    feature_names = None
    model_loaded = False
    model_metadata = {}
    
    # Update compatibility variables when accessed
    def _update_legacy_vars():
        global model, scaler, label_encoders, feature_names, model_loaded, model_metadata
        model = model_manager.model
        scaler = model_manager.scaler
        label_encoders = model_manager.label_encoders
        feature_names = model_manager.feature_names
        model_loaded = model_manager.is_loaded
        model_metadata = {
            'model_name': model_manager.model_name,
            'version': model_manager.version,
            'training_date': model_manager.training_date,
            'performance_metrics': model_manager.performance_metrics
        }
    
    # Initialize legacy variables
    _update_legacy_vars()
    
    # Re-export for backward compatibility
    __all__ = [
        'app', 'CustomerData', 'BatchCustomerData', 'PredictionResponse',
        'BatchPredictionResponse', 'ModelManager', 'get_model_manager',
        'model_manager', 'model', 'scaler', 'label_encoders', 
        'feature_names', 'model_metadata', 'model_loaded', 'analytics_db'
    ]
    
except ImportError as e:
    print(f"Error importing new modular app: {e}")
    print("Please ensure the new app structure is properly set up.")
    sys.exit(1)

if __name__ == "__main__":
    import uvicorn
    print("Starting legacy compatibility mode...")
    print("Consider migrating to: uvicorn app.main:app")
    uvicorn.run(app, host="0.0.0.0", port=8000)