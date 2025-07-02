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
    
    # Import model-related components for backward compatibility
    from app.services.model_manager import (
    ModelManager, get_model_manager, is_model_loaded, model_manager,
    model, scaler, label_encoders, feature_names, model_metadata, model_loaded
)
    from app.models.schemas import (
        CustomerData, BatchCustomerData, PredictionResponse, BatchPredictionResponse
    )
    
    # Re-export for backward compatibility
    __all__ = [
        'app', 'CustomerData', 'BatchCustomerData', 'PredictionResponse',
        'BatchPredictionResponse', 'ModelManager', 'get_model_manager',
        'is_model_loaded', 'model_manager', 'model', 'scaler', 'label_encoders', 
        'feature_names', 'model_metadata', 'model_loaded'
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