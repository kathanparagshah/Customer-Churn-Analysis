#!/usr/bin/env python3

import sys
sys.path.append('deployment')

from app import ModelManager, CustomerData
import pandas as pd

# Create ModelManager instance
manager = ModelManager()

print("ModelManager initialized successfully")
print(f"Model loaded: {manager.model is not None}")
print(f"Scaler loaded: {manager.scaler is not None}")
print(f"Feature names: {manager.feature_names}")

if manager.scaler:
    print(f"Scaler feature names: {manager.scaler.feature_names_in_}")

# Test customer data
customer_data = CustomerData(
    CreditScore=600,
    Geography="Germany", 
    Gender="Male",
    Age=40,
    Tenure=3,
    Balance=60000,
    NumOfProducts=2,
    HasCrCard=1,
    IsActiveMember=0,
    EstimatedSalary=50000
)

print("\nTesting preprocessing...")
try:
    features = manager.preprocess_customer_data(customer_data)
    print("✅ Preprocessing successful!")
    print(f"Features shape: {features.shape}")
    print(f"Features: {features[0]}")
    
    # Test prediction
    print("\nTesting prediction...")
    prediction_result = manager.predict_single(customer_data)
    print("✅ Prediction successful!")
    print(f"Prediction result: {prediction_result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()