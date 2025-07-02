#!/usr/bin/env python3

import sys
sys.path.append('/Users/kathan/Downloads/Customer Churn Analysis/deployment')

from app import ModelManager
import json

# Test data
test_customer = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
}

print("Testing direct ModelManager prediction...")
print(f"Test customer type: {type(test_customer)}")
print(f"Test customer: {test_customer}")

try:
    # Create ModelManager instance
    manager = ModelManager()
    
    # Test predict_single directly
    result = manager.predict_single(test_customer, threshold=0.5)
    print(f"Prediction successful: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()