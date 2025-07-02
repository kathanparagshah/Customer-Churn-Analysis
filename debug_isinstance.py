#!/usr/bin/env python3

import requests
import json

# Test data
sample_customer = {
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

print("Testing isinstance with dict...")
test_dict = {"test": "value"}
print(f"isinstance(test_dict, dict): {isinstance(test_dict, dict)}")
print(f"type(test_dict): {type(test_dict)}")

print("\nTesting API call...")
try:
    response = requests.post(
        "http://localhost:8000/predict?threshold=0.5",
        json=sample_customer,
        timeout=10
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")