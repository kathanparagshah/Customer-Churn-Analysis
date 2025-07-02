#!/usr/bin/env python3

import requests
import json

# Test data for a high-risk customer
test_customer = {
    "CreditScore": 600,
    "Geography": "Germany", 
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 50000
}

try:
    print("🧪 Testing single prediction...")
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_customer,
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"Churn Probability: {result['churn_probability']:.3f}")
        print(f"Churn Prediction: {result['churn_prediction']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API. Make sure it's running on localhost:8000")
except Exception as e:
    print(f"❌ Error: {str(e)}")