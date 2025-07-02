#!/usr/bin/env python3

import requests
import json

def test_api_with_debug():
    """Test API and check what's happening with type checking"""
    
    # Sample customer data
    customer_data = {
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
    
    print(f"Sending data type: {type(customer_data)}")
    print(f"Data: {customer_data}")
    
    try:
        response = requests.post(
            "http://localhost:8000/predict?threshold=0.5",
            json=customer_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_with_debug()