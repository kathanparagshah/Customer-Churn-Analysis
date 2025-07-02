#!/usr/bin/env python3
"""
Test script for batch churn prediction API endpoint
"""

import requests
import json

def test_batch_prediction():
    """Test the batch prediction endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/predict/batch"
    
    # Sample batch data
    batch_data = {
        "customers": [
            {
                "CreditScore": 600,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 50000.0
            },
            {
                "CreditScore": 850,
                "Geography": "France",
                "Gender": "Female",
                "Age": 25,
                "Tenure": 8,
                "Balance": 120000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 80000.0
            },
            {
                "CreditScore": 700,
                "Geography": "Spain",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 5,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 1,
                "EstimatedSalary": 60000.0
            }
        ]
    }
    
    print("🧪 Testing batch prediction...")
    
    try:
        # Make the request
        response = requests.post(url, json=batch_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            print(f"Number of predictions: {len(result['predictions'])}")
            
            for i, prediction in enumerate(result['predictions']):
                print(f"\nCustomer {i+1}:")
                print(f"  Churn Probability: {prediction['churn_probability']:.3f}")
                print(f"  Churn Prediction: {prediction['churn_prediction']}")
                print(f"  Risk Level: {prediction['risk_level']}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    test_batch_prediction()