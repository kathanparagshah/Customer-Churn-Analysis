#!/usr/bin/env python3
"""
API Integration Test Script

This script helps test the API integration between frontend and backend.
Run this script to verify your backend is working correctly before deploying.
"""

import requests
import json
import sys
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed backend URL

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            print("✅ Model info endpoint working")
            data = response.json()
            print(f"   Model version: {data.get('model_version', 'Unknown')}")
            return True
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info endpoint error: {e}")
        return False

def test_single_prediction():
    """Test single customer prediction"""
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
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=sample_customer,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Single prediction working")
            data = response.json()
            print(f"   Prediction: {data.get('prediction')}")
            print(f"   Churn probability: {data.get('churn_probability', 0):.3f}")
            print(f"   Risk level: {data.get('risk_level')}")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
        return False

def test_batch_prediction():
    """Test batch customer prediction"""
    sample_customers = [
        {
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
        },
        {
            "CreditScore": 608,
            "Geography": "Spain",
            "Gender": "Female",
            "Age": 41,
            "Tenure": 1,
            "Balance": 83807.86,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 1,
            "EstimatedSalary": 112542.58
        }
    ]
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"customers": sample_customers},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Batch prediction working")
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"   Processed {len(predictions)} customers")
            if predictions:
                print(f"   First prediction: {predictions[0].get('prediction')}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_cors():
    """Test CORS configuration"""
    try:
        # Test preflight request
        response = requests.options(
            f"{API_BASE_URL}/predict",
            headers={
                "Origin": "https://customer-churn-analysis-kgz3.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        if response.status_code in [200, 204]:
            print("✅ CORS configuration working")
            return True
        else:
            print(f"❌ CORS preflight failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ CORS test error: {e}")
        return False

def main():
    """Run all API tests"""
    print(f"Testing API at: {API_BASE_URL}")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_model_info,
        test_single_prediction,
        test_batch_prediction,
        test_cors
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your API is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check your backend configuration.")
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_BASE_URL = sys.argv[1]
    
    exit_code = main()
    sys.exit(exit_code)