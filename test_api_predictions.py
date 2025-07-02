#!/usr/bin/env python3
"""
Test API Predictions

This script tests the deployed API to verify it correctly
identifies customers who will churn vs those who will stay.
"""

import requests
import json
import pandas as pd
from pathlib import Path

def test_api_predictions():
    """Test the API predictions with sample data."""
    print("🌐 Testing Churn Prediction API")
    print("=" * 40)
    
    # API endpoint
    api_url = "http://localhost:8000"
    
    # Test if API is running
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API is not running or not healthy")
            return
        print("✅ API is running and healthy")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API. Make sure it's running on localhost:8000")
        return
    
    # Test cases: customers likely to churn vs likely to stay
    test_customers = [
        {
            "name": "High Risk Churner",
            "data": {
                "CreditScore": 400,  # Low credit score
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 55,  # Older age
                "Tenure": 1,  # Short tenure
                "Balance": 0,  # No balance
                "NumOfProducts": 1,  # Few products
                "HasCrCard": 0,  # No credit card
                "IsActiveMember": 0,  # Inactive
                "EstimatedSalary": 30000  # Low salary
            },
            "expected": "High churn probability"
        },
        {
            "name": "Low Risk Customer",
            "data": {
                "CreditScore": 800,  # High credit score
                "Geography": "France",
                "Gender": "Male",
                "Age": 35,  # Middle age
                "Tenure": 8,  # Long tenure
                "Balance": 100000,  # High balance
                "NumOfProducts": 3,  # Multiple products
                "HasCrCard": 1,  # Has credit card
                "IsActiveMember": 1,  # Active
                "EstimatedSalary": 80000  # High salary
            },
            "expected": "Low churn probability"
        },
        {
            "name": "Medium Risk Customer",
            "data": {
                "CreditScore": 650,  # Average credit score
                "Geography": "Spain",
                "Gender": "Female",
                "Age": 42,  # Middle age
                "Tenure": 5,  # Medium tenure
                "Balance": 50000,  # Medium balance
                "NumOfProducts": 2,  # Average products
                "HasCrCard": 1,  # Has credit card
                "IsActiveMember": 0,  # Inactive
                "EstimatedSalary": 60000  # Medium salary
            },
            "expected": "Medium churn probability"
        },
        {
            "name": "Extreme Churner Profile",
            "data": {
                "CreditScore": 350,  # Very low credit score
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 65,  # Very old
                "Tenure": 0,  # No tenure
                "Balance": 0,  # No balance
                "NumOfProducts": 1,  # Minimum products
                "HasCrCard": 0,  # No credit card
                "IsActiveMember": 0,  # Inactive
                "EstimatedSalary": 15000  # Very low salary
            },
            "expected": "Very high churn probability"
        },
        {
            "name": "Loyal Customer Profile",
            "data": {
                "CreditScore": 850,  # Excellent credit score
                "Geography": "France",
                "Gender": "Female",
                "Age": 30,  # Young
                "Tenure": 10,  # Very long tenure
                "Balance": 150000,  # Very high balance
                "NumOfProducts": 4,  # Many products
                "HasCrCard": 1,  # Has credit card
                "IsActiveMember": 1,  # Very active
                "EstimatedSalary": 120000  # Very high salary
            },
            "expected": "Very low churn probability"
        }
    ]
    
    print(f"\n🧪 Testing {len(test_customers)} customer profiles...\n")
    
    results = []
    
    for i, customer in enumerate(test_customers, 1):
        print(f"👤 Test {i}: {customer['name']}")
        print(f"Expected: {customer['expected']}")
        
        try:
            # Make prediction request
            response = requests.post(
                f"{api_url}/predict",
                json=customer['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                churn_prob = result.get('churn_probability', 0)
                churn_pred = result.get('churn_prediction', 0)
                risk_level = result.get('risk_level', 'Unknown')
                confidence = result.get('confidence', 0)
                
                print(f"📊 Results:")
                print(f"   Churn Probability: {churn_prob:.3f} ({churn_prob*100:.1f}%)")
                print(f"   Churn Prediction: {'Will Churn' if churn_pred == 1 else 'Will Stay'}")
                print(f"   Risk Level: {risk_level}")
                print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                
                # Assess if prediction makes sense
                if "High" in customer['expected'] or "Very high" in customer['expected']:
                    if churn_prob >= 0.6:
                        print("   ✅ CORRECT: High churn probability as expected")
                        assessment = "Correct"
                    else:
                        print("   ❌ INCORRECT: Expected high churn probability")
                        assessment = "Incorrect"
                elif "Low" in customer['expected'] or "Very low" in customer['expected']:
                    if churn_prob <= 0.4:
                        print("   ✅ CORRECT: Low churn probability as expected")
                        assessment = "Correct"
                    else:
                        print("   ❌ INCORRECT: Expected low churn probability")
                        assessment = "Incorrect"
                else:  # Medium
                    if 0.3 <= churn_prob <= 0.7:
                        print("   ✅ CORRECT: Medium churn probability as expected")
                        assessment = "Correct"
                    else:
                        print("   ❌ INCORRECT: Expected medium churn probability")
                        assessment = "Incorrect"
                
                results.append({
                    'customer': customer['name'],
                    'expected': customer['expected'],
                    'churn_probability': churn_prob,
                    'churn_prediction': churn_pred,
                    'risk_level': risk_level,
                    'confidence': confidence,
                    'assessment': assessment
                })
                
            else:
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                results.append({
                    'customer': customer['name'],
                    'expected': customer['expected'],
                    'error': f"API Error: {response.status_code}",
                    'assessment': "Error"
                })
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request Error: {e}")
            results.append({
                'customer': customer['name'],
                'expected': customer['expected'],
                'error': str(e),
                'assessment': "Error"
            })
        
        print()  # Empty line for readability
    
    # Summary
    print("📈 Test Summary:")
    print("=" * 20)
    
    correct_predictions = sum(1 for r in results if r.get('assessment') == 'Correct')
    total_predictions = len([r for r in results if 'error' not in r])
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"Correct Predictions: {correct_predictions}/{total_predictions} ({accuracy*100:.1f}%)")
        
        if accuracy >= 0.8:
            print("🟢 EXCELLENT: API predictions are highly accurate")
        elif accuracy >= 0.6:
            print("🟡 GOOD: API predictions are reasonably accurate")
        else:
            print("🔴 POOR: API predictions need improvement")
    
    # Test batch prediction
    print("\n🔄 Testing Batch Prediction...")
    
    batch_data = {
        "customers": [customer['data'] for customer in test_customers[:3]]
    }
    
    try:
        batch_response = requests.post(
            f"{api_url}/predict/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        if batch_response.status_code == 200:
            batch_result = batch_response.json()
            print(f"✅ Batch prediction successful")
            print(f"   Processed: {batch_result.get('summary', {}).get('total_customers', 0)} customers")
            
            predictions = batch_result.get('predictions', [])
            for i, pred in enumerate(predictions):
                prob = pred.get('churn_probability', 0)
                risk = pred.get('risk_level', 'Unknown')
                print(f"   Customer {i+1}: {prob:.3f} probability, {risk} risk")
        else:
            print(f"❌ Batch prediction failed: {batch_response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction error: {e}")
    
    # Save results
    results_path = Path('reports/api_prediction_test.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    
    return results

if __name__ == "__main__":
    test_api_predictions()