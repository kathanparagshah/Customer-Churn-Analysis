#!/usr/bin/env python3
"""
Test script to compare prediction results with different thresholds.
Demonstrates how higher thresholds reduce false alarms.
"""

import requests
import json

def test_threshold_comparison():
    """Test the same customer data with different thresholds"""
    
    # API endpoint
    base_url = "http://localhost:8000"
    
    # Sample customer data (likely to be a borderline case)
    customer_data = {
        "CreditScore": 650,
        "Geography": "France", 
        "Gender": "Female",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0
    }
    
    # Test different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("🔍 Testing Threshold Impact on False Alarms")
    print("=" * 60)
    print(f"Customer Profile: {customer_data['Age']} year old {customer_data['Gender']} from {customer_data['Geography']}")
    print(f"Credit Score: {customer_data['CreditScore']}, Balance: ${customer_data['Balance']:,.2f}")
    print("\n" + "-" * 60)
    
    results = []
    
    for threshold in thresholds:
        try:
            # Make prediction with specific threshold
            response = requests.post(
                f"{base_url}/predict?threshold={threshold}",
                json=customer_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'threshold': threshold,
                    'churn_probability': result['churn_probability'],
                    'churn_prediction': result['churn_prediction'],
                    'risk_level': result['risk_level'],
                    'confidence': result['confidence']
                })
                
                # Format output
                prediction_text = "🚨 WILL CHURN" if result['churn_prediction'] else "✅ WILL STAY"
                print(f"Threshold {threshold:3.1f}: {prediction_text} | "
                      f"Probability: {result['churn_probability']:.3f} | "
                      f"Risk: {result['risk_level']:6s} | "
                      f"Confidence: {result['confidence']:.3f}")
            else:
                print(f"❌ Error with threshold {threshold}: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error with threshold {threshold}: {e}")
            return
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 ANALYSIS:")
    
    if results:
        prob = results[0]['churn_probability']  # Same for all thresholds
        print(f"• Churn Probability: {prob:.3f} (constant across all thresholds)")
        
        # Count predictions at each threshold
        churn_predictions = sum(1 for r in results if r['churn_prediction'])
        stay_predictions = len(results) - churn_predictions
        
        print(f"• Predictions across thresholds:")
        print(f"  - Will Churn: {churn_predictions}/{len(results)} thresholds")
        print(f"  - Will Stay:  {stay_predictions}/{len(results)} thresholds")
        
        # Find threshold where prediction changes
        for i, result in enumerate(results):
            if i > 0 and result['churn_prediction'] != results[i-1]['churn_prediction']:
                print(f"  - Prediction changes between {results[i-1]['threshold']} and {result['threshold']}")
        
        print(f"\n💡 INSIGHT:")
        if prob < 0.7:
            print(f"   With default threshold (0.7), this customer is predicted to STAY")
            print(f"   This reduces false alarms compared to lower thresholds")
        else:
            print(f"   This customer has high churn probability and will be flagged at most thresholds")
            
    print("\n🎯 RECOMMENDATION:")
    print("   • Use threshold 0.7-0.8 to minimize false alarms")
    print("   • Higher thresholds = fewer false positives but might miss some real churners")
    print("   • Lower thresholds = catch more churners but more false alarms")

if __name__ == "__main__":
    test_threshold_comparison()