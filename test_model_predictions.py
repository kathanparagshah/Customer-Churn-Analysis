#!/usr/bin/env python3
"""
Test Model Predictions

This script tests the churn prediction model to verify it correctly
identifies customers who will churn vs those who will stay.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_model_and_data():
    """Load the trained model and test data."""
    # Load the trained model
    model_path = Path('deployment/models/churn_model.pkl')
    if not model_path.exists():
        model_path = Path('models/churn_model.pkl')
    
    if not model_path.exists():
        print("❌ Model file not found. Please train the model first.")
        return None, None, None, None
    
    print(f"📦 Loading model from {model_path}")
    model_data = joblib.load(model_path)
    
    # Extract components
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    
    # Load test data
    data_path = Path('data/processed/churn_features_selected.parquet')
    if not data_path.exists():
        data_path = Path('data/processed/churn_cleaned.parquet')
    
    if not data_path.exists():
        print("❌ Test data not found.")
        return None, None, None, None
    
    print(f"📊 Loading test data from {data_path}")
    df = pd.read_parquet(data_path)
    
    return model, scaler, label_encoders, feature_names, df

def preprocess_test_data(df, scaler, label_encoders, feature_names):
    """Preprocess test data using the same transformations as training."""
    # Separate features and target
    target_col = 'Exited'
    feature_cols = [col for col in df.columns if col not in [target_col, 'CustomerId', 'Surname', 'RowNumber']]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col in label_encoders:
            # Handle unseen categories
            unique_values = set(X[col].astype(str))
            known_values = set(label_encoders[col].classes_)
            unseen_values = unique_values - known_values
            
            if unseen_values:
                print(f"⚠️  Found unseen values in {col}: {unseen_values}")
                # Replace unseen values with the most frequent class
                most_frequent = label_encoders[col].classes_[0]
                X[col] = X[col].astype(str).replace(list(unseen_values), most_frequent)
            
            X[col] = label_encoders[col].transform(X[col].astype(str))
    
    # Ensure we have the same features as training
    missing_features = set(feature_names) - set(X.columns)
    extra_features = set(X.columns) - set(feature_names)
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        for feature in missing_features:
            X[feature] = 0
    
    if extra_features:
        print(f"⚠️  Extra features (will be dropped): {extra_features}")
        X = X.drop(columns=list(extra_features))
    
    # Reorder columns to match training
    X = X[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def test_model_predictions():
    """Test the model predictions and analyze performance."""
    print("🔍 Testing Churn Prediction Model Performance")
    print("=" * 50)
    
    # Load model and data
    result = load_model_and_data()
    if result[0] is None:
        return
    
    model, scaler, label_encoders, feature_names, df = result
    
    # Preprocess data
    X_test, y_test = preprocess_test_data(df, scaler, label_encoders, feature_names)
    
    print(f"\n📊 Test Data Summary:")
    print(f"Total samples: {len(y_test)}")
    print(f"Churners (1): {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"Non-churners (0): {len(y_test) - sum(y_test)} ({(len(y_test) - sum(y_test))/len(y_test)*100:.1f}%)")
    
    # Make predictions
    print("\n🔮 Making Predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📈 Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1-Score: {f1:.3f} ({f1*100:.1f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                Stay  Churn")
    print(f"Actual Stay    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Churn   {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Detailed analysis
    print(f"\n🔍 Detailed Analysis:")
    
    # True Positives (Correctly predicted churners)
    tp_mask = (y_test == 1) & (y_pred == 1)
    tp_count = sum(tp_mask)
    
    # True Negatives (Correctly predicted non-churners)
    tn_mask = (y_test == 0) & (y_pred == 0)
    tn_count = sum(tn_mask)
    
    # False Positives (Incorrectly predicted as churners)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_count = sum(fp_mask)
    
    # False Negatives (Missed churners)
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_count = sum(fn_mask)
    
    print(f"✅ Correctly identified churners: {tp_count} out of {sum(y_test)} ({tp_count/max(sum(y_test), 1)*100:.1f}%)")
    print(f"✅ Correctly identified non-churners: {tn_count} out of {len(y_test) - sum(y_test)} ({tn_count/max(len(y_test) - sum(y_test), 1)*100:.1f}%)")
    print(f"❌ False alarms (predicted churn but stayed): {fp_count}")
    print(f"❌ Missed churners (predicted stay but churned): {fn_count}")
    
    # Probability distribution analysis
    print(f"\n📊 Prediction Probability Analysis:")
    
    # For actual churners
    churner_probs = y_pred_proba[y_test == 1]
    if len(churner_probs) > 0:
        print(f"Actual churners - Avg probability: {churner_probs.mean():.3f}, Min: {churner_probs.min():.3f}, Max: {churner_probs.max():.3f}")
    
    # For actual non-churners
    non_churner_probs = y_pred_proba[y_test == 0]
    if len(non_churner_probs) > 0:
        print(f"Actual non-churners - Avg probability: {non_churner_probs.mean():.3f}, Min: {non_churner_probs.min():.3f}, Max: {non_churner_probs.max():.3f}")
    
    # Sample predictions
    print(f"\n🎯 Sample Predictions:")
    sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
    
    for i, idx in enumerate(sample_indices[:5]):
        actual = "Churn" if y_test.iloc[idx] == 1 else "Stay"
        predicted = "Churn" if y_pred[idx] == 1 else "Stay"
        probability = y_pred_proba[idx]
        status = "✅" if y_test.iloc[idx] == y_pred[idx] else "❌"
        
        print(f"{status} Sample {i+1}: Actual={actual}, Predicted={predicted}, Probability={probability:.3f}")
    
    # Overall assessment
    print(f"\n🎯 Model Assessment:")
    
    if accuracy >= 0.8:
        print("🟢 EXCELLENT: Model shows high accuracy in predictions")
    elif accuracy >= 0.7:
        print("🟡 GOOD: Model shows decent accuracy but could be improved")
    elif accuracy >= 0.6:
        print("🟠 FAIR: Model shows moderate accuracy, needs improvement")
    else:
        print("🔴 POOR: Model accuracy is low, significant improvement needed")
    
    if recall >= 0.7:
        print("🟢 EXCELLENT: Model catches most churners")
    elif recall >= 0.5:
        print("🟡 GOOD: Model catches a reasonable number of churners")
    else:
        print("🔴 POOR: Model misses too many churners")
    
    if precision >= 0.7:
        print("🟢 EXCELLENT: Model has low false alarm rate")
    elif precision >= 0.5:
        print("🟡 GOOD: Model has moderate false alarm rate")
    else:
        print("🔴 POOR: Model has high false alarm rate")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_test),
        'churners': int(sum(y_test)),
        'non_churners': int(len(y_test) - sum(y_test))
    }

if __name__ == "__main__":
    results = test_model_predictions()
    
    if results:
        # Save results
        results_path = Path('reports/model_prediction_test.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {results_path}")