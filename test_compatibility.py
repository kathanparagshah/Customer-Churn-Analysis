#!/usr/bin/env python3
"""
Test script to verify XGBoost and scikit-learn compatibility.
This script tests the key components that might fail in CI.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__} imported successfully")
        
        import xgboost as xgb
        print(f"✓ XGBoost {xgb.__version__} imported successfully")
        
        from models.train_churn import XGBClassifierWrapper
        print("✓ XGBClassifierWrapper imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_xgb_wrapper():
    """Test XGBClassifierWrapper functionality."""
    print("\nTesting XGBClassifierWrapper...")
    
    try:
        from models.train_churn import XGBClassifierWrapper
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Create wrapper instance
        wrapper = XGBClassifierWrapper(random_state=42, n_estimators=10)
        print("✓ XGBClassifierWrapper instantiated")
        
        # Test sklearn_tags method
        if hasattr(wrapper, '__sklearn_tags__'):
            tags = wrapper.__sklearn_tags__()
            print(f"✓ __sklearn_tags__ method works: {type(tags)}")
        else:
            print("✗ __sklearn_tags__ method missing")
            return False
        
        # Test basic fit/predict
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        wrapper.fit(X, y)
        predictions = wrapper.predict(X)
        print(f"✓ Fit and predict work: {len(predictions)} predictions")
        
        return True
    except Exception as e:
        print(f"✗ XGBClassifierWrapper test failed: {e}")
        return False

def test_sklearn_compatibility():
    """Test scikit-learn compatibility features."""
    print("\nTesting scikit-learn compatibility...")
    
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
        from models.train_churn import XGBClassifierWrapper
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        wrapper = XGBClassifierWrapper(random_state=42, n_estimators=10)
        
        # Test cross-validation (this uses sklearn_tags internally)
        scores = cross_val_score(wrapper, X, y, cv=3)
        print(f"✓ Cross-validation works: mean score = {scores.mean():.3f}")
        
        return True
    except Exception as e:
        print(f"✗ sklearn compatibility test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("=== XGBoost/scikit-learn Compatibility Test ===")
    
    tests = [
        test_imports,
        test_xgb_wrapper,
        test_sklearn_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n=== Results ===")
    if all(results):
        print("✓ All compatibility tests passed!")
        return 0
    else:
        print("✗ Some compatibility tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())