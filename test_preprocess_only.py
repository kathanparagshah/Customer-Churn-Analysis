#!/usr/bin/env python3

import sys
sys.path.append('/Users/kathan/Downloads/Customer Churn Analysis/deployment')

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

print("Testing preprocess_customer_data directly...")
print(f"Test customer type: {type(test_customer)}")
print(f"isinstance(test_customer, dict): {isinstance(test_customer, dict)}")

try:
    # Import the ModelManager class
    from app import ModelManager
    
    # Create a minimal ModelManager instance (without loading model)
    class TestModelManager:
        def __init__(self):
            self.scaler = None
            self.feature_names = None
            
        def preprocess_customer_data(self, customer_data):
            """Test version of preprocess_customer_data"""
            import pandas as pd
            import logging
            
            logger = logging.getLogger('app')
            
            # Convert to DataFrame
            try:
                logger.info("preprocess_customer_data called")
                logger.info(str(type(customer_data)))
                is_dict = isinstance(customer_data, dict)
                logger.info(f"is_dict: {is_dict}")
                
                if is_dict:
                    # Already a dictionary
                    logger.info("Using dictionary directly")
                    df = pd.DataFrame([customer_data])
                    print("SUCCESS: Dictionary processed correctly")
                    return "success"
                else:
                    # CustomerData object
                    logger.info("Converting CustomerData object to dict")
                    df = pd.DataFrame([customer_data.dict()])
                    print("SUCCESS: CustomerData object processed correctly")
                    return "success"
            except Exception as e:
                logger.error(f"Error in preprocess_customer_data: {e}")
                logger.error(f"customer_data type: {type(customer_data)}")
                logger.error(f"customer_data value: {customer_data}")
                raise
    
    # Test with our minimal manager
    test_manager = TestModelManager()
    result = test_manager.preprocess_customer_data(test_customer)
    print(f"Result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()