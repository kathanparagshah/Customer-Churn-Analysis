#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Load the model data
model_data = joblib.load('deployment/models/churn_model.pkl')
scaler = model_data['scaler']
feature_names = model_data['feature_names']

print("Expected feature names:", feature_names)
print("Scaler feature names:", scaler.feature_names_in_)

# Test customer data
customer_data = CustomerData(
    CreditScore=600,
    Geography="Germany", 
    Gender="Male",
    Age=40,
    Tenure=3,
    Balance=60000,
    NumOfProducts=2,
    HasCrCard=1,
    IsActiveMember=0,
    EstimatedSalary=50000
)

# Replicate the preprocessing logic
df = pd.DataFrame([customer_data.dict()])

# Create one-hot encoded categorical features
df['Geography_Germany'] = 1 if df['Geography'].iloc[0] == 'Germany' else 0
df['Geography_Spain'] = 1 if df['Geography'].iloc[0] == 'Spain' else 0
df['Gender_Male'] = 1 if df['Gender'].iloc[0] == 'Male' else 0

# Create feature dict
feature_dict = {}

# Add numeric features with '_scaled' suffix
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
for feature in numeric_features:
    feature_dict[f'{feature}_scaled'] = df[feature].iloc[0]

# Add categorical features
feature_dict['Geography_Germany'] = df['Geography_Germany'].iloc[0]
feature_dict['Geography_Spain'] = df['Geography_Spain'].iloc[0]
feature_dict['Gender_Male'] = df['Gender_Male'].iloc[0]

# Add binary features
feature_dict['HasCrCard'] = df['HasCrCard'].iloc[0]
feature_dict['IsActiveMember'] = df['IsActiveMember'].iloc[0]

# Create DataFrame
feature_df = pd.DataFrame([feature_dict])
print("\nFeature DataFrame columns:", list(feature_df.columns))
print("Feature DataFrame values:", feature_df.iloc[0].to_dict())

# Reorder columns
feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
print("\nReordered DataFrame columns:", list(feature_df.columns))
print("Reordered DataFrame values:", feature_df.iloc[0].to_dict())

# Try scaling
try:
    scaled_features = scaler.transform(feature_df)
    print("\n✅ Scaling successful!")
    print("Scaled features shape:", scaled_features.shape)
    print("Scaled features:", scaled_features[0])
except Exception as e:
    print(f"\n❌ Scaling failed: {e}")