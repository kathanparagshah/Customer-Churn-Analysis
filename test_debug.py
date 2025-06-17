import sys
sys.path.append('/Users/kathan/Downloads/Customer Churn Analysis/src')
from features.create_features import FeatureEngineer
import pandas as pd
import numpy as np

# Create sample data exactly like the test fixture
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'CreditScore': np.random.randint(300, 850, n_samples),
    'Geography_France': np.random.choice([0, 1], n_samples),
    'Geography_Germany': np.random.choice([0, 1], n_samples),
    'Geography_Spain': np.random.choice([0, 1], n_samples),
    'Gender_Female': np.random.choice([0, 1], n_samples),
    'Gender_Male': np.random.choice([0, 1], n_samples),
    'Age': np.random.randint(18, 80, n_samples),
    'Tenure': np.random.randint(0, 10, n_samples),
    'Balance': np.random.uniform(0, 200000, n_samples),
    'NumOfProducts': np.random.randint(1, 4, n_samples),
    'HasCrCard': np.random.choice([0, 1], n_samples),
    'IsActiveMember': np.random.choice([0, 1], n_samples),
    'EstimatedSalary': np.random.uniform(10000, 150000, n_samples),
    'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

# Test exactly like the test does
feature_engineer = FeatureEngineer()
enhanced_data = feature_engineer.create_age_groups(data)

# Check that age group columns are created
age_group_cols = [col for col in enhanced_data.columns if col.startswith('AgeGroup_')]
print(f"Age group columns found: {age_group_cols}")
print(f"Number of age group columns: {len(age_group_cols)}")
print(f"Length check passes: {len(age_group_cols) > 0}")

# Check that age groups are mutually exclusive
age_group_sum = enhanced_data[age_group_cols].sum(axis=1)
print(f"\nAge group sum data type: {age_group_sum.dtype}")
print(f"Unique sums: {age_group_sum.unique()}")
print(f"All sums equal 1: {(age_group_sum == 1).all()}")
print(f"Type of (age_group_sum == 1).all(): {type((age_group_sum == 1).all())}")
print(f"Value of (age_group_sum == 1).all(): {(age_group_sum == 1).all()}")

# Check if there are any problematic rows
problematic = age_group_sum != 1
print(f"\nNumber of problematic rows: {problematic.sum()}")
if problematic.any():
    print("Problematic indices:", problematic[problematic].index.tolist())
    print("Their sums:", age_group_sum[problematic].tolist())