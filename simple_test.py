import sys
sys.path.append('/Users/kathan/Downloads/Customer Churn Analysis/src')
from features.create_features import FeatureEngineer
import pandas as pd
import numpy as np

# Create sample data like the test fixture
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

fe = FeatureEngineer()
result = fe.create_age_groups(data)
age_cols = [col for col in result.columns if col.startswith('AgeGroup_') and not col.endswith('_num')]
row_sums = result[age_cols].sum(axis=1)

print('All row sums equal 1?', (row_sums == 1).all())
print('Number of problematic rows:', (row_sums != 1).sum())
print('Row sums data type:', row_sums.dtype)
print('Unique row sums:', row_sums.unique())