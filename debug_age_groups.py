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

# Create feature engineer and apply age groups
fe = FeatureEngineer()
result = fe.create_age_groups(data)

# Find age group columns (exclude numeric version)
age_cols = [col for col in result.columns if col.startswith('AgeGroup_') and not col.endswith('_num')]
print('All columns:', result.columns.tolist())
print('Age columns:', age_cols)
print('\nData shape:', result.shape)
print('\nData:')
print(result[['Age'] + age_cols])
print('\nData types:')
print(result[age_cols].dtypes)
print('\nRow sums:')
row_sums = result[age_cols].sum(axis=1)
print(row_sums)
print('\nRow sums type:', type(row_sums.iloc[0]))
print('\nAll row sums equal 1?', (row_sums == 1).all())
print('Number of rows where sum != 1:', (row_sums != 1).sum())
if (row_sums != 1).any():
    print('Problematic rows:')
    problematic_indices = row_sums[row_sums != 1].index
    print(result.loc[problematic_indices, ['Age'] + age_cols])
    print('Their row sums:', row_sums[problematic_indices])

# Let's also test pd.cut directly
print('\n--- Testing pd.cut directly ---')
age_groups = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['Young', 'Adult', 'MiddleAge', 'Senior', 'Elder'])
print('Age groups:', age_groups)
dummies = pd.get_dummies(age_groups, prefix='AgeGroup', dummy_na=False)
print('Dummies:')
print(dummies)
print('Dummies row sums:', dummies.sum(axis=1))