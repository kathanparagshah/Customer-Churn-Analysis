import pandas as pd
import numpy as np

# Test data from the failing test
data = pd.Series([1, 2, 3, 4, 5, 100, 200])

# IQR calculation
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 0.5 * IQR
upper_bound = Q3 + 0.5 * IQR

print(f'Data: {data.tolist()}')
print(f'Q1: {Q1}, Q3: {Q3}, IQR: {IQR}')
print(f'Lower bound: {lower_bound}, Upper bound: {upper_bound}')

# Find outliers
outlier_mask = (data < lower_bound) | (data > upper_bound)
outliers = data[outlier_mask]

print(f'Outlier mask: {outlier_mask.tolist()}')
print(f'Outliers: {outliers.tolist()}')
print(f'Count: {len(outliers)}')
print(f'Values in outliers: {outliers.values}')