import pandas as pd
import numpy as np

# Test data from the failing test
data = pd.Series([1, 2, 3, 4, 5, 100, 200])
threshold = 2

# Modified Z-score calculation
clean_series = data.dropna()
median = clean_series.median()
mad = np.median(np.abs(clean_series - median))

print(f'Data: {data.tolist()}')
print(f'Median: {median}, MAD: {mad}')

# Avoid division by zero
if mad == 0:
    mad = np.std(clean_series)
    print(f'MAD was 0, using std: {mad}')
    
if mad == 0:
    print('Both MAD and std are 0, no outliers can be detected')
else:
    # Modified z-scores
    modified_z_scores = 0.6745 * (clean_series - median) / mad
    print(f'Modified Z-scores: {modified_z_scores.tolist()}')
    print(f'Threshold: {threshold}')
    
    # Find outliers
    outlier_mask = np.abs(modified_z_scores) > threshold
    outliers = clean_series[outlier_mask]
    
    print(f'Outlier mask: {outlier_mask.tolist()}')
    print(f'Outliers: {outliers.tolist()}')
    print(f'Count: {len(outliers)}')
    print(f'Values in outliers: {outliers.values}')