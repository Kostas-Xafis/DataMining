import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

bancrupt = 'X65'
df = pd.read_csv('./training_companydata.csv', na_values=['?'])

bancrupt_column = df[bancrupt]
df = df.drop(columns=[bancrupt])

# Drop Columns with High Missing Values
null_threshold = 0.2
null_perc = df.isnull().mean()
to_drop_missing = null_perc.where(null_perc > null_threshold).dropna().index
print("Columns with high missing values: ", to_drop_missing.tolist())
df = df.drop(columns=to_drop_missing)

# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Drop the columns with high correlation
correlation_threshold = 0.99
correlation_matrix = df.corr().abs()
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

to_drop_corr = [column for column in upper.columns if any(upper[column] >= correlation_threshold)]
print("Columns with high correlation: ", to_drop_corr)
df = df.drop(columns=to_drop_corr)

# Drop columns with multiple 0 values
zero_threshold = 0.1 # More than 10% of the values are 0
to_drop_zero = df.columns[df.eq(0).mean().ge(zero_threshold)]
print("Columns with high zero values: ", to_drop_zero)
df = df.drop(columns=to_drop_zero)

# MinMax Data normalization: Performs worse than Z-score normalization
# df = (df-df.min()) / (df.max()-df.min())

# Z-score Data normalization
df = (df-df.mean()) / df.std()

# Append the bancrupt column back to the dataframe
df = pd.concat([df, bancrupt_column], axis=1)

# Save the prepped data
df.to_csv('./training_companydata_prepped.csv', index=False, float_format='%.5f')

