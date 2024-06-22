import pandas as pd
from sklearn.preprocessing import StandardScaler

#StandardScaler
# Creating a sample DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [10, 20, 30, 40, 50],
    'Feature3': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Applying standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Standardized DataFrame:\n", df_standardized)