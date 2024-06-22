
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Applying min-max scaling
scaler = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Min-Max Scaled DataFrame:\n", df_minmax_scaled)


# RobustScaler
from sklearn.preprocessing import RobustScaler

# Applying robust scaling
scaler = RobustScaler()
df_robust_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Robust Scaled DataFrame:\n", df_robust_scaled)