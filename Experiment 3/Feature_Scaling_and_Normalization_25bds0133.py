import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

input_file = 'fraudTrain.csv'
output_file = 'fraudTrain_scaled.csv'

df = pd.read_csv(input_file)

candidate_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
numeric_cols = [c for c in candidate_cols if c in df.columns]

minmax_suffix = '_minmax'
standard_suffix = '_standard'
robust_suffix = '_robust'

minmax_columns = [c + minmax_suffix for c in numeric_cols]
standard_columns = [c + standard_suffix for c in numeric_cols]
robust_columns = [c + robust_suffix for c in numeric_cols]

original_data = df[numeric_cols]
original_preview = original_data.head()

print("Numeric columns used:", numeric_cols)
print("\nOriginal data (first 5 rows):")
print(original_preview)

min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

minmax_values = min_max_scaler.fit_transform(original_data)
standard_values = standard_scaler.fit_transform(original_data)
robust_values = robust_scaler.fit_transform(original_data)

df_minmax = pd.DataFrame(minmax_values, columns=minmax_columns)
df_standard = pd.DataFrame(standard_values, columns=standard_columns)
df_robust = pd.DataFrame(robust_values, columns=robust_columns)

result = pd.concat([original_data, df_minmax, df_standard, df_robust], axis=1)

minmax_preview = df_minmax.head()
standard_preview = df_standard.head()
robust_preview = df_robust.head()

print("\nMin-Max scaled data (first 5 rows):")
print(minmax_preview)

print("\nStandardized data (first 5 rows):")
print(standard_preview)

print("\nRobust scaled data (first 5 rows):")
print(robust_preview)

result.to_csv(output_file, index=False)

saved_message = "\nSaved all scaled features to " + output_file
print(saved_message)
