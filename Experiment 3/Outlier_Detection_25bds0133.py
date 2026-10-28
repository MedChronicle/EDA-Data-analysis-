import pandas as pd
import numpy as np

input_file = 'fraudTrain.csv'
outliers_file = 'fraudTrain_outliers.csv'
cleaned_file = 'fraudTrain_cleaned.csv'

amount_column = 'amt'
zscore_column = 'amt_zscore'
lower_quantile = 0.25
upper_quantile = 0.75
iqr_multiplier = 1.5
zscore_threshold = 3
top_rows = 10

df = pd.read_csv(input_file)
original_shape = df.shape

print("Dataset shape:", original_shape)

Q1 = df[amount_column].quantile(lower_quantile)
Q3 = df[amount_column].quantile(upper_quantile)
IQR = Q3 - Q1
lower_bound = Q1 - iqr_multiplier * IQR
upper_bound = Q3 + iqr_multiplier * IQR

quartile_message = f"\nQ1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}"
bounds_message = f"IQR outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"

print(quartile_message)
print(bounds_message)

below_lower = df[amount_column] < lower_bound
above_upper = df[amount_column] > upper_bound
iqr_outliers = df[below_lower | above_upper]

total_rows = len(df)
iqr_outlier_count = len(iqr_outliers)
iqr_message = f"IQR method flagged {iqr_outlier_count} outlier transactions out of {total_rows}"
top_iqr_outliers = iqr_outliers[[amount_column]].sort_values(amount_column, ascending=False).head(top_rows)

print(iqr_message)
print(top_iqr_outliers)

mean_amt = df[amount_column].mean()
std_amt = df[amount_column].std()
df[zscore_column] = (df[amount_column] - mean_amt) / std_amt

is_z_outlier = np.abs(df[zscore_column]) > zscore_threshold
z_outliers = df[is_z_outlier]

z_outlier_count = len(z_outliers)
z_message = f"\nZ-score method (|z| > {zscore_threshold}) flagged {z_outlier_count} outlier transactions"
top_z_outliers = z_outliers[[amount_column, zscore_column]].sort_values(amount_column, ascending=False).head(top_rows)

print(z_message)
print(top_z_outliers)

within_lower = df[amount_column] >= lower_bound
within_upper = df[amount_column] <= upper_bound
df_clean = df[within_lower & within_upper].copy()

cleaned_shape = df_clean.shape
removed_rows = len(df) - len(df_clean)
cleaned_message = f"\nCleaned dataset shape: {cleaned_shape} (removed {removed_rows} rows using the IQR rule)"

print(cleaned_message)

iqr_outliers.to_csv(outliers_file, index=False)
df_clean.to_csv(cleaned_file, index=False)

outliers_saved_message = "\nSaved flagged outliers to " + outliers_file
cleaned_saved_message = "Saved cleaned (outlier-free) dataset to " + cleaned_file

print(outliers_saved_message)
print(cleaned_saved_message)
