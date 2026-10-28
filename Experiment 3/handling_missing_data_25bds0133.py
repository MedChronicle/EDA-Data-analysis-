import numpy as np
import pandas as pd

newline = "\n"

start_value = 15
stop_value = 30
row_count = 5
col_count = 3
fruit_names = ['apple', 'banana', 'kiwi', 'grapes', 'mango']
store_names = ['store1', 'store2', 'store3']

data = np.arange(start_value, stop_value).reshape(row_count, col_count)
dfx = pd.DataFrame(data, index=fruit_names, columns=store_names)

print(dfx, newline)

new_row_values = np.arange(15, 19)
apple_store4_value = 20.

dfx['store4'] = np.nan
dfx.loc['watermelon'] = new_row_values
dfx.loc['oranges'] = np.nan
dfx['store5'] = np.nan
dfx.loc['apple', 'store4'] = apple_store4_value

print(dfx, newline)

null_mask = dfx.isnull()
not_null_mask = dfx.notnull()
null_counts = dfx.isnull().sum()
total_nulls = null_counts.sum()
non_null_counts = dfx.count()

print(null_mask, newline)
print(not_null_mask, newline)
print(null_counts, newline)
print(total_nulls, newline)
print(non_null_counts, newline)

store4_column = dfx.store4
store4_not_null = store4_column[store4_column.notnull()]
store4_dropped = store4_column.dropna()

print(store4_not_null, newline)
print(store4_dropped, newline)

dropped_any = dfx.dropna()
dropped_all_rows = dfx.dropna(how='all')
dropped_all_columns = dfx.dropna(how='all', axis=1)
column_threshold = 5
dropped_by_threshold = dfx.dropna(thresh=column_threshold, axis=1)

print(dropped_any, newline)
print(dropped_all_rows, newline)
print(dropped_all_columns, newline)
print(dropped_by_threshold, newline)

ar1 = np.array([100, 200, np.nan, 300])
ser1 = pd.Series(ar1)
array_mean = ar1.mean()
series_mean = ser1.mean()

print(array_mean, series_mean, newline)

ser2 = dfx.store4
ser2_sum = ser2.sum()
ser2_mean = ser2.mean()
ser2_cumsum = ser2.cumsum()

print(ser2_sum)
print(ser2_mean)
print(ser2_cumsum, newline)

fill_zero_value = 0
filledDf = dfx.fillna(fill_zero_value)
original_means = dfx.mean(numeric_only=True)
filled_means = filledDf.mean(numeric_only=True)

print(filledDf, newline)
print(original_means, newline)
print(filled_means, newline)

forward_filled = dfx.store4.ffill()
backward_filled = dfx.store4.bfill()

print(forward_filled, newline)
print(backward_filled, newline)

fill_values = {'store4': 0, 'store5': -1}
filled_by_column = dfx.fillna(fill_values)

print(filled_by_column, newline)

interpolated_store1 = dfx.store1.interpolate()
interpolated_store4 = dfx.store4.interpolate()

print(interpolated_store1, newline)
print(interpolated_store4, newline)

filled_with_means = dfx.fillna(original_means)

print(filled_with_means, newline)

replacement_value = 999
replaced_nan = dfx.replace(np.nan, replacement_value)

print(replaced_nan, newline)

rows_allowed_missing = 3
required_non_null = len(dfx) - rows_allowed_missing
dropped_sparse_columns = dfx.dropna(axis=1, thresh=required_non_null)

print(dropped_sparse_columns, newline)

dfy = dfx.copy()
store6_values = [1, 2, np.nan, 4, np.nan, 6, np.nan]
dfy['store6'] = store6_values

print(dfy, newline)

store6_median = dfy.store6.median()
store6_mode = dfy.store6.mode()[0]
store6_filled_median = dfy.store6.fillna(store6_median)
store6_filled_mode = dfy.store6.fillna(store6_mode)

print(store6_filled_median, newline)
print(store6_filled_mode, newline)
