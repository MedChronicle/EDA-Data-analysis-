import pandas as pd
import numpy as np

df1SE = pd.DataFrame({'StudentID': [59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79],
                      'ScoreSE': [48, 61, 74, 35, 82, 90, 57, 42, 69, 76, 88]})
df2SE = pd.DataFrame({'StudentID': [52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80],
                      'ScoreSE': [85, 90, 42, 71, 63, 58, 36, 49, 82, 91, 54, 73, 38, 61, 25]})

df1ML = pd.DataFrame({'StudentID': [51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79],
                      'ScoreML': [44, 52, 61, 79, 48, 83, 39, 71, 68, 55, 84, 79, 95, 27, 46]})
df2ML = pd.DataFrame({'StudentID': [52, 54, 56, 58, 60, 62, 64, 66, 68, 70],
                      'ScoreML': [88, 41, 75, 94, 84, 91, 42, 46, 85, 76]})

dfSE = pd.concat([df1SE, df2SE], ignore_index=True)
dfML = pd.concat([df1ML, df2ML], ignore_index=True)

side_by_side = pd.concat([dfML, dfSE], axis=1)
inner_merged = dfSE.merge(dfML, how='inner')
left_merged = dfSE.merge(dfML, how='left')
right_merged = dfSE.merge(dfML, how='right')

print(side_by_side)
print(inner_merged)
print(left_merged)
print(right_merged)

orders = pd.DataFrame({
    'OrderID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Company': ['Acme', 'Bolt', 'Acme', 'Crane', 'Bolt', 'Delta', 'Acme', 'Crane', 'Delta', 'Bolt', 'Acme', 'Delta', 'Crane', 'Bolt', 'Acme'],
    'Product': ['Widget', 'Gear', 'Widget', 'Bolt', 'Gear', 'Sprocket', 'Bolt', 'Widget', 'Gear', 'Sprocket', 'Widget', 'Bolt', 'Gear', 'Widget', 'Sprocket'],
    'UnitPrice': [12.5, 8.0, 12.5, 5.0, 8.0, 15.0, 5.0, 12.5, 8.0, 15.0, 12.5, 5.0, 8.0, 12.5, 15.0],
    'Quantity': [10, 25, 15, 40, 30, 12, 20, 18, 22, 9, 14, 35, 27, 16, 11]
})

preview_rows = 10
orders_preview = orders.head(preview_rows)
print(orders_preview)

orders['TotalPrice'] = orders['UnitPrice'] * orders['Quantity']
orders_with_total = orders.head(preview_rows)
company_counts = orders['Company'].value_counts()
orders_summary = orders.describe()

print(orders_with_total)
print(company_counts)
print(orders_summary)

rows = 3
cols = 5
data = np.arange(rows * cols).reshape((rows, cols))
indexers = ['Temperature', 'Pressure', 'Visibility']
cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bengaluru']

dframe1 = pd.DataFrame(data, index=indexers, columns=cities)
stacked = dframe1.stack()
unstacked = stacked.unstack()

print(dframe1)
print(stacked)
print(unstacked)

series1 = pd.Series([100, 200, 300, 400], index=['ones', 'twos', 'threes', 'fours'])
series2 = pd.Series([500, 600, 700], index=['fives', 'sixs', 'sevens'])
group_keys = ['GroupA', 'GroupB']

frame2 = pd.concat([series1, series2], keys=group_keys)
frame2_unstacked = frame2.unstack()

print(frame2_unstacked)
