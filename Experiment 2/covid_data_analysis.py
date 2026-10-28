import pandas as pd
import numpy as np

csv_file = "covid-data.csv"
df = pd.read_csv(csv_file)

print("TASK -> LOAD THE DATASET")

first_rows = df.head()
last_rows = df.tail()

print("FIRST 5 ROWS -> ")
print(first_rows)

print("LAST 5 ROWS -> ")
print(last_rows)

shape_of_data = df.shape
print("Dataset Shape is:")
print(shape_of_data)

print("Q1. How many records are present in the dataset?")
total_rows = len(df)
rows_answer = "Answer: " + str(total_rows) + " records"
print(rows_answer)

print("Q2. How many features (columns) are available?")
total_cols = len(df.columns)
cols_answer = "Answer: " + str(total_cols) + " columns"
print(cols_answer)

print("TASK  DISPLAY METADATA")

column_names = df.columns
column_types = df.dtypes
empty_line = ""

print("Column Names:")
print(column_names)
print(empty_line)

print("Data Types of Columns:")
print(column_types)

print("Dataset Info:")
df.info()

summary_stats = df.describe()
print("Descriptive Statistics:")
print(summary_stats)

print("Q1. What is the average new cases reported per record?")
avg_cases = df["new_cases"].mean()
avg_cases_answer = "Answer: " + str(avg_cases)
print(avg_cases_answer)

print("Q2. Which feature has the highest standard deviation?")
std_values = df.std(numeric_only=True)
print("Standard deviation for each column:")
print(std_values)

print("Q3. How many unique countries/locations are in the dataset?")
unique_countries = df["location"].nunique()
countries_answer = "Answer: " + str(unique_countries) + " countries"
print(countries_answer)

print("Q4. Which country recorded the single highest total cases value, and what was it?")
max_cases_index = df["total_cases"].idxmax()
max_cases_row = df.loc[max_cases_index]
max_location = str(max_cases_row["location"])
max_total_cases = str(max_cases_row["total_cases"])
max_cases_answer = "Answer: " + max_location + " with " + max_total_cases + " total cases"
print(max_cases_answer)

missing_stuff = df.isnull().sum()
print("Missing values count for each column:")
print(missing_stuff)

missing_pct = (missing_stuff / total_rows) * 100
print("Missing values percentage:")
print(missing_pct)

missing_sorted = missing_stuff.sort_values(ascending=False)
print("Missing values sorted:")
print(missing_sorted)

print("What percentage of data is missing overall?")
total_missing = missing_stuff.sum()
total_cells = total_rows * total_cols
overall_percent = (total_missing / total_cells) * 100
overall_answer = "Answer: " + str(overall_percent) + "%"
print(overall_answer)
