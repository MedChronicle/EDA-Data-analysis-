import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer

newline = "\n"

ages = [25, np.nan, 30, 45, 22]
salaries = [50000, 60000, np.nan, 80000, 45000]
categories = ["A", "B", "A", np.nan, "B"]

df = pd.DataFrame({"Age": ages, "Salary": salaries, "Category": categories})

print(df, newline)

age_column = ["Age"]
salary_column = ["Salary"]
category_column = ["Category"]
num_cols = ["Age", "Salary"]
constant_fill_value = 0

mean_imputer = SimpleImputer(strategy="mean")
median_imputer = SimpleImputer(strategy="median")
mode_imputer = SimpleImputer(strategy="most_frequent")
const_imputer = SimpleImputer(strategy="constant", fill_value=constant_fill_value)

df["Age_Mean"] = mean_imputer.fit_transform(df[age_column]).ravel()
df["Salary_Median"] = median_imputer.fit_transform(df[salary_column]).ravel()
df["Category_Mode"] = mode_imputer.fit_transform(df[category_column]).ravel()
df["Age_Const"] = const_imputer.fit_transform(df[age_column]).ravel()

max_iterations = 10
random_seed = 42
mice_imputer = IterativeImputer(max_iter=max_iterations, random_state=random_seed)
df_mice = df[num_cols].copy()
df_mice.iloc[:, :] = mice_imputer.fit_transform(df_mice)
df["Age_MICE"] = df_mice["Age"]
df["Salary_MICE"] = df_mice["Salary"]

neighbor_count = 2
knn_imputer = KNNImputer(n_neighbors=neighbor_count)
df_knn = df[num_cols].copy()
df_knn.iloc[:, :] = knn_imputer.fit_transform(df_knn)
df["Age_KNN"] = df_knn["Age"]
df["Salary_KNN"] = df_knn["Salary"]

max_columns_option = None
display_width = 160
pd.set_option("display.max_columns", max_columns_option)
pd.set_option("display.width", display_width)

print(df, newline)

departments = ["Sales", "Sales", "IT", "IT", "HR", "HR"]
scores = [80, np.nan, 70, 75, np.nan, 60]

df2 = pd.DataFrame({"Dept": departments, "Score": scores})

print(df2, newline)

group_mean_fill = lambda s: s.fillna(s.mean())
df2["Score_GroupMean"] = df2.groupby("Dept")["Score"].transform(group_mean_fill)

print(df2, newline)

missing_suffix = "_missing"
indicator = df[num_cols].isnull().astype(int).add_suffix(missing_suffix)
df = pd.concat([df, indicator], axis=1)

print(df)
