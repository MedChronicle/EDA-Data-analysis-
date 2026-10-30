import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import stats
except ImportError:
    stats = None

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "SuicideChina.csv"
OUT_DIR = Path("eda_output")
OUT_DIR.mkdir(exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=130)
    plt.close()


section("1. LOAD THE DATASET")
df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

print("First 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())
print("\nRandom sample of 5 rows:\n", df.sample(5, random_state=42))
print(f"\nShape: {df.shape}  ->  {df.shape[0]:,} records, {df.shape[1]} features")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

section("2. METADATA")
print("Columns:", list(df.columns))
print("\nData types:\n", df.dtypes)
print()
df.info()

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
low_card_num = [c for c in num_cols if df[c].nunique() <= 10]
print(f"\nNumeric columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns that look like codes (<=10 unique): {low_card_num}")

print("\nUnique values per column:")
print(df.nunique().sort_values())

print("\nDescriptive statistics (numeric):\n", df.describe().T)
if cat_cols:
    print("\nDescriptive statistics (categorical):\n", df[cat_cols].describe().T)

section("3. KEY QUESTIONS")
if "Age" in df.columns:
    print(f"Average age : {df['Age'].mean():.2f}")
    print(f"Median age  : {df['Age'].median():.2f}")
    print(f"Mode age    : {df['Age'].mode().iloc[0]}")
    print(f"Age range   : {df['Age'].min()} - {df['Age'].max()}")

std = df[num_cols].std().sort_values(ascending=False)
print("\nStd deviation per numeric column (high -> low):\n", std)
print(f"Highest std deviation: {std.index[0]} ({std.iloc[0]:.3f})")

cv = (df[num_cols].std() / df[num_cols].mean().replace(0, np.nan)).abs()
print("\nCoefficient of variation (std/mean):\n", cv.sort_values(ascending=False))

for col in ["Sex", "Urban", "Education", "Occupation", "Method"]:
    if col in df.columns:
        counts = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(normalize=True, dropna=False) * 100
        print(f"\nRecords by {col}:\n", pd.DataFrame({"count": counts, "percent": pct.round(2)}))

section("4. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = missing / len(df) * 100
miss_df = pd.DataFrame({"missing": missing, "percent": missing_pct.round(2)})
miss_df = miss_df.sort_values("missing", ascending=False)
print(miss_df)

total_missing = int(missing.sum())
total_cells = df.size
print(f"\nOverall missing: {total_missing:,} of {total_cells:,} cells "
      f"({total_missing / total_cells * 100:.3f}%)")
print(f"Rows with at least one missing value: {df.isnull().any(axis=1).sum():,}")
print(f"Completely empty columns: {df.columns[df.isnull().all()].tolist()}")

if total_missing:
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
    plt.title("Missing-value map (light = missing)")
    save_fig("missing_heatmap.png")

section("5. DATA QUALITY")
dups = df.duplicated().sum()
print(f"Fully duplicated rows: {dups:,} ({dups / len(df) * 100:.2f}%)")

const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print(f"Constant columns: {const_cols}")

for c in cat_cols:
    s = df[c].dropna().astype(str)
    if (s != s.str.strip()).any():
        print(f"Column '{c}' has leading/trailing whitespace")

section("6. OUTLIER DETECTION (IQR and Z-score)")
outlier_rows = []
for c in num_cols:
    if c in low_card_num:
        continue
    s = df[c].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_out = ((s < lo) | (s > hi)).sum()
    z_out = (np.abs((s - s.mean()) / s.std()) > 3).sum() if s.std() > 0 else 0
    outlier_rows.append([c, lo, hi, iqr_out, z_out])
if outlier_rows:
    print(pd.DataFrame(outlier_rows,
                       columns=["column", "lower_fence", "upper_fence",
                                "IQR_outliers", "Z>3_outliers"]).round(2))

section("7. DISTRIBUTION SHAPE")
shape_df = pd.DataFrame({"skew": df[num_cols].skew(),
                         "kurtosis": df[num_cols].kurt()}).round(3)
print(shape_df)

section("8. CORRELATION")
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    print(corr.round(3))

    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                 .stack().rename("corr").reset_index())
    pairs["abs"] = pairs["corr"].abs()
    print("\nTop 10 strongest correlations:\n",
          pairs.sort_values("abs", ascending=False).head(10)
               .drop(columns="abs").round(3).to_string(index=False))

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    save_fig("correlation_heatmap.png")

section("9. GROUP ANALYSIS")
if "Age" in df.columns:
    bins = [0, 17, 29, 44, 59, 74, 120]
    labels = ["<18", "18-29", "30-44", "45-59", "60-74", "75+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    print("Records by age group:\n", df["AgeGroup"].value_counts().sort_index())

    for g in ["Sex", "Urban"]:
        if g in df.columns:
            print(f"\nAge summary by {g}:\n",
                  df.groupby(g)["Age"].agg(["count", "mean", "median", "std", "min", "max"]).round(2))

if "Year" in df.columns:
    print("\nRecords per year:\n", df["Year"].value_counts().sort_index())

for a, b in [("Sex", "Urban"), ("Sex", "Method"), ("Urban", "Method"),
             ("Sex", "Died"), ("Urban", "Died")]:
    if a in df.columns and b in df.columns:
        ct = pd.crosstab(df[a], df[b], margins=True)
        print(f"\nCross-tab {a} x {b}:\n", ct)
        print("Row percentages:\n", (pd.crosstab(df[a], df[b], normalize="index") * 100).round(2))

section("10. STATISTICAL TESTS")
if stats is None:
    print("scipy not installed - run: pip install scipy")
else:
    cats = [c for c in df.columns if df[c].nunique() <= 15 and c != "AgeGroup"]
    results = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            ct = pd.crosstab(df[a], df[b])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, dof, _ = stats.chi2_contingency(ct)
                n = ct.values.sum()
                cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
                results.append([a, b, round(chi2, 2), dof, p, round(cramers_v, 3)])
    if results:
        res = pd.DataFrame(results, columns=["var1", "var2", "chi2", "dof", "p_value", "cramers_v"])
        print("Chi-square tests (sorted by effect size):\n",
              res.sort_values("cramers_v", ascending=False).head(15).to_string(index=False))

    if "Age" in df.columns:
        for g in [c for c in df.columns if df[c].nunique() == 2 and c != "Age"]:
            grp = [x["Age"].dropna() for _, x in df.groupby(g)]
            if all(len(x) > 2 for x in grp):
                t, p_t = stats.ttest_ind(grp[0], grp[1], equal_var=False)
                u, p_u = stats.mannwhitneyu(grp[0], grp[1])
                print(f"\nAge by {g}: Welch t={t:.2f} (p={p_t:.4g}), "
                      f"Mann-Whitney U p={p_u:.4g}")

section("11. PLOTS")
plot_num = [c for c in num_cols if c not in low_card_num][:6]
if plot_num:
    fig, axes = plt.subplots(len(plot_num), 2, figsize=(11, 3.2 * len(plot_num)))
    axes = np.atleast_2d(axes)
    for ax_row, c in zip(axes, plot_num):
        sns.histplot(df[c].dropna(), kde=True, ax=ax_row[0])
        ax_row[0].set_title(f"{c} distribution")
        sns.boxplot(x=df[c], ax=ax_row[1])
        ax_row[1].set_title(f"{c} boxplot")
    save_fig("numeric_distributions.png")

plot_cat = [c for c in cat_cols + low_card_num if df[c].nunique() <= 12][:6]
if plot_cat:
    cols = 2
    rows = int(np.ceil(len(plot_cat) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    for ax, c in zip(np.atleast_1d(axes).ravel(), plot_cat):
        df[c].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"{c} counts")
    save_fig("categorical_counts.png")

if "Age" in df.columns and "Sex" in df.columns:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x="Age", hue="Sex", fill=True, common_norm=False)
    plt.title("Age distribution by sex")
    save_fig("age_by_sex.png")

if "Year" in df.columns:
    plt.figure(figsize=(9, 4))
    df["Year"].value_counts().sort_index().plot(marker="o")
    plt.title("Records per year")
    save_fig("records_per_year.png")

print(f"Plots saved to: {OUT_DIR.resolve()}")

section("12. BASIC CLEANING AND EXPORT")
clean = df.drop_duplicates().copy()
for c in clean.select_dtypes(include=np.number).columns:
    if clean[c].isnull().any():
        clean[c] = clean[c].fillna(clean[c].median())
for c in clean.select_dtypes(exclude=np.number).columns:
    if clean[c].isnull().any():
        clean[c] = clean[c].fillna(clean[c].mode().iloc[0])
print(f"Rows before: {len(df):,} | after dedupe: {len(clean):,}")
print(f"Missing values after imputation: {int(clean.isnull().sum().sum())}")
clean.to_csv(OUT_DIR / "SuicideChina_clean.csv", index=False)

with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\n")
    f.write(f"Duplicates: {dups}\nMissing cells: {total_missing}\n")
    f.write(f"Numeric: {num_cols}\nCategorical: {cat_cols}\n")
print(f"Saved cleaned data and summary to {OUT_DIR.resolve()}")
print("\nDone.")
