from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from pandas.plotting import andrews_curves, parallel_coordinates
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import STOPWORDS, WordCloud

INPUT_PATH = "ecommerce_sales_data.csv"
PLOT_DIR = Path("plots")
NUMERIC_COLS = ["Price", "Quantity", "Discount_pct", "Sales", "Rating", "Customer_Age", "Delivery_Days"]
CATEGORY_COL = "Category"
COUNTRY_COL = "Country"
REVIEW_COL = "Review"
DATE_COL = "Order_Date"
TOP_COUNTRIES = 6
PIE_MAX_SLICES = 8
PAIRPLOT_SAMPLE = 200
PARALLEL_SAMPLE = 150
SCATTER_SAMPLE = 5000
RANDOM_STATE = 42


def save_figure(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run_step(label, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as exc:
        print(f"Skipped {label}: {exc}")


def sample_frame(data, n):
    return data.sample(n=min(n, len(data)), random_state=RANDOM_STATE)


def minmax(data):
    span = (data.max() - data.min()).replace(0, 1)
    return (data - data.min()) / span


def zscore(data):
    return (data - data.mean()) / data.std().replace(0, 1)


def profile(df):
    print("Shape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe(include="all").T.to_string())
    missing = df.isna().sum()
    print("\nMissing values:")
    print(missing[missing > 0].to_string() if missing.any() else "None")


def clean(df):
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"\nRemoved {before - len(df)} duplicate row(s)")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df[CATEGORY_COL] = df[CATEGORY_COL].fillna("Unknown").astype(str)
    return df


def plot_pie(series, title, filename):
    series = series.sort_values(ascending=False)
    if len(series) > PIE_MAX_SLICES:
        head = series.iloc[: PIE_MAX_SLICES - 1]
        other = pd.Series({"Other": series.iloc[PIE_MAX_SLICES - 1:].sum()})
        series = pd.concat([head, other])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        series.values,
        labels=series.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2", len(series)),
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title(title)
    save_figure(fig, filename)


def plot_bar(series, title, ylabel, filename):
    series = series.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(series.index.astype(str), series.values, color=sns.color_palette("viridis", len(series)))
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, filename)


def plot_histogram(data, column, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data[column], bins=25, kde=True, color="teal", ax=ax)
    ax.axvline(data[column].mean(), color="crimson", linestyle="--", label=f"Mean {data[column].mean():.2f}")
    ax.axvline(data[column].median(), color="navy", linestyle=":", label=f"Median {data[column].median():.2f}")
    ax.set_title(f"Distribution of {column} (skew {data[column].skew():.2f})")
    ax.legend()
    save_figure(fig, filename)


def plot_boxplot(data, group_col, value_col, filename):
    order = data.groupby(group_col)[value_col].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=group_col, y=value_col, order=order, hue=group_col, palette="Set3", legend=False, ax=ax)
    ax.set_title(f"{value_col} by {group_col}")
    ax.tick_params(axis="x", rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, filename)


def plot_heatmap(data, title, filename, fmt=".0f", cmap="coolwarm", center=None, mask=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(data, annot=True, cmap=cmap, fmt=fmt, center=center, mask=mask, linewidths=0.5, ax=ax)
    ax.set_title(title)
    save_figure(fig, filename)


def plot_choropleth(data, filename):
    totals = data.groupby(COUNTRY_COL)["Sales"].sum().reset_index()
    fig = px.choropleth(
        totals,
        locations=COUNTRY_COL,
        locationmode="country names",
        color="Sales",
        color_continuous_scale="Plasma",
        hover_name=COUNTRY_COL,
        title="Total Sales by Country",
    )
    path = PLOT_DIR / filename
    fig.write_html(path)
    print(f"Saved {path}")


def plot_wordcloud(reviews, filename):
    text = " ".join(reviews.dropna().astype(str))
    if not text.strip():
        raise ValueError("no review text available")
    cloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        colormap="viridis",
        stopwords=set(STOPWORDS),
        max_words=150,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Words in Customer Reviews")
    save_figure(fig, filename)


def plot_dendrogram(data, filename):
    profile_data = data.groupby(CATEGORY_COL)[NUMERIC_COLS].mean()
    if len(profile_data) < 2:
        raise ValueError("need at least two categories")
    linked = linkage(zscore(profile_data), method="ward")
    fig, ax = plt.subplots(figsize=(9, 6))
    dendrogram(linked, labels=profile_data.index.tolist(), leaf_rotation=45, ax=ax)
    ax.set_title("Dendrogram of Product Categories (standardized average profile)")
    ax.set_ylabel("Ward distance")
    save_figure(fig, filename)


def plot_scatter(data, x, y, filename):
    plot_data = sample_frame(data, SCATTER_SAMPLE)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_data, x=x, y=y, hue=CATEGORY_COL, palette="tab10", alpha=0.7, ax=ax)
    ax.set_title(f"{y} vs {x} (r = {data[x].corr(data[y]):.2f})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_figure(fig, filename)


def plot_correlation(data, filename):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    plot_heatmap(corr, "Correlation Heatmap of Numeric Features", filename, fmt=".2f", center=0, mask=mask)


def plot_pairplot(data, filename):
    grid = sns.pairplot(
        sample_frame(data, PAIRPLOT_SAMPLE)[NUMERIC_COLS + [CATEGORY_COL]],
        hue=CATEGORY_COL,
        palette="tab10",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 20},
    )
    grid.fig.suptitle("Pair Plot of Numeric Features by Category", y=1.02)
    save_figure(grid.fig, filename)


def plot_parallel(data, filename):
    sample = sample_frame(data, PARALLEL_SAMPLE)[NUMERIC_COLS + [CATEGORY_COL]].copy()
    sample[NUMERIC_COLS] = minmax(sample[NUMERIC_COLS])
    palette = sns.color_palette("tab10", sample[CATEGORY_COL].nunique())
    fig, ax = plt.subplots(figsize=(10, 6))
    parallel_coordinates(sample, CATEGORY_COL, color=palette, ax=ax, alpha=0.6)
    ax.set_title("Parallel Coordinates Plot (normalized features)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_figure(fig, filename)


def plot_andrews(data, filename):
    sample = sample_frame(data, PARALLEL_SAMPLE)[NUMERIC_COLS + [CATEGORY_COL]].copy()
    sample[NUMERIC_COLS] = zscore(sample[NUMERIC_COLS])
    fig, ax = plt.subplots(figsize=(10, 6))
    andrews_curves(sample, CATEGORY_COL, colormap="tab10", ax=ax, alpha=0.6)
    ax.set_title("Andrews Curves by Category (standardized features)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_figure(fig, filename)


def plot_monthly_trend(data, filename):
    months = pd.to_datetime(data[DATE_COL], errors="coerce").dt.to_period("M").dt.to_timestamp()
    monthly = data.assign(month=months).dropna(subset=["month"]).groupby("month")["Sales"].sum()
    if monthly.empty:
        raise ValueError(f"no valid dates in '{DATE_COL}'")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color="teal")
    ax.set_title("Monthly Total Sales")
    ax.set_ylabel("Sales")
    fig.autofmt_xdate()
    save_figure(fig, filename)


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 110

    df = pd.read_csv(INPUT_PATH)
    missing_cols = [c for c in NUMERIC_COLS + [CATEGORY_COL] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Required columns missing from {INPUT_PATH}: {missing_cols}")

    profile(df)
    df = clean(df)

    run_step(
        "pie chart",
        plot_pie,
        df.groupby(CATEGORY_COL)["Sales"].sum(),
        "Total Sales Share by Category",
        "01_pie_sales_by_category.png",
    )
    run_step(
        "bar chart",
        plot_bar,
        df.groupby(CATEGORY_COL)["Rating"].mean(),
        "Average Rating by Category",
        "Average Rating",
        "02_bar_avg_rating_by_category.png",
    )
    run_step("price histogram", plot_histogram, df, "Price", "03_histogram_price.png")
    run_step("sales histogram", plot_histogram, df, "Sales", "03_histogram_sales.png")

    if COUNTRY_COL in df.columns:
        top = df[COUNTRY_COL].value_counts().head(TOP_COUNTRIES).index
        pivot = df[df[COUNTRY_COL].isin(top)].pivot_table(
            index=CATEGORY_COL, columns=COUNTRY_COL, values="Sales", aggfunc="mean"
        )
        run_step(
            "sales heatmap",
            plot_heatmap,
            pivot,
            f"Average Sales by Category and Country (Top {TOP_COUNTRIES} Countries)",
            "04_heatmap_sales_pivot.png",
        )
        run_step("choropleth", plot_choropleth, df, "05_choropleth_sales_by_country.html")

    if REVIEW_COL in df.columns:
        run_step("word cloud", plot_wordcloud, df[REVIEW_COL], "06_wordcloud_reviews.png")

    run_step("dendrogram", plot_dendrogram, df, "07_dendrogram_categories.png")
    run_step("scatter plot", plot_scatter, df, "Price", "Sales", "08_scatter_price_vs_sales.png")
    run_step("correlation heatmap", plot_correlation, df[NUMERIC_COLS], "09_correlation_heatmap.png")
    run_step("pair plot", plot_pairplot, df, "10_pairplot.png")
    run_step("parallel coordinates", plot_parallel, df, "11_parallel_coordinates.png")
    run_step("andrews curves", plot_andrews, df, "12_andrews_curves.png")
    run_step("sales boxplot", plot_boxplot, df, CATEGORY_COL, "Sales", "13_boxplot_sales_by_category.png")

    if DATE_COL in df.columns:
        run_step("monthly trend", plot_monthly_trend, df, "14_monthly_sales_trend.png")

    print(f"\nAll available plots saved to the {PLOT_DIR}/ folder.")


if __name__ == "__main__":
    main()
