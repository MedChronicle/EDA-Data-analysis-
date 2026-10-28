import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import andrews_curves, parallel_coordinates
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from wordcloud import STOPWORDS, WordCloud
except ImportError:
    WordCloud = None
    STOPWORDS = set()

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})

DATA_FILE = "ecommerce_sales_data.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["Price", "Quantity", "Discount_pct", "Sales", "Rating", "Customer_Age", "Delivery_Days"]
STOP_EXTRA = {"product", "item", "one", "got", "get", "would", "also", "really", "very", "just"}


def save(name, fig=None):
    (fig or plt.gcf()).savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close("all")
    print("saved", name)


def palette(n):
    return sns.color_palette("tab10" if n <= 10 else "husl", n)


def zscore(frame):
    return (frame - frame.mean()) / frame.std(ddof=0).replace(0, 1)


df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]

print("Shape:", df.shape)
print("\nInfo:")
df.info()
print("\nDescribe:")
print(df.describe(include="all").T)
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].fillna("" if c == "Review" else "Unknown").astype(str).str.strip()

n_cat = df["Category"].nunique()
cat_colors = palette(n_cat)


def plot_pie(data, labels, title="Pie Chart", filename="01_pie_sales_by_category.png", max_slices=8):
    data, labels = pd.Series(data).reset_index(drop=True), pd.Series(labels).reset_index(drop=True)
    if len(data) > max_slices:
        data = pd.concat([data.head(max_slices - 1), pd.Series([data.iloc[max_slices - 1:].sum()])], ignore_index=True)
        labels = pd.concat([labels.head(max_slices - 1), pd.Series(["Other"])], ignore_index=True)
    plt.figure(figsize=(7.5, 7.5))
    plt.pie(data, labels=labels, autopct="%1.1f%%", startangle=90, pctdistance=0.8,
            colors=sns.color_palette("Set2", len(data)), wedgeprops={"width": 0.45, "edgecolor": "white"})
    plt.title(title)
    save(filename)


sales_by_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
plot_pie(sales_by_category.values, sales_by_category.index, title="Total Sales Share by Category")


def plot_bar(data, x, y, title="Bar Chart", filename="02_bar_avg_rating_by_category.png", xlabel=None):
    order = data.groupby(x)[y].mean().sort_values(ascending=False).index
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=data, y=x, x=y, order=order, hue=x, palette="viridis", legend=False,
                     errorbar="se", capsize=0.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    plt.xlabel(xlabel or f"Average {y}")
    plt.title(title)
    save(filename)


plot_bar(df, "Category", "Rating", title="Average Rating by Category (± standard error)")


def plot_histogram(data, column, filename=None):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], bins=25, kde=True, color="teal")
    plt.axvline(data[column].mean(), color="crimson", ls="--", label=f"mean {data[column].mean():.1f}")
    plt.axvline(data[column].median(), color="orange", ls="--", label=f"median {data[column].median():.1f}")
    plt.title(f"Distribution of {column} (skew {data[column].skew():.2f})")
    plt.legend()
    save(filename or f"03_histogram_{column.lower()}.png")


for col in ("Price", "Sales"):
    if col in df:
        plot_histogram(df, col)


def plot_distribution_grid(data, columns, filename="03_distribution_grid.png"):
    cols = 3
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, columns):
        sns.histplot(data[c], bins=25, kde=True, ax=ax, color="teal")
        ax.set_title(f"{c} (skew {data[c].skew():.2f})")
    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.suptitle("Numeric Feature Distributions", y=1.01)
    save(filename, fig)


plot_distribution_grid(df, numeric_cols)


def plot_boxplot(data, x, y, filename="04_boxplot_sales_by_category.png"):
    order = data.groupby(x)[y].median().sort_values(ascending=False).index
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y, order=order, hue=x, palette="Set2", legend=False, showfliers=False)
    if data[y].skew() > 2:
        plt.yscale("log")
    plt.xticks(rotation=40, ha="right")
    plt.title(f"{y} by {x} (ordered by median)")
    save(filename)


plot_boxplot(df, "Category", "Sales")


def heatmap(data, annot=True, cmap="YlOrRd", title="Heatmap", filename="05_heatmap_sales_pivot.png", fmt=".0f"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=annot, cmap=cmap, fmt=fmt, linewidths=0.5)
    plt.title(title)
    save(filename)


top_countries = df["Country"].value_counts().head(6).index
pivot = df[df["Country"].isin(top_countries)].pivot_table(
    index="Category", columns="Country", values="Sales", aggfunc="mean")
heatmap(pivot, title="Average Sales by Category and Country (Top 6 Countries)")

if px is not None:
    sales_by_country = df.groupby("Country")["Sales"].sum().reset_index()
    px.choropleth(sales_by_country, locations="Country", locationmode="country names", color="Sales",
                  color_continuous_scale="Plasma", title="Total Sales by Country"
                  ).write_html(OUT_DIR / "06_choropleth_sales_by_country.html", include_plotlyjs="cdn")
    px.sunburst(df, path=["Category", "Country"], values="Sales", title="Sales: Category to Country"
                ).write_html(OUT_DIR / "06_sunburst_sales.html", include_plotlyjs="cdn")
    print("saved interactive html plots")
else:
    print("plotly not installed, skipped interactive plots")


def plot_wordcloud(reviews, filename="07_wordcloud_reviews.png"):
    tokens = [t for t in re.findall(r"[a-z']{3,}", " ".join(reviews).lower())
              if t not in STOPWORDS and t not in STOP_EXTRA]
    if not tokens:
        print("no review text, skipped word cloud")
        return
    freq = Counter(tokens)
    if WordCloud is not None:
        wc = WordCloud(width=1200, height=650, background_color="white", colormap="viridis")
        plt.figure(figsize=(11, 6))
        plt.imshow(wc.generate_from_frequencies(freq), interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Frequent Words in Customer Reviews")
        save(filename)
    words, counts = zip(*freq.most_common(20))
    plt.figure(figsize=(9, 6))
    sns.barplot(x=list(counts), y=list(words), hue=list(words), palette="mako", legend=False)
    plt.xlabel("Frequency")
    plt.title("Top 20 Review Words")
    save("07_top_review_words.png")


if "Review" in df:
    plot_wordcloud(df["Review"].tolist())

cluster_data = zscore(df.groupby("Category")[numeric_cols].mean())
if len(cluster_data) >= 3:
    plt.figure(figsize=(10, 6))
    dendrogram(linkage(cluster_data, method="ward"), labels=cluster_data.index.tolist(), leaf_rotation=45)
    plt.title("Dendrogram of Product Categories (standardised average profile)")
    save("08_dendrogram_categories.png")
    g = sns.clustermap(cluster_data, cmap="vlag", center=0, annot=True, fmt=".1f", figsize=(10, 8))
    g.fig.suptitle("Category Profiles (z-scores)", y=1.02)
    save("08_clustermap_categories.png", g.fig)


def scatter(x, y, filename="09_scatter_price_vs_sales.png"):
    sample = df.sample(min(3000, len(df)), random_state=1)
    plt.figure(figsize=(9, 6.5))
    sns.scatterplot(data=sample, x=x.name, y=y.name, hue=df.loc[sample.index, "Category"],
                    palette=cat_colors, alpha=0.7, s=28)
    sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"lw": 1.5})
    r, p = stats.spearmanr(x, y)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title(f"{y.name} vs {x.name} (Spearman ρ = {r:.2f}, p = {p:.3g})")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save(filename)


scatter(df["Price"], df["Sales"])

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
for ax, method in zip(axes, ["pearson", "spearman"]):
    corr = df[numeric_cols].corr(method=method)
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(method.capitalize())
fig.suptitle("Correlation Heatmaps of Numeric Features")
save("10_correlation_heatmap.png", fig)

pairplot_sample = df.sample(min(200, len(df)), random_state=1)
g = sns.pairplot(pairplot_sample[numeric_cols + ["Category"]], hue="Category", palette=cat_colors,
                 corner=True, plot_kws={"alpha": 0.6, "s": 20})
g.fig.suptitle("Pair Plot of Numeric Features by Category", y=1.02)
save("11_pairplot.png", g.fig)

parallel_data = df.sample(min(150, len(df)), random_state=2)[numeric_cols + ["Category"]].copy()
span = (parallel_data[numeric_cols].max() - parallel_data[numeric_cols].min()).replace(0, 1)
parallel_data[numeric_cols] = (parallel_data[numeric_cols] - parallel_data[numeric_cols].min()) / span
plt.figure(figsize=(11, 6))
parallel_coordinates(parallel_data, "Category", color=cat_colors, alpha=0.6)
plt.title("Parallel Coordinates Plot (normalised features)")
plt.xticks(rotation=30)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
save("12_parallel_coordinates.png")

andrews_data = df.sample(min(150, len(df)), random_state=3)[numeric_cols + ["Category"]].copy()
andrews_data[numeric_cols] = zscore(andrews_data[numeric_cols])
plt.figure(figsize=(11, 6))
andrews_curves(andrews_data, "Category", color=cat_colors, alpha=0.6)
plt.title("Andrews Curves by Category (standardised)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
save("13_andrews_curves.png")

print("\nAll outputs saved to the plots/ folder.")
