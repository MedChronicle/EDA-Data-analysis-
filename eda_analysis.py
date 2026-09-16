import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates, andrews_curves

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("ecommerce_sales_data.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe(include="all"))
print("\nMissing values:")
print(df.isna().sum())

df["Rating"] = df["Rating"].fillna(df["Rating"].median())

numeric_cols = ["Price", "Quantity", "Discount_pct", "Sales", "Rating",
                "Customer_Age", "Delivery_Days"]


def plot_pie(data, labels, title="Pie Chart"):
    plt.figure(figsize=(7, 7))
    plt.pie(data, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=sns.color_palette("Set2", len(data)))
    plt.title(title)
    plt.tight_layout()
    plt.savefig("plots/01_pie_sales_by_category.png")
    plt.close()


sales_by_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
plot_pie(sales_by_category.values, sales_by_category.index, title="Total Sales Share by Category")


def plot_bar(categories, values, title="Bar Chart"):
    plt.figure(figsize=(9, 6))
    sns.barplot(x=list(categories), y=list(values), hue=list(categories),
                palette="viridis", legend=False)
    plt.title(title)
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Average Rating")
    plt.tight_layout()
    plt.savefig("plots/02_bar_avg_rating_by_category.png")
    plt.close()


avg_rating_by_category = df.groupby("Category")["Rating"].mean().sort_values(ascending=False)
plot_bar(avg_rating_by_category.index, avg_rating_by_category.values,
          title="Average Rating by Category")


def plot_histogram(df, column):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=25, kde=True, color="teal")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig("plots/03_histogram_price.png")
    plt.close()


plot_histogram(df, "Price")


def heatmap(data, annot=True, cmap="coolwarm"):
    plt.figure(figsize=(9, 6))
    sns.heatmap(data, annot=annot, cmap=cmap, fmt=".0f")
    plt.title("Average Sales by Category and Country (Top 6 Countries)")
    plt.tight_layout()
    plt.savefig("plots/04_heatmap_sales_pivot.png")
    plt.close()


top_countries = df["Country"].value_counts().head(6).index
pivot = df[df["Country"].isin(top_countries)].pivot_table(
    index="Category", columns="Country", values="Sales", aggfunc="mean"
)
heatmap(pivot, annot=True, cmap="coolwarm")

sales_by_country = df.groupby("Country")["Sales"].sum().reset_index()
fig = px.choropleth(
    sales_by_country,
    locations="Country",
    locationmode="country names",
    color="Sales",
    color_continuous_scale="Plasma",
    title="Total Sales by Country",
)
fig.write_html("plots/05_choropleth_sales_by_country.html")

text = " ".join(df["Review"].tolist())
wc = WordCloud(width=1000, height=600, background_color="white",
               colormap="viridis").generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Customer Reviews")
plt.tight_layout()
plt.savefig("plots/06_wordcloud_reviews.png")
plt.close()

cluster_data = df.groupby("Category")[numeric_cols].mean()
Z = linkage(cluster_data, method="ward")
plt.figure(figsize=(9, 6))
dendrogram(Z, labels=cluster_data.index.tolist(), leaf_rotation=45)
plt.title("Dendrogram of Product Categories (by average numeric profile)")
plt.tight_layout()
plt.savefig("plots/07_dendrogram_categories.png")
plt.close()


def scatter(x, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=df["Category"], palette="tab10", alpha=0.8)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title(f"{y.name} vs {x.name}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/08_scatter_price_vs_sales.png")
    plt.close()


scatter(df["Price"], df["Sales"])

plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("plots/09_correlation_heatmap.png")
plt.close()

pairplot_sample = df.sample(200, random_state=1)
g = sns.pairplot(pairplot_sample[numeric_cols + ["Category"]], hue="Category",
                  palette="tab10", corner=True, plot_kws={"alpha": 0.6, "s": 20})
g.fig.suptitle("Pair Plot of Numeric Features by Category", y=1.02)
g.savefig("plots/10_pairplot.png")
plt.close()

plt.figure(figsize=(10, 6))
parallel_data = df.sample(150, random_state=2)[numeric_cols + ["Category"]].copy()
for col in numeric_cols:
    parallel_data[col] = (parallel_data[col] - parallel_data[col].min()) / (
        parallel_data[col].max() - parallel_data[col].min()
    )
parallel_coordinates(parallel_data, "Category", color=sns.color_palette("tab10", 8))
plt.title("Parallel Coordinates Plot (normalized features)")
plt.xticks(rotation=30)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("plots/11_parallel_coordinates.png")
plt.close()

plt.figure(figsize=(10, 6))
andrews_data = df.sample(150, random_state=3)[numeric_cols + ["Category"]]
andrews_curves(andrews_data, "Category", colormap="tab10")
plt.title("Andrews Curves by Category")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("plots/12_andrews_curves.png")
plt.close()

print("\nAll plots saved to the plots/ folder.")
