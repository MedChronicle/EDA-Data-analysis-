import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates, andrews_curves

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"

OUT = "outputs"

df = pd.read_csv("data/security_incidents.csv", parse_dates=["date"])


def plot_pie(data, labels, title="Pie Chart", filename="pie_chart.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("viridis", len(labels))
    ax.pie(data, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=colors, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}", dpi=150)
    plt.close(fig)


def plot_bar(categories, values, title="Bar Chart", filename="bar_chart.png",
             xlabel="", ylabel="Count"):
    fig, ax = plt.subplots()
    colors = sns.color_palette("mako", len(categories))
    ax.bar(categories, values, color=colors, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}", dpi=150)
    plt.close(fig)


def plot_histogram(dataframe, column, title=None, filename=None, bins=30):
    title = title or f"Distribution of {column}"
    filename = filename or f"hist_{column}.png"
    fig, ax = plt.subplots()
    sns.histplot(dataframe[column], bins=bins, kde=True, color="steelblue", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}", dpi=150)
    plt.close(fig)


def heatmap(dataframe, annot=True, cmap="coolwarm", title="Correlation Heatmap",
            filename="correlation_heatmap.png"):
    numeric_df = dataframe.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=annot, cmap=cmap, fmt=".2f",
                linewidths=0.5, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}", dpi=150)
    plt.close(fig)


def scatter(x, y, title="Scatter Plot", xlabel="X", ylabel="Y",
            hue=None, filename="scatter_plot.png"):
    fig, ax = plt.subplots()
    if hue is not None:
        sns.scatterplot(x=x, y=y, hue=hue, palette="viridis", alpha=0.6, ax=ax)
    else:
        sns.scatterplot(x=x, y=y, alpha=0.6, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}", dpi=150)
    plt.close(fig)


def make_choropleth(dataframe):
    country_counts = dataframe.groupby("country").size().reset_index(name="incident_count")
    fig = px.choropleth(
        country_counts,
        locations="country",
        locationmode="country names",
        color="incident_count",
        color_continuous_scale="Reds",
        title="Security Incidents by Country",
    )
    fig.write_html(f"{OUT}/choropleth_map.html")
    try:
        fig.write_image(f"{OUT}/choropleth_map.png", scale=2)
    except Exception as exc:
        print(f"Skipped static choropleth export (Chrome not available): {exc}")


def make_wordcloud(dataframe):
    text = " ".join(dataframe["description"].tolist())
    wc = WordCloud(width=1200, height=700, background_color="black",
                    colormap="viridis", max_words=150).generate(text)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Incident Description Word Cloud", color="white")
    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(f"{OUT}/wordcloud.png", dpi=150, facecolor="black")
    plt.close(fig)


def make_dendrogram(dataframe):
    numeric_cols = ["duration_sec", "bytes_transferred", "packets", "risk_score"]
    grouped = dataframe.groupby("attack_type")[numeric_cols].mean()
    scaled = (grouped - grouped.mean()) / grouped.std()
    Z = linkage(scaled, method="ward")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=grouped.index.tolist(), ax=ax, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Dendrogram: Attack Types by Behavioral Similarity")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(f"{OUT}/dendrogram.png", dpi=150)
    plt.close(fig)


def make_pairplot(dataframe):
    cols = ["duration_sec", "bytes_transferred", "packets", "risk_score", "severity"]
    subset = dataframe[cols].sample(min(500, len(dataframe)), random_state=1)
    g = sns.pairplot(subset, hue="severity", palette="rocket",
                      hue_order=["Low", "Medium", "High", "Critical"],
                      diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
    g.fig.suptitle("Pair Plot of Numeric Features by Severity", y=1.02)
    g.savefig(f"{OUT}/pairplot.png", dpi=150)
    plt.close(g.fig)


def make_parallel_coordinates(dataframe):
    cols = ["duration_sec", "bytes_transferred", "packets", "risk_score", "severity"]
    subset = dataframe[cols].sample(min(300, len(dataframe)), random_state=1).copy()
    for c in ["duration_sec", "bytes_transferred", "packets", "risk_score"]:
        subset[c] = (subset[c] - subset[c].min()) / (subset[c].max() - subset[c].min())
    fig, ax = plt.subplots(figsize=(11, 6))
    parallel_coordinates(subset, "severity", color=sns.color_palette("cool", 4), ax=ax)
    ax.set_title("Parallel Coordinates: Normalized Incident Features by Severity")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(f"{OUT}/parallel_coordinates.png", dpi=150)
    plt.close(fig)


def make_andrews_curves(dataframe):
    cols = ["duration_sec", "bytes_transferred", "packets", "risk_score", "attack_type"]
    subset = dataframe[cols].sample(min(300, len(dataframe)), random_state=1).copy()
    for c in ["duration_sec", "bytes_transferred", "packets", "risk_score"]:
        subset[c] = (subset[c] - subset[c].min()) / (subset[c].max() - subset[c].min())
    fig, ax = plt.subplots(figsize=(11, 6))
    andrews_curves(subset, "attack_type", ax=ax, colormap="tab10")
    ax.set_title("Andrews Curves by Attack Type")
    fig.tight_layout()
    fig.savefig(f"{OUT}/andrews_curves.png", dpi=150)
    plt.close(fig)


def print_summary(dataframe):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {dataframe.shape}")
    print(f"\nColumns:\n{dataframe.dtypes}")
    print(f"\nMissing values:\n{dataframe.isnull().sum()}")
    print(f"\nNumeric summary:\n{dataframe.describe()}")
    print(f"\nAttack type counts:\n{dataframe['attack_type'].value_counts()}")
    print(f"\nSeverity counts:\n{dataframe['severity'].value_counts()}")


if __name__ == "__main__":
    print_summary(df)

    attack_counts = df["attack_type"].value_counts()
    plot_pie(attack_counts.values, attack_counts.index.tolist(),
             title="Share of Incidents by Attack Type")

    severity_counts = df["severity"].value_counts().reindex(["Low", "Medium", "High", "Critical"])
    plot_bar(severity_counts.index.tolist(), severity_counts.values,
              title="Incident Count by Severity Level", xlabel="Severity", ylabel="Number of Incidents")

    plot_histogram(df, "risk_score", title="Distribution of Risk Scores")
    plot_histogram(df, "duration_sec", title="Distribution of Incident Duration (sec)")

    heatmap(df, annot=True, cmap="coolwarm")

    scatter(df["duration_sec"], df["bytes_transferred"],
            title="Duration vs. Bytes Transferred by Severity",
            xlabel="Duration (sec)", ylabel="Bytes Transferred",
            hue=df["severity"])

    make_choropleth(df)
    make_wordcloud(df)
    make_dendrogram(df)
    make_pairplot(df)
    make_parallel_coordinates(df)
    make_andrews_curves(df)

    print("\nAll visualizations saved to the 'outputs' folder.")
