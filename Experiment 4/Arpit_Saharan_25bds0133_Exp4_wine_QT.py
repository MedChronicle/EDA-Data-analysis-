import os
from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set(style="ticks", color_codes=True)

DATA_PATH = "WineQT.csv"
PLOTS_DIR = "plots"
ID_COLUMN = "Id"
TARGET_COLUMN = "quality"
SEPARATOR_LINE = "=" * 60
PLOT_DPI = 150
BAR_COLOR = "steelblue"
SCATTER_COLOR = "mediumvioletred"
HEATMAP_COLORS = "rocket"
LOWER_QUANTILE = 0.25
UPPER_QUANTILE = 0.75
UNIVARIATE_COLUMNS = ["alcohol", "pH"]
PAIRPLOT_REG_COLUMNS = ["alcohol", "volatile acidity", "quality"]
PAIRPLOT_HUE_COLUMNS = ["citric acid", "sulphates", "alcohol"]
PEARSON_FEATURES = ["alcohol", "volatile acidity"]
COMPLETION_MESSAGE = "\n All plots saved to the 'plots' directory."


@dataclass
class DispersionSummary:
    column: str
    mean: float
    median: float
    mode: float
    std: float
    variance: float
    q1: float
    q3: float
    iqr: float


def save_figure(fig, filename: str) -> None:
    output_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def ensure_plots_dir(path: str = PLOTS_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    has_id_column = ID_COLUMN in df.columns
    if has_id_column:
        df = df.drop(columns=[ID_COLUMN])
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    row_count = df.shape[0]
    column_count = df.shape[1]
    preview_rows = 3

    print(SEPARATOR_LINE)
    print("DATASET OVERVIEW")
    print(SEPARATOR_LINE)
    print(f"\nShape: {row_count} rows x {column_count} columns\n")
    print("First 3 rows:")
    print(df.head(preview_rows))
    print("\nLast 3 rows:")
    print(df.tail(preview_rows))
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nDescriptive statistics:")
    print(df.describe())


def central_tendency_and_dispersion(df: pd.DataFrame, column: str) -> DispersionSummary:
    series = df[column]
    q1 = series.quantile(LOWER_QUANTILE)
    q3 = series.quantile(UPPER_QUANTILE)
    iqr = q3 - q1
    mode_value = series.mode().iloc[0]

    summary = DispersionSummary(
        column=column,
        mean=series.mean(),
        median=series.median(),
        mode=mode_value,
        std=series.std(),
        variance=series.var(),
        q1=q1,
        q3=q3,
        iqr=iqr,
    )

    header_message = f"\nCentral tendency & dispersion for '{column}':"
    print(header_message)
    for field, value in summary.__dict__.items():
        if field != "column":
            line_message = f"  {field:>10}: {value:.4f}"
            print(line_message)
    return summary


def pearson_test(df: pd.DataFrame, x: str, y: str):
    coefficient, p_value = stats.pearsonr(df[x], df[y])
    header_message = f"\nPearson correlation between '{x}' and '{y}':"
    coefficient_message = f"  coefficient: {coefficient:.4f}"
    p_value_message = f"  p-value: {p_value:.4e}"
    print(header_message)
    print(coefficient_message)
    print(p_value_message)
    return coefficient, p_value


def plot_distribution(df: pd.DataFrame, column: str, filename: str) -> None:
    figure_size = (8, 5)
    fig, ax = plt.subplots(figsize=figure_size)
    sns.histplot(df[column], kde=True, ax=ax, color=BAR_COLOR)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_quality_counts(df: pd.DataFrame, filename: str = "quality_counts.png") -> None:
    figure_size = (8, 5)
    quality_counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=figure_size)
    quality_counts.plot(kind="bar", ax=ax, color=BAR_COLOR)
    ax.set_title("Number of Wine Samples by Quality Score")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Number of Samples")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_boxplot(df: pd.DataFrame, column: str, filename: str) -> None:
    figure_size = (6, 4)
    fig, ax = plt.subplots(figsize=figure_size)
    sns.boxplot(x=df[column], ax=ax, color=BAR_COLOR)
    ax.set_title(f"Boxplot of {column}")
    fig.tight_layout()
    save_figure(fig, filename)


def univariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + SEPARATOR_LINE)
    print("UNIVARIATE ANALYSIS")
    print(SEPARATOR_LINE)
    for col in UNIVARIATE_COLUMNS:
        distribution_filename = f"dist_{col}.png"
        central_tendency_and_dispersion(df, col)
        plot_distribution(df, col, distribution_filename)
    plot_quality_counts(df)
    plot_boxplot(df, "alcohol", "box_alcohol.png")


def plot_scatter(df: pd.DataFrame, x: str, y: str, filename: str) -> None:
    figure_size = (7, 5)
    point_transparency = 0.6
    fig, ax = plt.subplots(figsize=figure_size)
    ax.scatter(df[x], df[y], alpha=point_transparency, color=SCATTER_COLOR)
    ax.set_title(f"Scatter Plot: {x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_grouped_boxplot(df: pd.DataFrame, group_col: str, value_col: str, filename: str) -> None:
    figure_size = (8, 5)
    fig, ax = plt.subplots(figsize=figure_size)
    sns.boxplot(x=group_col, y=value_col, data=df, ax=ax)
    ax.set_title(f"{value_col} distribution across {group_col}")
    fig.tight_layout()
    save_figure(fig, filename)


def bivariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + SEPARATOR_LINE)
    print("BIVARIATE ANALYSIS")
    print(SEPARATOR_LINE)
    plot_scatter(df, "alcohol", TARGET_COLUMN, "scatter_alcohol_quality.png")
    plot_scatter(df, "volatile acidity", TARGET_COLUMN, "scatter_volacid_quality.png")
    plot_grouped_boxplot(df, TARGET_COLUMN, "alcohol", "box_alcohol_by_quality.png")
    plot_grouped_boxplot(df, TARGET_COLUMN, "volatile acidity", "box_volacid_by_quality.png")


def plot_pairplot(df: pd.DataFrame, columns: Sequence[str], filename: str, hue: Optional[str] = None) -> None:
    plot_height = 2.2
    plot_kind = "reg" if hue is None else "scatter"
    grid = sns.pairplot(df, vars=list(columns), hue=hue, kind=plot_kind, height=plot_height)
    output_path = os.path.join(PLOTS_DIR, filename)
    grid.savefig(output_path, dpi=PLOT_DPI)
    plt.close("all")


def correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> pd.DataFrame:
    figure_size = (10, 8)
    correlation = df.corr(method="pearson", numeric_only=True)
    fig, ax = plt.subplots(figsize=figure_size)
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns,
                cmap=HEATMAP_COLORS, annot=False, ax=ax)
    ax.set_title("Pearson Correlation Heatmap")
    fig.tight_layout()
    save_figure(fig, filename)
    return correlation


def multivariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + SEPARATOR_LINE)
    print("MULTIVARIATE ANALYSIS")
    print(SEPARATOR_LINE)
    plot_pairplot(df, PAIRPLOT_REG_COLUMNS, "pairplot_reg.png")
    plot_pairplot(df, PAIRPLOT_HUE_COLUMNS, "pairplot_quality_hue.png", hue=TARGET_COLUMN)
    correlation = correlation_heatmap(df)
    for feature in PEARSON_FEATURES:
        pearson_test(df, feature, TARGET_COLUMN)
    quality_correlation = correlation[TARGET_COLUMN].drop(TARGET_COLUMN)
    top_correlated_with_quality = quality_correlation.sort_values(key=np.abs, ascending=False)
    print("\nFeatures most correlated with wine quality:")
    print(top_correlated_with_quality)


def main() -> None:
    ensure_plots_dir()
    df = load_dataset()
    inspect_dataset(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)
    print(COMPLETION_MESSAGE)


if __name__ == "__main__":
    main()
