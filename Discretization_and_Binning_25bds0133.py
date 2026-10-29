import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinningConfig:
    amount_col: str = "amt"
    target_col: str = "is_fraud"
    custom_edges: tuple = (0, 10, 50, 100, 500, float("inf"))
    custom_labels: tuple = ("Very Low", "Low", "Medium", "High", "Very High")
    quartile_labels: tuple = ("Q1 (cheapest 25%)", "Q2", "Q3", "Q4 (priciest 25%)")
    equal_width_bins: int = 5
    decile_count: int = 10
    extra_group_cols: list = field(default_factory=lambda: ["category", "gender"])


def load_data(path: Path, amount_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if amount_col not in df.columns:
        raise KeyError(f"Column '{amount_col}' missing from dataset")
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    missing = df[amount_col].isna().sum()
    if missing:
        logger.warning("Dropping %d rows with non-numeric '%s'", missing, amount_col)
        df = df.dropna(subset=[amount_col]).reset_index(drop=True)
    return df


def add_bins(df: pd.DataFrame, cfg: BinningConfig) -> pd.DataFrame:
    amt = df[cfg.amount_col]
    df["amt_bin"] = pd.cut(
        amt, bins=list(cfg.custom_edges), labels=list(cfg.custom_labels), include_lowest=True
    )
    df["amt_quartile"] = pd.qcut(amt, q=4, labels=list(cfg.quartile_labels))
    df["amt_equal_width"] = pd.cut(amt, bins=cfg.equal_width_bins, precision=2)
    df["amt_decile"] = pd.qcut(
        amt, q=cfg.decile_count, labels=False, duplicates="drop"
    ).astype("Int64") + 1
    df["amt_log"] = np.log1p(amt)
    return df


def describe_amount(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col]
    stats = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("value")
    stats.loc["skew"] = s.skew()
    stats.loc["kurtosis"] = s.kurt()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    stats.loc["iqr_upper_fence"] = q3 + 1.5 * iqr
    stats.loc["outlier_share_%"] = (s > q3 + 1.5 * iqr).mean() * 100
    return stats.round(3)


def count_table(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    counts = df[bin_col].value_counts(dropna=False).sort_index()
    out = counts.to_frame("count")
    out["share_%"] = (counts / counts.sum() * 100).round(2)
    out["cumulative_%"] = out["share_%"].cumsum().round(2)
    return out


def fraud_table(df: pd.DataFrame, bin_col: str, target: str) -> pd.DataFrame:
    grouped = df.groupby(bin_col, observed=True)[target]
    out = grouped.agg(transactions="count", frauds="sum", fraud_rate_pct="mean")
    out["fraud_rate_pct"] = (out["fraud_rate_pct"] * 100).round(3)
    total_frauds = out["frauds"].sum()
    out["share_of_all_fraud_%"] = (out["frauds"] / total_frauds * 100).round(2)
    baseline = df[target].mean()
    out["lift"] = (out["fraud_rate_pct"] / 100 / baseline).round(2)
    return out


def segment_table(df: pd.DataFrame, group_col: str, bin_col: str, target: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=group_col, columns=bin_col, values=target, aggfunc="mean", observed=True
    )
    return (pivot * 100).round(3)


def bin_summary(df: pd.DataFrame, bin_col: str, amount_col: str) -> pd.DataFrame:
    return (
        df.groupby(bin_col, observed=True)[amount_col]
        .agg(["min", "median", "mean", "max", "std", "sum"])
        .round(2)
    )


def report(title: str, table: Optional[pd.DataFrame]) -> None:
    if table is None or table.empty:
        return
    print(f"\n{title}")
    print("-" * len(title))
    print(table.to_string())


def run(input_path: Path, output_path: Path, cfg: BinningConfig) -> pd.DataFrame:
    df = load_data(input_path, cfg.amount_col)
    logger.info("Loaded dataset with shape %s", df.shape)

    df = add_bins(df, cfg)
    has_target = cfg.target_col in df.columns

    report("Amount statistics", describe_amount(df, cfg.amount_col))

    for title, col in [
        ("Custom amount bins", "amt_bin"),
        ("Amount quartiles", "amt_quartile"),
        ("Equal-width bins", "amt_equal_width"),
        ("Amount deciles", "amt_decile"),
    ]:
        report(f"{title}: distribution", count_table(df, col))

    report("Custom bins: amount statistics", bin_summary(df, "amt_bin", cfg.amount_col))

    if has_target:
        for title, col in [
            ("Custom bins", "amt_bin"),
            ("Quartiles", "amt_quartile"),
            ("Deciles", "amt_decile"),
        ]:
            report(f"{title}: fraud analysis", fraud_table(df, col, cfg.target_col))

        for group_col in cfg.extra_group_cols:
            if group_col in df.columns:
                report(
                    f"Fraud rate (%) by {group_col} and custom amount bin",
                    segment_table(df, group_col, "amt_bin", cfg.target_col),
                )
    else:
        logger.warning("Target column '%s' not found; skipping fraud analysis", cfg.target_col)

    df.to_csv(output_path, index=False)
    logger.info("Saved binned dataset to %s", output_path)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bin transaction amounts and analyse fraud rates.")
    parser.add_argument("-i", "--input", type=Path, default=Path("fraudTrain.csv"))
    parser.add_argument("-o", "--output", type=Path, default=Path("fraudTrain_binned.csv"))
    parser.add_argument("--amount-col", default="amt")
    parser.add_argument("--target-col", default="is_fraud")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BinningConfig(amount_col=args.amount_col, target_col=args.target_col)
    run(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
