import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT_PATH = "fraudTrain.csv"
FLAGGED_PATH = "fraudTrain_flagged.csv"
OUTLIERS_PATH = "fraudTrain_outliers.csv"
CLEANED_PATH = "fraudTrain_cleaned.csv"
AMOUNT_COL = "amt"
TARGET_COL = "is_fraud"
GROUP_COL = "category"
IQR_FACTOR = 1.5
Z_THRESHOLD = 3.0
MODIFIED_Z_THRESHOLD = 3.5
CONTAMINATION = 0.01
WINSOR_LIMITS = (0.01, 0.99)
MIN_VOTES = 3
PROTECT_FRAUD = True
RANDOM_STATE = 42


def iqr_bounds(series, factor=IQR_FACTOR):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def iqr_flags(series, factor=IQR_FACTOR):
    lower, upper = iqr_bounds(series, factor)
    return (series < lower) | (series > upper)


def grouped_iqr_flags(df, column, group, factor=IQR_FACTOR):
    grouped = df.groupby(group)[column]
    q1 = df[group].map(grouped.quantile(0.25))
    q3 = df[group].map(grouped.quantile(0.75))
    iqr = q3 - q1
    return (df[column] < q1 - factor * iqr) | (df[column] > q3 + factor * iqr)


def zscore(series):
    return (series - series.mean()) / series.std()


def modified_zscore(series):
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median) / mad


def isolation_flags(values):
    model = IsolationForest(
        contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1
    )
    return pd.Series(model.fit_predict(values) == -1, index=values.index)


def summarize(df, flag_cols):
    has_target = TARGET_COL in df.columns
    total_fraud = df[TARGET_COL].sum() if has_target else 0
    rows = []
    for col in flag_cols:
        mask = df[col]
        row = {"method": col.removeprefix("flag_"), "flagged": int(mask.sum()),
               "flagged_%": round(mask.mean() * 100, 3)}
        if has_target:
            frauds = int(df.loc[mask, TARGET_COL].sum())
            row["frauds_flagged"] = frauds
            row["fraud_rate_%"] = round(df.loc[mask, TARGET_COL].mean() * 100, 3) if mask.any() else 0.0
            row["share_of_all_fraud_%"] = round(frauds / total_fraud * 100, 2) if total_fraud else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


df = pd.read_csv(INPUT_PATH)
if AMOUNT_COL not in df.columns:
    raise KeyError(f"Column '{AMOUNT_COL}' not found in {INPUT_PATH}")

df[AMOUNT_COL] = pd.to_numeric(df[AMOUNT_COL], errors="coerce")
dropped = int(df[AMOUNT_COL].isna().sum())
if dropped:
    print(f"Dropping {dropped} row(s) with non-numeric '{AMOUNT_COL}'")
    df = df.dropna(subset=[AMOUNT_COL]).reset_index(drop=True)

print("Dataset shape:", df.shape)

amt = df[AMOUNT_COL]
log_amt = np.log1p(amt.clip(lower=0))

lower, upper = iqr_bounds(amt)
log_lower, log_upper = iqr_bounds(log_amt)
print(f"\nIQR bounds on raw amount:  [{lower:.2f}, {upper:.2f}]")
print(f"IQR bounds on log amount:  [{np.expm1(log_lower):.2f}, {np.expm1(log_upper):.2f}] (back-transformed)")
print(f"Skewness raw: {amt.skew():.2f} | log: {log_amt.skew():.2f}")

df["amt_log"] = log_amt
df["amt_zscore"] = zscore(amt)
df["amt_modified_zscore"] = modified_zscore(amt)
df["amt_winsorized"] = amt.clip(*amt.quantile(WINSOR_LIMITS))

df["flag_iqr"] = iqr_flags(amt)
df["flag_iqr_log"] = iqr_flags(log_amt)
df["flag_zscore"] = df["amt_zscore"].abs() > Z_THRESHOLD
df["flag_modified_zscore"] = df["amt_modified_zscore"].abs() > MODIFIED_Z_THRESHOLD
df["flag_isolation_forest"] = isolation_flags(log_amt.to_frame())
if GROUP_COL in df.columns:
    df["flag_iqr_by_category"] = grouped_iqr_flags(df, AMOUNT_COL, GROUP_COL)

flag_cols = [c for c in df.columns if c.startswith("flag_")]
df["outlier_votes"] = df[flag_cols].sum(axis=1)
df["is_consensus_outlier"] = df["outlier_votes"] >= MIN_VOTES

print("\nOutlier detection summary by method:")
print(summarize(df, flag_cols + ["is_consensus_outlier"]).to_string())

print(f"\nVote distribution (methods agreeing, out of {len(flag_cols)}):")
print(df["outlier_votes"].value_counts().sort_index().to_string())

print(f"\nTop 10 transactions by outlier votes and amount:")
top_cols = [c for c in [AMOUNT_COL, "amt_modified_zscore", "outlier_votes", GROUP_COL, TARGET_COL] if c in df.columns]
print(df.sort_values(["outlier_votes", AMOUNT_COL], ascending=False)[top_cols].head(10).to_string())

outliers = df[df["is_consensus_outlier"]]
keep = ~df["is_consensus_outlier"]
if PROTECT_FRAUD and TARGET_COL in df.columns:
    keep |= df[TARGET_COL] == 1
cleaned = df[keep]

print(f"\nConsensus outliers (>= {MIN_VOTES} votes): {len(outliers)}")
print(f"Cleaned dataset shape: {cleaned.shape} (removed {len(df) - len(cleaned)} rows)")

df.to_csv(FLAGGED_PATH, index=False)
outliers.to_csv(OUTLIERS_PATH, index=False)
cleaned.to_csv(CLEANED_PATH, index=False)
print(f"\nSaved {FLAGGED_PATH}, {OUTLIERS_PATH} and {CLEANED_PATH}")
