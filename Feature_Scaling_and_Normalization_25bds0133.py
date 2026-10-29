import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

INPUT_PATH = "fraudTrain.csv"
OUTPUT_PATH = "fraudTrain_scaled.csv"
CANDIDATE_COLS = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long"]
TARGET_COL = "is_fraud"

SCALERS = {
    "minmax": MinMaxScaler(),
    "standard": StandardScaler(),
    "robust": RobustScaler(),
}

df = pd.read_csv(INPUT_PATH)

numeric_cols = [c for c in CANDIDATE_COLS if c in df.columns]
if not numeric_cols:
    raise ValueError(f"None of the expected columns were found: {CANDIDATE_COLS}")

features = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
missing = int(features.isna().sum().sum())
if missing:
    print(f"Filling {missing} missing value(s) with column medians")
    features = features.fillna(features.median())

print("Numeric columns used:", numeric_cols)
print("\nOriginal data (first 5 rows):")
print(features.head())

scaled_frames = []
for name, scaler in SCALERS.items():
    scaler.set_output(transform="pandas")
    scaled = scaler.fit_transform(features).add_suffix(f"_{name}")
    scaled_frames.append(scaled)

    print(f"\n{name.capitalize()} scaled data (first 5 rows):")
    print(scaled.head())
    print(f"\n{name.capitalize()} scaling summary:")
    print(scaled.agg(["min", "max", "mean", "std"]).round(3))

result = pd.concat([features, *scaled_frames], axis=1)
if TARGET_COL in df.columns:
    result.insert(0, TARGET_COL, df[TARGET_COL])

result.to_csv(OUTPUT_PATH, index=False, float_format="%.6f")
print(f"\nSaved {result.shape[1]} columns and {result.shape[0]} rows to {OUTPUT_PATH}")
