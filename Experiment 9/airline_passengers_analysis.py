from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
RAW_FILE = OUT_DIR / "airline-passengers.csv"
HOLDOUT = 12
FORECAST_HORIZON = 24
SEED = 42

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.figsize": (14, 6), "figure.dpi": 110, "savefig.dpi": 150})


def save(name, fig):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def generate_data(n_months=144, seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1949-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    trend = np.linspace(100, 500, n_months)
    season = 50 * np.sin(2 * np.pi * t / 12) * np.linspace(1.0, 1.8, n_months)
    noise = rng.normal(0, 15, n_months)
    passengers = np.clip(trend + season + noise, 50, None).round().astype(int)
    return pd.DataFrame({"Date": dates, "Passengers": passengers})


def load_data():
    if not RAW_FILE.exists():
        generate_data().to_csv(RAW_FILE, index=False)
        print("generated synthetic data ->", RAW_FILE)
    df = pd.read_csv(RAW_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    df = df.asfreq("MS")
    missing = int(df["Passengers"].isna().sum())
    if missing:
        print(f"interpolating {missing} missing months")
        df["Passengers"] = df["Passengers"].interpolate(method="time")
    return df


def decompose(series, period=12):
    trend = series.rolling(period, center=True).mean().rolling(2, center=True).mean().shift(-1)
    ratio = series / trend
    index = ratio.groupby(series.index.month).mean()
    index = index / index.mean()
    seasonal = pd.Series(index.reindex(series.index.month).values, index=series.index)
    residual = series / (trend * seasonal)
    return trend, seasonal, residual, index


def forecast_trend_seasonal(train, horizon, period=12):
    trend, _, _, index = decompose(train, period)
    valid = trend.dropna()
    x = np.arange(len(train))[train.index.get_indexer(valid.index)]
    slope, intercept = np.polyfit(x, valid.values, 1)
    future_idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    future_x = np.arange(len(train), len(train) + horizon)
    return pd.Series((intercept + slope * future_x) * index.reindex(future_idx.month).values, index=future_idx)


def forecast_seasonal_naive(train, horizon, period=12):
    future_idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    last = train.iloc[-period:].values
    return pd.Series(np.resize(last, horizon), index=future_idx)


def metrics(actual, pred):
    err = actual.values - pred.values
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAPE_%": float(np.abs(err / actual.values).mean() * 100)}


df = load_data()
s = df["Passengers"]

print(df.shape)
print(df.head())
print(df.dtypes)
print(df.tail(3))
print(df.sample(5, random_state=0).sort_index())
print(df.loc["1955-06"])
print(df.loc["1955"].describe().T)
print("\nSummary:")
print(s.describe().round(2))
print("Skew:", round(s.skew(), 3), "| Autocorr lag-1:", round(s.autocorr(1), 3),
      "| Autocorr lag-12:", round(s.autocorr(12), 3))

df["Rolling_Mean_12M"] = s.rolling(12).mean()
df["Rolling_Std_12M"] = s.rolling(12).std()
df["YoY_Growth_%"] = s.pct_change(12) * 100
df["MoM_Growth_%"] = s.pct_change() * 100
df["Cumulative_Passengers"] = s.cumsum()
df["Year"], df["Month"] = df.index.year, df.index.month

fig, ax = plt.subplots()
ax.plot(s.index, s, color="steelblue", lw=1.5)
peak = s.idxmax()
ax.scatter([peak], [s.max()], color="crimson", zorder=3)
ax.annotate(f"Peak: {s.max()}", (peak, s.max()), textcoords="offset points", xytext=(-40, 8), color="crimson")
ax.set(title=f"Monthly Airline Passengers ({s.index.year.min()}-{s.index.year.max()})",
       xlabel="Date", ylabel="Number of Passengers")
save("passengers_timeseries.png", fig)

fig, ax = plt.subplots()
ax.plot(s.index, s, alpha=0.5, label="Monthly Passengers")
ax.plot(df.index, df["Rolling_Mean_12M"], color="crimson", lw=2, label="12-Month Rolling Mean")
ax.fill_between(df.index, df["Rolling_Mean_12M"] - df["Rolling_Std_12M"],
                df["Rolling_Mean_12M"] + df["Rolling_Std_12M"], color="crimson", alpha=0.12, label="±1 rolling std")
ax.set(title="Airline Passengers with 12-Month Rolling Mean (Trend)", xlabel="Date", ylabel="Number of Passengers")
ax.legend()
save("passengers_trend.png", fig)

yearly = s.groupby(s.index.year).agg(["mean", "min", "max"])
fig, ax = plt.subplots()
bars = ax.bar(yearly.index, yearly["mean"], color="mediumseagreen")
ax.errorbar(yearly.index, yearly["mean"], yerr=[yearly["mean"] - yearly["min"], yearly["max"] - yearly["mean"]],
            fmt="none", ecolor="black", capsize=4, lw=1)
ax.bar_label(bars, fmt="%.0f", padding=14, fontsize=8)
ax.set_xticks(yearly.index)
ax.set(title="Average Yearly Passenger Count (whiskers = monthly min/max)", xlabel="Year", ylabel="Average Passengers")
save("passengers_yearly_avg.png", fig)

fig, ax = plt.subplots()
ax.plot(df.index, df["Cumulative_Passengers"], color="darkorange", lw=2)
ax.set(title="Cumulative Sum of Airline Passengers Over Time", xlabel="Date", ylabel="Cumulative Passengers")
save("passengers_cumulative.png", fig)

trend, seasonal, residual, season_index = decompose(s)
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
for ax, (name, data, color) in zip(axes, [("Observed", s, "steelblue"), ("Trend", trend, "crimson"),
                                           ("Seasonal index", seasonal, "seagreen"),
                                           ("Residual ratio", residual, "grey")]):
    ax.plot(data.index, data, color=color)
    ax.set_ylabel(name)
axes[2].axhline(1, color="black", lw=0.8, ls="--")
axes[3].axhline(1, color="black", lw=0.8, ls="--")
axes[0].set_title("Multiplicative Decomposition (classical)")
save("passengers_decomposition.png", fig)

pivot = df.pivot_table(index="Year", columns="Month", values="Passengers")
pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.4, ax=ax)
ax.set_title("Passengers by Year and Month")
save("passengers_seasonal_heatmap.png", fig)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=df, x="Month", y="Passengers", hue="Month", palette="coolwarm", legend=False, ax=axes[0])
axes[0].set(title="Monthly Distribution (all years)", xticks=range(12),
            xticklabels=[pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)])
axes[1].bar(season_index.index, season_index.values, color="seagreen")
axes[1].axhline(1, color="black", lw=0.8, ls="--")
axes[1].set(title="Seasonal Index by Month (1.0 = average)", xlabel="Month", xticks=range(1, 13))
save("passengers_seasonality.png", fig)

fig, ax = plt.subplots()
ax.plot(df.index, df["YoY_Growth_%"], color="purple")
ax.axhline(0, color="black", lw=0.8)
ax.set(title="Year-over-Year Growth", xlabel="Date", ylabel="Growth (%)")
save("passengers_yoy_growth.png", fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pd.plotting.autocorrelation_plot(s, ax=axes[0])
axes[0].set_title("Autocorrelation")
lags = range(1, 37)
axes[1].bar(lags, [s.autocorr(l) for l in lags], color="steelblue")
axes[1].axvline(12, color="crimson", ls="--", lw=1)
axes[1].set(title="Autocorrelation by Lag (spike at 12 = yearly seasonality)", xlabel="Lag (months)")
save("passengers_autocorrelation.png", fig)

train, test = s.iloc[:-HOLDOUT], s.iloc[-HOLDOUT:]
pred_ts = forecast_trend_seasonal(train, HOLDOUT)
pred_naive = forecast_seasonal_naive(train, HOLDOUT)
scores = pd.DataFrame({"Trend x Seasonal": metrics(test, pred_ts),
                       "Seasonal Naive": metrics(test, pred_naive)}).T.round(2)
print(f"\nHoldout evaluation (last {HOLDOUT} months):")
print(scores)

future = forecast_trend_seasonal(s, FORECAST_HORIZON)
resid_std = float(np.log(residual.dropna()).std())
band = pd.DataFrame({"lo": future * np.exp(-1.96 * resid_std), "hi": future * np.exp(1.96 * resid_std)})

fig, ax = plt.subplots()
ax.plot(s.index, s, color="steelblue", label="Observed")
ax.plot(pred_ts.index, pred_ts, color="darkorange", lw=2, label="Holdout forecast (trend x seasonal)")
ax.plot(pred_naive.index, pred_naive, color="grey", ls=":", label="Holdout forecast (seasonal naive)")
ax.plot(future.index, future, color="crimson", lw=2, label=f"{FORECAST_HORIZON}-month forecast")
ax.fill_between(future.index, band["lo"], band["hi"], color="crimson", alpha=0.15, label="~95% band")
ax.axvline(train.index[-1], color="black", ls="--", lw=0.8)
ax.set(title=f"Forecast (holdout MAPE {scores.loc['Trend x Seasonal', 'MAPE_%']:.1f}% vs naive "
             f"{scores.loc['Seasonal Naive', 'MAPE_%']:.1f}%)", xlabel="Date", ylabel="Number of Passengers")
ax.legend()
save("passengers_forecast.png", fig)

df.to_csv(OUT_DIR / "airline-passengers-analyzed.csv")
future.rename("Forecast").to_frame().join(band).round(1).to_csv(OUT_DIR / "airline-passengers-forecast.csv")
scores.to_csv(OUT_DIR / "forecast_metrics.csv")
print("\nAll outputs saved to", OUT_DIR.resolve())
