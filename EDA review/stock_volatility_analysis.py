import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
import os
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

FILE = "stock_market_vitality_eda.csv"
TRADING_DAYS = 252
COST_BPS = 10          # one-way trading cost for strategy backtests
SEED = 42

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 11, "axes.labelsize": 9})
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def section(title):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(name)
    plt.show()
    plt.close(fig)


def make_grid(count, cols=5, width=4.0, height=3.2, **kwargs):
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * width, rows * height), **kwargs)
    axes = np.atleast_1d(axes).ravel()
    for extra in axes[count:]:
        extra.axis("off")
    return fig, axes


def ljung_box(series, lags=10):
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    n_obs = len(x)
    x = x - x.mean()
    denom = np.sum(x ** 2)
    acf = np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, lags + 1)])
    q_stat = n_obs * (n_obs + 2) * np.sum(acf ** 2 / (n_obs - np.arange(1, lags + 1)))
    return q_stat, 1 - stats.chi2.cdf(q_stat, lags)


def arch_lm(series, lags=5):
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    sq = (x - x.mean()) ** 2
    y = sq[lags:]
    columns = [np.ones(len(y))]
    for i in range(1, lags + 1):
        columns.append(sq[lags - i:len(sq) - i])
    X = np.column_stack(columns)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    lm_stat = len(y) * (1 - resid.var() / y.var())
    return lm_stat, 1 - stats.chi2.cdf(lm_stat, lags)


def hurst(series, max_lag=100):
    x = np.asarray(series, float)
    lag_range = np.arange(2, max_lag)
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lag_range]
    return np.polyfit(np.log(lag_range), np.log(tau), 1)[0]


def max_drawdown(returns):
    growth = (1 + np.asarray(returns, float) / 100).cumprod()
    running_peak = np.maximum.accumulate(growth)
    return (growth / running_peak - 1).min() * 100


def performance(returns):
    returns = returns.dropna()
    avg = returns.mean()
    vol = returns.std()
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))   # downside deviation vs 0 target
    var95 = returns.quantile(.05)
    var99 = returns.quantile(.01)
    return pd.Series({
        "Mean_%": avg,
        "Median_%": returns.median(),
        "Std_%": vol,
        "Variance": returns.var(),
        "CV": vol / abs(avg) if avg != 0 else np.nan,
        "Skew": returns.skew(),
        "ExKurt": returns.kurt(),
        "Min_%": returns.min(),
        "Max_%": returns.max(),
        "P95_%": returns.quantile(.95),
        "Ann_Ret_%": avg * TRADING_DAYS,
        "Ann_Vol_%": vol * np.sqrt(TRADING_DAYS),
        "Sharpe": avg * TRADING_DAYS / (vol * np.sqrt(TRADING_DAYS)),
        "Sortino": avg * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS)),
        "MaxDD_%": max_drawdown(returns),
        "VaR95_%": var95,
        "VaR99_%": var99,
        "CVaR95_%": returns[returns <= var95].mean(),
        "WinRate_%": (returns > 0).mean() * 100,
    })


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def fit_garch(x, gjr=False):
    """Gaussian-QMLE GARCH(1,1) / GJR-GARCH(1,1) in %-returns. Returns dict or None."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    var0 = x.var()

    def recursion(p):
        omega, alpha, beta, gamma = p
        s2 = np.empty(len(x))
        s2[0] = var0
        for t in range(1, len(x)):
            neg = x[t - 1] < 0
            s2[t] = omega + (alpha + gamma * neg) * x[t - 1] ** 2 + beta * s2[t - 1]
        return s2

    def nll(p):
        s2 = recursion(p)
        if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
            return 1e10
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + x ** 2 / s2)

    g0 = 0.05 if gjr else 0.0
    cons = [{"type": "ineq", "fun": lambda p: 0.999 - (p[1] + p[2] + 0.5 * p[3])}]
    bnds = [(1e-6, None), (0, 0.5), (0, 0.999), (0, 0.5) if gjr else (0, 0)]
    res = optimize.minimize(nll, [var0 * 0.05, 0.05, 0.90, g0], bounds=bnds,
                            constraints=cons, method="SLSQP")
    if not res.success and not np.isfinite(res.fun):
        return None
    omega, alpha, beta, gamma = res.x
    s2 = recursion(res.x)
    persistence = alpha + beta + 0.5 * gamma
    k_params = 4 if gjr else 3
    return {"omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma,
            "Persistence": persistence,
            "HalfLife_days": np.log(.5) / np.log(persistence) if 0 < persistence < 1 else np.nan,
            "LongRun_Vol_%": np.sqrt(omega / (1 - persistence)) if persistence < 1 else np.nan,
            "LogLik": -res.fun, "AIC": 2 * k_params + 2 * res.fun,
            "sigma": np.sqrt(s2), "x": x,
            "std_resid": x / np.sqrt(s2), "last_s2": s2[-1], "params": res.x}


def christoffersen(hits):
    """Christoffersen independence test for VaR breach sequence."""
    h = np.asarray(hits, int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0)); n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0)); n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    if min(n00 + n01, n10 + n11) == 0 or (n01 + n11) == 0:
        return np.nan, np.nan
    pi01, pi11 = n01 / (n00 + n01), n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    def ll(p, a, b):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return a * np.log(1 - p) + b * np.log(p)
    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi01, n00, n01) - ll(pi11, n10, n11))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def qlike(actual, pred):
    a, p = np.asarray(actual, float) ** 2, np.asarray(pred, float) ** 2
    ok = (a > 0) & (p > 0)
    return np.mean(a[ok] / p[ok] - np.log(a[ok] / p[ok]) - 1)


section("1. LOAD, CLEAN, VALIDATE")
raw = pd.read_csv(FILE)
raw.columns = raw.columns.str.strip()
df = raw.copy()
df["Date"] = pd.to_datetime(df["Date"])
df["Symbol"] = df["Symbol"].astype(str).str.strip()
df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

print("Duplicate Symbol-Date rows:", df.duplicated(["Symbol", "Date"]).sum())
df = df.drop_duplicates(["Symbol", "Date"], keep="last").reset_index(drop=True)

base_cols = ["Open", "High", "Low", "Close", "Volume"]
print("Missing in raw price columns:", df[base_cols].isna().sum().to_dict())
df[base_cols] = df.groupby("Symbol")[base_cols].ffill()

checks = {
    "High >= Low": (df.High >= df.Low).all(),
    "High >= max(Open, Close)": (df.High >= df[["Open", "Close"]].max(axis=1)).all(),
    "Low <= min(Open, Close)": (df.Low <= df[["Open", "Close"]].min(axis=1)).all(),
    "Prices > 0": (df[["Open", "High", "Low", "Close"]] > 0).all().all(),
    "Volume > 0": (df.Volume > 0).all(),
    "Equal observations per stock": df.groupby("Symbol").size().nunique() == 1,
    "Dates strictly increasing": df.groupby("Symbol")["Date"].apply(lambda s: s.is_monotonic_increasing).all(),
}
for name, passed in checks.items():
    print(f"{name:35s}:", "PASS" if passed else "FAIL")
assert all(checks.values())

by_symbol = df.groupby("Symbol")
df["Daily_Return"] = by_symbol["Close"].pct_change() * 100
df["Absolute_Return"] = df["Daily_Return"].abs()
df["Log_Return"] = by_symbol["Close"].transform(lambda s: np.log(s).diff()) * 100
df["Volatility_20D"] = by_symbol["Daily_Return"].transform(lambda s: s.rolling(20).std())
df["MA_20"] = by_symbol["Close"].transform(lambda s: s.rolling(20).mean())
df["MA_50"] = by_symbol["Close"].transform(lambda s: s.rolling(50).mean())

supplied = raw.assign(Date=pd.to_datetime(raw["Date"]))
supplied = supplied.sort_values(["Symbol", "Date"]).reset_index(drop=True)
for col in ["Daily_Return", "Absolute_Return", "Volatility_20D", "MA_20", "MA_50"]:
    same = np.allclose(df[col].fillna(0), supplied[col].fillna(0))
    print(f"{col:16s} matches supplied CSV column:", same)

df["EWMA_Vol"] = by_symbol["Daily_Return"].transform(lambda s: s.ewm(span=30, min_periods=20).std())
df["Range_Pct"] = (df.High - df.Low) / df.Close * 100
df["Gap_Pct"] = (df.Open / by_symbol["Close"].shift(1) - 1) * 100

parkinson = np.log(df.High / df.Low) ** 2 / (4 * np.log(2))
garman_klass = (0.5 * np.log(df.High / df.Low) ** 2
                - (2 * np.log(2) - 1) * np.log(df.Close / df.Open) ** 2)
df["Park_Vol"] = np.sqrt(parkinson.groupby(df.Symbol).transform(lambda s: s.rolling(20).mean())) * 100
df["GK_Vol"] = np.sqrt(
    garman_klass.groupby(df.Symbol).transform(lambda s: s.rolling(20).mean()).clip(lower=0)
) * 100

std20 = by_symbol["Close"].transform(lambda s: s.rolling(20).std())
df["BB_Up"] = df.MA_20 + 2 * std20
df["BB_Lo"] = df.MA_20 - 2 * std20

price_change = by_symbol["Close"].diff()
avg_gain = price_change.clip(lower=0).groupby(df.Symbol).transform(lambda s: s.rolling(14).mean())
avg_loss = (-price_change.clip(upper=0)).groupby(df.Symbol).transform(lambda s: s.rolling(14).mean())
df["RSI"] = 100 - 100 / (1 + avg_gain / avg_loss)

ema12 = by_symbol["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
ema26 = by_symbol["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df.groupby("Symbol")["MACD"].transform(lambda s: s.ewm(span=9, adjust=False).mean())

df["Drawdown"] = (df.Close / by_symbol["Close"].cummax() - 1) * 100
df["Vol_Z"] = by_symbol["Volume"].transform(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
df["Amihud"] = df.Absolute_Return / (df.Close * df.Volume) * 1e9
df["Year"] = df.Date.dt.year
df["Month"] = df.Date.dt.month
df["DOW"] = df.Date.dt.dayofweek

new_cols = [c for c in df.columns if c not in raw.columns]
print("Engineered columns added:", new_cols)


def wide(column):
    return df.pivot(index="Date", columns="Symbol", values=column)


prices = wide("Close")
returns = wide("Daily_Return")
vol20 = wide("Volatility_20D")
abs_ret = wide("Absolute_Return")
volume = wide("Volume")

symbols = list(prices.columns)
n = len(symbols)
market = returns.mean(axis=1)
ret_long = df.dropna(subset=["Daily_Return"])
colors = dict(zip(symbols, sns.color_palette("tab10", n)))


section("2. DATASET DESCRIPTION")
print("Shape (raw):", raw.shape, "| Stocks:", n, "| Observations per stock:", len(prices))
print("\nData types:\n", df[raw.columns].dtypes.to_string())
print("\nHead:\n", df[raw.columns].head().to_string())
print("\nTail:\n", df[raw.columns].tail().to_string())

summary = df.groupby("Symbol").agg(
    Start=("Date", "min"),
    End=("Date", "max"),
    Obs=("Date", "count"),
    Avg_Close=("Close", "mean"),
    Avg_Volume=("Volume", "mean"),
)
summary["Frequency"] = "Daily"
summary["Source"] = "NSE"
print("\nDataset summary:\n", summary.to_string())
print("\nSummary statistics:\n", df[base_cols].describe(percentiles=[.25, .5, .75]).T.to_string())

missing_after = df.isna().sum()
print("\nMissing values after feature engineering:\n", missing_after[missing_after > 0].to_string())
print("\nTrading days per year:\n", df[df.Symbol == symbols[0]].groupby("Year").size().to_string())


section("3. PERFORMANCE AND RISK METRICS")
metrics = returns.apply(performance).T
metrics["Calmar"] = metrics["Ann_Ret_%"] / metrics["MaxDD_%"].abs()
print(metrics.round(4).T.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics["Ann_Ret_%"].sort_values().plot.barh(ax=axes[0], color="seagreen", title="Annualised Return (%)")
metrics["Ann_Vol_%"].sort_values().plot.barh(ax=axes[1], color="indianred", title="Annualised Volatility (%)")
metrics["Sharpe"].sort_values().plot.barh(ax=axes[2], color="steelblue", title="Sharpe Ratio (rf = 0)")
save_fig(fig, "fig_01_performance.png")

fig, axes = make_grid(n)
for ax, sym in zip(axes, symbols):
    sns.histplot(returns[sym].dropna(), bins=50, kde=True, stat="density", ax=ax, color=colors[sym])
    xs = np.linspace(returns[sym].min(), returns[sym].max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, returns[sym].mean(), returns[sym].std()), "k--", lw=1)
    ax.axvline(returns[sym].mean(), color="red", lw=1)
    ax.set_title(sym)
    ax.set_xlabel("Daily return (%)")
fig.suptitle("Return Distributions vs Normal Curve", fontsize=14)
save_fig(fig, "fig_02_distributions.png")


section("4. NORMALITY AND TAIL BEHAVIOUR")
normality_rows = {}
for sym in symbols:
    r = returns[sym].dropna()
    normality_rows[sym] = {
        "JB_p": stats.jarque_bera(r)[1],
        "DAgostino_p": stats.normaltest(r)[1],
        "Shapiro_p": stats.shapiro(r)[1],
        "KS_p": stats.kstest((r - returns[sym].mean()) / returns[sym].std(), "norm")[1],
    }
normality = pd.DataFrame(normality_rows).T
normality["Normal_at_5%"] = (normality[["JB_p", "DAgostino_p", "Shapiro_p"]] > .05).all(axis=1)
print(normality.round(4).to_string())

pooled = ret_long["Daily_Return"].values
t_df, t_loc, t_scale = stats.t.fit(pooled)
print(f"\nPooled Student-t fit: df={t_df:.2f}, loc={t_loc:.4f}, scale={t_scale:.4f}")
obs_extreme = (np.abs((pooled - pooled.mean()) / pooled.std()) > 3).mean() * 100
norm_extreme = 2 * stats.norm.sf(3) * 100
print("Share of |z|>3 days: observed %.3f%% vs normal %.3f%%" % (obs_extreme, norm_extreme))

fig, axes = make_grid(n)
for ax, sym in zip(axes, symbols):
    stats.probplot(returns[sym].dropna(), dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(2)
    ax.set_title(sym)
    ax.set_xlabel("")
    ax.set_ylabel("")
fig.suptitle("Normal Q-Q Plots of Daily Returns", fontsize=14)
save_fig(fig, "fig_03_qq.png")


section("5. TIME SERIES, TRENDS AND TECHNICAL INDICATORS")
fig, axes = make_grid(n, cols=2, width=8, height=3.4, sharex=True)
for ax, sym in zip(axes, symbols):
    d = df[df.Symbol == sym]
    ax.plot(d.Date, d.Close, lw=1, label="Close", color="black")
    ax.plot(d.Date, d.MA_20, lw=1, label="MA20")
    ax.plot(d.Date, d.MA_50, lw=1, label="MA50")
    ax.fill_between(d.Date, d.BB_Lo, d.BB_Up, alpha=.15, color="grey", label="Bollinger")
    ax.scatter(d.loc[d.Close.idxmax(), "Date"], d.Close.max(), color="green", zorder=5)
    ax.scatter(d.loc[d.Close.idxmin(), "Date"], d.Close.min(), color="red", zorder=5)
    ax.set_title(sym)
    ax.legend(fontsize=6, ncol=4)
save_fig(fig, "fig_04_price_ma_bollinger.png")

rebased = prices / prices.iloc[0] * 100
cum_growth = (1 + returns.fillna(0) / 100).cumprod()
drawdown = (cum_growth / cum_growth.cummax() - 1) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for sym in symbols:
    axes[0].plot(rebased.index, rebased[sym], lw=1.1, label=sym, color=colors[sym])
axes[0].axhline(100, color="k", ls="--", lw=.8)
axes[0].set_title("Rebased Prices (Start = 100)")
axes[0].legend(ncol=2, fontsize=7)
for sym in symbols:
    axes[1].plot(drawdown.index, drawdown[sym], lw=.9, label=sym, color=colors[sym])
axes[1].set_title("Drawdown from Running Peak (%)")
save_fig(fig, "fig_05_rebased_drawdown.png")

peak_rows = {}
for sym in symbols:
    peak_rows[sym] = {
        "Peak": prices[sym].max(),
        "Peak_Date": prices[sym].idxmax().date(),
        "Trough": prices[sym].min(),
        "Trough_Date": prices[sym].idxmin().date(),
        "Final_Rebased": rebased[sym].iloc[-1],
        "Max_Drawdown_%": drawdown[sym].min(),
        "Hurst": hurst(np.log(prices[sym].values)),
    }
peaks = pd.DataFrame(peak_rows).T

ma_signal = (df.MA_20 > df.MA_50).astype(int)
crossings = ma_signal.groupby(df.Symbol).diff()
peaks["Golden_Crosses"] = crossings.eq(1).groupby(df.Symbol).sum()
peaks["Death_Crosses"] = crossings.eq(-1).groupby(df.Symbol).sum()
peaks["Pct_Days_MA20>MA50"] = ma_signal.groupby(df.Symbol).mean() * 100
print(peaks.to_string())
print("\nHurst ~0.5 => random walk, >0.5 trending, <0.5 mean reverting")


section("6. VOLATILITY ANALYSIS")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for sym in symbols:
    axes[0].plot(vol20.index, vol20[sym], lw=.8, label=sym, color=colors[sym])
axes[0].set_title("20-Day Rolling Volatility (%)")
axes[0].legend(ncol=2, fontsize=7)

estimators = pd.DataFrame({
    "Rolling 20D": wide("Volatility_20D").mean(axis=1),
    "EWMA (span 30)": wide("EWMA_Vol").mean(axis=1),
    "Parkinson 20D": wide("Park_Vol").mean(axis=1),
    "Garman-Klass 20D": wide("GK_Vol").mean(axis=1),
})
estimators.plot(ax=axes[1], lw=1)
axes[1].set_title("Average Market Volatility: Four Estimators (%)")
save_fig(fig, "fig_06_volatility_estimators.png")
print("Estimator correlation:\n", estimators.corr().round(3).to_string())
print("Estimator means:\n", estimators.mean().round(4).to_string())

market_vol = vol20.mean(axis=1).dropna()
low_cut, high_cut = market_vol.quantile([.25, .75])
regime = pd.cut(market_vol, [-np.inf, low_cut, high_cut, np.inf], labels=["Low", "Normal", "High"])
regime_by_year = pd.crosstab(regime.index.year, regime, normalize="index") * 100

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
axes[0].plot(market_vol.index, market_vol, color="navy", lw=1)
axes[0].axhline(high_cut, color="red", ls="--", label="75th pct: high regime")
axes[0].axhline(low_cut, color="green", ls="--", label="25th pct: low regime")
axes[0].set_title("Market Volatility Regimes")
axes[0].legend(fontsize=8)
regime_by_year.plot.bar(stacked=True, ax=axes[1], color=["seagreen", "goldenrod", "indianred"])
axes[1].set_title("Regime Share by Year (%)")
sns.heatmap(returns.groupby(returns.index.year).std(), annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[2])
axes[2].set_title("Yearly Volatility (Std of Daily Returns, %)")
save_fig(fig, "fig_07_regimes.png")
print("Regime share by year (%):\n", regime_by_year.round(1).to_string())

lag_range = range(1, 31)
fig, axes = plt.subplots(1, 3, figsize=(20, 4.8))
acf_inputs = [market, market.abs()]
acf_titles = ["ACF of Market Return", "ACF of |Market Return| (clustering)"]
for ax, series, title in zip(axes[:2], acf_inputs, acf_titles):
    acf_values = [series.autocorr(lag) for lag in lag_range]
    ax.bar(list(lag_range), acf_values, color="steelblue")
    ax.axhline(1.96 / np.sqrt(len(series)), color="red", ls="--")
    ax.axhline(-1.96 / np.sqrt(len(series)), color="red", ls="--")
    ax.set_title(title)
    ax.set_xlabel("Lag (days)")

persistence = pd.DataFrame({
    sym: {f"AC_vol_lag{lag}": vol20[sym].autocorr(lag) for lag in (1, 5, 20)}
    for sym in symbols
}).T
persistence.plot.bar(ax=axes[2], title="Autocorrelation of 20D Volatility")
save_fig(fig, "fig_08_clustering.png")

lb_rows = {}
for sym in symbols:
    lb_ret = ljung_box(returns[sym])
    lb_sq = ljung_box(returns[sym].dropna() ** 2)
    arch = arch_lm(returns[sym])
    lb_rows[sym] = {
        "LB_ret_Q": lb_ret[0],
        "LB_ret_p": lb_ret[1],
        "LB_sq_Q": lb_sq[0],
        "LB_sq_p": lb_sq[1],
        "ARCH_LM": arch[0],
        "ARCH_p": arch[1],
    }
lb_table = pd.DataFrame(lb_rows).T
print("\nLjung-Box (10 lags) and ARCH-LM (5 lags) tests:\n", lb_table.round(4).to_string())

leverage = {}
for sym in symbols:
    next_abs = abs_ret[sym].shift(-1)
    leverage[sym] = {
        "Corr(r_t, |r_t+1|)": returns[sym].corr(next_abs),
        "NextAbs_after_down": next_abs[returns[sym] < 0].mean(),
        "NextAbs_after_up": next_abs[returns[sym] > 0].mean(),
        "Vol_persist_lag1": vol20[sym].autocorr(1),
    }
print("\nLeverage / asymmetry check:\n", pd.DataFrame(leverage).T.round(4).to_string())


section("7. CORRELATION AND DEPENDENCE")
corr_frame = df[["Daily_Return", "Absolute_Return", "Volume", "Volatility_20D", "Range_Pct", "RSI", "Vol_Z"]].dropna()
corr_frame.columns = ["Return", "Abs Return", "Volume", "Volatility", "Range %", "RSI", "Volume Z"]

fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))
sns.heatmap(corr_frame.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Pearson: Market Variables")
sns.heatmap(corr_frame.corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Spearman: Market Variables")
stock_corr = returns.corr()
sns.heatmap(stock_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2], annot_kws={"size": 7})
axes[2].set_title("Pearson: Stock Returns")
save_fig(fig, "fig_09_correlations.png")

r_vol_abs, p_vol_abs = stats.pearsonr(corr_frame["Volume"], corr_frame["Abs Return"])
rho_vol_abs, p_rho = stats.spearmanr(corr_frame["Volume"], corr_frame["Abs Return"])
print(f"Volume vs |Return|: Pearson r={r_vol_abs:.4f} (p={p_vol_abs:.4g}), Spearman rho={rho_vol_abs:.4f} (p={p_rho:.4g})")

upper = stock_corr.where(np.triu(np.ones(stock_corr.shape, bool), 1)).stack()
pair_pvals = {}
for pair in upper.index:
    pair_data = returns[list(pair)].dropna().values.T
    pair_pvals[pair] = stats.pearsonr(*pair_data)[1]
n_sig = sum(p < .05 for p in pair_pvals.values())
print(f"Pairs: {len(upper)} | positive: {(upper > 0).sum()} | significant at 5%: {n_sig}")
print("Strongest pair:", upper.idxmax(), round(upper.max(), 4), "| Weakest pair:", upper.idxmin(), round(upper.min(), 4))
print("Mean pairwise correlation: %.4f" % upper.mean())

capm_rows = {}
for sym in symbols:
    pair = pd.concat([market, returns[sym]], axis=1).dropna()
    fit = stats.linregress(pair.iloc[:, 0], pair.iloc[:, 1])
    capm_rows[sym] = {
        "Beta": fit.slope,
        "Alpha_%/day": fit.intercept,
        "R2": fit.rvalue ** 2,
        "p_beta": fit.pvalue,
    }
capm = pd.DataFrame(capm_rows).T
print("\nBeta vs equal-weighted market:\n", capm.round(4).to_string())

vol_z = wide("Vol_Z")
lead_lag = {}
for lag in range(-5, 6):
    lead_lag[lag] = np.mean([vol_z[sym].corr(abs_ret[sym].shift(lag)) for sym in symbols])
lead_lag = pd.Series(lead_lag)
rolling_corr = pd.DataFrame({sym: returns[sym].rolling(60).corr(market) for sym in symbols})

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
bar_colors = np.where(lead_lag.index == 0, "firebrick", "steelblue")
lead_lag.plot.bar(ax=axes[0], color=bar_colors, title="Lead-Lag: Volume Z vs |Return| (lag in days)")
rolling_corr.mean(axis=1).plot(ax=axes[1], color="purple", title="Average Rolling 60D Correlation with Market")
axes[1].axhline(0, color="k", lw=.8)
save_fig(fig, "fig_10_leadlag_rolling.png")


section("8. HYPOTHESIS TESTS")
ttest_rows = {}
for sym in symbols:
    t_val, p_val = stats.ttest_1samp(returns[sym].dropna(), 0)
    ttest_rows[sym] = {"t_stat": t_val, "p_value": p_val}
ttests = pd.DataFrame(ttest_rows).T
ttests["p_BH_adj"] = bh_adjust(ttests.p_value.values)
ttests["Mean_ret_sig_at_5%"] = ttests.p_value < .05
ttests["Sig_after_FDR"] = ttests.p_BH_adj < .05
print("One-sample t-test (mean daily return = 0):\n", ttests.round(4).to_string())

stock_groups = [returns[sym].dropna() for sym in symbols]
print("\nANOVA, equal mean returns across stocks: F=%.4f p=%.4g" % stats.f_oneway(*stock_groups))
print("Kruskal-Wallis, same return distribution: H=%.4f p=%.4g" % stats.kruskal(*stock_groups))
print("Levene, equal variance across stocks:     W=%.4f p=%.4g" % stats.levene(*stock_groups))

weekday_groups = [ret_long[ret_long.DOW == d]["Daily_Return"] for d in range(5)]
print("Day-of-week effect (Kruskal): H=%.4f p=%.4g" % stats.kruskal(*weekday_groups))
month_groups = [ret_long[ret_long.Month == m]["Daily_Return"] for m in range(1, 13)]
print("Month effect (Kruskal):       H=%.4f p=%.4g" % stats.kruskal(*month_groups))
year_groups = [ret_long[ret_long.Year == y]["Daily_Return"] for y in sorted(ret_long.Year.unique())]
print("Year effect on variance (Levene): W=%.4f p=%.4g" % stats.levene(*year_groups))

most_volatile = metrics["Std_%"].idxmax()
least_volatile = metrics["Std_%"].idxmin()
var_ratio = metrics.loc[most_volatile, "Variance"] / metrics.loc[least_volatile, "Variance"]
dof = len(returns) - 2
f_p = 2 * min(stats.f.sf(var_ratio, dof, dof), stats.f.cdf(var_ratio, dof, dof))
print(f"F-test {most_volatile} vs {least_volatile}: variance ratio={var_ratio:.3f}, p={f_p:.4g}")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
weekday_names = dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]))
ret_long.groupby("DOW")["Daily_Return"].mean().rename(weekday_names).plot.bar(
    ax=axes[0], color="teal", title="Mean Return by Weekday (%)")
year_month = market.groupby([market.index.year, market.index.month]).mean().unstack()
sns.heatmap(year_month, cmap="RdYlGn", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=axes[1])
axes[1].set_title("Market Mean Daily Return: Year x Month (%)")
sns.boxplot(data=ret_long, x="Symbol", y="Daily_Return", hue="Symbol", palette="Set2", legend=False, ax=axes[2])
axes[2].tick_params(axis="x", rotation=60)
axes[2].set_title("Return Box Plots")
save_fig(fig, "fig_11_seasonality_boxplots.png")


section("9. OUTLIER AND ANOMALY DETECTION")


def iqr_summary(series):
    q1 = series.quantile(.25)
    q3 = series.quantile(.75)
    spread = q3 - q1
    lower = q1 - 1.5 * spread
    upper_fence = q3 + 1.5 * spread
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": spread,
        "Lower": lower,
        "Upper": upper_fence,
        "IQR_Outliers": ((series < lower) | (series > upper_fence)).sum(),
    }


outliers = pd.DataFrame({sym: iqr_summary(returns[sym].dropna()) for sym in symbols}).T
outliers["Z>3_Outliers"] = ((returns - returns.mean()) / returns.std()).abs().gt(3).sum()
outliers["Outlier_%"] = outliers.IQR_Outliers / returns.count() * 100
print(outliers.round(4).to_string())

anomaly_features = ["Daily_Return", "Range_Pct", "Vol_Z", "Volatility_20D"]
anom = df.dropna(subset=anomaly_features).copy()
forest = IsolationForest(n_estimators=200, contamination=.01, random_state=42)
anom["Anomaly"] = forest.fit_predict(StandardScaler().fit_transform(anom[anomaly_features])) == -1
print("\nIsolation Forest anomalies:", anom.Anomaly.sum(), "of", len(anom))
print(anom[anom.Anomaly].groupby("Symbol").size().sort_values(ascending=False).to_string())

top_days = ret_long.nlargest(10, "Absolute_Return")[["Date", "Symbol", "Daily_Return", "Volume"]]
print("\nTop 10 extreme days:\n", top_days.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
normal_pts = anom[~anom.Anomaly]
anomaly_pts = anom[anom.Anomaly]
axes[0].scatter(normal_pts.Daily_Return, normal_pts.Range_Pct, s=3, alpha=.3, color="steelblue", label="Normal")
axes[0].scatter(anomaly_pts.Daily_Return, anomaly_pts.Range_Pct, s=12, color="red", label="Anomaly")
axes[0].set_xlabel("Daily Return (%)")
axes[0].set_ylabel("Intraday Range (%)")
axes[0].set_title("Isolation Forest Anomalies")
axes[0].legend()
outliers[["IQR_Outliers", "Z>3_Outliers"]].plot.bar(ax=axes[1], title="Outlier Counts by Method")
save_fig(fig, "fig_12_outliers.png")


section("10. RISK PROFILING: PCA, K-MEANS, HIERARCHICAL CLUSTERING")
profile = pd.DataFrame({
    "Ann_Ret": metrics["Ann_Ret_%"],
    "Ann_Vol": metrics["Ann_Vol_%"],
    "Sharpe": metrics["Sharpe"],
    "MaxDD": metrics["MaxDD_%"],
    "VaR95": metrics["VaR95_%"],
    "Skew": metrics["Skew"],
    "ExKurt": metrics["ExKurt"],
    "LogVolume": np.log(volume.mean()),
    "Amihud": df.groupby("Symbol")["Amihud"].mean(),
})
scaled = StandardScaler().fit_transform(profile)
pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(scaled)
print("PCA explained variance ratio:", pca.explained_variance_ratio_.round(4))
loadings = pd.DataFrame(pca.components_.T, index=profile.columns, columns=["PC1", "PC2"])
print("PCA loadings:\n", loadings.round(3).to_string())

silhouettes = {}
for k in (2, 3, 4):
    km_labels = KMeans(k, n_init=20, random_state=42).fit_predict(scaled)
    silhouettes[k] = silhouette_score(scaled, km_labels)
print("Silhouette by k:", {k: round(v, 4) for k, v in silhouettes.items()})

best_k = max(silhouettes, key=silhouettes.get)
labels = KMeans(best_k, n_init=20, random_state=42).fit_predict(scaled)
vol_order = pd.Series(profile.Ann_Vol.values).groupby(labels).mean().sort_values().index
cluster_names = {
    2: ["Lower Risk", "Higher Risk"],
    3: ["Low Risk", "Moderate Risk", "High Risk"],
    4: ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
}[best_k]
profile["Cluster"] = [cluster_names[list(vol_order).index(lbl)] for lbl in labels]

risk_cols = profile[["Ann_Vol", "MaxDD", "VaR95", "ExKurt"]].copy()
risk_cols["MaxDD"] = risk_cols.MaxDD.abs()
risk_cols["VaR95"] = risk_cols.VaR95.abs()
profile["Risk_Score"] = risk_cols.rank().mean(axis=1)
print(f"\nChosen k={best_k}\n", profile.sort_values("Risk_Score", ascending=False).round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
axes[0].scatter(components[:, 0], components[:, 1], c=labels, cmap="viridis", s=140)
for i, sym in enumerate(symbols):
    axes[0].annotate(sym, (components[i, 0], components[i, 1]), xytext=(5, 5),
                     textcoords="offset points", fontsize=9)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
axes[0].set_title("PCA of Risk-Return Profiles with K-Means Clusters")
dendrogram(linkage(scaled, "ward"), labels=symbols, ax=axes[1], leaf_rotation=45)
axes[1].set_title("Hierarchical Clustering (Ward)")
save_fig(fig, "fig_13_risk_profiling.png")


section("11. PORTFOLIO OPTIMISATION")
mu = returns.mean().values * TRADING_DAYS
cov_sample = returns.cov().values * TRADING_DAYS
lw = LedoitWolf().fit(returns.fillna(0).values)
cov = lw.covariance_ * TRADING_DAYS                 # shrinkage: far more stable than the sample covariance
print(f"Ledoit-Wolf shrinkage intensity: {lw.shrinkage_:.3f}")
rng = np.random.default_rng(42)

random_weights = rng.dirichlet(np.ones(n), 6000)
random_ret = random_weights @ mu
random_vol = np.sqrt(np.einsum("ij,jk,ik->i", random_weights, cov, random_weights))

constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
bounds = [(0, 1)] * n
equal_w = np.ones(n) / n

min_var = optimize.minimize(lambda w: np.sqrt(w @ cov @ w), equal_w,
                            bounds=bounds, constraints=constraints, method="SLSQP")
max_sharpe = optimize.minimize(lambda w: -(w @ mu) / np.sqrt(w @ cov @ w), equal_w,
                               bounds=bounds, constraints=constraints, method="SLSQP")
print("Optimiser success:", min_var.success, max_sharpe.success)

frontier = []
for target in np.linspace(mu.min() * 1.02, mu.max() * .98, 30):
    target_cons = constraints + ({"type": "eq", "fun": lambda w, t=target: w @ mu - t},)
    sol = optimize.minimize(lambda w: np.sqrt(w @ cov @ w), equal_w,
                            bounds=bounds, constraints=target_cons, method="SLSQP")
    if sol.success:
        frontier.append((np.sqrt(sol.x @ cov @ sol.x), target))
frontier = np.array(frontier)

def risk_parity_weights(c):
    f = lambda w: np.sum((w * (c @ w) / (w @ c @ w) - 1 / len(w)) ** 2)
    return optimize.minimize(f, np.ones(len(c)) / len(c), bounds=[(0, 1)] * len(c),
                             constraints=constraints, method="SLSQP").x


# cap single-name weight at 40% (or equal weight if too few stocks) to avoid corner solutions
cap = max(.4, 1.5 / n)
capped_bounds = [(0, cap)] * n
max_sharpe_cap = optimize.minimize(lambda w: -(w @ mu) / np.sqrt(w @ cov @ w), equal_w,
                                   bounds=capped_bounds, constraints=constraints, method="SLSQP")
weights = pd.DataFrame({
    "Equal_Weight": equal_w,
    "Min_Variance": min_var.x,
    "Risk_Parity": risk_parity_weights(cov),
    "Max_Sharpe": max_sharpe.x,
    "Max_Sharpe_Capped": max_sharpe_cap.x,
}, index=symbols)
print((weights * 100).round(2).to_string())

filled = returns.fillna(0).values
backtest = pd.DataFrame({name: filled @ weights[name].values for name in weights}, index=returns.index)
port_metrics = backtest.apply(performance).T[["Ann_Ret_%", "Ann_Vol_%", "Sharpe", "Sortino", "MaxDD_%", "VaR95_%", "CVaR95_%"]]
print("\nPortfolio comparison (in-sample, daily rebalanced):\n", port_metrics.round(4).to_string())
div_ratio = (equal_w * returns.std().values).sum() / backtest["Equal_Weight"].std()
print("Diversification ratio (equal weight): %.4f" % div_ratio)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
scatter = axes[0].scatter(random_vol, random_ret, c=random_ret / random_vol, cmap="viridis", s=5)
axes[0].plot(frontier[:, 0], frontier[:, 1], "r-", lw=2, label="Efficient frontier")
axes[0].scatter(np.sqrt(min_var.x @ cov @ min_var.x), min_var.x @ mu,
                marker="*", s=300, color="blue", label="Min variance")
axes[0].scatter(np.sqrt(max_sharpe.x @ cov @ max_sharpe.x), max_sharpe.x @ mu,
                marker="*", s=300, color="gold", edgecolor="k", label="Max Sharpe")
axes[0].set_xlabel("Annual volatility (%)")
axes[0].set_ylabel("Annual return (%)")
axes[0].set_title("Efficient Frontier (6000 random portfolios)")
axes[0].legend()
plt.colorbar(scatter, ax=axes[0], label="Sharpe")
(weights * 100).plot.bar(ax=axes[1], title="Portfolio Weights (%)")
((1 + backtest / 100).cumprod() * 100).plot(ax=axes[2], title="Growth of 100 (in-sample)")
save_fig(fig, "fig_14_portfolio.png")


section("12A. GARCH / GJR-GARCH VOLATILITY MODELS")
garch_rows, garch_fits = {}, {}
for sym in symbols:
    base_fit = fit_garch(returns[sym], gjr=False)
    gjr_fit = fit_garch(returns[sym], gjr=True)
    if base_fit is None or gjr_fit is None:
        continue
    garch_fits[sym] = gjr_fit if gjr_fit["AIC"] < base_fit["AIC"] else base_fit
    lb_std = ljung_box(garch_fits[sym]["std_resid"] ** 2)[1]
    garch_rows[sym] = {
        "alpha": base_fit["alpha"], "beta": base_fit["beta"], "Persist": base_fit["Persistence"],
        "HalfLife_d": base_fit["HalfLife_days"], "LongRun_Vol_%": base_fit["LongRun_Vol_%"],
        "GJR_gamma": gjr_fit["gamma"], "AIC_GARCH": base_fit["AIC"], "AIC_GJR": gjr_fit["AIC"],
        "Best": "GJR" if gjr_fit["AIC"] < base_fit["AIC"] else "GARCH",
        "ResidSq_LB_p": lb_std,
    }
garch_table = pd.DataFrame(garch_rows).T
print(garch_table.to_string())
print("\nResidSq_LB_p > 0.05 means the model removed the volatility clustering.")
print("GJR_gamma > 0 means bad news raises volatility more than good news (leverage effect).")

section("12B. VOLATILITY FORECASTING (WALK-FORWARD, EMBARGOED)")
HORIZON = 10
model_df = df.copy()
g_ret = model_df.groupby("Symbol")["Daily_Return"]
model_df["Ret_5"] = g_ret.transform(lambda s: s.rolling(5).mean())
model_df["Vol_5D"] = g_ret.transform(lambda s: s.rolling(5).std())
model_df["Vol_60D"] = g_ret.transform(lambda s: s.rolling(60).std())
model_df["Down_Vol_20D"] = g_ret.transform(lambda s: s.where(s < 0).rolling(20, min_periods=5).std())
model_df["Vol_Ratio"] = model_df["Vol_5D"] / model_df["Vol_60D"].replace(0, np.nan)
model_df["Mkt_Vol"] = model_df.Date.map(vol20.mean(axis=1))
model_df["Rel_Vol"] = model_df["Volatility_20D"] / model_df["Mkt_Vol"].replace(0, np.nan)
model_df["Log_Vol20"] = np.log(model_df["Volatility_20D"].clip(lower=1e-6))
# forward realised volatility over the NEXT HORIZON days (strictly after today)
model_df["Target"] = g_ret.transform(lambda s: s.rolling(HORIZON).std().shift(-HORIZON))
model_df["Log_Target"] = np.log(model_df["Target"].clip(lower=1e-6))

features = ["Volatility_20D", "Vol_5D", "Vol_60D", "Down_Vol_20D", "Vol_Ratio", "Rel_Vol", "Mkt_Vol",
            "EWMA_Vol", "Park_Vol", "GK_Vol", "Absolute_Return", "Range_Pct", "Vol_Z", "RSI",
            "Ret_5", "Drawdown", "Log_Vol20"]
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["Target", "Log_Target"])

all_dates = np.sort(model_df.Date.unique())
n_folds, first = 4, 0.5
edges = np.linspace(int(len(all_dates) * first), len(all_dates), n_folds + 1).astype(int)
fold_preds = []
for i in range(n_folds):
    a_idx, b_idx = edges[i], edges[i + 1]
    train_end = all_dates[max(a_idx - HORIZON, 0)]          # embargo = forecast horizon
    tr_f = model_df[model_df.Date < train_end]
    te_f = model_df[(model_df.Date >= all_dates[a_idx]) & (model_df.Date <= all_dates[b_idx - 1])].copy()
    sc = StandardScaler().fit(tr_f[features])
    ridge_f = Ridge(alpha=10.0).fit(sc.transform(tr_f[features]), tr_f.Log_Target)
    rf_f = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=30,
                                 max_features=.5, n_jobs=-1, random_state=SEED).fit(tr_f[features], tr_f.Log_Target)
    gb_f = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=60,
                                         l2_regularization=5.0, random_state=SEED).fit(tr_f[features], tr_f.Log_Target)
    # log-target models: bias-correct exp() with the training residual variance
    resid_var = {}
    for nm, mdl, X in (("Ridge", ridge_f, sc.transform(tr_f[features])), ("RF", rf_f, tr_f[features]),
                       ("GB", gb_f, tr_f[features])):
        resid_var[nm] = np.var(tr_f.Log_Target - mdl.predict(X))
    te_f["P_Baseline"] = te_f.Volatility_20D
    te_f["P_EWMA"] = te_f.EWMA_Vol
    te_f["P_Ridge"] = np.exp(ridge_f.predict(sc.transform(te_f[features])) + resid_var["Ridge"] / 2)
    te_f["P_RandomForest"] = np.exp(rf_f.predict(te_f[features]) + resid_var["RF"] / 2)
    te_f["P_GradBoost"] = np.exp(gb_f.predict(te_f[features]) + resid_var["GB"] / 2)
    te_f["P_Ensemble"] = te_f[["P_Ridge", "P_RandomForest", "P_GradBoost"]].mean(axis=1)
    te_f["fold"] = i + 1
    fold_preds.append(te_f)
    if i == n_folds - 1:
        last_rf, last_gb, last_ridge = rf_f, gb_f, ridge_f
vol_oos = pd.concat(fold_preds)
pred_cols = ["P_Baseline", "P_EWMA", "P_Ridge", "P_RandomForest", "P_GradBoost", "P_Ensemble"]
print(f"Walk-forward test: {vol_oos.Date.min().date()} to {vol_oos.Date.max().date()} | "
      f"{len(vol_oos)} rows | {n_folds} folds | embargo {HORIZON} days")

score_rows = {}
for col in pred_cols:
    pred = vol_oos[col]
    score_rows[col[2:]] = {
        "MAE": mean_absolute_error(vol_oos.Target, pred),
        "RMSE": np.sqrt(np.mean((vol_oos.Target - pred) ** 2)),
        "R2": r2_score(vol_oos.Target, pred),
        "QLIKE": qlike(vol_oos.Target, pred),
        "Corr": np.corrcoef(vol_oos.Target, pred)[0, 1],
    }
scores = pd.DataFrame(score_rows).T
print(scores.round(4).to_string())
fold_r2 = pd.DataFrame({col[2:]: [r2_score(x.Target, x[col]) for _, x in vol_oos.groupby("fold")]
                        for col in pred_cols}, index=[f"fold{i}" for i in range(1, n_folds + 1)])
print("\nR2 by fold (stability):\n", fold_r2.round(4).to_string())
best_vol_model = scores["QLIKE"].idxmin()
imp_gain = scores.loc["Baseline", "MAE"] - scores.loc[best_vol_model, "MAE"]
print(f"\nBest by QLIKE: {best_vol_model} | MAE improvement vs current 20D vol: "
      f"{imp_gain / scores.loc['Baseline', 'MAE'] * 100:.1f}%")

from sklearn.inspection import permutation_importance
last_te = vol_oos[vol_oos.fold == n_folds]
perm = permutation_importance(last_rf, last_te[features].sample(min(3000, len(last_te)), random_state=SEED),
                              np.log(last_te.Target.clip(lower=1e-6)).loc[
                                  last_te[features].sample(min(3000, len(last_te)), random_state=SEED).index],
                              n_repeats=5, random_state=SEED, n_jobs=-1)
importance = pd.Series(perm.importances_mean, index=features).sort_values()

predictions = {"Random Forest": vol_oos["P_RandomForest"].values}
test = vol_oos
ridge = last_ridge
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
axes[0].scatter(test.Target, predictions["Random Forest"], s=4, alpha=.3, color="teal")
limits = [test.Target.min(), test.Target.max()]
axes[0].plot(limits, limits, "r--")
axes[0].set_xlabel("Actual next-10D volatility")
axes[0].set_ylabel("Predicted")
axes[0].set_title("Random Forest (walk-forward): Actual vs Predicted")
importance.plot.barh(ax=axes[1], color="darkorange", title="Permutation Importance (last fold, held-out)")
scores["QLIKE"].sort_values().plot.barh(ax=axes[2], color="slateblue", title="QLIKE Loss (lower = better)")
save_fig(fig, "fig_15_forecasting.png")


section("13. VOLUME AND LIQUIDITY ANALYSIS")
turnover = (df.Close * df.Volume).groupby(df.Symbol).mean() / 1e7
liquidity = pd.DataFrame({
    "Avg_Volume": volume.mean(),
    "Volume_CV": volume.std() / volume.mean(),
    "Avg_Turnover_Cr": turnover,
    "Amihud_x1e9": df.groupby("Symbol")["Amihud"].mean(),
    "Spike_Days_Z>2": (wide("Vol_Z") > 2).sum(),
})
liquidity["Illiquidity_Rank"] = liquidity["Amihud_x1e9"].rank(ascending=False)
print(liquidity.sort_values("Illiquidity_Rank").round(4).to_string())

spike_df = df.dropna(subset=["Vol_Z", "Absolute_Return"])
abs_high = spike_df[spike_df.Vol_Z > 2].Absolute_Return
abs_low = spike_df[spike_df.Vol_Z <= 2].Absolute_Return
mw_p = stats.mannwhitneyu(abs_high, abs_low)[1]
print("\nMean |return| on volume-spike days: %.4f vs other days: %.4f | Mann-Whitney p=%.4g" % (
    abs_high.mean(), abs_low.mean(), mw_p))

quintile_labels = ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
by_quintile = spike_df.groupby(pd.qcut(spike_df.Vol_Z, 5, labels=quintile_labels))["Absolute_Return"].mean()
print("Mean |return| by volume-Z quintile:\n", by_quintile.round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
sns.heatmap(volume.groupby(volume.index.year).mean() / 1e6, annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
axes[0].set_title("Average Daily Volume by Year (millions)")
liquidity["Amihud_x1e9"].sort_values().plot.barh(
    ax=axes[1], color="darkcyan", title="Amihud Illiquidity (higher = less liquid)")
by_quintile.plot.bar(ax=axes[2], color="coral", title="Mean |Return| by Volume Quintile")
save_fig(fig, "fig_16_liquidity.png")


section("14. RULE-BASED STRATEGY BACKTEST (1-DAY SIGNAL LAG)")
rsi = wide("RSI")
ma_position = (wide("MA_20") > wide("MA_50")).astype(float).shift(1)
rsi_raw = np.where(rsi < 30, 1.0, np.where(rsi > 70, 0.0, np.nan))
rsi_position = pd.DataFrame(rsi_raw, index=rsi.index, columns=rsi.columns)
rsi_position = rsi_position.ffill().fillna(0).shift(1)

def net_of_costs(gross, position):
    trades = position.fillna(0).diff().abs().fillna(0)
    return gross - trades * COST_BPS / 100          # returns are in %, cost in bps

strategies = {
    "Buy & Hold": returns,
    "MA20/50 Trend": net_of_costs(returns * ma_position, ma_position),
    "RSI 30/70": net_of_costs(returns * rsi_position, rsi_position),
}
print(f"Strategy returns are NET of {COST_BPS} bps one-way costs.")
sharpe_table = pd.DataFrame({name: rets.apply(performance).T["Sharpe"] for name, rets in strategies.items()})
return_table = pd.DataFrame({name: rets.apply(performance).T["Ann_Ret_%"] for name, rets in strategies.items()})
dd_table = pd.DataFrame({name: rets.apply(performance).T["MaxDD_%"] for name, rets in strategies.items()})

print("Sharpe:\n", sharpe_table.round(3).to_string(), "\n\nAnnual return (%):\n", return_table.round(2).to_string())
avg_table = pd.DataFrame({
    "Sharpe": sharpe_table.mean(),
    "Ann_Ret_%": return_table.mean(),
    "MaxDD_%": dd_table.mean(),
})
print("\nAverage across stocks:\n", avg_table.round(3).to_string())
beat_counts = {}
for name in list(strategies)[1:]:
    beat_counts[name] = int((sharpe_table[name] > sharpe_table["Buy & Hold"]).sum())
print("Stocks where strategy beat buy & hold (Sharpe):", beat_counts)


section("15. BOOTSTRAP CONFIDENCE, VaR BACKTEST AND OUT-OF-SAMPLE PORTFOLIO")
bootstrap_rows = {}
for sym in symbols:
    x = returns[sym].dropna().values
    resamples = x[rng.integers(0, len(x), (2000, len(x)))]
    boot_sharpe = resamples.mean(axis=1) / resamples.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    bootstrap_rows[sym] = {
        "Sharpe": x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "CI_2.5%": np.percentile(boot_sharpe, 2.5),
        "CI_97.5%": np.percentile(boot_sharpe, 97.5),
        "P(Sharpe>0)": (boot_sharpe > 0).mean(),
    }
bootstrap = pd.DataFrame(bootstrap_rows).T
print("Bootstrap Sharpe (2000 resamples):\n", bootstrap.round(4).to_string())


def kupiec(breaches, total, p=.05):
    breaches = np.clip(breaches, 1e-9, total - 1e-9)
    null_ll = (total - breaches) * np.log(1 - p) + breaches * np.log(p)
    alt_ll = (total - breaches) * np.log(1 - breaches / total) + breaches * np.log(breaches / total)
    lr = -2 * null_ll + 2 * alt_ll
    return lr, 1 - stats.chi2.cdf(lr, 1)


rolling_var = returns.rolling(250).quantile(.05).shift(1)
var_backtest = pd.DataFrame({
    sym: {
        "Test_Days": int(rolling_var[sym].notna().sum()),
        "Breaches": int((returns[sym] < rolling_var[sym]).sum()),
    }
    for sym in symbols
}).T
var_backtest["Breach_%"] = var_backtest.Breaches / var_backtest.Test_Days * 100
var_backtest["Kupiec_LR"] = [kupiec(row.Breaches, row.Test_Days)[0] for row in var_backtest.itertuples()]
var_backtest["Kupiec_p"] = [kupiec(row.Breaches, row.Test_Days)[1] for row in var_backtest.itertuples()]
var_backtest["Christoffersen_p"] = [
    christoffersen((returns[sym] < rolling_var[sym])[rolling_var[sym].notna()].astype(int).values)[1]
    for sym in symbols]
lam = 0.94
ewma_sigma = np.sqrt((returns ** 2).ewm(alpha=1 - lam, adjust=False).mean()).shift(1)
z05 = stats.norm.ppf(.05)
ewma_var = ewma_sigma * z05
valid = ewma_var.notna() & (np.arange(len(returns))[:, None] > 250)
var_backtest["EWMA_Breach_%"] = [((returns[sym] < ewma_var[sym])[valid[sym]]).mean() * 100 for sym in symbols]
print("\n95% Historical VaR backtest (250D window, expected breach rate 5%):\n", var_backtest.round(4).to_string())
print("Kupiec_p < 0.05 => wrong breach frequency; Christoffersen_p < 0.05 => breaches are clustered.")
print("EWMA_Breach_% is the RiskMetrics (lambda=0.94) normal VaR breach rate for comparison.")

split_at = int(len(returns) * .7)
ret_train = returns.iloc[:split_at]
ret_test = returns.iloc[split_at:]
mu_train = ret_train.mean().values * TRADING_DAYS
cov_train = ret_train.cov().values * TRADING_DAYS
cov_train = LedoitWolf().fit(ret_train.fillna(0).values).covariance_ * TRADING_DAYS
train_opt = optimize.minimize(lambda w: -(w @ mu_train) / np.sqrt(w @ cov_train @ w), equal_w,
                              bounds=bounds, constraints=constraints, method="SLSQP")
train_minvar = optimize.minimize(lambda w: np.sqrt(w @ cov_train @ w), equal_w,
                                 bounds=bounds, constraints=constraints, method="SLSQP")
train_rp = risk_parity_weights(cov_train)

test_filled = ret_test.fillna(0).values
oos = pd.DataFrame({
    "Equal_Weight": test_filled @ equal_w,
    "Max_Sharpe (train weights)": test_filled @ train_opt.x,
    "Min_Var (train weights)": test_filled @ train_minvar.x,
    "Risk_Parity (train weights)": test_filled @ train_rp,
}, index=ret_test.index)
print("\nTrain weights (%):\n", pd.Series(train_opt.x * 100, index=symbols).round(2).to_string())
oos_metrics = oos.apply(performance).T
print("\nOut-of-sample performance:\n", oos_metrics[["Ann_Ret_%", "Ann_Vol_%", "Sharpe", "MaxDD_%"]].round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
sharpe_table.plot.bar(ax=axes[0], title="Strategy vs Buy & Hold: Sharpe")
lower_err = bootstrap["Sharpe"] - bootstrap["CI_2.5%"]
upper_err = bootstrap["CI_97.5%"] - bootstrap["Sharpe"]
axes[1].errorbar(range(n), bootstrap["Sharpe"], yerr=[lower_err, upper_err], fmt="o", capsize=4, color="navy")
axes[1].axhline(0, color="red", ls="--")
axes[1].set_xticks(range(n))
axes[1].set_xticklabels(symbols, rotation=60)
axes[1].set_title("Bootstrap 95% CI for Sharpe Ratio")
((1 + oos / 100).cumprod() * 100).plot(ax=axes[2], title="Out-of-Sample Growth of 100")
save_fig(fig, "fig_17_validation.png")


section("16. AUTOMATED KEY FINDINGS")
best_stock = metrics["Sharpe"].idxmax()
worst_stock = metrics["Sharpe"].idxmin()
n_non_normal = int((~normality["Normal_at_5%"]).sum())
n_arch = int((lb_table["ARCH_p"] < .05).sum())
n_ci_excludes_zero = int(((bootstrap["CI_2.5%"] > 0) | (bootstrap["CI_97.5%"] < 0)).sum())
cluster_summary = "; ".join(
    f"{name}: {', '.join(profile.index[profile.Cluster == name])}" for name in cluster_names
)

print(f"1. Dataset: {n} NSE stocks, {len(prices)} daily observations each, "
      f"{prices.index.min().date()} to {prices.index.max().date()}.")
print(f"2. Highest annualised return: {metrics['Ann_Ret_%'].idxmax()} ({metrics['Ann_Ret_%'].max():.2f}%); "
      f"lowest: {metrics['Ann_Ret_%'].idxmin()} ({metrics['Ann_Ret_%'].min():.2f}%).")
print(f"3. Most volatile: {most_volatile} ({metrics.loc[most_volatile, 'Std_%']:.3f}% daily); "
      f"most stable: {least_volatile} ({metrics.loc[least_volatile, 'Std_%']:.3f}% daily).")
print(f"4. Best Sharpe: {best_stock} ({metrics.loc[best_stock, 'Sharpe']:.3f}); "
      f"worst Sharpe: {worst_stock} ({metrics.loc[worst_stock, 'Sharpe']:.3f}).")
print(f"5. Deepest drawdown: {metrics['MaxDD_%'].idxmin()} ({metrics['MaxDD_%'].min():.2f}%).")
print(f"6. Normality rejected for {n_non_normal} of {n} stocks; pooled t-fit df = {t_df:.2f}.")
print(f"7. Mean pairwise stock correlation = {upper.mean():.4f}; Volume vs |Return| Pearson r = {r_vol_abs:.4f}.")
print(f"8. ARCH effects significant (5%) in {n_arch} of {n} stocks; "
      f"|return| autocorrelation lag 1 = {market.abs().autocorr(1):.4f}.")
print(f"9. Risk clusters (k={best_k}): " + cluster_summary)
print(f"10. In-sample Max-Sharpe portfolio Sharpe = {port_metrics.loc['Max_Sharpe', 'Sharpe']:.3f} (optimistic) "
      f"vs equal weight {port_metrics.loc['Equal_Weight', 'Sharpe']:.3f}.")
print(f"11. Walk-forward volatility forecast R2: baseline {scores.loc['Baseline', 'R2']:.3f}, "
      f"ridge {scores.loc['Ridge', 'R2']:.3f}, random forest {scores.loc['RandomForest', 'R2']:.3f}, "
      f"gradient boosting {scores.loc['GradBoost', 'R2']:.3f}, ensemble {scores.loc['Ensemble', 'R2']:.3f}; "
      f"best by QLIKE: {best_vol_model}.")
if len(garch_table):
    print(f"11b. GARCH persistence averages {garch_table['Persist'].astype(float).mean():.3f} "
          f"(half-life {garch_table['HalfLife_d'].astype(float).median():.1f} days); "
          f"leverage (GJR) preferred in {int((garch_table['Best'] == 'GJR').sum())} of {len(garch_table)} stocks.")
print(f"12. Volume-spike days show mean |return| {abs_high.mean():.3f}% vs {abs_low.mean():.3f}% on other days.")
print(f"13. Average Sharpe: buy & hold {sharpe_table['Buy & Hold'].mean():.3f}, "
      f"MA trend {sharpe_table['MA20/50 Trend'].mean():.3f}, RSI {sharpe_table['RSI 30/70'].mean():.3f}.")
print(f"14. Stocks whose bootstrap Sharpe CI excludes zero: {n_ci_excludes_zero} of {n}.")
print("15. Out-of-sample Sharpe: " + ", ".join(f"{k} {v:.3f}" for k, v in oos_metrics["Sharpe"].items()) + ".")
print(f"16. Mean-return t-tests significant before / after FDR correction: "
      f"{int(ttests['Mean_ret_sig_at_5%'].sum())} / {int(ttests['Sig_after_FDR'].sum())} of {n}.")
print("\nAll figures saved as fig_*.png in the working directory.")
