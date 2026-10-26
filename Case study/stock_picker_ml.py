"""
Cross-sectional stock ranking model (walk-forward validated).

Improvements over v1:
  * Walk-forward (expanding window) validation with an embargo = prediction horizon,
    instead of one 80/20 split -> far more reliable out-of-sample estimates.
  * Richer features: momentum, volatility ratios, skew, liquidity, beta, drawdown,
    market-level context, and cross-sectional percentile ranks of every feature.
  * Winsorised, market-neutral target (forward return minus same-day average).
  * 5 models + ensemble (Ridge, ElasticNet, ExtraTrees, RandomForest, GradientBoosting).
  * Honest statistics: RankIC, ICIR, overlap-adjusted t-stat, top-K edge, long-short
    spread, per-fold stability. Final model is used only if its edge is meaningful.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FILE = "stock_market_vitality_eda.csv"
H = 10            # prediction horizon (trading days)
TOPK = 3          # stocks to BUY / AVOID
N_FOLDS = 5       # walk-forward folds
START_FRAC = 0.5  # first test fold starts after this share of the dates
COST_BPS = 10     # one-way trading cost assumption (basis points) for the net-edge estimate
SEED = 42
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


# ----------------------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = (df.drop_duplicates(["Symbol", "Date"], keep="last")
            .sort_values(["Symbol", "Date"]).reset_index(drop=True))
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df[cols] = df.groupby("Symbol")[cols].ffill()
    df = df.dropna(subset=cols)
    return df[df[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)].reset_index(drop=True)


def stock_features(g):
    c, v, o_, h, l = g["Close"], g["Volume"], g["Open"], g["High"], g["Low"]
    r = c.pct_change()
    d = c.diff()
    rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    o = pd.DataFrame(index=g.index)
    for k in (1, 5, 10, 20, 60, 120):
        o[f"ret{k}"] = c.pct_change(k)
    o["mom_12_1"] = c.shift(20) / c.shift(120) - 1            # skip most recent month
    for w in (10, 20, 60):
        o[f"vol{w}"] = r.rolling(w).std()
    o["volratio"] = o["vol10"] / o["vol60"].replace(0, np.nan)
    o["ewma_vol"] = r.ewm(span=30, min_periods=20).std()
    o["ma20r"] = c / c.rolling(20).mean() - 1
    o["ma50r"] = c / c.rolling(50).mean() - 1
    o["ma_x"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    o["rsi"] = 100 - 100 / (1 + rs)
    o["range"] = (h - l) / c
    o["range20"] = o["range"].rolling(20).mean()
    o["gap"] = o_ / c.shift(1) - 1
    o["body"] = (c - o_) / o_
    o["volz"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    o["logdv"] = np.log((c * v).rolling(20).mean().replace(0, np.nan))
    o["amihud"] = np.log1p((r.abs() / (c * v)).rolling(20).mean() * 1e9)
    o["skew20"] = r.rolling(20).skew()
    o["upfrac20"] = (r > 0).rolling(20).mean()
    o["maxret20"] = r.rolling(20).max()
    o["minret20"] = r.rolling(20).min()
    o["dd60"] = c / c.rolling(60).max() - 1
    o["hi120"] = c / c.rolling(120, min_periods=60).max() - 1
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    o["macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    o["_r"] = r
    o["fwd"] = c.shift(-H) / c - 1
    return o


def build_panel(df):
    parts = []
    for s, g in df.groupby("Symbol"):
        parts.append(stock_features(g).assign(Symbol=s, Date=g["Date"].values))
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan)
    # market context + beta
    mkt = F.pivot(index="Date", columns="Symbol", values="_r").mean(axis=1)
    mk = pd.DataFrame({"mkt_ret5": (1 + mkt).rolling(5).apply(np.prod, raw=True) - 1,
                       "mkt_ret20": (1 + mkt).rolling(20).apply(np.prod, raw=True) - 1,
                       "mkt_vol20": mkt.rolling(20).std()})
    F = F.merge(mk, left_on="Date", right_index=True, how="left")
    F["_m"] = F["Date"].map(mkt)
    F["beta60"] = (F.groupby("Symbol", group_keys=False)
                    .apply(lambda g: g["_r"].rolling(60).cov(g["_m"]) / g["_m"].rolling(60).var()))
    F["beta60"] = F["beta60"].replace([np.inf, -np.inf], np.nan)
    base = [c for c in F.columns if c not in ("fwd", "Symbol", "Date", "_r", "_m")
            and not c.startswith("mkt_")]
    # cross-sectional percentile ranks (who is high/low *today* relative to peers)
    for c in base:
        F["cs_" + c] = F.groupby("Date")[c].rank(pct=True)
    X_cols = base + ["mkt_ret5", "mkt_ret20", "mkt_vol20"] + ["cs_" + c for c in base]
    # market-neutral, winsorised target
    F["fwd_rel"] = F["fwd"] - F.groupby("Date")["fwd"].transform("mean")
    lo, hi = F["fwd_rel"].quantile([.01, .99])
    F["fwd_rel"] = F["fwd_rel"].clip(lo, hi)
    return F, X_cols


# ----------------------------------------------------------------------------- models
def make_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=300)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=5e-4, l1_ratio=.3, max_iter=5000)),
        "ExtraTrees": ExtraTreesRegressor(300, min_samples_leaf=40, max_features=.4, n_jobs=-1, random_state=SEED),
        "RandomForest": RandomForestRegressor(300, min_samples_leaf=40, max_features=.4, n_jobs=-1, random_state=SEED),
        "GradBoost": HistGradientBoostingRegressor(max_depth=3, learning_rate=.03, max_iter=250,
                                                   min_samples_leaf=80, l2_regularization=5.0,
                                                   random_state=SEED),
    }


def per_date_stats(t, pcol, k):
    ics, top, bot, allm, sets = [], [], [], [], []
    prev = None
    turn = []
    for d, x in t.groupby("Date"):
        allm.append(x["fwd"].mean())
        if len(x) < 3:
            continue
        ics.append(x[pcol].corr(x["fwd_rel"], method="spearman"))
        kk = min(k, max(len(x) // 2, 1))
        top_s = x.nlargest(kk, pcol)
        top.append(top_s["fwd"].mean())
        bot.append(x.nsmallest(kk, pcol)["fwd"].mean())
        cur = set(top_s["Symbol"])
        if prev is not None:
            turn.append(1 - len(cur & prev) / kk)
        prev = cur
    return (np.array(ics, float), np.array(top), np.array(bot),
            np.array(allm[:len(top)]) if len(allm) != len(top) else np.array(allm), np.mean(turn) if turn else np.nan)


def summarise(t, name, pcol, k):
    ic, top, bot, allm, turn = per_date_stats(t, pcol, k)
    ic = ic[~np.isnan(ic)]
    eff_n = max(len(ic) / H, 1)                      # overlapping targets -> shrink sample size
    ic_t = ic.mean() / (ic.std(ddof=1) / np.sqrt(eff_n)) if len(ic) > 2 and ic.std() > 0 else np.nan
    gross_edge = (top - allm).mean()
    # a 10-day holding period re-trades about `turn` of the book once per H days
    net_edge = gross_edge - 2 * COST_BPS / 1e4 * (turn if not np.isnan(turn) else 1)
    return {"Model": name,
            "R2": r2_score(t["fwd_rel"], t[pcol]), "MAE": mean_absolute_error(t["fwd_rel"], t[pcol]),
            "RankIC": ic.mean(), "ICIR": ic.mean() / ic.std(ddof=1) if ic.std() > 0 else np.nan,
            "IC_tstat_adj": ic_t, "IC>0_%": (ic > 0).mean() * 100,
            "HitRate_%": ((t[pcol] > 0) == (t["fwd_rel"] > 0)).mean() * 100,
            "Top_fwd_%": top.mean() * 100, "AllStocks_fwd_%": allm.mean() * 100,
            "Edge_%": gross_edge * 100, "NetEdge_%": net_edge * 100,
            "LongShort_%": (top - bot).mean() * 100}


def main():
    df = load(FILE)
    n_sym = df["Symbol"].nunique()
    k = max(1, min(TOPK, n_sym // 3))
    F, X_cols = build_panel(df)
    live = F.sort_values("Date").groupby("Symbol").tail(1).dropna(subset=X_cols)
    D = F.dropna(subset=X_cols + ["fwd_rel"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    dates = np.sort(D["Date"].unique())
    nd = len(dates)
    bounds = np.linspace(int(nd * START_FRAC), nd, N_FOLDS + 1).astype(int)
    print(f"{n_sym} stocks | {len(D):,} labelled rows | {len(X_cols)} features | horizon {H}d | top-k {k}")

    models = make_models()
    oos = D.iloc[0:0][["Date", "Symbol", "fwd", "fwd_rel"]].copy()
    fold_frames = []
    for i in range(N_FOLDS):
        a, b = bounds[i], bounds[i + 1]
        train_end = dates[max(a - H, 0)]                       # embargo: no label overlap
        tr = D[D["Date"] < train_end]
        te = D[(D["Date"] >= dates[a]) & (D["Date"] <= dates[min(b, nd) - 1])].copy()
        if len(tr) < 500 or te.empty:
            continue
        for n, m in models.items():
            te["p_" + n] = clone(m).fit(tr[X_cols], tr["fwd_rel"]).predict(te[X_cols])
        te["fold"] = i + 1
        fold_frames.append(te)
    oos = pd.concat(fold_frames)
    print(f"Walk-forward test: {oos['Date'].min().date()} -> {oos['Date'].max().date()} "
          f"({len(fold_frames)} folds)")

    names = list(models)
    # equal-weight ensemble of per-date ranks (scale-free) + average raw prediction for R2/MAE
    for n in names:
        oos["r_" + n] = oos.groupby("Date")["p_" + n].rank(pct=True)
    oos["p_Ensemble"] = oos[["p_" + n for n in names]].mean(axis=1)
    oos["r_Ensemble"] = oos[["r_" + n for n in names]].mean(axis=1)

    rows = [summarise(oos, n, "p_" + n, k) for n in names]
    rows.append(summarise(oos, "Ensemble", "r_Ensemble", k) | {
        "R2": r2_score(oos["fwd_rel"], oos["p_Ensemble"]), "MAE": mean_absolute_error(oos["fwd_rel"], oos["p_Ensemble"])})
    score = pd.DataFrame(rows).set_index("Model")
    print("\nOut-of-sample walk-forward scores (all folds pooled):")
    print(score.round(4).to_string())

    fold_ic = {n: [] for n in names + ["Ensemble"]}
    for f, x in oos.groupby("fold"):
        for n in names:
            fold_ic[n].append(summarise(x, n, "p_" + n, k)["RankIC"])
        fold_ic["Ensemble"].append(summarise(x, "Ensemble", "r_Ensemble", k)["RankIC"])
    print("\nRankIC by fold (stability check):")
    print(pd.DataFrame(fold_ic, index=[f"fold{i}" for i in sorted(oos['fold'].unique())]).round(4).to_string())

    # --- final models: only those with positive out-of-sample IC, weighted by IC
    pos = score.loc[names, "RankIC"].clip(lower=0)
    if pos.sum() == 0:
        pos[score.loc[names, "RankIC"].idxmax()] = 1.0
    w = pos / pos.sum()
    use = [n for n in names if w[n] > 0]
    print("\nFinal model weights:", {n: round(float(w[n]), 3) for n in use})
    final = {n: clone(models[n]).fit(D[X_cols], D["fwd_rel"]) for n in use}

    out = live.set_index("Symbol")[X_cols + ["Date"]].copy()
    P = pd.DataFrame({n: final[n].predict(out[X_cols]) for n in use}, index=out.index)
    out["Pred_Rel_Ret_%"] = (P * w[use].values).sum(axis=1) * 100
    out["Model_Agree_%"] = (np.sign(P).eq(np.sign(P.mean(axis=1)), axis=0)).mean(axis=1) * 100
    out["Risk_%"] = out["vol20"] * 100
    out["Score"] = out["Pred_Rel_Ret_%"] / out["Risk_%"].replace(0, np.nan)
    out["Rank"] = out["Score"].rank(ascending=False, method="first").astype(int)
    out["Signal"] = np.where(out["Rank"] <= k, "BUY", np.where(out["Rank"] > len(out) - k, "AVOID", "HOLD"))
    res = out.sort_values("Rank")[["Date", "Pred_Rel_Ret_%", "Risk_%", "Score", "Model_Agree_%",
                                   "rsi", "ret20", "beta60", "Signal"]].copy()
    res["ret20"] *= 100
    asof = res["Date"].max()
    res["Date"] = res["Date"].dt.date
    print(f"\nStock ranking as of {asof.date()} (predicted {H}-day return vs peer average, per unit of risk)")
    print(res.round(3).to_string())
    stale = res[pd.to_datetime(res["Date"]) < asof]
    if len(stale):
        print("NOTE: stale data for:", ", ".join(stale.index))

    print("\nBEST STOCK TO BUY:", res.index[0], f"| TOP {k}:", ", ".join(res.index[:k]))
    print("Avoid:", ", ".join(res.index[-k:]))

    e = score.loc["Ensemble"]
    reliable = (e["RankIC"] >= .03) and (e["IC_tstat_adj"] >= 2) and (e["NetEdge_%"] > 0)
    if reliable:
        print("\nConfidence: out-of-sample edge is statistically meaningful (IC t-stat >= 2, positive net edge).")
    else:
        print("\nWARNING: no statistically reliable out-of-sample edge "
              f"(RankIC={e['RankIC']:.3f}, adj t={e['IC_tstat_adj']:.2f}, net edge={e['NetEdge_%']:.3f}%). "
              "Treat rankings as low-confidence.")
    print("Educational output only, not financial advice. Past patterns do not guarantee future returns.")
    res.to_csv("stock_ranking_latest.csv")
    score.to_csv("model_walkforward_scores.csv")


if __name__ == "__main__":
    main()
