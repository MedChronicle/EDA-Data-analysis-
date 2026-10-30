import argparse
import logging
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    classification_report,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
PALETTE = "Set2"
LINE = "=" * 78

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("iris_eda")


@dataclass
class Config:
    out_dir: Path = Path("eda_output")
    seed: int = 42
    dpi: int = 150
    alpha: float = 0.05
    cv_splits: int = 5
    cv_repeats: int = 3
    skip_tsne: bool = False
    sections: list = field(default_factory=list)


class Report:
    def __init__(self, path):
        self.path = path
        self.buffer = []

    def section(self, title, body=""):
        block = f"\n{LINE}\n{title.upper()}\n{LINE}\n{body}".rstrip()
        self.buffer.append(block)
        print(block)

    def save(self):
        self.path.write_text("\n".join(self.buffer) + "\n", encoding="utf-8")
        log.info("Report written to %s", self.path)


def pretty(name):
    return name.replace("_", " ").title()


def save_fig(fig, cfg, name):
    path = cfg.out_dir / "plots" / f"{name}.png"
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path.name)


def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df[TARGET] = pd.Categorical(df["target"].map(dict(enumerate(iris.target_names))), categories=list(iris.target_names), ordered=True)
    df = df.drop(columns=["target"])
    df.columns = FEATURES + [TARGET]
    return df


def data_overview(df, rep):
    buf = [
        f"Shape: {df.shape}",
        f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB",
        f"Missing values: {int(df.isnull().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "Head:",
        df.head().to_string(),
        "",
        "Dtypes:",
        df.dtypes.to_string(),
        "",
        "Class distribution:",
        pd.concat([df[TARGET].value_counts(), df[TARGET].value_counts(normalize=True).mul(100).round(2).rename("percent")], axis=1).to_string(),
    ]
    rep.section("Dataset overview", "\n".join(buf))


def descriptive_stats(df, rep):
    desc = df[FEATURES].describe().T
    desc["range"] = desc["max"] - desc["min"]
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["cv_%"] = desc["std"] / desc["mean"] * 100
    desc["skew"] = df[FEATURES].skew()
    desc["kurtosis"] = df[FEATURES].kurt()
    desc["sem"] = df[FEATURES].sem()
    ci = stats.t.interval(0.95, len(df) - 1, loc=desc["mean"], scale=desc["sem"])
    desc["ci95_low"], desc["ci95_high"] = ci
    rep.section("Descriptive statistics", desc.round(3).to_string())
    by_species = df.groupby(TARGET, observed=True)[FEATURES].agg(["mean", "std", "median"]).round(3)
    rep.section("Statistics by species", by_species.T.to_string())


def outlier_analysis(df, rep, cfg):
    rows = []
    for f in FEATURES:
        q1, q3 = df[f].quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_mask = (df[f] < q1 - 1.5 * iqr) | (df[f] > q3 + 1.5 * iqr)
        z_mask = np.abs(stats.zscore(df[f])) > 3
        mad = stats.median_abs_deviation(df[f], scale="normal")
        mod_z = (df[f] - df[f].median()) / (mad if mad else 1)
        rows.append({"feature": f, "iqr_outliers": int(iqr_mask.sum()), "zscore_outliers": int(z_mask.sum()), "mad_outliers": int((np.abs(mod_z) > 3.5).sum())})
    table = pd.DataFrame(rows).set_index("feature")

    x = df[FEATURES].values
    centered = x - x.mean(axis=0)
    inv_cov = np.linalg.inv(np.cov(x, rowvar=False))
    d2 = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    threshold = stats.chi2.ppf(0.999, df=len(FEATURES))
    flagged = df.assign(mahalanobis_d2=d2)[d2 > threshold]
    body = table.to_string() + f"\n\nMultivariate outliers (Mahalanobis, chi2 p<0.001, threshold={threshold:.2f}): {len(flagged)}"
    if len(flagged):
        body += "\n" + flagged.round(3).to_string()
    rep.section("Outlier analysis", body)
    return d2, threshold


def normality_and_variance(df, rep, cfg):
    rows = []
    for f in FEATURES:
        w, p = stats.shapiro(df[f])
        row = {"feature": f, "overall_W": w, "overall_p": p}
        for s, g in df.groupby(TARGET, observed=True)[f]:
            row[f"{s}_p"] = stats.shapiro(g)[1]
        groups = [g.values for _, g in df.groupby(TARGET, observed=True)[f]]
        row["levene_p"] = stats.levene(*groups, center="median")[1]
        row["bartlett_p"] = stats.bartlett(*groups)[1]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("feature")
    rep.section("Normality (Shapiro-Wilk) and variance homogeneity", table.round(4).to_string() + f"\n\nAlpha = {cfg.alpha}")


def group_comparisons(df, rep, cfg):
    rows = []
    tukey_blocks = []
    levels = list(df[TARGET].cat.categories)
    for f in FEATURES:
        groups = [df.loc[df[TARGET] == s, f].values for s in levels]
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        grand = df[f].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((df[f] - grand) ** 2).sum()
        eta2 = ss_between / ss_total
        omega2 = (ss_between - (len(groups) - 1) * (ss_total - ss_between) / (len(df) - len(groups))) / (ss_total + (ss_total - ss_between) / (len(df) - len(groups)))
        size = "large" if eta2 >= 0.14 else "medium" if eta2 >= 0.06 else "small"
        rows.append({"feature": f, "F": f_stat, "anova_p": p_anova, "H": h_stat, "kruskal_p": p_kw, "eta_sq": eta2, "omega_sq": omega2, "effect": size})
        res = stats.tukey_hsd(*groups)
        pairs = []
        for i, j in combinations(range(len(levels)), 2):
            ci = res.confidence_interval()
            pairs.append({"pair": f"{levels[i]} vs {levels[j]}", "mean_diff": groups[i].mean() - groups[j].mean(), "p_adj": res.pvalue[i, j], "ci_low": ci.low[i, j], "ci_high": ci.high[i, j], "significant": res.pvalue[i, j] < cfg.alpha})
        tukey_blocks.append(f"\n[{f}]\n" + pd.DataFrame(pairs).round(4).to_string(index=False))
    table = pd.DataFrame(rows).set_index("feature")
    rep.section("Group comparisons (ANOVA, Kruskal-Wallis, effect sizes)", table.round(4).to_string())
    rep.section("Tukey HSD post-hoc tests", "\n".join(tukey_blocks))
    return table


def correlation_analysis(df, rep, cfg):
    pearson = df[FEATURES].corr(method="pearson")
    spearman = df[FEATURES].corr(method="spearman")
    pvals = pd.DataFrame(np.ones((len(FEATURES), len(FEATURES))), index=FEATURES, columns=FEATURES)
    for a, b in combinations(FEATURES, 2):
        p = stats.pearsonr(df[a], df[b])[1]
        pvals.loc[a, b] = pvals.loc[b, a] = p
    vif = pd.Series(np.diag(np.linalg.inv(pearson.values)), index=FEATURES, name="VIF")
    within = {s: g[FEATURES].corr().round(2) for s, g in df.groupby(TARGET, observed=True)}
    body = "Pearson:\n" + pearson.round(3).to_string() + "\n\nSpearman:\n" + spearman.round(3).to_string() + "\n\nPearson p-values:\n" + pvals.round(5).to_string() + "\n\nVariance Inflation Factors:\n" + vif.round(2).to_string()
    for s, c in within.items():
        body += f"\n\nWithin-class correlation ({s}):\n" + c.to_string()
    rep.section("Correlation analysis", body)
    return pearson, spearman


def pca_analysis(df, rep, cfg):
    x = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA(random_state=cfg.seed).fit(x)
    evr = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=[f"PC{i + 1}" for i in range(len(FEATURES))])
    summary = pd.DataFrame({"explained_%": evr * 100, "cumulative_%": np.cumsum(evr) * 100, "eigenvalue": pca.explained_variance_}, index=loadings.columns)
    rep.section("PCA (standardised)", summary.round(3).to_string() + "\n\nLoadings:\n" + loadings.round(3).to_string())
    return pca, x, loadings


def clustering_analysis(df, x, rep, cfg):
    y_true = df[TARGET].cat.codes.values
    rows = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=cfg.seed).fit(x)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(x, km.labels_), "ARI": adjusted_rand_score(y_true, km.labels_), "NMI": normalized_mutual_info_score(y_true, km.labels_)})
    table = pd.DataFrame(rows).set_index("k")
    km3 = KMeans(n_clusters=3, n_init=20, random_state=cfg.seed).fit(x)
    crosstab = pd.crosstab(df[TARGET], km3.labels_, rownames=["species"], colnames=["cluster"])
    rep.section("Clustering (K-Means)", table.round(4).to_string() + "\n\nSpecies vs cluster (k=3):\n" + crosstab.to_string())
    return table, km3.labels_


def classification_benchmark(df, rep, cfg):
    x, y = df[FEATURES].values, df[TARGET].cat.codes.values
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg.seed)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=cfg.seed)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "LDA": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=cfg.seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=cfg.seed),
    }
    cv = RepeatedStratifiedKFold(n_splits=cfg.cv_splits, n_repeats=cfg.cv_repeats, random_state=cfg.seed)
    rows, scores = [], {}
    for name, model in models.items():
        res = cross_validate(model, x, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        scores[name] = res["test_accuracy"]
        rows.append({"model": name, "acc_mean": res["test_accuracy"].mean(), "acc_std": res["test_accuracy"].std(), "f1_macro": res["test_f1_macro"].mean()})
    table = pd.DataFrame(rows).set_index("model").sort_values("acc_mean", ascending=False)
    best = table.index[0]
    y_pred = cross_val_predict(models[best], x, y, cv=StratifiedKFold(cfg.cv_splits, shuffle=True, random_state=cfg.seed))
    names = list(df[TARGET].cat.categories)
    report = classification_report(y, y_pred, target_names=names, digits=3)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, stratify=y, random_state=cfg.seed)
    rf = RandomForestClassifier(n_estimators=300, random_state=cfg.seed).fit(x_tr, y_tr)
    perm = permutation_importance(rf, x_te, y_te, n_repeats=30, random_state=cfg.seed)
    importance = pd.DataFrame({"gini": rf.feature_importances_, "perm_mean": perm.importances_mean, "perm_std": perm.importances_std}, index=FEATURES).sort_values("perm_mean", ascending=False)
    rep.section("Classification benchmark", table.round(4).to_string() + f"\n\nBest model: {best}\n\n{report}\nFeature importance (Random Forest):\n" + importance.round(4).to_string())
    return table, scores, models[best], best, y, y_pred, importance


def plot_class_distribution(df, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df[TARGET].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=axes[0])
    for container in axes[0].containers:
        axes[0].bar_label(container, fontsize=11)
    axes[0].set_title("Species Distribution")
    axes[0].set_xlabel("Species")
    axes[0].set_ylabel("Count")
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette(PALETTE, len(counts)), startangle=90, wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Class Balance")
    save_fig(fig, cfg, "class_distribution")


def plot_histograms(df, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, f in zip(axes.flat, FEATURES):
        sns.histplot(data=df, x=f, hue=TARGET, kde=True, element="step", palette=PALETTE, ax=ax)
        ax.axvline(df[f].mean(), color="black", linestyle="--", linewidth=1, label="Overall mean")
        ax.set_title(f"Distribution of {pretty(f)}")
    save_fig(fig, cfg, "histograms")


def plot_ecdf(df, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, f in zip(axes.flat, FEATURES):
        sns.ecdfplot(data=df, x=f, hue=TARGET, palette=PALETTE, ax=ax, linewidth=2)
        ax.set_title(f"ECDF of {pretty(f)}")
    save_fig(fig, cfg, "ecdf")


def plot_distribution_grid(df, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, f in zip(axes.flat, FEATURES):
        sns.violinplot(data=df, x=TARGET, y=f, hue=TARGET, palette=PALETTE, inner=None, cut=0, legend=False, ax=ax, alpha=0.6)
        sns.boxplot(data=df, x=TARGET, y=f, width=0.15, showcaps=False, boxprops={"facecolor": "white", "zorder": 3}, showfliers=False, whiskerprops={"linewidth": 1.5}, ax=ax)
        sns.stripplot(data=df, x=TARGET, y=f, color="black", size=2.5, alpha=0.5, jitter=0.2, ax=ax)
        ax.set_title(f"{pretty(f)} by Species")
    save_fig(fig, cfg, "violin_box_strip")


def plot_boxen_swarm(df, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, f in zip(axes.flat, FEATURES):
        sns.boxenplot(data=df, x=TARGET, y=f, hue=TARGET, palette=PALETTE, legend=False, ax=ax)
        sns.swarmplot(data=df, x=TARGET, y=f, color="black", size=2.5, alpha=0.6, ax=ax)
        ax.set_title(f"{pretty(f)} Boxen and Swarm")
    save_fig(fig, cfg, "boxen_swarm")


def plot_pairplot(df, cfg):
    g = sns.pairplot(df, hue=TARGET, palette=PALETTE, diag_kind="kde", corner=False, height=2.3, plot_kws={"alpha": 0.8, "s": 35})
    g.map_lower(sns.kdeplot, levels=4, color=".2", linewidths=0.6)
    g.figure.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=14, fontweight="bold")
    save_fig(g.figure, cfg, "pairplot")


def plot_correlations(pearson, spearman, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mask = np.triu(np.ones_like(pearson, dtype=bool), k=1)
    for ax, mat, title in zip(axes, (pearson, spearman), ("Pearson", "Spearman")):
        sns.heatmap(mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f"{title} Correlation")
        ax.set_xticklabels([pretty(t.get_text()) for t in ax.get_xticklabels()], rotation=45, ha="right")
        ax.set_yticklabels([pretty(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    save_fig(fig, cfg, "correlation_heatmaps")


def plot_clustermap(df, cfg):
    scaled = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
    lut = dict(zip(df[TARGET].cat.categories, sns.color_palette(PALETTE, 3)))
    g = sns.clustermap(scaled, row_colors=[lut[v] for v in df[TARGET]], cmap="vlag", figsize=(9, 10), yticklabels=False, method="ward", center=0)
    g.figure.suptitle("Hierarchical Clustermap (z-scored)", y=1.02, fontweight="bold")
    save_fig(g.figure, cfg, "clustermap")


def plot_scatter_relationships(df, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (xf, yf) in zip(axes, (("sepal_length", "sepal_width"), ("petal_length", "petal_width"))):
        sns.scatterplot(data=df, x=xf, y=yf, hue=TARGET, style=TARGET, s=80, palette=PALETTE, ax=ax)
        for s, g in df.groupby(TARGET, observed=True):
            if len(g) > 2:
                cov = np.cov(g[xf], g[yf])
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * 2.4477 * np.sqrt(vals)
                color = sns.color_palette(PALETTE, 3)[list(df[TARGET].cat.categories).index(s)]
                ax.add_patch(matplotlib.patches.Ellipse((g[xf].mean(), g[yf].mean()), w, h, angle=angle, fill=False, edgecolor=color, linewidth=1.8, linestyle="--"))
        ax.set_title(f"{pretty(xf)} vs {pretty(yf)} (95% ellipses)")
    save_fig(fig, cfg, "scatter_ellipses")


def plot_jointplot(df, cfg):
    g = sns.jointplot(data=df, x="petal_length", y="petal_width", hue=TARGET, palette=PALETTE, height=7, kind="scatter", marginal_kws={"fill": True})
    g.plot_joint(sns.kdeplot, levels=4, linewidths=0.8)
    g.figure.suptitle("Petal Length vs Width Joint Distribution", y=1.02, fontweight="bold")
    save_fig(g.figure, cfg, "jointplot")


def plot_feature_means(df, cfg):
    means = df.groupby(TARGET, observed=True)[FEATURES].mean()
    sems = df.groupby(TARGET, observed=True)[FEATURES].sem()
    fig, ax = plt.subplots(figsize=(11, 6))
    means.plot(kind="bar", yerr=sems * 1.96, capsize=4, colormap=PALETTE, ax=ax, edgecolor="white")
    ax.set_title("Mean Feature Values by Species (95% CI)")
    ax.set_ylabel("Mean Value")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, cfg, "feature_means")


def plot_parallel_coordinates(df, cfg):
    scaled = df.copy()
    scaled[FEATURES] = (scaled[FEATURES] - scaled[FEATURES].min()) / (scaled[FEATURES].max() - scaled[FEATURES].min())
    scaled[TARGET] = scaled[TARGET].astype(str)
    fig, ax = plt.subplots(figsize=(11, 6))
    parallel_coordinates(scaled, TARGET, color=sns.color_palette(PALETTE, 3), alpha=0.5, ax=ax)
    ax.set_title("Parallel Coordinates (min-max scaled)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Scaled Value")
    save_fig(fig, cfg, "parallel_coordinates")


def plot_radar(df, cfg):
    means = df.groupby(TARGET, observed=True)[FEATURES].mean()
    norm = (means - means.min()) / (means.max() - means.min())
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for color, (s, row) in zip(sns.color_palette(PALETTE, len(norm)), norm.iterrows()):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, label=s, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([pretty(f) for f in FEATURES])
    ax.set_title("Normalised Feature Radar", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save_fig(fig, cfg, "radar_chart")


def plot_3d_scatter(df, cfg):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for color, (s, g) in zip(sns.color_palette(PALETTE, 3), df.groupby(TARGET, observed=True)):
        ax.scatter(g["sepal_length"], g["sepal_width"], g["petal_length"], label=s, color=color, s=50, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.set_zlabel("Petal Length")
    ax.set_title("3D Feature Space")
    ax.view_init(elev=22, azim=-55)
    ax.legend()
    save_fig(fig, cfg, "scatter_3d")


def plot_mahalanobis(df, d2, threshold, cfg):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = df[TARGET].cat.codes.map(dict(enumerate(sns.color_palette(PALETTE, 3))))
    ax.scatter(range(len(df)), d2, c=list(colors), s=30)
    ax.axhline(threshold, color="red", linestyle="--", label=f"chi2 99.9% = {threshold:.2f}")
    ax.set_title("Mahalanobis Distance per Observation")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Squared Distance")
    ax.legend()
    save_fig(fig, cfg, "mahalanobis")


def plot_pca(df, pca, x, loadings, cfg):
    comps = pca.transform(x)
    evr = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].bar(range(1, len(evr) + 1), evr * 100, color="steelblue", alpha=0.8, label="Individual")
    axes[0].plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, "o-", color="crimson", label="Cumulative")
    axes[0].set_title("Scree Plot")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_xticks(range(1, len(evr) + 1))
    axes[0].legend()
    sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=df[TARGET].values, style=df[TARGET].values, s=90, palette=PALETTE, ax=axes[1])
    axes[1].set_title(f"PCA Projection ({evr[:2].sum() * 100:.2f}% variance)")
    axes[1].set_xlabel(f"PC1 ({evr[0] * 100:.2f}%)")
    axes[1].set_ylabel(f"PC2 ({evr[1] * 100:.2f}%)")
    scale = np.abs(comps[:, :2]).max()
    sns.scatterplot(x=comps[:, 0] / scale, y=comps[:, 1] / scale, hue=df[TARGET].values, palette=PALETTE, s=30, alpha=0.4, legend=False, ax=axes[2])
    for f in FEATURES:
        vx, vy = loadings.loc[f, "PC1"], loadings.loc[f, "PC2"]
        axes[2].arrow(0, 0, vx, vy, color="black", head_width=0.03, length_includes_head=True)
        axes[2].text(vx * 1.12, vy * 1.12, pretty(f), ha="center", fontsize=10)
    axes[2].set_title("PCA Biplot")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    save_fig(fig, cfg, "pca_analysis")


def plot_lda_tsne(df, cfg):
    x = StandardScaler().fit_transform(df[FEATURES])
    y = df[TARGET].cat.codes.values
    lda = LinearDiscriminantAnalysis(n_components=2).fit(x, y)
    z = lda.transform(x)
    panels = [("LDA Projection", z, "LD1", "LD2")]
    if not cfg.skip_tsne:
        emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=cfg.seed).fit_transform(x)
        panels.append(("t-SNE Embedding", emb, "t-SNE 1", "t-SNE 2"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6), squeeze=False)
    for ax, (title, data, xl, yl) in zip(axes[0], panels):
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=df[TARGET].values, style=df[TARGET].values, s=80, palette=PALETTE, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    save_fig(fig, cfg, "lda_tsne")


def plot_clustering(df, x, table, labels, pca, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].plot(table.index, table["inertia"], "o-", color="steelblue")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(table.index, table["silhouette"], "o-", color="darkgreen", label="Silhouette")
    axes[1].plot(table.index, table["ARI"], "s--", color="darkorange", label="ARI vs species")
    axes[1].set_title("Cluster Quality")
    axes[1].set_xlabel("k")
    axes[1].legend()
    comps = pca.transform(x)
    sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=labels, palette="Dark2", s=70, ax=axes[2])
    axes[2].set_title("K-Means (k=3) on PCA Space")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    save_fig(fig, cfg, "clustering")


def plot_model_results(table, scores, y, y_pred, best, names, importance, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    data = pd.DataFrame({k: v for k, v in scores.items()}).melt(var_name="model", value_name="accuracy")
    order = table.index.tolist()
    sns.boxplot(data=data, x="accuracy", y="model", order=order, hue="model", palette="Set3", legend=False, ax=axes[0])
    axes[0].set_title("Cross-Validated Accuracy")
    ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=names, cmap="Blues", ax=axes[1], colorbar=False)
    axes[1].set_title(f"Confusion Matrix ({best})")
    imp = importance.sort_values("perm_mean")
    axes[2].barh(imp.index, imp["perm_mean"], xerr=imp["perm_std"], color="slateblue", alpha=0.8)
    axes[2].set_title("Permutation Importance (Random Forest)")
    axes[2].set_xlabel("Mean Accuracy Decrease")
    save_fig(fig, cfg, "model_results")


def plot_effect_sizes(effects, cfg):
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = effects.sort_values("eta_sq")
    ax.barh(ordered.index, ordered["eta_sq"], color=sns.color_palette("viridis", len(ordered)))
    for x0, color in ((0.01, "grey"), (0.06, "orange"), (0.14, "red")):
        ax.axvline(x0, color=color, linestyle=":", linewidth=1)
    ax.set_title("Effect Size (Eta Squared) of Species on Each Feature")
    ax.set_xlabel("Eta Squared")
    ax.set_xlim(0, 1)
    save_fig(fig, cfg, "effect_sizes")


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced exploratory data analysis of the Iris dataset")
    parser.add_argument("--out", type=Path, default=Path("eda_output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--skip-tsne", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(out_dir=args.out, seed=args.seed, dpi=args.dpi, alpha=args.alpha, cv_splits=args.cv_splits, cv_repeats=args.cv_repeats, skip_tsne=args.skip_tsne)
    (cfg.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({"axes.titlesize": 13, "axes.titleweight": "bold", "figure.figsize": (10, 6)})

    rep = Report(cfg.out_dir / "eda_report.txt")
    df = load_data()
    df.to_csv(cfg.out_dir / "iris_clean.csv", index=False)

    data_overview(df, rep)
    descriptive_stats(df, rep)
    d2, threshold = outlier_analysis(df, rep, cfg)
    normality_and_variance(df, rep, cfg)
    effects = group_comparisons(df, rep, cfg)
    pearson, spearman = correlation_analysis(df, rep, cfg)
    pca, x_scaled, loadings = pca_analysis(df, rep, cfg)
    cluster_table, labels = clustering_analysis(df, x_scaled, rep, cfg)
    model_table, scores, best_model, best_name, y, y_pred, importance = classification_benchmark(df, rep, cfg)

    plot_class_distribution(df, cfg)
    plot_histograms(df, cfg)
    plot_ecdf(df, cfg)
    plot_distribution_grid(df, cfg)
    plot_boxen_swarm(df, cfg)
    plot_pairplot(df, cfg)
    plot_correlations(pearson, spearman, cfg)
    plot_clustermap(df, cfg)
    plot_scatter_relationships(df, cfg)
    plot_jointplot(df, cfg)
    plot_feature_means(df, cfg)
    plot_parallel_coordinates(df, cfg)
    plot_radar(df, cfg)
    plot_3d_scatter(df, cfg)
    plot_mahalanobis(df, d2, threshold, cfg)
    plot_pca(df, pca, x_scaled, loadings, cfg)
    plot_lda_tsne(df, cfg)
    plot_clustering(df, x_scaled, cluster_table, labels, pca, cfg)
    plot_model_results(model_table, scores, y, y_pred, best_name, list(df[TARGET].cat.categories), importance, cfg)
    plot_effect_sizes(effects, cfg)

    rep.section("Completed", f"All outputs saved to: {cfg.out_dir.resolve()}")
    rep.save()


if __name__ == "__main__":
    main()
