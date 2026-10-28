import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import DecisionBoundaryDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, learning_curve,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
log = logging.getLogger("iris")


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        log.warning("%s not found - using sklearn's built-in iris dataset", path)
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.copy()
        df.columns = COLS
        df["species"] = df["species"].map(dict(enumerate(d.target_names)))
        return df

    df = pd.read_csv(p, header=None, names=COLS)
    if pd.to_numeric(df.iloc[0, :4], errors="coerce").isna().any():
        df = df.iloc[1:].reset_index(drop=True)
    df[COLS[:4]] = df[COLS[:4]].apply(pd.to_numeric, errors="coerce")
    df["species"] = df["species"].astype(str).str.strip()
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna()
    log.info("Rows: %d (dropped %d with missing values)", len(df), n0 - len(df))
    log.info("Duplicate rows: %d", df.duplicated().sum())
    log.info("Class balance:\n%s", df["species"].value_counts().to_string())
    return df.reset_index(drop=True)


def plot_eda(df: pd.DataFrame, out: Path) -> None:
    g = sns.pairplot(df, hue="species", corner=True, diag_kind="kde")
    g.figure.suptitle("Pairplot", y=1.02)
    g.savefig(out / "eda_pairplot.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(df[COLS[:4]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(out / "eda_correlation.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict, out: Path) -> None:
    names = list(results)
    means = [results[n]["f1_macro_mean"] for n in names]
    stds = [results[n]["f1_macro_std"] for n in names]
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.array(names)[order], np.array(means)[order],
            xerr=np.array(stds)[order], color="steelblue", capsize=4)
    ax.set(xlabel="Macro-F1 (nested CV, mean ± std)", title="Model comparison")
    lo = max(0.0, min(means) - 0.1)
    ax.set_xlim(lo, 1.0)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, classes, title, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, norm, fmt in zip(axes, [None, "true"], ["d", ".2f"]):
        cm = confusion_matrix(y_true, y_pred, normalize=norm)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual",
               title="Counts" if norm is None else "Row-normalised")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(model, X_test, y_test, classes, out: Path) -> dict:
    proba = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=range(len(classes)))
    aucs = {}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        aucs[c] = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{c} (AUC={aucs[c]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="One-vs-rest ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150)
    plt.close(fig)
    return aucs


def plot_learning_curve(model, X, y, cv, out: Path) -> None:
    sizes, tr, va = learning_curve(model, X, y, cv=cv, n_jobs=-1,
                                   train_sizes=np.linspace(0.2, 1.0, 6),
                                   scoring="accuracy", random_state=RANDOM_STATE,
                                   shuffle=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for arr, lbl in [(tr, "Train"), (va, "Validation")]:
        ax.plot(sizes, arr.mean(1), "o-", label=lbl)
        ax.fill_between(sizes, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), alpha=0.15)
    ax.set(xlabel="Training samples", ylabel="Accuracy", title="Learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)
    plt.close(fig)


def plot_decision_boundary(model, X_train, y_train, classes, out: Path) -> None:
    pca_pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(2, random_state=RANDOM_STATE))])
    Z = pca_pipe.fit_transform(X_train)
    clf = clone(model.named_steps["clf"]).fit(Z, y_train)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    DecisionBoundaryDisplay.from_estimator(clf, Z, response_method="predict",
                                           alpha=0.25, cmap="Set2", ax=ax)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y_train, cmap="Set2", edgecolor="k", s=35)
    ax.legend(sc.legend_elements()[0], classes, title="Species")
    ev = pca_pipe["pca"].explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ev[0]:.0%})", ylabel=f"PC2 ({ev[1]:.0%})",
           title="Decision boundary (PCA projection)")
    fig.tight_layout()
    fig.savefig(out / "decision_boundary.png", dpi=150)
    plt.close(fig)


def build_candidates() -> dict:
    def pipe(clf):
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])

    cands = {
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {"clf__C": [0.1, 1, 10, 100]}),
        "KNN": (
            pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]}),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            {"clf__max_depth": [2, 3, 4, 5, None], "clf__min_samples_leaf": [1, 2, 4]}),
        "Random Forest": (
            pipe(RandomForestClassifier(random_state=RANDOM_STATE)),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 3, 5]}),
        "SVM (RBF)": (
            pipe(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]}),
    }
    ens = pipe(VotingClassifier(
        [("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
         ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
         ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))],
        voting="soft"))
    cands["Soft Voting Ensemble"] = (ens, {})
    return cands


def make_searcher(estimator, grid, cv, fast: bool):
    if not grid or fast:
        return estimator
    return GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro", n_jobs=-1)


def train(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = validate(load_data(args.data))
    plot_eda(df, out)

    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    X = df[COLS[:4]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    outer = StratifiedKFold(args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE + 1)

    candidates = build_candidates()
    results = {}
    log.info("Nested %d-fold CV (tuning=%s)", args.cv_folds, args.tune)
    for name, (est, grid) in candidates.items():
        searcher = make_searcher(est, grid, inner, fast=not args.tune)
        cv_res = cross_validate(searcher, X_train, y_train, cv=outer,
                                scoring=["accuracy", "f1_macro"], n_jobs=1)
        results[name] = {
            "accuracy_mean": cv_res["test_accuracy"].mean(),
            "accuracy_std": cv_res["test_accuracy"].std(),
            "f1_macro_mean": cv_res["test_f1_macro"].mean(),
            "f1_macro_std": cv_res["test_f1_macro"].std(),
        }
        r = results[name]
        log.info("  %-22s acc %.4f ± %.4f | macro-F1 %.4f ± %.4f", name,
                 r["accuracy_mean"], r["accuracy_std"], r["f1_macro_mean"], r["f1_macro_std"])
    plot_model_comparison(results, out)

    best_name = max(results, key=lambda n: (round(results[n]["f1_macro_mean"], 4),
                                            -results[n]["f1_macro_std"]))
    est, grid = candidates[best_name]
    searcher = make_searcher(est, grid, inner, fast=not args.tune)
    searcher.fit(X_train, y_train)
    best_model = searcher.best_estimator_ if isinstance(searcher, GridSearchCV) else searcher
    best_params = searcher.best_params_ if isinstance(searcher, GridSearchCV) else {}
    log.info("Best model: %s %s", best_name, best_params)

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nBest model: {best_name}  {best_params}")
    print(f"Test accuracy: {test_acc:.4f} | Test macro-F1: {test_f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plot_confusion(y_test, y_pred, le.classes_, f"{best_name} ({test_acc:.2%})", out)
    aucs = plot_roc(best_model, X_test, y_test, le.classes_, out)
    plot_learning_curve(best_model, X_train, y_train, outer, out)
    plot_decision_boundary(best_model, X_train, y_train, le.classes_, out)

    pi = permutation_importance(best_model, X_test, y_test, n_repeats=30,
                                random_state=RANDOM_STATE, scoring="accuracy")
    importance = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    print("Permutation importance (accuracy drop):")
    print(importance.round(4).to_string())

    final_model = clone(best_model).fit(X, y)
    joblib.dump({"model": final_model, "label_encoder": le, "features": list(X.columns),
                 "model_name": best_name, "params": best_params},
                out / "iris_classifier.joblib")
    (out / "metrics.json").write_text(json.dumps({
        "best_model": best_name, "best_params": best_params,
        "test_accuracy": test_acc, "test_macro_f1": test_f1,
        "roc_auc_ovr": aucs, "cv_results": results,
        "permutation_importance": importance.to_dict(),
    }, indent=2, default=str))
    print(f"\nArtifacts written to {out.resolve()}")


def predict(args) -> None:
    bundle = joblib.load(args.model)
    vals = [float(v) for v in args.predict.split(",")]
    if len(vals) != len(bundle["features"]):
        raise SystemExit(f"Expected {len(bundle['features'])} values: {bundle['features']}")
    row = pd.DataFrame([vals], columns=bundle["features"])
    model, le = bundle["model"], bundle["label_encoder"]
    proba = model.predict_proba(row)[0]
    print(f"Prediction: {le.inverse_transform([proba.argmax()])[0]}")
    for c, p in sorted(zip(le.classes_, proba), key=lambda t: -t[1]):
        print(f"  {c:18s} {p:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Advanced Iris classifier")
    ap.add_argument("--data", default="iris_data.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="grid-search hyper-parameters (nested CV)")
    ap.add_argument("--predict", help='comma-separated features, e.g. "5.1,3.5,1.4,0.2"')
    ap.add_argument("--model", default="outputs/iris_classifier.joblib")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_theme(style="whitegrid")
    predict(args) if args.predict else train(args)


if __name__ == "__main__":
    main()
