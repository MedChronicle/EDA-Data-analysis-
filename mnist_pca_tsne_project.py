import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    silhouette_score,
)

try:
    import umap
except ImportError:
    umap = None

warnings.filterwarnings("ignore")

SEED = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_mnist():
    try:
        bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = bunch.data.astype(np.float64)
        y = bunch.target.astype(int)
        dim = (28, 28)
        name = "MNIST"
    except Exception as e:
        print(f"OpenML unreachable ({e}); using sklearn's digits dataset as fallback.")
        bunch = load_digits()
        X = bunch.data.astype(np.float64)
        y = bunch.target.astype(int)
        dim = (8, 8)
        name = "sklearn digits"
    return X, y, dim, name


def subsample(X, y, size, seed=SEED):
    if size >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=seed)
    return Xs, ys


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}", dpi=130)
    plt.close()


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def evaluate(model, Xtr, Xte, ytr, yte, label):
    _, train_time = timed(model.fit, Xtr, ytr)
    preds, predict_time = timed(model.predict, Xte)
    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average="macro")
    print(f"{label:<34} acc={acc:.4f}  macroF1={f1:.4f}  train={train_time:.3f}s  predict={predict_time:.3f}s")
    return {
        "model": label,
        "accuracy": acc,
        "macro_f1": f1,
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "preds": preds,
    }


X, y, IMG_DIM, DATA_NAME = load_mnist()
N_SAMPLES, N_FEATURES = X.shape
N_CLASSES = len(np.unique(y))

print("=" * 70)
print("PART A: DATA UNDERSTANDING")
print("=" * 70)
print(f"Dataset: {DATA_NAME}")
print(f"Number of samples: {N_SAMPLES}")
print(f"Number of features: {N_FEATURES}")
print(f"Number of classes: {N_CLASSES}")
print(f"Image dimensions: {IMG_DIM[0]} x {IMG_DIM[1]}")
print(f"Pixel range: {X.min():.0f} - {X.max():.0f}")
print(f"Sparsity (share of zero pixels): {(X == 0).mean() * 100:.2f}%")

class_counts = pd.Series(y).value_counts().sort_index()
print("\nClass distribution:")
print(pd.DataFrame({"count": class_counts, "percent": (class_counts / N_SAMPLES * 100).round(2)}))

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
rng = np.random.RandomState(0)
sample_idx = rng.choice(N_SAMPLES, 10, replace=False)
for ax, idx in zip(axes.flat, sample_idx):
    ax.imshow(X[idx].reshape(IMG_DIM), cmap="gray")
    ax.set_title(str(y[idx]))
    ax.axis("off")
save("01_sample_digits.png")

plt.figure(figsize=(7, 4))
plt.bar(class_counts.index, class_counts.values, color="steelblue")
plt.xticks(range(N_CLASSES))
plt.xlabel("Digit")
plt.ylabel("Count")
plt.title("Class Distribution")
save("01b_class_distribution.png")

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for digit, ax in zip(range(N_CLASSES), axes.flat):
    ax.imshow(X[y == digit].mean(axis=0).reshape(IMG_DIM), cmap="magma")
    ax.set_title(f"Mean of {digit}")
    ax.axis("off")
save("01c_mean_digits.png")

X = X / X.max()

SAMPLE_SIZE = min(10000, N_SAMPLES)
X_sub, y_sub = subsample(X, y, SAMPLE_SIZE)
print(f"\nUsing a stratified subsample of {len(X_sub)} images for PCA, t-SNE and classification.")

print("\n" + "=" * 70)
print("PART B: PCA")
print("=" * 70)
pca_full = PCA(random_state=SEED).fit(X_sub)
var_ratio = pca_full.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)


def components_for(threshold):
    return int(np.argmax(cum_var >= threshold) + 1)


n90, n95, n99 = components_for(0.90), components_for(0.95), components_for(0.99)
print(f"Components for 90% variance: {n90}")
print(f"Components for 95% variance: {n95}")
print(f"Components for 99% variance: {n99}")
print(f"Variance captured by first 2 components: {cum_var[1] * 100:.2f}%")
print(f"Variance captured by first 10 components: {cum_var[9] * 100:.2f}%")
print(f"Compression at 95% variance: {N_FEATURES / n95:.1f}x fewer dimensions")

pca_2d = PCA(n_components=2, random_state=SEED)
X_pca_2d = pca_2d.fit_transform(X_sub)

N_PCA50 = min(50, N_FEATURES)
pca_50d = PCA(n_components=N_PCA50, random_state=SEED)
X_pca_50d = pca_50d.fit_transform(X_sub)

n_scree = min(50, len(var_ratio))
plt.figure(figsize=(7, 5))
plt.plot(range(1, n_scree + 1), var_ratio[:n_scree], marker="o", markersize=3)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(alpha=0.3)
save("02_scree_plot.png")

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var)
for thresh, n_c in zip([0.90, 0.95, 0.99], [n90, n95, n99]):
    plt.axhline(thresh, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(n_c, color="gray", linestyle="--", linewidth=0.8)
    plt.annotate(f"{int(thresh * 100)}%: {n_c}", (n_c, thresh), textcoords="offset points", xytext=(6, -12), fontsize=8)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.grid(alpha=0.3)
save("03_cumulative_variance.png")

plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_sub, cmap="tab10", s=6, alpha=0.6)
plt.colorbar(scatter, label="Digit")
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}%)")
plt.title("2D PCA Projection")
save("04_pca_2d_projection.png")

n_eigen = 10
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for i, ax in enumerate(axes.flat[:n_eigen]):
    ax.imshow(pca_full.components_[i].reshape(IMG_DIM), cmap="RdBu_r")
    ax.set_title(f"PC{i + 1} ({var_ratio[i] * 100:.1f}%)", fontsize=9)
    ax.axis("off")
plt.suptitle("Eigen-digits (principal component images)")
save("04b_eigen_digits.png")

recon_levels = [c for c in [2, 10, 30, N_PCA50, n95] if c <= N_FEATURES]
recon_levels = sorted(set(recon_levels))
demo_idx = [np.where(y_sub == d)[0][0] for d in range(5)]
fig, axes = plt.subplots(len(recon_levels) + 1, 5, figsize=(9, 1.9 * (len(recon_levels) + 1)))
for j, idx in enumerate(demo_idx):
    axes[0, j].imshow(X_sub[idx].reshape(IMG_DIM), cmap="gray")
    axes[0, j].axis("off")
    if j == 0:
        axes[0, j].set_title("Original", fontsize=8, loc="left")
for i, k in enumerate(recon_levels, start=1):
    p = PCA(n_components=k, random_state=SEED).fit(X_sub)
    rec = p.inverse_transform(p.transform(X_sub[demo_idx]))
    for j in range(5):
        axes[i, j].imshow(rec[j].reshape(IMG_DIM), cmap="gray")
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(f"{k} components", fontsize=8, loc="left")
save("04c_reconstructions.png")

comp_grid = [c for c in [2, 5, 10, 20, 30, 50, 75, 100, 150] if c <= N_FEATURES]
recon_mse = []
for k in comp_grid:
    p = PCA(n_components=k, random_state=SEED).fit(X_sub)
    rec = p.inverse_transform(p.transform(X_sub))
    recon_mse.append(np.mean((X_sub - rec) ** 2))
plt.figure(figsize=(7, 5))
plt.plot(comp_grid, recon_mse, marker="o")
plt.yscale("log")
plt.xlabel("Number of Components")
plt.ylabel("Reconstruction MSE (log scale)")
plt.title("Reconstruction Error vs PCA Components")
plt.grid(alpha=0.3, which="both")
save("04d_reconstruction_error.png")

print("\nReconstruction MSE by component count:")
print(pd.DataFrame({"components": comp_grid, "mse": np.round(recon_mse, 6)}).to_string(index=False))

print("\n" + "=" * 70)
print("PART C: t-SNE (AND UMAP IF INSTALLED)")
print("=" * 70)
TSNE_SIZE = min(2000, len(X_sub))
X_tsne_sub, y_tsne_sub = subsample(X_sub, y_sub, TSNE_SIZE)
X_pca50_tsne_sub = pca_50d.transform(X_tsne_sub)

perplexities = [5, 30, 50]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
sc = None
tsne_rows = []
best_tsne = None
for col, perp in enumerate(perplexities):
    for row, (data, tag) in enumerate([(X_tsne_sub, "Original"), (X_pca50_tsne_sub, f"PCA-{N_PCA50}D")]):
        model = TSNE(n_components=2, perplexity=perp, random_state=SEED, init="pca", learning_rate="auto")
        emb, secs = timed(model.fit_transform, data)
        sil = silhouette_score(emb, y_tsne_sub)
        tsne_rows.append({"input": tag, "perplexity": perp, "kl_divergence": model.kl_divergence_, "silhouette": sil, "seconds": secs})
        sc = axes[row, col].scatter(emb[:, 0], emb[:, 1], c=y_tsne_sub, cmap="tab10", s=6, alpha=0.7)
        axes[row, col].set_title(f"{tag} data, perplexity={perp}\nsilhouette={sil:.3f}, {secs:.1f}s")
        if best_tsne is None or sil > best_tsne[0]:
            best_tsne = (sil, emb, tag, perp)

fig.colorbar(sc, ax=axes.ravel().tolist(), label="Digit", fraction=0.02)
plt.savefig(f"{OUT_DIR}/05_tsne_comparison.png", dpi=130)
plt.close()

tsne_df = pd.DataFrame(tsne_rows).round(4)
print(tsne_df.to_string(index=False))
print(f"\nBest t-SNE by silhouette: {best_tsne[2]} input, perplexity={best_tsne[3]} (silhouette={best_tsne[0]:.3f})")

pca2_sil = silhouette_score(pca_2d.transform(X_tsne_sub), y_tsne_sub)
print(f"Silhouette of 2D PCA projection for comparison: {pca2_sil:.3f}")

if umap is not None:
    um_emb, um_secs = timed(umap.UMAP(n_components=2, random_state=SEED).fit_transform, X_pca50_tsne_sub)
    um_sil = silhouette_score(um_emb, y_tsne_sub)
    plt.figure(figsize=(7, 6))
    s = plt.scatter(um_emb[:, 0], um_emb[:, 1], c=y_tsne_sub, cmap="tab10", s=6, alpha=0.7)
    plt.colorbar(s, label="Digit")
    plt.title(f"UMAP on PCA-{N_PCA50}D (silhouette={um_sil:.3f}, {um_secs:.1f}s)")
    save("05b_umap.png")
    print(f"UMAP: silhouette={um_sil:.3f}, time={um_secs:.1f}s")
else:
    print("UMAP skipped (pip install umap-learn to enable).")

print("\n" + "=" * 70)
print("PART D: DIGIT RECOGNITION")
print("=" * 70)
Xtr, Xte, ytr, yte = train_test_split(X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=SEED)

pca_clf = PCA(n_components=N_PCA50, random_state=SEED).fit(Xtr)
Xtr_pca = pca_clf.transform(Xtr)
Xte_pca = pca_clf.transform(Xte)

print("\nKNN: original vs PCA")
res_orig = evaluate(KNeighborsClassifier(n_neighbors=5), Xtr, Xte, ytr, yte, f"KNN on Original ({N_FEATURES}D)")
res_pca = evaluate(KNeighborsClassifier(n_neighbors=5), Xtr_pca, Xte_pca, ytr, yte, f"KNN on PCA ({N_PCA50}D)")

print("\nMulti-model comparison")
model_zoo = {
    "KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Logistic Regression": lambda: LogisticRegression(max_iter=300, n_jobs=-1),
    "SVM (RBF)": lambda: SVC(kernel="rbf", C=5, gamma="scale"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
}

records = []
store = {}
for name, make in model_zoo.items():
    for space, (A, B) in {"Original": (Xtr, Xte), f"PCA-{N_PCA50}": (Xtr_pca, Xte_pca)}.items():
        label = f"{name} | {space}"
        r = evaluate(make(), A, B, ytr, yte, label)
        store[label] = r["preds"]
        records.append({k: v for k, v in r.items() if k != "preds"} | {"classifier": name, "space": space})

results_df = pd.DataFrame(records)
results_df = results_df[["classifier", "space", "accuracy", "macro_f1", "train_time_s", "predict_time_s"]]
results_df = results_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
print("\nLeaderboard:")
print(results_df.round(4).to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

pivot = results_df.pivot(index="classifier", columns="space", values="accuracy")
pivot.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Accuracy")
plt.ylim(max(0, pivot.values.min() - 0.05), 1.0)
plt.title("Accuracy: Original vs PCA Features")
plt.xticks(rotation=20)
plt.grid(alpha=0.3, axis="y")
save("06_model_comparison.png")

time_pivot = results_df.pivot(index="classifier", columns="space", values="train_time_s")
time_pivot.plot(kind="bar", figsize=(8, 5), logy=True)
plt.ylabel("Training time (s, log scale)")
plt.title("Training Time: Original vs PCA Features")
plt.xticks(rotation=20)
plt.grid(alpha=0.3, axis="y")
save("06b_training_time.png")

best_row = results_df.iloc[0]
best_label = f"{best_row['classifier']} | {best_row['space']}"
best_preds = store[best_label]
print(f"\nBest model: {best_label} (accuracy={best_row['accuracy']:.4f})")
print("\nClassification report:")
print(classification_report(yte, best_preds, digits=4))

cm = confusion_matrix(yte, best_preds)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix: {best_label}")
save("07_confusion_matrix.png")

per_class = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(7, 4))
plt.bar(range(N_CLASSES), per_class, color="seagreen")
plt.xticks(range(N_CLASSES))
plt.ylim(max(0, per_class.min() - 0.05), 1.0)
plt.xlabel("Digit")
plt.ylabel("Per-class accuracy")
plt.title("Per-class Accuracy (best model)")
plt.grid(alpha=0.3, axis="y")
save("07b_per_class_accuracy.png")

off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
top_pairs = np.dstack(np.unravel_index(np.argsort(off_diag.ravel())[::-1][:5], off_diag.shape))[0]
print("Most confused digit pairs (true -> predicted):")
for t, p in top_pairs:
    print(f"  {t} -> {p}: {off_diag[t, p]} times")

wrong = np.where(best_preds != yte)[0][:10]
if len(wrong):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(Xte[i].reshape(IMG_DIM), cmap="gray")
        ax.set_title(f"true {yte[i]} / pred {best_preds[i]}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong):]:
        ax.axis("off")
    plt.suptitle("Misclassified Examples")
    save("07c_misclassified.png")

print("\n" + "=" * 70)
print("PART E: ACCURACY VS NUMBER OF PCA COMPONENTS")
print("=" * 70)
sweep_rows = []
for k in comp_grid:
    p = PCA(n_components=k, random_state=SEED).fit(Xtr)
    A, B = p.transform(Xtr), p.transform(Xte)
    r = evaluate(KNeighborsClassifier(n_neighbors=5), A, B, ytr, yte, f"KNN | {k} components")
    sweep_rows.append({"components": k, "accuracy": r["accuracy"], "predict_time_s": r["predict_time_s"], "variance_kept": p.explained_variance_ratio_.sum()})
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(f"{OUT_DIR}/component_sweep.csv", index=False)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(sweep_df["components"], sweep_df["accuracy"], marker="o", color="tab:blue")
ax1.axhline(res_orig["accuracy"], color="tab:red", linestyle="--", label=f"Original {N_FEATURES}D")
ax1.set_xlabel("PCA components")
ax1.set_ylabel("KNN accuracy", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(sweep_df["components"], sweep_df["predict_time_s"], marker="s", color="tab:orange")
ax2.set_ylabel("Predict time (s)", color="tab:orange")
ax1.legend(loc="lower right")
plt.title("KNN Accuracy and Speed vs PCA Components")
ax1.grid(alpha=0.3)
save("08_component_sweep.png")

k_best = int(sweep_df.loc[sweep_df["accuracy"].idxmax(), "components"])
print(f"\nBest component count for KNN: {k_best} (accuracy={sweep_df['accuracy'].max():.4f})")

print("\n" + "=" * 70)
print("PART F: CROSS-VALIDATION")
print("=" * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_models = {
    f"PCA-{N_PCA50} + KNN": make_pipeline(PCA(n_components=N_PCA50, random_state=SEED), KNeighborsClassifier(n_neighbors=5)),
    f"PCA-{N_PCA50} + SVM": make_pipeline(PCA(n_components=N_PCA50, random_state=SEED), SVC(kernel="rbf", C=5, gamma="scale")),
    f"PCA-{N_PCA50} + Scaler + LogReg": make_pipeline(PCA(n_components=N_PCA50, random_state=SEED), StandardScaler(), LogisticRegression(max_iter=300)),
}
cv_rows = []
for name, pipe in cv_models.items():
    scores, secs = timed(cross_val_score, pipe, X_sub, y_sub, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_rows.append({"pipeline": name, "mean_acc": scores.mean(), "std_acc": scores.std(), "min_acc": scores.min(), "max_acc": scores.max(), "seconds": secs})
    print(f"{name:<34} mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores, 4)}")
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(f"{OUT_DIR}/cross_validation.csv", index=False)

plt.figure(figsize=(8, 4.5))
plt.barh(cv_df["pipeline"], cv_df["mean_acc"], xerr=cv_df["std_acc"], color="slateblue", capsize=4)
plt.xlim(max(0, cv_df["mean_acc"].min() - 0.05), 1.0)
plt.xlabel("5-fold CV accuracy")
plt.title("Cross-validated Accuracy")
plt.grid(alpha=0.3, axis="x")
save("09_cross_validation.png")

with open(f"{OUT_DIR}/results_summary.txt", "w") as f:
    f.write(f"Dataset: {DATA_NAME}\nSamples: {N_SAMPLES}\nFeatures: {N_FEATURES}\nClasses: {N_CLASSES}\nImage dimensions: {IMG_DIM}\n\n")
    f.write(f"Components for 90% variance: {n90}\n")
    f.write(f"Components for 95% variance: {n95}\n")
    f.write(f"Components for 99% variance: {n99}\n\n")
    f.write(f"KNN Original ({N_FEATURES}D): accuracy={res_orig['accuracy']:.4f}, train_time={res_orig['train_time_s']:.4f}s, predict_time={res_orig['predict_time_s']:.4f}s\n")
    f.write(f"KNN PCA ({N_PCA50}D): accuracy={res_pca['accuracy']:.4f}, train_time={res_pca['train_time_s']:.4f}s, predict_time={res_pca['predict_time_s']:.4f}s\n")
    f.write(f"Speedup (train): {res_orig['train_time_s'] / max(res_pca['train_time_s'], 1e-9):.2f}x\n")
    f.write(f"Speedup (predict): {res_orig['predict_time_s'] / max(res_pca['predict_time_s'], 1e-9):.2f}x\n\n")
    f.write(f"Best model: {best_label} (accuracy={best_row['accuracy']:.4f})\n")
    f.write(f"Best KNN component count: {k_best}\n\n")
    f.write("Leaderboard:\n" + results_df.round(4).to_string(index=False) + "\n\n")
    f.write("Cross-validation:\n" + cv_df.round(4).to_string(index=False) + "\n\n")
    f.write("t-SNE runs:\n" + tsne_df.to_string(index=False) + "\n")

print("\nAll plots, tables and results saved in the outputs/ directory.")
