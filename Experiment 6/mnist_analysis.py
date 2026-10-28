import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

RNG = 42
OUT = "outputs"
DPI = 150
SAMPLE_IMAGE_COUNT = 10
SCREE_MAX_COMPONENTS = 50
PCA_REDUCED_COMPONENTS = 50
VARIANCE_THRESHOLDS = [(0.90, "green"), (0.95, "orange"), (0.99, "red")]
TSNE_MAX_SAMPLES = 3000
TSNE_DEFAULT_PERPLEXITY = 30
TSNE_PERPLEXITIES = [5, 15, 30, 50, 100]
TSNE_MAX_ITER = 1000
CLASSIFIER_MAX_SAMPLES = 8000
TEST_SIZE = 0.25
KNN_NEIGHBORS = 5
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
SECTION_LINE = "=" * 70
SUMMARY_LINE = "-" * 70

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
np.random.seed(RNG)
os.makedirs(OUT, exist_ok=True)


def load_from_kaggle_csv(path="train.csv"):
    df = pd.read_csv(path)
    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float64)
    return X, y, (28, 28)


def load_mnist():
    try:
        print("Attempting to download full MNIST (70,000 x 784) from OpenML...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="auto")
        y = y.astype(int)
        print("  -> Success: full MNIST loaded from OpenML.")
        return X.astype(np.float64), y, (28, 28), "OpenML mnist_784 (full MNIST)"
    except Exception as e:
        print(f"  -> No internet access to OpenML ({type(e).__name__}). "
              f"Falling back to offline UCI 'Optical Digits' dataset.")
        data = load_digits()
        X = data.data.astype(np.float64)
        y = data.target.astype(int)
        source_name = "sklearn load_digits (UCI Optical Recognition of Handwritten Digits)"
        return X, y, (8, 8), source_name


def n_components_for(cumulative_variance, threshold):
    n_components = int(np.argmax(cumulative_variance >= threshold) + 1)
    return n_components


def run_tsne(X, perplexity, label):
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=RNG, max_iter=TSNE_MAX_ITER)
    embedding = tsne.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"  t-SNE [{label}, perplexity={perplexity}] done in {elapsed:.1f}s")
    return embedding, elapsed


def evaluate(name, model, Xtr, Xte, ytr, yte, feature_desc, results):
    start_time = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - start_time

    start_time = time.time()
    pred = model.predict(Xte)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(yte, pred)
    per_sample_ms = predict_time / len(yte) * 1000

    print(f"\n[{name} | {feature_desc}]")
    print(f"  Accuracy       : {accuracy*100:.2f}%")
    print(f"  Training time  : {train_time:.3f}s")
    print(f"  Prediction time: {predict_time:.3f}s ({per_sample_ms:.3f} ms/sample)")

    results.append({
        "Classifier": name, "Features": feature_desc, "Dimensions": Xtr.shape[1],
        "Accuracy": accuracy, "Train_Time_s": train_time, "Predict_Time_s": predict_time,
    })
    return pred


print(SECTION_LINE)
print("PART A: DATA UNDERSTANDING AND PREPROCESSING")
print(SECTION_LINE)

X_raw, y, IMG_SHAPE, SOURCE = load_mnist()
n_samples, n_features = X_raw.shape
n_classes = len(np.unique(y))
pixel_min = X_raw.min()
pixel_max = X_raw.max()
class_labels = sorted(np.unique(y).tolist())

print(f"\nData source           : {SOURCE}")
print(f"Number of samples     : {n_samples}")
print(f"Number of classes     : {n_classes}  -> {class_labels}")
print(f"Image dimensions      : {IMG_SHAPE[0]} x {IMG_SHAPE[1]} = {n_features} pixels (features)")
print(f"Raw pixel value range : [{pixel_min:.1f}, {pixel_max:.1f}]")

rng = np.random.RandomState(RNG)
sample_idx = rng.choice(n_samples, SAMPLE_IMAGE_COUNT, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(X_raw[idx].reshape(IMG_SHAPE), cmap="gray")
    ax.set_title(f"Label: {y[idx]}", fontsize=10)
    ax.axis("off")
fig.suptitle(f"Sample Handwritten Digits ({SOURCE})", fontsize=12)
plt.tight_layout(h_pad=2.0)
sample_digits_path = f"{OUT}/01_sample_digits.png"
plt.savefig(sample_digits_path, dpi=DPI)
plt.close()
print(f"\nSaved: {sample_digits_path}")

X_norm = X_raw / pixel_max
print(f"Normalized pixel range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")


print("\n" + SECTION_LINE)
print("PART B: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print(SECTION_LINE)

max_components = min(n_samples, n_features)
pca_full = PCA(n_components=max_components, random_state=RNG)
pca_full.fit(X_norm)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

threshold_counts = []
for threshold, color in VARIANCE_THRESHOLDS:
    component_count = n_components_for(cumulative, threshold)
    threshold_counts.append((threshold, component_count, color))
    print(f"Components needed to retain {int(threshold*100)}% variance: {component_count}")
print(f"(Original feature space: {n_features} dimensions)")

n_show = min(SCREE_MAX_COMPONENTS, max_components)
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_show + 1), explained[:n_show], "o-", linewidth=1.5, markersize=4)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title(f"Scree Plot (first {n_show} components)")
plt.tight_layout()
scree_path = f"{OUT}/02_scree_plot.png"
plt.savefig(scree_path, dpi=DPI)
plt.close()
print(f"Saved: {scree_path}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, max_components + 1), cumulative, linewidth=2)
for threshold, component_count, color in threshold_counts:
    plt.axhline(threshold, color=color, linestyle="--", linewidth=1,
                label=f"{int(threshold*100)}% @ {component_count} PCs")
    plt.axvline(component_count, color=color, linestyle="--", linewidth=1)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance vs. Number of Components")
plt.legend()
plt.tight_layout()
cumulative_path = f"{OUT}/03_cumulative_variance.png"
plt.savefig(cumulative_path, dpi=DPI)
plt.close()
print(f"Saved: {cumulative_path}")

pca_2 = PCA(n_components=2, random_state=RNG)
X_pca_2d = pca_2.fit_transform(X_norm)

n_50 = min(PCA_REDUCED_COMPONENTS, max_components)
pca_50 = PCA(n_components=n_50, random_state=RNG)
X_pca_50d = pca_50.fit_transform(X_norm)

variance_2d = pca_2.explained_variance_ratio_.sum() * 100
variance_50d = pca_50.explained_variance_ratio_.sum() * 100

print(f"\n2D PCA projection variance captured : {variance_2d:.2f}%")
print(f"{n_50}D PCA projection variance captured: {variance_50d:.2f}%")

pc1_variance = pca_2.explained_variance_ratio_[0] * 100
pc2_variance = pca_2.explained_variance_ratio_[1] * 100

plt.figure(figsize=(8, 7))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap="tab10", s=8, alpha=0.6)
plt.colorbar(scatter, label="Digit", ticks=range(10))
plt.xlabel(f"PC1 ({pc1_variance:.1f}% var)")
plt.ylabel(f"PC2 ({pc2_variance:.1f}% var)")
plt.title("2D PCA Projection of Handwritten Digits")
plt.tight_layout()
pca_2d_path = f"{OUT}/04_pca_2d_projection.png"
plt.savefig(pca_2d_path, dpi=DPI)
plt.close()
print(f"Saved: {pca_2d_path}")


print("\n" + SECTION_LINE)
print("PART C: t-SNE VISUALIZATION")
print(SECTION_LINE)

TSNE_N = min(TSNE_MAX_SAMPLES, n_samples)
sub_idx = rng.choice(n_samples, TSNE_N, replace=False)
X_sub_raw = X_norm[sub_idx]
X_sub_pca50 = X_pca_50d[sub_idx]
y_sub = y[sub_idx]
print(f"\nUsing a random subsample of {TSNE_N} points for all t-SNE runs "
      f"(t-SNE does not scale well to very large N).")

emb_raw, t_raw = run_tsne(X_sub_raw, TSNE_DEFAULT_PERPLEXITY, "raw pixels")
emb_pca, t_pca = run_tsne(X_sub_pca50, TSNE_DEFAULT_PERPLEXITY, "PCA-50")

tsne_panels = [
    (emb_raw, f"t-SNE on Raw {n_features}-D Pixel Data", t_raw),
    (emb_pca, f"t-SNE on PCA-Reduced {n_50}-D Data", t_pca),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (emb, title, elapsed) in zip(axes, tsne_panels):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_sub, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(f"{title}\n(fit time: {elapsed:.1f}s)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
fig.colorbar(sc, ax=axes, label="Digit", ticks=range(10), fraction=0.025)
tsne_compare_path = f"{OUT}/05_tsne_raw_vs_pca.png"
plt.savefig(tsne_compare_path, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {tsne_compare_path}")

perplexities = [p for p in TSNE_PERPLEXITIES if p < TSNE_N]
fig, axes = plt.subplots(1, len(perplexities), figsize=(4.5 * len(perplexities), 4.5))
if len(perplexities) == 1:
    axes = [axes]
perp_times = {}
for ax, perplexity in zip(axes, perplexities):
    emb_p, elapsed = run_tsne(X_sub_pca50, perplexity, "PCA-50")
    perp_times[perplexity] = elapsed
    ax.scatter(emb_p[:, 0], emb_p[:, 1], c=y_sub, cmap="tab10", s=6, alpha=0.7)
    ax.set_title(f"Perplexity = {perplexity}\n({elapsed:.1f}s)")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("t-SNE Embeddings Across Perplexity Values (input: PCA-50 features)", y=1.03)
plt.tight_layout()
perplexity_path = f"{OUT}/06_tsne_perplexity_comparison.png"
plt.savefig(perplexity_path, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {perplexity_path}")


print("\n" + SECTION_LINE)
print("PART D: DIGIT RECOGNITION ANALYSIS (KNN & SVM)")
print(SECTION_LINE)

CLF_N = min(CLASSIFIER_MAX_SAMPLES, n_samples)
clf_idx = rng.choice(n_samples, CLF_N, replace=False)
X_clf_raw = X_norm[clf_idx]
X_clf_pca = X_pca_50d[clf_idx]
y_clf = y[clf_idx]

Xtr_raw, Xte_raw, Xtr_pca, Xte_pca, ytr, yte = train_test_split(
    X_clf_raw, X_clf_pca, y_clf, test_size=TEST_SIZE, random_state=RNG, stratify=y_clf
)
print(f"\nClassifier subsample size: {CLF_N}  (train={len(ytr)}, test={len(yte)})")

results = []
raw_description = f"Raw {n_features}-D pixels"
pca_description = f"PCA {n_50}-D"

pred_knn_raw = evaluate("KNN (k=5)", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS),
                        Xtr_raw, Xte_raw, ytr, yte, raw_description, results)
pred_knn_pca = evaluate("KNN (k=5)", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS),
                        Xtr_pca, Xte_pca, ytr, yte, pca_description, results)
pred_svm_raw = evaluate("SVM (RBF)", SVC(kernel=SVM_KERNEL, gamma=SVM_GAMMA, random_state=RNG),
                        Xtr_raw, Xte_raw, ytr, yte, raw_description, results)
pred_svm_pca = evaluate("SVM (RBF)", SVC(kernel=SVM_KERNEL, gamma=SVM_GAMMA, random_state=RNG),
                        Xtr_pca, Xte_pca, ytr, yte, pca_description, results)

results_df = pd.DataFrame(results)
results_path = f"{OUT}/classifier_results.csv"
results_df.to_csv(results_path, index=False)

print("\n" + SUMMARY_LINE)
print("SUMMARY TABLE")
print(SUMMARY_LINE)
print(results_df.to_string(index=False))

metrics = [("Accuracy", "Accuracy", "{:.1%}"),
           ("Train_Time_s", "Training Time (s)", "{:.2f}s"),
           ("Predict_Time_s", "Prediction Time (s)", "{:.3f}s")]
bar_labels = [f"{r.Classifier}\n{r.Features}" for r in results_df.itertuples()]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (column, title, value_format) in zip(axes, metrics):
    bars = ax.bar(bar_labels, results_df[column], color=BAR_COLORS)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, results_df[column]):
        bar_center = bar.get_x() + bar.get_width() / 2
        ax.text(bar_center, bar.get_height(), value_format.format(value),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
comparison_path = f"{OUT}/07_classifier_comparison.png"
plt.savefig(comparison_path, dpi=DPI)
plt.close()
print(f"\nSaved: {comparison_path}")

confusion = confusion_matrix(yte, pred_svm_pca)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: SVM on PCA-Reduced Features")
plt.tight_layout()
confusion_path = f"{OUT}/08_confusion_matrix_svm_pca.png"
plt.savefig(confusion_path, dpi=DPI)
plt.close()
print(f"Saved: {confusion_path}")

print("\n" + SECTION_LINE)
print("ALL DONE. Figures + results CSV written to ./outputs/")
print(SECTION_LINE)
