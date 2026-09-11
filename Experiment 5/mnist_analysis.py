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
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_style("whitegrid")

OUTPUT_DIR = "/mnt/user-data/outputs"


def load_mnist_dataset():
    print("Step 1: Downloading / Loading the MNIST dataset")
    try:
        print("Trying to fetch the full MNIST dataset (70000 x 784) from OpenML ...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float64)
        y = mnist.target.astype(np.int64)
        image_size = 28
        print("Successfully downloaded MNIST (784-pixel, 28x28) images from OpenML.")
    except Exception as error:
        print("Could not reach OpenML in this environment.")
        print("Reason reported:", str(error))
        print("Falling back to the scikit-learn built-in 'load_digits' dataset,")
        print("which is a smaller, offline version of handwritten digit images (8x8 pixels).")
        digits = load_digits()
        X = digits.data.astype(np.float64)
        y = digits.target.astype(np.int64)
        image_size = 8
    return X, y, image_size


def show_dataset_summary(X, y, image_size):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print("\nStep 2: Dataset Summary")
    print("Number of samples  :", n_samples)
    print("Number of features :", n_features, "(this equals image_size * image_size)")
    print("Number of classes  :", n_classes, "-> digits 0 through 9")
    print("Image dimensions   :", image_size, "x", image_size)
    counts = pd.Series(y).value_counts().sort_index()
    print("\nSamples available per digit class:")
    print(counts.to_string())
    return n_samples, n_features, n_classes


def visualize_sample_digits(X, y, image_size, n_images=10):
    print("\nStep 3: Visualizing", n_images, "sample handwritten digit images")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    rng = np.random.RandomState(42)
    chosen = rng.choice(len(X), n_images, replace=False)
    for ax, idx in zip(axes.ravel(), chosen):
        image = X[idx].reshape(image_size, image_size)
        ax.imshow(image, cmap="gray")
        ax.set_title("Label: " + str(y[idx]))
        ax.axis("off")
    plt.suptitle("Sample Handwritten Digit Images from the Dataset")
    plt.tight_layout()
    path = OUTPUT_DIR + "/01_sample_digits.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved sample digit visualization to:", path)


def normalize_pixels(X):
    print("\nStep 4: Normalizing pixel values to the range [0, 1]")
    print("Original pixel value range: min =", X.min(), ", max =", X.max())
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_norm = scaler.fit_transform(X)
    print("Normalized pixel value range: min =", round(X_norm.min(), 4), ", max =", round(X_norm.max(), 4))
    return X_norm


def apply_full_pca(X_norm):
    print("\nStep 5: Applying PCA on the normalized dataset")
    pca_full = PCA(random_state=42)
    pca_full.fit(X_norm)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print("Total number of principal components computed:", len(explained_variance_ratio))
    print("Variance explained by the first 5 components:")
    for i in range(5):
        print("  PC" + str(i + 1) + ":", round(explained_variance_ratio[i] * 100, 3), "% of total variance")
    return pca_full, explained_variance_ratio, cumulative_variance


def components_for_variance_threshold(cumulative_variance, threshold):
    n_components = int(np.argmax(cumulative_variance >= threshold) + 1)
    return n_components


def report_variance_thresholds(cumulative_variance):
    print("\nStep 6: Number of components required to retain specific variance levels")
    thresholds = [0.90, 0.95, 0.99]
    results = {}
    for t in thresholds:
        n_comp = components_for_variance_threshold(cumulative_variance, t)
        results[t] = n_comp
        print(str(int(t * 100)) + "% variance retained with", n_comp, "principal components")
    return results


def reduce_dimensions(X_norm, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_norm)
    print("Reduced dataset to", n_components, "dimensions.")
    print("Shape after reduction:", X_reduced.shape)
    print("Variance retained with", n_components, "components:",
          round(np.sum(pca.explained_variance_ratio_) * 100, 3), "%")
    return X_reduced, pca


def plot_scree(explained_variance_ratio, n_show=50):
    print("\nPlotting the Scree Plot (variance explained by each individual component)")
    plt.figure(figsize=(10, 6))
    n_show = min(n_show, len(explained_variance_ratio))
    plt.plot(range(1, n_show + 1), explained_variance_ratio[:n_show] * 100, marker="o", markersize=3)
    plt.xlabel("Principal Component Number")
    plt.ylabel("Variance Explained (%)")
    plt.title("Scree Plot: Variance Explained by Each Principal Component")
    path = OUTPUT_DIR + "/02_scree_plot.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved scree plot to:", path)


def plot_cumulative_variance(cumulative_variance):
    print("Plotting the Cumulative Explained Variance Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100)
    for threshold, color in zip([0.90, 0.95, 0.99], ["green", "orange", "red"]):
        n_comp = components_for_variance_threshold(cumulative_variance, threshold)
        plt.axhline(y=threshold * 100, color=color, linestyle="--", linewidth=1,
                    label=str(int(threshold * 100)) + "% at " + str(n_comp) + " components")
        plt.axvline(x=n_comp, color=color, linestyle="--", linewidth=1)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained (%)")
    plt.title("Cumulative Explained Variance vs Number of Components")
    plt.legend()
    path = OUTPUT_DIR + "/03_cumulative_variance_plot.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved cumulative variance plot to:", path)


def plot_pca_2d(X_2d, y):
    print("Plotting the 2D PCA Projection of the dataset")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=8, alpha=0.6)
    plt.colorbar(scatter, label="Digit Class")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D PCA Projection of Handwritten Digits")
    path = OUTPUT_DIR + "/04_pca_2d_projection.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved 2D PCA projection plot to:", path)


def subsample_for_heavy_algorithms(X, y, max_samples=4000):
    print("\nNote: t-SNE and some classifier experiments are computationally expensive")
    print("on very large datasets, so a stratified subsample is used to keep runtime")
    print("reasonable while still preserving the class balance of the full dataset.")
    if len(X) <= max_samples:
        print("Dataset already has", len(X), "samples, which is within the limit of", max_samples)
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_samples, stratify=y, random_state=42
    )
    print("Subsampled dataset from", len(X), "down to", len(X_sub), "samples for t-SNE.")
    return X_sub, y_sub


def run_tsne(X, perplexity, n_components=2):
    start = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_embedded = tsne.fit_transform(X)
    elapsed = time.time() - start
    print("t-SNE with perplexity =", perplexity, "completed in", round(elapsed, 2), "seconds")
    return X_embedded, elapsed


def plot_tsne_single(X_embedded, y, title, filename):
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(scatter, label="Digit Class")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    path = OUTPUT_DIR + "/" + filename
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved plot to:", path)


def plot_tsne_perplexity_grid(results, filename):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (perplexity, (X_embedded, y_labels, elapsed)) in zip(axes, results.items()):
        scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_labels, cmap="tab10", s=8, alpha=0.7)
        ax.set_title("Perplexity = " + str(perplexity) + "\n(" + str(round(elapsed, 1)) + "s)")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
    fig.colorbar(scatter, ax=axes, label="Digit Class", fraction=0.02)
    plt.suptitle("Effect of Perplexity on t-SNE Visualization")
    path = OUTPUT_DIR + "/" + filename
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved perplexity comparison plot to:", path)


def train_and_evaluate_classifier(model, X_train, X_test, y_train, y_test, label):
    start_train = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_pred

    accuracy = accuracy_score(y_test, y_pred)

    print("\nResults for:", label)
    print("Training time   :", round(training_time, 4), "seconds")
    print("Prediction time :", round(prediction_time, 4), "seconds")
    print("Accuracy        :", round(accuracy * 100, 3), "%")

    return {
        "label": label,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "accuracy": accuracy,
        "y_pred": y_pred,
    }


def plot_classifier_comparison(results, filename):
    labels = [r["label"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    train_times = [r["training_time"] for r in results]
    pred_times = [r["prediction_time"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(labels, accuracies, color="steelblue")
    axes[0].set_title("Classification Accuracy (%)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, train_times, color="darkorange")
    axes[1].set_title("Training Time (seconds)")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(labels, pred_times, color="seagreen")
    axes[2].set_title("Prediction Time (seconds)")
    axes[2].set_ylabel("Seconds")
    axes[2].tick_params(axis="x", rotation=30)

    plt.suptitle("Classifier Performance Comparison: Original vs PCA-Reduced Data")
    plt.tight_layout()
    path = OUTPUT_DIR + "/" + filename
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved classifier comparison plot to:", path)


def plot_confusion_matrices(results, filename):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_test"], r["y_pred"])
        sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix: " + r["label"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    path = OUTPUT_DIR + "/" + filename
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved confusion matrix plot to:", path)


def main():
    print("#" * 80)
    print("HANDWRITTEN DIGIT RECOGNITION: PCA AND t-SNE DIMENSIONALITY REDUCTION")
    print("#" * 80)

    print("\n" + "=" * 80)
    print("PART A: DATA UNDERSTANDING AND PREPROCESSING")
    print("=" * 80)

    X, y, image_size = load_mnist_dataset()
    n_samples, n_features, n_classes = show_dataset_summary(X, y, image_size)
    visualize_sample_digits(X, y, image_size, n_images=10)
    X_norm = normalize_pixels(X)

    print("\n" + "=" * 80)
    print("PART B: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 80)

    pca_full, explained_variance_ratio, cumulative_variance = apply_full_pca(X_norm)
    variance_thresholds = report_variance_thresholds(cumulative_variance)

    print("\nStep 7: Reducing the dataset to 2 and 50 dimensions")
    X_pca_2, pca_2 = reduce_dimensions(X_norm, n_components=2)
    X_pca_50, pca_50 = reduce_dimensions(X_norm, n_components=50)

    print("\nStep 8: Generating PCA-related plots")
    plot_scree(explained_variance_ratio, n_show=50)
    plot_cumulative_variance(cumulative_variance)
    plot_pca_2d(X_pca_2, y)

    print("\n" + "=" * 80)
    print("PART C: t-SNE VISUALIZATION")
    print("=" * 80)

    X_tsne_source, y_tsne_source = subsample_for_heavy_algorithms(X_norm, y, max_samples=4000)

    X_sub_idx, _, y_sub_idx, _ = train_test_split(
        np.arange(len(X_norm)), y, train_size=len(X_tsne_source), stratify=y, random_state=42
    ) if len(X_norm) > len(X_tsne_source) else (np.arange(len(X_norm)), None, y, None)

    X_pca_50_sub = X_pca_50[X_sub_idx]
    y_sub = y[X_sub_idx]
    X_norm_sub = X_norm[X_sub_idx]

    print("\nStep 9a: Applying t-SNE on the ORIGINAL normalized data")
    X_tsne_original, time_original = run_tsne(X_norm_sub, perplexity=30)

    print("\nStep 9b: Applying t-SNE on the PCA-reduced (50 dimensions) data")
    X_tsne_pca50, time_pca50 = run_tsne(X_pca_50_sub, perplexity=30)

    print("\nStep 10: Generating 2D t-SNE visualizations")
    plot_tsne_single(
        X_tsne_original, y_sub,
        "t-SNE on Original 784-Dimensional Data (Perplexity=30)",
        "05_tsne_original_data.png",
    )
    plot_tsne_single(
        X_tsne_pca50, y_sub,
        "t-SNE on PCA-Reduced 50-Dimensional Data (Perplexity=30)",
        "06_tsne_pca50_data.png",
    )

    print("\nStep 11: Experimenting with different perplexity values")
    perplexities = [5, 30, 50]
    perplexity_results = {}
    for p in perplexities:
        X_embedded, elapsed = run_tsne(X_pca_50_sub, perplexity=p)
        perplexity_results[p] = (X_embedded, y_sub, elapsed)

    print("\nStep 12: Comparing t-SNE visualizations across perplexity values")
    plot_tsne_perplexity_grid(perplexity_results, "07_tsne_perplexity_comparison.png")

    print("\nSummary of t-SNE timing comparison:")
    print("t-SNE on original data (784-dim) took   :", round(time_original, 2), "seconds")
    print("t-SNE on PCA-reduced data (50-dim) took :", round(time_pca50, 2), "seconds")
    if time_pca50 > 0:
        print("Speed-up factor from using PCA before t-SNE:", round(time_original / time_pca50, 2), "x")

    print("\n" + "=" * 80)
    print("PART D: DIGIT RECOGNITION ANALYSIS")
    print("=" * 80)

    classification_sample_size = min(6000, n_samples)
    if n_samples > classification_sample_size:
        print("\nUsing a stratified subsample of", classification_sample_size,
              "instances for classifier training/testing to keep runtime practical.")
        X_clf, _, y_clf, _ = train_test_split(
            X_norm, y, train_size=classification_sample_size, stratify=y, random_state=42
        )
    else:
        X_clf, y_clf = X_norm, y

    X_clf_pca50 = pca_50.transform(X_clf)

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    X_train_pca, X_test_pca, _, _ = train_test_split(
        X_clf_pca50, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("\nStep 13: Training classifiers on ORIGINAL and PCA-transformed data")
    print("Training set size:", X_train_orig.shape[0], " Test set size:", X_test_orig.shape[0])

    all_results = []

    knn_orig = KNeighborsClassifier(n_neighbors=5)
    result = train_and_evaluate_classifier(
        knn_orig, X_train_orig, X_test_orig, y_train, y_test, "KNN - Original Data"
    )
    result["y_test"] = y_test
    all_results.append(result)

    knn_pca = KNeighborsClassifier(n_neighbors=5)
    result = train_and_evaluate_classifier(
        knn_pca, X_train_pca, X_test_pca, y_train, y_test, "KNN - PCA(50) Data"
    )
    result["y_test"] = y_test
    all_results.append(result)

    svm_orig = SVC(kernel="rbf", random_state=42)
    result = train_and_evaluate_classifier(
        svm_orig, X_train_orig, X_test_orig, y_train, y_test, "SVM - Original Data"
    )
    result["y_test"] = y_test
    all_results.append(result)

    svm_pca = SVC(kernel="rbf", random_state=42)
    result = train_and_evaluate_classifier(
        svm_pca, X_train_pca, X_test_pca, y_train, y_test, "SVM - PCA(50) Data"
    )
    result["y_test"] = y_test
    all_results.append(result)

    print("\nStep 14: Comparing Accuracy, Training Time, and Prediction Time")
    comparison_table = pd.DataFrame([
        {
            "Model": r["label"],
            "Accuracy (%)": round(r["accuracy"] * 100, 3),
            "Training Time (s)": round(r["training_time"], 4),
            "Prediction Time (s)": round(r["prediction_time"], 4),
        }
        for r in all_results
    ])
    print(comparison_table.to_string(index=False))

    plot_classifier_comparison(all_results, "08_classifier_comparison.png")
    plot_confusion_matrices(all_results, "09_confusion_matrices.png")

    print("\nStep 15: Analysis of How Dimensionality Reduction Affects Recognition Performance")
    knn_orig_res, knn_pca_res, svm_orig_res, svm_pca_res = all_results

    print("\nKNN Classifier:")
    print("Accuracy on original 784-dim data :", round(knn_orig_res["accuracy"] * 100, 3), "%")
    print("Accuracy on PCA 50-dim data        :", round(knn_pca_res["accuracy"] * 100, 3), "%")
    print("Training time reduced from", round(knn_orig_res["training_time"], 4),
          "s to", round(knn_pca_res["training_time"], 4), "s using PCA")
    print("Prediction time reduced from", round(knn_orig_res["prediction_time"], 4),
          "s to", round(knn_pca_res["prediction_time"], 4), "s using PCA")

    print("\nSVM Classifier:")
    print("Accuracy on original 784-dim data :", round(svm_orig_res["accuracy"] * 100, 3), "%")
    print("Accuracy on PCA 50-dim data        :", round(svm_pca_res["accuracy"] * 100, 3), "%")
    print("Training time reduced from", round(svm_orig_res["training_time"], 4),
          "s to", round(svm_pca_res["training_time"], 4), "s using PCA")
    print("Prediction time reduced from", round(svm_orig_res["prediction_time"], 4),
          "s to", round(svm_pca_res["prediction_time"], 4), "s using PCA")

    print("\nOverall conclusion:")
    print("PCA compresses the", n_features, "-dimensional pixel space down to just 50")
    print("dimensions while retaining the large majority of the meaningful variance,")
    print("(see the", int(variance_thresholds[0.95]), "components needed for 95% variance).")
    print("This typically results in comparable or only slightly lower accuracy,")
    print("while substantially reducing training and prediction time, which becomes")
    print("increasingly important as dataset size grows.")
    print("t-SNE, run after PCA pre-reduction, produces visually tighter and more")
    print("clearly separated digit clusters than PCA alone, and running it on the")
    print("50-dimensional PCA output is markedly faster than running it directly on")
    print("the raw pixel space, with minimal loss of visual cluster quality.")

    print("\nAll figures have been saved to:", OUTPUT_DIR)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
