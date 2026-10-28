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
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

RANDOM_SEED = 42
OUTPUT_DIR = "outputs"
PLOT_DPI = 150

SAMPLE_IMAGE_COUNT = 10
PRINT_COMPONENT_COUNT = 5
SCREE_COMPONENT_COUNT = 50
VARIANCE_THRESHOLDS = [0.90, 0.95, 0.99]
THRESHOLD_COLORS = ["green", "orange", "red"]
SMALL_PCA_COMPONENTS = 2
LARGE_PCA_COMPONENTS = 50

TSNE_MAX_SAMPLES = 4000
CLASSIFICATION_MAX_SAMPLES = 6000
DEFAULT_PERPLEXITY = 30
PERPLEXITIES = [5, 30, 50]

TEST_SIZE = 0.2
KNN_NEIGHBORS = 5
SVM_KERNEL = "rbf"

SECTION_LINE_WIDE = "=" * 80
BANNER_LINE = "#" * 80

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")


def build_output_path(filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    return output_path


def save_current_figure(filename):
    output_path = build_output_path(filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()
    return output_path


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
    counts = pd.Series(y).value_counts().sort_index()

    print("\nStep 2: Dataset Summary")
    print("Number of samples  :", n_samples)
    print("Number of features :", n_features, "(this equals image_size * image_size)")
    print("Number of classes  :", n_classes, "-> digits 0 through 9")
    print("Image dimensions   :", image_size, "x", image_size)
    print("\nSamples available per digit class:")
    print(counts.to_string())
    return n_samples, n_features, n_classes


def visualize_sample_digits(X, y, image_size, n_images=SAMPLE_IMAGE_COUNT):
    figure_size = (12, 5)
    rows = 2
    cols = 5

    print("\nStep 3: Visualizing", n_images, "sample handwritten digit images")
    fig, axes = plt.subplots(rows, cols, figsize=figure_size)
    rng = np.random.RandomState(RANDOM_SEED)
    chosen = rng.choice(len(X), n_images, replace=False)
    for ax, idx in zip(axes.ravel(), chosen):
        image = X[idx].reshape(image_size, image_size)
        ax.imshow(image, cmap="gray")
        ax.set_title("Label: " + str(y[idx]))
        ax.axis("off")
    plt.suptitle("Sample Handwritten Digit Images from the Dataset")
    path = save_current_figure("01_sample_digits.png")
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
    pca_full = PCA(random_state=RANDOM_SEED)
    pca_full.fit(X_norm)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("Total number of principal components computed:", len(explained_variance_ratio))
    print("Variance explained by the first 5 components:")
    for i in range(PRINT_COMPONENT_COUNT):
        percentage = round(explained_variance_ratio[i] * 100, 3)
        print("  PC" + str(i + 1) + ":", percentage, "% of total variance")
    return pca_full, explained_variance_ratio, cumulative_variance


def components_for_variance_threshold(cumulative_variance, threshold):
    n_components = int(np.argmax(cumulative_variance >= threshold) + 1)
    return n_components


def report_variance_thresholds(cumulative_variance):
    results = {}
    print("\nStep 6: Number of components required to retain specific variance levels")
    for threshold in VARIANCE_THRESHOLDS:
        n_comp = components_for_variance_threshold(cumulative_variance, threshold)
        results[threshold] = n_comp
        print(str(int(threshold * 100)) + "% variance retained with", n_comp, "principal components")
    return results


def reduce_dimensions(X_norm, n_components):
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_reduced = pca.fit_transform(X_norm)
    retained_percentage = round(np.sum(pca.explained_variance_ratio_) * 100, 3)

    print("Reduced dataset to", n_components, "dimensions.")
    print("Shape after reduction:", X_reduced.shape)
    print("Variance retained with", n_components, "components:", retained_percentage, "%")
    return X_reduced, pca


def plot_scree(explained_variance_ratio, n_show=SCREE_COMPONENT_COUNT):
    figure_size = (10, 6)
    n_show = min(n_show, len(explained_variance_ratio))
    x_values = range(1, n_show + 1)
    y_values = explained_variance_ratio[:n_show] * 100

    print("\nPlotting the Scree Plot (variance explained by each individual component)")
    plt.figure(figsize=figure_size)
    plt.plot(x_values, y_values, marker="o", markersize=3)
    plt.xlabel("Principal Component Number")
    plt.ylabel("Variance Explained (%)")
    plt.title("Scree Plot: Variance Explained by Each Principal Component")
    path = save_current_figure("02_scree_plot.png")
    print("Saved scree plot to:", path)


def plot_cumulative_variance(cumulative_variance):
    figure_size = (10, 6)
    x_values = range(1, len(cumulative_variance) + 1)

    print("Plotting the Cumulative Explained Variance Plot")
    plt.figure(figsize=figure_size)
    plt.plot(x_values, cumulative_variance * 100)
    for threshold, color in zip(VARIANCE_THRESHOLDS, THRESHOLD_COLORS):
        n_comp = components_for_variance_threshold(cumulative_variance, threshold)
        line_label = str(int(threshold * 100)) + "% at " + str(n_comp) + " components"
        plt.axhline(y=threshold * 100, color=color, linestyle="--", linewidth=1, label=line_label)
        plt.axvline(x=n_comp, color=color, linestyle="--", linewidth=1)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained (%)")
    plt.title("Cumulative Explained Variance vs Number of Components")
    plt.legend()
    path = save_current_figure("03_cumulative_variance_plot.png")
    print("Saved cumulative variance plot to:", path)


def plot_pca_2d(X_2d, y):
    figure_size = (10, 8)

    print("Plotting the 2D PCA Projection of the dataset")
    plt.figure(figsize=figure_size)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=8, alpha=0.6)
    plt.colorbar(scatter, label="Digit Class")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D PCA Projection of Handwritten Digits")
    path = save_current_figure("04_pca_2d_projection.png")
    print("Saved 2D PCA projection plot to:", path)


def get_stratified_indices(y, max_samples):
    total_samples = len(y)
    all_indices = np.arange(total_samples)
    if total_samples <= max_samples:
        return all_indices
    selected_indices, _ = train_test_split(
        all_indices, train_size=max_samples, stratify=y, random_state=RANDOM_SEED
    )
    return selected_indices


def select_tsne_subsample(y, max_samples=TSNE_MAX_SAMPLES):
    total_samples = len(y)

    print("\nNote: t-SNE and some classifier experiments are computationally expensive")
    print("on very large datasets, so a stratified subsample is used to keep runtime")
    print("reasonable while still preserving the class balance of the full dataset.")

    selected_indices = get_stratified_indices(y, max_samples)

    if total_samples <= max_samples:
        print("Dataset already has", total_samples, "samples, which is within the limit of", max_samples)
    else:
        print("Subsampled dataset from", total_samples, "down to", len(selected_indices), "samples for t-SNE.")
    return selected_indices


def run_tsne(X, perplexity, n_components=2):
    start = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    X_embedded = tsne.fit_transform(X)
    elapsed = time.time() - start

    print("t-SNE with perplexity =", perplexity, "completed in", round(elapsed, 2), "seconds")
    return X_embedded, elapsed


def plot_tsne_single(X_embedded, y, title, filename):
    figure_size = (9, 7)

    plt.figure(figsize=figure_size)
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(scatter, label="Digit Class")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    path = save_current_figure(filename)
    print("Saved plot to:", path)


def plot_tsne_perplexity_grid(results, filename):
    plot_count = len(results)
    figure_size = (6 * plot_count, 6)
    colorbar_fraction = 0.02

    fig, axes = plt.subplots(1, plot_count, figsize=figure_size)
    if plot_count == 1:
        axes = [axes]
    for ax, (perplexity, (X_embedded, y_labels, elapsed)) in zip(axes, results.items()):
        scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_labels, cmap="tab10", s=8, alpha=0.7)
        ax.set_title("Perplexity = " + str(perplexity) + "\n(" + str(round(elapsed, 1)) + "s)")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
    fig.colorbar(scatter, ax=axes, label="Digit Class", fraction=colorbar_fraction)
    plt.suptitle("Effect of Perplexity on t-SNE Visualization")
    output_path = build_output_path(filename)
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()
    print("Saved perplexity comparison plot to:", output_path)


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

    result = {
        "label": label,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "accuracy": accuracy,
        "y_pred": y_pred,
        "y_test": y_test,
    }
    return result


def plot_classifier_comparison(results, filename):
    figure_size = (18, 5)
    label_rotation = 30

    labels = [r["label"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    train_times = [r["training_time"] for r in results]
    pred_times = [r["prediction_time"] for r in results]

    panels = [
        (accuracies, "steelblue", "Classification Accuracy (%)", "Accuracy (%)"),
        (train_times, "darkorange", "Training Time (seconds)", "Seconds"),
        (pred_times, "seagreen", "Prediction Time (seconds)", "Seconds"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=figure_size)
    for ax, (values, color, title, y_label) in zip(axes, panels):
        ax.bar(labels, values, color=color)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.tick_params(axis="x", rotation=label_rotation)

    plt.suptitle("Classifier Performance Comparison: Original vs PCA-Reduced Data")
    path = save_current_figure(filename)
    print("Saved classifier comparison plot to:", path)


def plot_confusion_matrices(results, filename):
    plot_count = len(results)
    figure_size = (7 * plot_count, 6)

    fig, axes = plt.subplots(1, plot_count, figsize=figure_size)
    if plot_count == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_test"], r["y_pred"])
        sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix: " + r["label"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    path = save_current_figure(filename)
    print("Saved confusion matrix plot to:", path)


def report_pca_effect(name, original_result, pca_result):
    original_accuracy = round(original_result["accuracy"] * 100, 3)
    pca_accuracy = round(pca_result["accuracy"] * 100, 3)
    original_train = round(original_result["training_time"], 4)
    pca_train = round(pca_result["training_time"], 4)
    original_predict = round(original_result["prediction_time"], 4)
    pca_predict = round(pca_result["prediction_time"], 4)

    print("\n" + name + " Classifier:")
    print("Accuracy on original 784-dim data :", original_accuracy, "%")
    print("Accuracy on PCA 50-dim data        :", pca_accuracy, "%")
    print("Training time reduced from", original_train, "s to", pca_train, "s using PCA")
    print("Prediction time reduced from", original_predict, "s to", pca_predict, "s using PCA")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(BANNER_LINE)
    print("HANDWRITTEN DIGIT RECOGNITION: PCA AND t-SNE DIMENSIONALITY REDUCTION")
    print(BANNER_LINE)

    print("\n" + SECTION_LINE_WIDE)
    print("PART A: DATA UNDERSTANDING AND PREPROCESSING")
    print(SECTION_LINE_WIDE)

    X, y, image_size = load_mnist_dataset()
    n_samples, n_features, n_classes = show_dataset_summary(X, y, image_size)
    visualize_sample_digits(X, y, image_size)
    X_norm = normalize_pixels(X)

    print("\n" + SECTION_LINE_WIDE)
    print("PART B: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print(SECTION_LINE_WIDE)

    pca_full, explained_variance_ratio, cumulative_variance = apply_full_pca(X_norm)
    variance_thresholds = report_variance_thresholds(cumulative_variance)

    print("\nStep 7: Reducing the dataset to 2 and 50 dimensions")
    X_pca_2, pca_2 = reduce_dimensions(X_norm, n_components=SMALL_PCA_COMPONENTS)
    X_pca_50, pca_50 = reduce_dimensions(X_norm, n_components=LARGE_PCA_COMPONENTS)

    print("\nStep 8: Generating PCA-related plots")
    plot_scree(explained_variance_ratio)
    plot_cumulative_variance(cumulative_variance)
    plot_pca_2d(X_pca_2, y)

    print("\n" + SECTION_LINE_WIDE)
    print("PART C: t-SNE VISUALIZATION")
    print(SECTION_LINE_WIDE)

    tsne_indices = select_tsne_subsample(y)
    X_pca_50_sub = X_pca_50[tsne_indices]
    X_norm_sub = X_norm[tsne_indices]
    y_sub = y[tsne_indices]

    print("\nStep 9a: Applying t-SNE on the ORIGINAL normalized data")
    X_tsne_original, time_original = run_tsne(X_norm_sub, perplexity=DEFAULT_PERPLEXITY)

    print("\nStep 9b: Applying t-SNE on the PCA-reduced (50 dimensions) data")
    X_tsne_pca50, time_pca50 = run_tsne(X_pca_50_sub, perplexity=DEFAULT_PERPLEXITY)

    print("\nStep 10: Generating 2D t-SNE visualizations")
    original_title = "t-SNE on Original 784-Dimensional Data (Perplexity=30)"
    pca50_title = "t-SNE on PCA-Reduced 50-Dimensional Data (Perplexity=30)"
    plot_tsne_single(X_tsne_original, y_sub, original_title, "05_tsne_original_data.png")
    plot_tsne_single(X_tsne_pca50, y_sub, pca50_title, "06_tsne_pca50_data.png")

    print("\nStep 11: Experimenting with different perplexity values")
    perplexity_results = {}
    for perplexity in PERPLEXITIES:
        X_embedded, elapsed = run_tsne(X_pca_50_sub, perplexity=perplexity)
        perplexity_results[perplexity] = (X_embedded, y_sub, elapsed)

    print("\nStep 12: Comparing t-SNE visualizations across perplexity values")
    plot_tsne_perplexity_grid(perplexity_results, "07_tsne_perplexity_comparison.png")

    print("\nSummary of t-SNE timing comparison:")
    print("t-SNE on original data (784-dim) took   :", round(time_original, 2), "seconds")
    print("t-SNE on PCA-reduced data (50-dim) took :", round(time_pca50, 2), "seconds")
    if time_pca50 > 0:
        speed_up = round(time_original / time_pca50, 2)
        print("Speed-up factor from using PCA before t-SNE:", speed_up, "x")

    print("\n" + SECTION_LINE_WIDE)
    print("PART D: DIGIT RECOGNITION ANALYSIS")
    print(SECTION_LINE_WIDE)

    classification_sample_size = min(CLASSIFICATION_MAX_SAMPLES, n_samples)
    if n_samples > classification_sample_size:
        print("\nUsing a stratified subsample of", classification_sample_size,
              "instances for classifier training/testing to keep runtime practical.")
    classification_indices = get_stratified_indices(y, classification_sample_size)
    X_clf = X_norm[classification_indices]
    y_clf = y[classification_indices]
    X_clf_pca50 = pca_50.transform(X_clf)

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_clf
    )
    X_train_pca, X_test_pca, _, _ = train_test_split(
        X_clf_pca50, y_clf, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_clf
    )

    print("\nStep 13: Training classifiers on ORIGINAL and PCA-transformed data")
    print("Training set size:", X_train_orig.shape[0], " Test set size:", X_test_orig.shape[0])

    experiments = [
        ("KNN - Original Data", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS), X_train_orig, X_test_orig),
        ("KNN - PCA(50) Data", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS), X_train_pca, X_test_pca),
        ("SVM - Original Data", SVC(kernel=SVM_KERNEL, random_state=RANDOM_SEED), X_train_orig, X_test_orig),
        ("SVM - PCA(50) Data", SVC(kernel=SVM_KERNEL, random_state=RANDOM_SEED), X_train_pca, X_test_pca),
    ]

    all_results = []
    for label, model, X_train, X_test in experiments:
        result = train_and_evaluate_classifier(model, X_train, X_test, y_train, y_test, label)
        all_results.append(result)

    print("\nStep 14: Comparing Accuracy, Training Time, and Prediction Time")
    comparison_rows = [
        {
            "Model": r["label"],
            "Accuracy (%)": round(r["accuracy"] * 100, 3),
            "Training Time (s)": round(r["training_time"], 4),
            "Prediction Time (s)": round(r["prediction_time"], 4),
        }
        for r in all_results
    ]
    comparison_table = pd.DataFrame(comparison_rows)
    print(comparison_table.to_string(index=False))

    plot_classifier_comparison(all_results, "08_classifier_comparison.png")
    plot_confusion_matrices(all_results, "09_confusion_matrices.png")

    print("\nStep 15: Analysis of How Dimensionality Reduction Affects Recognition Performance")
    knn_orig_res, knn_pca_res, svm_orig_res, svm_pca_res = all_results
    report_pca_effect("KNN", knn_orig_res, knn_pca_res)
    report_pca_effect("SVM", svm_orig_res, svm_pca_res)

    components_for_95 = int(variance_thresholds[0.95])

    print("\nOverall conclusion:")
    print("PCA compresses the", n_features, "-dimensional pixel space down to just 50")
    print("dimensions while retaining the large majority of the meaningful variance,")
    print("(see the", components_for_95, "components needed for 95% variance).")
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
