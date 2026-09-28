# 📊 Exploratory Data Analysis (EDA) — Python Lab & Projects

A structured collection of **Exploratory Data Analysis (EDA), data preprocessing, visualization, dimensionality reduction, and introductory machine-learning practicals implemented in Python**.

This repository contains work organized by experiment/lab number, along with standalone scripts, datasets, generated visualizations, and a larger end-to-end EDA project. It is designed as both an academic practical repository and a reference for common data-analysis workflows.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Experiment-wise Documentation](#-experiment-wise-documentation)
  - [Experiment 1 — Python Fundamentals](#experiment-1--python-fundamentals)
  - [Experiment 2 — Exploratory Data Analysis](#experiment-2--exploratory-data-analysis)
  - [Experiment 3 — Data Preprocessing](#experiment-3--data-preprocessing)
  - [Experiment 4 — Wine Quality Analysis](#experiment-4--wine-quality-analysis)
  - [Experiment 5 — MNIST PCA and t-SNE](#experiment-5--mnist-pca-and-t-sne)
  - [Experiment 6 — MNIST Analysis and Classification](#experiment-6--mnist-analysis-and-classification)
  - [Experiment 7 — Iris Classification](#experiment-7--iris-classification)
  - [Experiment 8 — E-Commerce EDA](#experiment-8--e-commerce-eda)
  - [Experiment 9 — Airline Passenger Time-Series Analysis](#experiment-9--airline-passenger-time-series-analysis)
  - [EDA Project — Security Incident Analysis](#eda-project--security-incident-analysis)
- [Standalone Files](#-standalone-files)
- [Datasets](#-datasets)
- [Visualization Outputs](#-visualization-outputs)
- [Concepts Covered](#-concepts-covered)
- [Technologies and Libraries](#-technologies-and-libraries)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Project Workflow](#-project-workflow)
- [Notes](#-notes)
- [License](#-license)

---

# 🔍 Overview

The repository follows the typical data-analysis pipeline:

```text
Raw Data
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Missing-Value Handling
   ↓
Imputation
   ↓
Outlier Detection
   ↓
Feature Scaling / Normalization
   ↓
Exploratory Data Analysis
   ↓
Visualization
   ↓
Dimensionality Reduction
   ↓
Classification / Pattern Analysis
   ↓
Interpretation
```

The repository contains both **small focused practical programs** and **larger analysis projects**.

---

# 📁 Repository Structure

```text
EDA-Data-analysis-/
│
├── Experiment 1/
│   └── Python programming fundamentals
│
├── Experiment 2/
│   ├── COVID-19 analysis
│   ├── Happiness analysis
│   └── Suicide statistics analysis
│
├── Experiment 3/
│   ├── Missing-data handling
│   ├── Imputation
│   ├── Feature scaling
│   ├── Outlier detection
│   └── Data-analysis assignments
│
├── Experiment 4/
│   └── Wine Quality EDA
│
├── Experiment 5/
│   └── MNIST PCA and t-SNE analysis
│
├── Experiment 6/
│   └── MNIST dimensionality reduction + classification
│
├── Experiment 7/
│   └── Iris classification
│
├── Experiment 8/
│   └── E-Commerce sales EDA and visualizations
│
├── Experiment 9/
│   └── Airline passenger time-series analysis
│
├── eda_project/
│   ├── data/
│   │   └── security_incidents.csv
│   ├── outputs/
│   │   └── generated charts
│   ├── generate_dataset.py
│   ├── eda_analysis.py
│   └── EDA_notebook.ipynb
│
├── *.csv                         # Supporting datasets
├── *.py                          # Standalone practical/analysis scripts
├── LICENSE
├── .gitignore
└── README.md
```

---

# 🧪 Experiment-wise Documentation

## Experiment 1 — Python Fundamentals

**Folder:** `Experiment 1/`

This experiment contains introductory Python programs used as the programming foundation for later data-analysis work.

### Files

| File | Description |
|---|---|
| `01_variables_and_types.py` | Demonstrates Python variables and common data types. |
| `02_conditionals.py` | Demonstrates conditional statements such as `if`, `elif`, and `else`. |
| `03_functions.py` | Demonstrates defining and calling Python functions. |
| `04_data_structures.py` | Demonstrates lists, tuples, sets, dictionaries, and related operations. |
| `05_strings.py` | Covers string creation, slicing, formatting, and manipulation. |
| `06_file_handling.py` | Demonstrates creating, reading, writing, appending, and working with files. |
| `07_exceptions.py` | Demonstrates exception handling using `try`, `except`, and related constructs. |
| `08_classes.py` | Introduces object-oriented programming and Python classes. |
| `09_modules_and_math.py` | Demonstrates modules and mathematical operations. |
| `10_simple_algorithms.py` | Contains basic algorithmic/problem-solving programs. |

### Main Concepts

- Python syntax
- Variables and data types
- Conditions
- Functions
- Data structures
- Strings
- File handling
- Exception handling
- Classes and objects
- Modules
- Basic algorithms

---

## Experiment 2 — Exploratory Data Analysis

**Folder:** `Experiment 2/`

This experiment introduces EDA using multiple datasets. The scripts inspect datasets, calculate descriptive statistics, identify data characteristics, and answer dataset-specific analytical questions.

### Files

| File | Description |
|---|---|
| `happiness_analysis.py` | Performs EDA on the happiness dataset, including shape, metadata, descriptive statistics, and categorical analysis. |
| `happiness.csv` | Dataset used for the happiness analysis. |
| `suicide_china_analysis.py` | Performs statistical exploration of the China suicide dataset. |
| `SuicideChina.csv` | Dataset used for suicide-statistics analysis. |
| `covid_data_analysis.py` | Performs EDA on COVID-19 data, including records, features, statistics, and country/location analysis. |
| `covid-data.csv` | COVID-19 dataset used by the analysis script. |

### Main Concepts

- Dataset loading with Pandas
- Dataset shape and structure
- Column inspection
- Data types
- `describe()`
- Mean and standard deviation
- Frequency counts
- Unique values
- Basic statistical questions
- Missing-value inspection

---

## Experiment 3 — Data Preprocessing

**Folder:** `Experiment 3/`

This experiment focuses on preparing raw data for analysis and machine learning.

### Files

| File | Description |
|---|---|
| `handling_missing_data_25bds0133.py` | Demonstrates detection, counting, dropping, and filling of missing values. |
| `imputation_tecnhique_25bds0133.py` | Demonstrates mean, median, mode, constant, iterative/MICE-style, and KNN imputation techniques. |
| `Feature_Scaling_and_Normalization_25bds0133.py` | Applies Min-Max scaling, Standard scaling, and Robust scaling to numerical features. |
| `Outlier_Detection_25bds0133.py` | Detects transaction-value outliers using IQR and Z-score methods and creates cleaned/output datasets. |
| `25bds0133.py` | Performs EDA on the employee dataset, including descriptive statistics and missing-value analysis. |
| `25bds0133(Moodle).py` | Assignment/Moodle-oriented data manipulation and merging exercises. |

### Main Concepts

- Missing-value detection
- `dropna()`
- `fillna()`
- Forward fill / backward fill
- Mean/median/mode imputation
- Iterative imputation
- KNN imputation
- Min-Max normalization
- Standardization
- Robust scaling
- IQR outlier detection
- Z-score outlier detection
- Dataset cleaning

> **Note:** Some Experiment 3 scripts expect external files such as `fraudTrain.csv`. Make sure the required dataset is available in the working directory before running those scripts.

---

## Experiment 4 — Wine Quality Analysis

**Folder:** `Experiment 4/`

This experiment performs a structured EDA of the **Wine Quality** dataset.

### Files

| File | Description |
|---|---|
| `WineQT.csv` | Wine-quality dataset containing physicochemical properties and quality-related information. |
| `Arpit_Saharan_25bds0133_Exp4_wine_QT.py` | Performs dataset inspection, descriptive statistics, dispersion analysis, and visualization. |

### Analysis Areas

- Dataset shape
- Data types
- Missing values
- Descriptive statistics
- Mean
- Median
- Mode
- Standard deviation
- Variance
- Quartiles
- IQR
- Distribution analysis
- Visualization

The script can generate a `plots/` directory when executed.

---

## Experiment 5 — MNIST PCA and t-SNE

**Folder:** `Experiment 5/`

This experiment studies dimensionality reduction on handwritten-digit data.

### Main Script

`mnist_analysis.py`

### Techniques

- MNIST loading
- PCA
- Explained variance
- Scree plot
- Cumulative variance
- 2D PCA projection
- t-SNE
- t-SNE perplexity comparison
- Classification
- Confusion matrices

### Output Files

| Output | Purpose |
|---|---|
| `01_sample_digits.png` | Example handwritten digit images. |
| `02_scree_plot.png` | PCA component variance visualization. |
| `03_cumulative_variance_plot.png` | Cumulative explained variance. |
| `04_pca_2d_projection.png` | Two-dimensional PCA representation. |
| `05_tsne_original_data.png` | t-SNE visualization of original/high-dimensional data. |
| `06_tsne_pca50_data.png` | t-SNE after PCA reduction. |
| `07_tsne_perplexity_comparison.png` | Comparison of t-SNE results for different perplexity settings. |
| `08_classifier_comparison.png` | Comparison of classification models. |
| `09_confusion_matrices.png` | Confusion matrices for classifier evaluation. |

---

## Experiment 6 — MNIST Analysis and Classification

**Folder:** `Experiment 6/`

Experiment 6 contains another structured MNIST analysis workflow with dimensionality reduction and classification.

### Files

| File | Description |
|---|---|
| `mnist_analysis.py` | Loads MNIST/digit data, performs PCA and t-SNE, and evaluates classifiers. |
| `classifier_results.csv` | Stores classifier evaluation results. |
| `01_sample_digits.png` | Sample digit visualization. |
| `03_cumulative_variance.png` | PCA cumulative variance visualization. |
| `04_pca_2d_projection.png` | 2D PCA projection. |
| `05_tsne_raw_vs_pca.png` | Comparison of t-SNE representations from raw and PCA-reduced data. |
| `06_tsne_perplexity_comparison.png` | t-SNE perplexity comparison. |
| `07_classifier_comparison.png` | Classifier performance comparison. |
| `08_confusion_matrix_svm_pca.png` | Confusion matrix for SVM using PCA features. |

### Main Concepts

- Dimensionality reduction
- PCA
- t-SNE
- Variance analysis
- KNN
- SVM
- Classification accuracy
- Confusion matrix
- Model comparison

---

## Experiment 7 — Iris Classification

**Folder:** `Experiment 7/`

This experiment applies multiple machine-learning classifiers to the classic Iris dataset.

### Files

| File | Description |
|---|---|
| `classify_iris.py` | Loads and preprocesses Iris data, scales features, trains multiple classifiers, and evaluates their performance. |
| `iris.data.csv` | Iris dataset containing sepal/petal measurements and species labels. |

### Models Used

The classification script includes:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

### Evaluation

- Train/test split
- Feature standardization
- Cross-validation
- Accuracy
- Classification report
- Confusion matrix

---

## Experiment 8 — E-Commerce EDA

**Folder:** `Experiment 8/`

This is a comprehensive EDA experiment based on e-commerce sales data.

### Files

| File | Description |
|---|---|
| `eda_analysis.py` | Main EDA script for sales, ratings, price, reviews, categories, and relationships between variables. |
| `ecommerce_sales_data.csv` | E-commerce dataset used for the analysis. |

### Visualizations

| File | Visualization |
|---|---|
| `01_pie_sales_by_category.png` | Sales distribution by category. |
| `02_bar_avg_rating_by_category.png` | Average rating by category. |
| `03_histogram_price.png` | Price distribution. |
| `04_heatmap_sales_pivot.png` | Pivot-table heatmap of sales information. |
| `06_wordcloud_reviews.png` | Word cloud generated from review-related text. |
| `07_dendrogram_categories.png` | Hierarchical clustering/dendrogram visualization. |
| `08_scatter_price_vs_sales.png` | Relationship between product price and sales. |
| `10_pairplot.png` | Pairwise relationships between selected numerical variables. |
| `12_andrews_curves.png` | Andrews curves for multivariate pattern exploration. |

### Main Concepts

- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Category-level comparison
- Correlation/relationships
- Text visualization
- Hierarchical clustering visualization
- Pair plots
- Andrews curves
- Sales analysis

---

## Experiment 9 — Airline Passenger Time-Series Analysis

**Folder:** `Experiment 9/`

This experiment focuses on time-series exploration using monthly airline passenger data.

### Files

| File | Description |
|---|---|
| `airline_passengers_analysis.py` | Generates/loads monthly passenger data and performs time-based analysis and visualization. |
| `airline-passengers.csv` | Airline passenger time-series dataset. |
| `airline-passengers-analyzed.csv` | Processed/analyzed passenger dataset. |
| `passengers_timeseries.png` | Passenger time-series visualization. |
| `passengers_trend.png` | Trend visualization. |
| `passengers_yearly_avg.png` | Yearly average passenger visualization. |
| `passengers_cumulative.png` | Cumulative passenger visualization. |

### Main Concepts

- Date parsing
- Datetime indexing
- Monthly observations
- Time-series visualization
- Trend analysis
- Yearly aggregation
- Cumulative analysis
- Seasonal pattern exploration

---

# 🔐 EDA Project — Security Incident Analysis

**Folder:** `eda_project/`

This is the larger, more complete EDA project in the repository.

It analyzes a synthetic **cybersecurity/security incident dataset** containing information such as attack type, severity, country, protocol, risk score, duration, and related incident attributes.

## Project Structure

```text
eda_project/
│
├── data/
│   └── security_incidents.csv
│
├── outputs/
│   ├── hist_duration_sec.png
│   ├── hist_risk_score.png
│   ├── bar_chart.png
│   ├── pie_chart.png
│   ├── scatter_plot.png
│   ├── correlation_heatmap.png
│   ├── pairplot.png
│   ├── dendrogram.png
│   ├── parallel_coordinates.png
│   ├── andrews_curves.png
│   └── wordcloud.png
│
├── generate_dataset.py
├── eda_analysis.py
└── EDA_notebook.ipynb
```

### `generate_dataset.py`

Generates a reproducible synthetic security-incident dataset.

The dataset includes categories such as:

- DDoS
- Phishing
- Malware
- Brute Force
- SQL Injection
- Ransomware
- Port Scan

It also generates attributes related to:

- Severity
- Country
- Network protocol
- Risk score
- Incident duration
- Tags
- Date/time information

A fixed random seed is used to make the generated data reproducible.

### `data/security_incidents.csv`

Contains the generated security-incident records used by the main EDA analysis.

### `eda_analysis.py`

The main analysis script.

It uses:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- SciPy
- WordCloud
- Pandas plotting utilities

The script produces multiple statistical and visual analyses.

### `EDA_notebook.ipynb`

A notebook version of the EDA workflow, useful for interactive exploration and presentation.

### `outputs/`

Contains the generated visualization results.

| Output | Purpose |
|---|---|
| `hist_duration_sec.png` | Distribution of incident duration. |
| `hist_risk_score.png` | Distribution of risk scores. |
| `bar_chart.png` | Category/count comparison. |
| `pie_chart.png` | Proportional category distribution. |
| `scatter_plot.png` | Relationship between numerical variables. |
| `correlation_heatmap.png` | Correlation between numerical features. |
| `pairplot.png` | Pairwise feature relationships. |
| `dendrogram.png` | Hierarchical clustering structure. |
| `parallel_coordinates.png` | Multivariate comparison across categories. |
| `andrews_curves.png` | Multivariate class-pattern visualization. |
| `wordcloud.png` | Text/tag frequency visualization. |

---

# 📄 Standalone Files

The repository root also contains several scripts and datasets that complement the numbered experiments.

## Python Scripts

| File | Purpose |
|---|---|
| `mnist_pca_tsne_project.py` | Standalone MNIST dimensionality-reduction project using PCA and t-SNE. |
| `happiness_analysis.py` | Happiness dataset analysis. |
| `suicide_china_analysis.py` | China suicide-statistics analysis. |
| `eda_analysis.py` | General e-commerce-style EDA workflow. |
| `imputation_tecnhique_25bds0133.py` | Missing-value imputation demonstrations. |
| `Feature_Scaling_and_Normalization_25bds0133.py` | Feature scaling and normalization demonstrations. |
| `Discretization_and_Binning_25bds0133.py` | Continuous-variable discretization and binning. |
| `handling_missing_data_25bds0133.py` | Missing-data inspection and handling. |
| `Outlier_Detection_25bds0133.py` | IQR and Z-score based outlier detection. |
| `iris_eda.py` | Iris dataset exploratory analysis. |
| `25bds0133.py` | Employee dataset analysis/assignment work. |
| `25bds0133(Moodle).py` | Moodle-oriented assignment work. |
| `covid_data_analysis.py` | COVID-19 dataset analysis. |
| `Arpit_Saharan_25bds0133_Exp4_wine_QT.py` | Wine Quality EDA implementation. |

---

# 🗃️ Datasets

| Dataset | Used For |
|---|---|
| `WineQT.csv` | Wine Quality analysis |
| `covid-data.csv` | COVID-19 EDA |
| `Covid Data(25BDS0133)` | Additional COVID-related coursework |
| `happiness.csv` | Happiness analysis |
| `SuicideChina.csv` | Suicide-statistics analysis |
| `employee_dataset.csv` | Employee/HR-style EDA |
| `ecommerce_sales_data.csv` | E-commerce sales analysis |
| `iris.data.csv` | Iris classification |
| `airline-passengers.csv` | Airline passenger time-series analysis |
| `airline-passengers-analyzed.csv` | Processed airline passenger data |
| `security_incidents.csv` | Cybersecurity incident EDA |
| `classifier_results.csv` | MNIST classifier results |

---

# 📊 Visualization Outputs

The repository contains a wide range of visualization techniques:

### Basic Charts

- Pie charts
- Bar charts
- Histograms
- Scatter plots

### Statistical Visualizations

- Correlation heatmaps
- Pair plots
- Distribution plots

### Multivariate Visualizations

- Parallel coordinates
- Andrews curves
- Dendrograms

### Text Visualization

- Word clouds

### Dimensionality Reduction

- PCA projections
- t-SNE projections
- t-SNE perplexity comparisons

### Model Evaluation

- Classifier comparison charts
- Confusion matrices

---

# 🧠 Concepts Covered

This repository covers the following major areas:

## Python

- Variables
- Data types
- Conditions
- Functions
- Data structures
- Strings
- File handling
- Exception handling
- Classes
- Modules

## Data Manipulation

- Pandas DataFrames
- NumPy arrays
- Filtering
- Sorting
- Grouping
- Merging
- Aggregation
- Datetime handling

## Data Cleaning

- Missing-value detection
- Missing-value removal
- Missing-value filling
- Forward/backward filling
- Imputation
- Duplicate/invalid-data handling

## Data Preprocessing

- Min-Max scaling
- Standardization
- Robust scaling
- Discretization
- Binning
- Outlier detection
- IQR
- Z-score

## Exploratory Data Analysis

- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Categorical analysis

## Machine Learning

- Train/test splitting
- Feature scaling
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- SVM
- Cross-validation
- Accuracy
- Classification reports
- Confusion matrices

## Dimensionality Reduction

- PCA
- Explained variance
- Scree plots
- Cumulative variance
- t-SNE
- Perplexity comparison

## Time-Series Analysis

- Datetime indexing
- Trend analysis
- Seasonal exploration
- Monthly aggregation
- Yearly averages
- Cumulative analysis

---

# 🛠️ Technologies and Libraries

| Technology | Usage |
|---|---|
| **Python** | Main programming language |
| **NumPy** | Numerical computation |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical visualization |
| **Scikit-learn** | Preprocessing, dimensionality reduction, and machine learning |
| **SciPy** | Statistical and hierarchical-clustering utilities |
| **Plotly** | Interactive visualization |
| **WordCloud** | Text-frequency visualization |
| **Joblib** | Model/scaler persistence in applicable scripts |
| **Jupyter Notebook** | Interactive EDA workflow |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/MedChronicle/EDA-Data-analysis-.git
cd EDA-Data-analysis-
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install the main dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly wordcloud jupyter joblib
```

If a particular experiment requires an additional package, install that package before running the corresponding script.

---

# ▶️ How to Run

Most scripts can be executed directly from their respective folders.

For example:

```bash
cd "Experiment 2"
python happiness_analysis.py
```

For the security EDA project:

```bash
cd eda_project
python generate_dataset.py
python eda_analysis.py
```

To open the notebook:

```bash
jupyter notebook EDA_notebook.ipynb
```

---

# 🔄 Recommended Workflow

For a new dataset, the practical workflow demonstrated throughout this repository is:

### 1. Load

```python
df = pd.read_csv("data.csv")
```

### 2. Inspect

```python
df.head()
df.shape
df.info()
df.describe()
```

### 3. Check Data Quality

```python
df.isnull().sum()
df.duplicated().sum()
```

### 4. Clean

Handle:

- Missing values
- Invalid values
- Duplicate records
- Incorrect data types

### 5. Preprocess

Apply:

- Imputation
- Scaling
- Normalization
- Discretization
- Outlier treatment

### 6. Explore

Perform:

- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Correlation analysis

### 7. Visualize

Create appropriate charts to identify patterns, distributions, relationships, and anomalies.

### 8. Reduce Dimensions / Model

Where required:

- PCA
- t-SNE
- Classification
- Clustering-oriented visualizations

### 9. Interpret

Use the statistical results and visualizations to identify meaningful patterns and communicate findings.

---

# ⚠️ Notes

1. **Run scripts from the directory containing their required dataset** unless the script specifies another path.
2. Some scripts expect datasets that are not included in the repository, such as `fraudTrain.csv`. These files must be supplied separately.
3. Some MNIST scripts may download data from an external source; internet access may therefore be required for the full dataset-loading path.
4. Generated plots may be written to an output directory created by the script.
5. The repository contains both repeated/standalone versions of some experiments because the work was developed across multiple practical assignments.
6. File names containing student IDs or assignment labels are retained as part of the original academic submission structure.

---

# 🎓 Academic Purpose

This repository was created as a practical collection for learning and demonstrating:

- Exploratory Data Analysis
- Data preprocessing
- Statistical analysis
- Data visualization
- Dimensionality reduction
- Basic machine learning
- Time-series analysis
- Python programming for data science

It can be used as a reference for understanding the complete journey from **raw data → preprocessing → analysis → visualization → machine-learning-oriented exploration**.

---

# 📜 License

This project is distributed under the license specified in the [`LICENSE`](LICENSE) file.

---

## 👨‍💻 Author

**Arpit Saharan**

Data Science / Computer Science student

GitHub: [MedChronicle](https://github.com/MedChronicle)

---

⭐ If this repository helps you understand EDA or data preprocessing, consider giving it a star!
