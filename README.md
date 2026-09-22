# 📊 Exploratory Data Analysis (EDA)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?logo=plotly&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Active-success)

A structured collection of **Exploratory Data Analysis (EDA)** lab assignments and practical work, built in Python. This repository walks through the full data-preprocessing pipeline — cleaning, handling missing values, imputation, discretization, normalization, outlier detection, statistical testing, dimensionality reduction, and visualization — applied across several real-world and synthetic datasets (COVID-19, wine quality, world happiness, suicide statistics, employee records, Iris, MNIST digits, and cybersecurity incidents).

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Techniques Covered](#-techniques-covered)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Purpose](#-purpose)
- [Author](#-author)
- [Last Updated](#-last-updated)
- [License](#-license)

---

## 🔍 Overview

This repository is a practical, hands-on record of core EDA concepts applied to multiple datasets. Each script is self-contained and focuses on one stage of the data preprocessing/analysis workflow, making it easy to study, reuse, or extend individual techniques. It includes dedicated experiments on **dimensionality reduction (PCA & t-SNE)** and classification, alongside the original preprocessing scripts, and now a full **multi-chart visualization project on a cybersecurity incidents dataset**.

---

## 📁 Repository Structure

```
EDA-Data-analysis-/
│
├── Experiment 1/                                        # Lab experiment 1 files
├── Experiment 2/                                        # Lab experiment 2 files
├── Experiment 3/                                        # Lab experiment 3 files
├── Experiment 4/                                        # Lab experiment 4 files
├── Experiment 5/                                        # Lab experiment 5 files
├── Experiment 6/                                        # Lab experiment 6 files
├── Experiment 7/                                        # Lab experiment 7 files
├── Experiment 8/                                        # Cybersecurity incidents EDA project
│   ├── EDA_notebook.ipynb                                #   Notebook walkthrough of the full analysis
│   ├── eda_analysis.py                                   #   Script version: generates all visualizations
│   ├── generate_dataset.py                               #   Builds the synthetic security-incidents dataset
│   ├── data/
│   │   └── security_incidents.csv                        #   2,000-row synthetic security incident log
│   └── outputs/                                          #   Saved charts (pie, bar, heatmap, dendrogram, etc.)
│
├── handling_missing_data_25bds0133.py                    # Detecting & handling missing values
├── imputation_tecnhique_25bds0133.py                     # Imputation strategies (mean/median/mode)
├── Discretization_and_Binning_25bds0133.py               # Binning continuous variables
├── Feature_Scaling_and_Normalization_25bds0133.py        # Min-Max / Z-score scaling
├── Outlier_Detection_25bds0133.py                        # Outlier identification & treatment
├── Arpit_Saharan_25bds0133_Exp4_wine_QT.py               # Wine quality EDA
├── covid_data_analysis.py                                # COVID-19 EDA & visualization
├── happiness_analysis.py                                 # World happiness EDA
├── suicide_china_analysis.py                             # Suicide statistics EDA
├── iris_eda.py                                           # Iris dataset EDA, statistical tests & PCA
├── mnist_pca_tsne_project.py                             # MNIST digits: PCA, t-SNE & KNN classification
├── 25bds0133.py / 25bds0133(Moodle).py                   # Assignment submission scripts
│
├── WineQT.csv                                            # Wine quality dataset
├── covid-data.csv                                        # COVID-19 dataset
├── Covid Data(25BDS0133)                                 # Additional COVID data file
├── happiness.csv                                         # World happiness dataset
├── SuicideChina.csv                                       # Suicide statistics dataset
├── employee_dataset.csv                                  # Employee records dataset
├── .gitignore                                            # Git ignore rules
├── LICENSE                                               # MIT License
└── README.md                                             # Project documentation
```

---

## 🗂️ Datasets

| Dataset                    | Description                                                              |
| --------------------------- | ------------------------------------------------------------------------ |
| `WineQT.csv`                | Physicochemical properties and quality ratings of wine samples           |
| `covid-data.csv`            | COVID-19 case and statistics data                                        |
| `Covid Data(25BDS0133)`     | Additional COVID-19 data file                                            |
| `happiness.csv`             | World happiness index and related indicators                            |
| `SuicideChina.csv`          | Suicide statistics data (China)                                          |
| `employee_dataset.csv`      | Employee records for HR-style analysis                                  |
| Iris (via `scikit-learn`)   | Classic flower measurement dataset used for statistical EDA & PCA        |
| MNIST / Digits (via `scikit-learn`) | Handwritten digit images used for dimensionality reduction & classification |
| `security_incidents.csv` (Experiment 8) | Synthetic, 2,000-row cybersecurity incident log — attack type, severity, country, protocol, ports, duration, bytes transferred, packets, risk score, and description tags. Generated by `generate_dataset.py` |

---

## 🧠 Techniques Covered

| Technique                                  | Script                                                                                                                    |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Handling missing data                       | `handling_missing_data_25bds0133.py`                                                                                      |
| Imputation (mean / median / mode)           | `imputation_tecnhique_25bds0133.py`                                                                                       |
| Discretization & binning                    | `Discretization_and_Binning_25bds0133.py`                                                                                 |
| Feature scaling & normalization             | `Feature_Scaling_and_Normalization_25bds0133.py`                                                                          |
| Outlier detection (IQR method)              | `Outlier_Detection_25bds0133.py`, `iris_eda.py`                                                                           |
| Statistical testing (Shapiro-Wilk, ANOVA)   | `iris_eda.py`                                                                                                              |
| Dimensionality reduction (PCA, t-SNE)       | `iris_eda.py`, `mnist_pca_tsne_project.py`                                                                                |
| Classification (KNN) & accuracy benchmarking| `mnist_pca_tsne_project.py`                                                                                                |
| Dataset-specific EDA & visualization         | `covid_data_analysis.py`, `happiness_analysis.py`, `suicide_china_analysis.py`, `Arpit_Saharan_25bds0133_Exp4_wine_QT.py`, `iris_eda.py` |
| Multi-chart visualization suite (pie, bar, histogram, correlation heatmap, choropleth map, word cloud, dendrogram/hierarchical clustering, scatter, pair plot, parallel coordinates, Andrews curves) | `Experiment 8/eda_analysis.py`, `Experiment 8/EDA_notebook.ipynb` |

---

## 🛠️ Technologies Used

| Tool             | Purpose                                  |
| ---------------- | ----------------------------------------- |
| **Python**       | Core programming language                 |
| **Pandas**       | Data manipulation and analysis            |
| **NumPy**        | Numerical computations                    |
| **Matplotlib**   | Data visualization                        |
| **Seaborn**      | Statistical data visualization            |
| **Scikit-learn** | PCA, t-SNE, KNN classification, datasets  |
| **SciPy**        | Statistical tests (Shapiro-Wilk, ANOVA) & hierarchical clustering (dendrograms) |
| **Plotly**       | Interactive choropleth mapping            |
| **WordCloud**    | Text-frequency visualization              |

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/MedChronicle/EDA-Data-analysis-.git
cd EDA-Data-analysis-
```

### 2. (Optional) Create a virtual environment

```
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly wordcloud
```

---

## ▶️ Usage

Run any script directly with Python:

```
python filename.py
```

**Example:**

```
python covid_data_analysis.py
python iris_eda.py
python mnist_pca_tsne_project.py
```

> **Note:** Most scripts expect their corresponding CSV file (e.g. `WineQT.csv`, `covid-data.csv`, `happiness.csv`) to be present in the same directory. `iris_eda.py` and `mnist_pca_tsne_project.py` fetch their datasets directly through `scikit-learn`, so no CSV download is needed for those — `mnist_pca_tsne_project.py` will save its plots and results into an `outputs/` folder created automatically.

### Experiment 8 — Cybersecurity Incidents EDA

```
cd "Experiment 8"
python generate_dataset.py      # (re)generates data/security_incidents.csv
python eda_analysis.py          # runs the full analysis and saves charts to outputs/
```

Or open `EDA_notebook.ipynb` to step through the same analysis interactively. The script/notebook covers dataset overview and summary statistics, a pie chart of attack-type share, a bar chart of incidents by severity, histograms of risk score and incident duration, a correlation heatmap, a scatter plot of duration vs. bytes transferred (colored by severity), a choropleth map of incidents by country, a word cloud of incident descriptions, a dendrogram clustering attack types by behavioral similarity, a pair plot and parallel-coordinates plot by severity, and Andrews curves by attack type.

---

## 🎯 Purpose

These scripts were developed as part of **EDA lab/practical assignments** to build a solid, hands-on understanding of:

- Data cleaning and preprocessing
- Handling missing/inconsistent data
- Statistical and visual data analysis
- Dimensionality reduction and its effect on downstream classification
- Multi-chart visualization techniques for exploratory storytelling
- Preparing data for downstream machine learning tasks

---

## 👤 Author

**Arpit Saharan**

---

## 🕒 Last Updated

**22 September 2026**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](https://github.com/MedChronicle/EDA-Data-analysis-/blob/main/LICENSE) file for details.
