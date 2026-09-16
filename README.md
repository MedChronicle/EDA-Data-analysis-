# 📊 Exploratory Data Analysis (EDA)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Active-success)

A structured collection of **Exploratory Data Analysis (EDA)** lab assignments and practical work, built in Python. This repository walks through the full data-preprocessing pipeline — cleaning, handling missing values, imputation, discretization, normalization, outlier detection, statistical testing, dimensionality reduction, and visualization — applied across several real-world datasets (COVID-19, wine quality, world happiness, suicide statistics, employee records, Iris, and MNIST digits).

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

This repository is a practical, hands-on record of core EDA concepts applied to multiple datasets. Each script is self-contained and focuses on one stage of the data preprocessing/analysis workflow, making it easy to study, reuse, or extend individual techniques. It now also includes dedicated experiments on **dimensionality reduction (PCA & t-SNE)** and classification, alongside the original preprocessing scripts.

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
├── SuicideChina.csv                                      # Suicide statistics dataset
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
| **SciPy**        | Statistical tests (Shapiro-Wilk, ANOVA)   |

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
pip install pandas numpy matplotlib seaborn scikit-learn scipy
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

---

## 🎯 Purpose

These scripts were developed as part of **EDA lab/practical assignments** to build a solid, hands-on understanding of:

- Data cleaning and preprocessing
- Handling missing/inconsistent data
- Statistical and visual data analysis
- Dimensionality reduction and its effect on downstream classification
- Preparing data for downstream machine learning tasks

---

## 👤 Author

**Arpit Saharan**

---

## 🕒 Last Updated

**16 September 2026**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](https://github.com/MedChronicle/EDA-Data-analysis-/blob/main/LICENSE) file for details.
