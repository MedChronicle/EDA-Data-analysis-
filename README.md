# 📊 EDA Data Analysis

A collection of Exploratory Data Analysis (EDA) lab assignments and practical work in Python, covering the full data-preprocessing pipeline — cleaning, missing-value handling, imputation, discretization, normalization, outlier detection, analysis, and visualization — applied to several real-world datasets (COVID-19, wine quality, world happiness, suicide statistics, employee records, and more).

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Techniques Covered](#-techniques-covered)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Purpose](#-purpose)
- [License](#-license)

## 🔍 Overview

Each script in this repository is self-contained and focuses on one stage of the EDA / data-preprocessing workflow, making it easy to study, reuse, or extend individual techniques. The `Experiment` folders group work by lab session.

## 📁 Repository Structure

```
EDA-Data-analysis-/
│
├── Experiment 1/                                   # Lab experiment 1 files
├── Experiment 2/                                   # Lab experiment 2 files
├── Experiment 3/                                   # Lab experiment 3 files
├── Experiment 4/                                   # Lab experiment 4 files
├── Experiment 5/                                   # Lab experiment 5 files
├── Experiment 6/                                   # Lab experiment 6 files
├── Experiment 7/                                   # Lab experiment 7 files
│
├── handling_missing_data_25bds0133.py              # Detecting & handling missing values
├── imputation_tecnhique_25bds0133.py               # Imputation strategies (mean/median/mode)
├── Discretization_and_Binning_25bds0133.py         # Binning continuous variables
├── Feature_Scaling_and_Normalization_25bds0133.py  # Min-Max / Z-score scaling
├── Outlier_Detection_25bds0133.py                  # Outlier identification & treatment
├── Arpit_Saharan_25bds0133_Exp4_wine_QT.py         # Wine quality EDA
├── covid_data_analysis.py                          # COVID-19 EDA & visualization
├── happiness_analysis.py                           # World happiness EDA
├── suicide_china_analysis.py                       # Suicide statistics EDA
├── iris_eda.py                                     # Iris dataset EDA
├── mnist_pca_tsne_project.py                       # MNIST dimensionality reduction (PCA/t-SNE)
├── 25bds0133.py / 25bds0133(Moodle).py             # Assignment submission scripts
│
├── WineQT.csv                                      # Wine quality dataset
├── covid-data.csv                                  # COVID-19 dataset
├── Covid Data(25BDS0133)                           # Additional COVID data file
├── happiness.csv                                   # World happiness dataset
├── SuicideChina.csv                                # Suicide statistics dataset
├── employee_dataset.csv                            # Employee records dataset
│
├── LICENSE                                         # MIT License
└── README.md                                       # Project documentation
```

## 🗂️ Datasets

| Dataset                  | Description                                                    |
| ------------------------ | ---------------------------------------------------------------|
| `WineQT.csv`              | Physicochemical properties and quality ratings of wine samples |
| `covid-data.csv`          | COVID-19 case and statistics data                               |
| `Covid Data(25BDS0133)`   | Additional COVID-19 data file                                   |
| `happiness.csv`           | World happiness index and related indicators                    |
| `SuicideChina.csv`        | Suicide statistics data (China)                                 |
| `employee_dataset.csv`    | Employee records for HR-style analysis                          |

## 🧠 Techniques Covered

| Technique                            | Script(s)                                                                                                         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Handling missing data                | `handling_missing_data_25bds0133.py`                                                                              |
| Imputation (mean / median / mode)    | `imputation_tecnhique_25bds0133.py`                                                                               |
| Discretization & binning             | `Discretization_and_Binning_25bds0133.py`                                                                         |
| Feature scaling & normalization      | `Feature_Scaling_and_Normalization_25bds0133.py`                                                                  |
| Outlier detection                    | `Outlier_Detection_25bds0133.py`                                                                                  |
| Dimensionality reduction (PCA/t-SNE) | `mnist_pca_tsne_project.py`                                                                                       |
| Dataset-specific EDA & visualization | `covid_data_analysis.py`, `happiness_analysis.py`, `suicide_china_analysis.py`, `iris_eda.py`, `Arpit_Saharan_25bds0133_Exp4_wine_QT.py` |

## 🛠️ Technologies Used

| Tool           | Purpose                        |
| -------------- | ------------------------------- |
| Python         | Core programming language       |
| Pandas         | Data manipulation and analysis  |
| NumPy          | Numerical computations          |
| Matplotlib     | Data visualization              |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MedChronicle/EDA-Data-analysis-.git
cd EDA-Data-analysis-
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

## ▶️ Usage

Run any script directly with Python:

```bash
python filename.py
```

**Example:**

```bash
python covid_data_analysis.py
```

> **Note:** Most scripts expect their corresponding CSV file (e.g. `WineQT.csv`, `covid-data.csv`, `happiness.csv`) to be present in the same directory. Make sure the relevant dataset is downloaded/available before running a script.

## 🎯 Purpose

These scripts were developed as part of EDA lab/practical assignments to build a solid, hands-on understanding of:

- Data cleaning and preprocessing
- Handling missing or inconsistent data
- Statistical and visual data analysis
- Preparing data for downstream machine learning tasks

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
