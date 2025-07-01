# ML-Therapy-Prediction

Machine Learning Pipeline for Clinical Therapy Scheme Prediction

---

## Project Overview

**ML-Therapy-Prediction** is a modular machine learning project for predicting neoadjuvant and adjuvant therapy schemes based on clinical data.  
The pipeline includes robust data cleaning, feature engineering, correlation-based reduction, single-label and multi-label model training, and evaluation.

Key capabilities:
- Automated data preprocessing, including missing value imputation and categorical encoding.
- Feature selection by removing highly correlated categorical variables using Cramér's V and numerical correlation using Pearson's coefficient.
- Support for both single therapy prediction and multi-label classification tasks.
- Baseline model benchmarking using LazyPredict.
- Custom pipelines with feature importance visualization and optional hyperparameter tuning.

---


## Directory Structure
```plaintext
ML-Therapy-Prediction/
├── Single_Therapy_ML/
│   ├── main.py
│   ├── pipeline_utils.py
│   ├── preprocess.py        
│   └── correlation_visualizer.py
│
├── Multi_Therapy_ML/
│   ├── main.py
│   ├── correlation_visualizer.py
│   ├── pipeline_utils.py     
│   └── preprocess.py
|
└── README.md               # Project documentation (this file)
```
## Requirements

- Python ≥ 3.8  
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `seaborn`  
- `matplotlib`  
- `xgboost`  
- `scikit-multilearn`  
- `lazypredict`

Install all dependencies with:
pip install -r requirements.txt

## Usage

### 1. Data Preprocessing

Preprocess raw data and generate cleaned and encoded datasets.

This script will:
- Clean and standardize string fields
- Handle missing values with median or mode imputation
- Remove redundant categorical features based on **Cramér’s V**
- Remove redundant numerical features based on **Pearson correlation**

Outputs:
- Cleaned Excel file
- Encoded numerical dataset
- Cramér’s V and correlation matrix printed to console and visualized as a heatmap.

### 2. Single-Label Classification

Train and evaluate single-label models.
cd Single_Therapy_ML
python main.py
This module:
- Runs LazyPredict for quick benchmarking
- Trains a chosen classifier (e.g., RandomForest, Bagging, ExtraTrees)
- Plots and exports feature importance
- Compares actual vs. predicted labels in an Excel report.

### 3. Multi-Label Classification

Train and evaluate multi-label models.
cd ../Multi_Therapy_ML
python main.py
This module:
- Encodes multiple target columns using `OneHotEncoder`
- Supports **MultiOutputClassifier**, **ClassifierChain**, and **LabelPowerset**
- Benchmarks each approach for Subset Accuracy and Hamming Loss
- Plots and exports top features for interpretability
- Performs hyperparameter tuning with `GridSearchCV`.

## Results

All results including:
- Cleaned datasets
- Feature importance ranking
- Prediction vs. actual comparison  
are automatically saved in the `results/` directory for further analysis or publication.


