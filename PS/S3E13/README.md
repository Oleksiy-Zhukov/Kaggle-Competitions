# [PS S3E13] Simple EDA + Ensemble for Kaggle Playground Series - Classification with a Tabular Vector Borne Disease Dataset

<img src="https://github.com/Oleksiy-Zhukov/Kaggle-Competitions/assets/75014961/74161f7c-b244-4fe0-90c5-2270ba50e3d1">

## Introduction

Welcome to the README for the Kaggle Playground Series competition titled "Classification with a Tabular Vector Borne Disease Dataset." This repository contains the code and analysis for the competition entry, which achieved a top 5 result out of 934 teams with a best MAP@3 score of 0.51644.

The goal of this competition was to explore, analyze, and build an ensemble model to predict the classification task associated with a tabular dataset related to vector-borne diseases. The competition dataset is synthetically generated from real-world data, allowing participants to quickly iterate through various model and feature engineering ideas while keeping test labels private.

## Kaggle Notebook
In the Kaggle Notebook [link](https://www.kaggle.com/code/zhukovoleksiy/5-solution-ps3e13-ensemble/notebook), I followed a structured approach to tackle the competition's task. Here's an outline of the key sections:

### 1. Import & Load Data
In this initial section, we imported necessary libraries and loaded the dataset, enabling us to proceed with data exploration and analysis.

To run the Kaggle Notebook and reproduce the analysis, make sure to have the following libraries and dependencies installed. You can use the [`requirements.txt`](https://github.com/Oleksiy-Zhukov/Kaggle-Competitions/blob/main/PS/S3E13/requirements.txt) file to install the necessary packages using `pip`:

The Kaggle Notebook utilizes various libraries for data preprocessing, model building, feature engineering, and visualization. Below is a breakdown of the libraries used in the import section:

* `numpy`: A library for numerical computing in Python.
* `pandas`: A powerful library for data manipulation and analysis.
* `plotly`: A library for interactive and declarative data visualization.
* `matplotlib`: A popular plotting library in Python for creating static, interactive, and animated plots.
* `seaborn`: A data visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.
* `category_encoders`: A library for encoding categorical features in various ways.
* `imbalanced-learn`: A library for dealing with imbalanced datasets using under-sampling and over-sampling techniques.
* `optuna`: An automatic hyperparameter optimization library.
* `joblib`: A library for lightweight pipelining in Python.
* `xgboost`: A gradient boosting library that provides an efficient and scalable implementation of gradient boosting.
* `lightgbm`: A gradient boosting library that is optimized for performance and handles large datasets efficiently.
* `catboost`: A gradient boosting library that is designed for categorical data and handles categorical features efficiently.
* `umap-learn`: A library for dimensionality reduction that provides manifold learning techniques.

These libraries are crucial for conducting the exploratory data analysis, feature engineering, model selection, hyperparameter tuning, and model evaluation in the Kaggle Notebook.

Please ensure you have installed the specified versions of these libraries before running the notebook.

*Notes*:
In the import section, we suppress warnings to ensure a clean output during the notebook execution. Additionally, you may notice that some libraries are imported but not explicitly used in the provided notebook content. These imports may have been used in other parts of the notebook, such as commented-out code or code blocks removed from the provided content.

### 3. Basic EDA
We performed Exploratory Data Analysis (EDA) to gain insights into the dataset's characteristics. This involved visualizing distributions, examining feature correlations, and identifying potential outliers.

<img src="https://github.com/Oleksiy-Zhukov/Kaggle-Competitions/assets/75014961/354066db-9bc8-4342-96c0-8b1667db13c2">


### 4. Feature Engineering
The Feature Engineering phase aimed to prepare the data for model building. We created new features, handled missing values, and preprocessed the dataset to enhance model performance.

### 5. Model Building
We employed a diverse set of machine learning models for the ensemble. The models used were:

Support Vector Classifier (SVC)
Logistic Regression (LR)
XGBoost (XGB)
LightGBM (LGB)
CatBoost (Cat)
Balanced Random Forest (BRF)
Random Forest (RF)
### 5. Model Optimization
To maximize model performance, we used Optuna to optimize hyperparameters for each model. Additionally, we optimized the ensemble weights to achieve the best possible combination of individual models.

### 6. Make Submission
After training and optimizing the models, we made the final predictions and submitted the results to Kaggle for evaluation.
https://www.kaggle.com/competitions/playground-series-s3e13

https://www.kaggle.com/competitions/playground-series-s3e13/data
