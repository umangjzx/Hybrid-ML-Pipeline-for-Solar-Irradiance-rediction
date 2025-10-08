☀️ Solar Irradiance Predictor
🌍 AI & Hybrid Models for Accurate and Explainable Solar Irradiance Prediction

This project is a Streamlit web application that predicts solar irradiance (W/m²) using advanced machine learning ensembles such as LightGBM, XGBoost, CatBoost, Random Forest, and more.
It provides interactive Exploratory Data Analysis (EDA), feature engineering, model comparison, performance visualization, and real-time prediction capabilities.

🚀 Features

✅ Upload & Analyze Solar Data

Upload one or multiple CSV files of solar plant data

Automated cleaning and feature extraction

Memory usage, missing value analysis, and data type summary

✅ Comprehensive EDA

Statistical summaries and correlation matrices

Target variable distribution

Time series and hourly pattern analysis

Temperature vs Irradiance relationships

✅ Advanced Feature Engineering

Automatic extraction of temporal, temperature-based, and lag features

Handles missing data and prepares time-series-aware train-test splits

✅ Multi-Model Ensemble Training

Supports models like LightGBM, XGBoost, CatBoost, Random Forest, Ridge, Lasso, and more

Evaluates using RMSE, MAE, R², and MAPE metrics

Visual comparison of model performances

✅ Interactive Results & Visualization

Performance tables and ranked comparison

RMSE and R² plots

Prediction vs Actual scatter

Residual and error distribution plots

Time series prediction visualizations

✅ Prediction Dashboard

Predict solar irradiance from new input data

Supports both manual and dataset-based sample input

Displays confidence level based on R² score

✅ Model Export

Download the trained best model as a .pkl file for reuse

🧠 Machine Learning Models Used
Category	Models
Tree-based Ensembles	LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Gradient Boosting
Linear Models	Ridge, Lasso, ElasticNet
Others (Full Mode)	AdaBoost, Decision Tree, KNN
🧩 Tech Stack
Component	Technology
Frontend / UI	Streamlit

Backend / ML Engine	Python (scikit-learn, LightGBM, XGBoost, CatBoost)
Visualization	Matplotlib, Seaborn
Data Handling	Pandas, NumPy
Export	Pickle serialization for trained models
