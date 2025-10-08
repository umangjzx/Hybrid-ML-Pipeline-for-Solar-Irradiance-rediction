# ☀️ Solar Irradiance Predictor  
### 🌍 AI & Hybrid Models for Accurate and Explainable Solar Irradiance Prediction  

This project is a **Streamlit web application** that predicts **solar irradiance (W/m²)** using advanced machine learning ensembles such as **LightGBM**, **XGBoost**, **CatBoost**, **Random Forest**, and more.  

It provides **interactive Exploratory Data Analysis (EDA)**, **feature engineering**, **model comparison**, **performance visualization**, and **real-time prediction capabilities** — all within a user-friendly interface.  

---

## 🚀 Features  

### ✅ Upload & Analyze Solar Data  
- Upload one or multiple CSV files of solar plant data  
- Automated data cleaning and feature extraction  
- Memory usage and missing value analysis  
- Data type summary and dataset statistics  

### ✅ Comprehensive EDA  
- Statistical summaries and correlation matrices  
- Target variable distribution visualization  
- Time series and hourly pattern analysis  
- Temperature vs Irradiance relationship plots  

### ✅ Advanced Feature Engineering  
- Automatic extraction of temporal, temperature-based, and lag features  
- Intelligent handling of missing data  
- Time-series-aware train-test splitting  

### ✅ Multi-Model Ensemble Training  
- Supports models like **LightGBM, XGBoost, CatBoost, Random Forest, Ridge, Lasso**, and more  
- Evaluation metrics: **RMSE**, **MAE**, **R²**, and **MAPE**  
- Visual comparison of model performance with plots and rankings  

### ✅ Interactive Results & Visualization  
- Performance summary tables and ranked comparisons  
- RMSE and R² score plots  
- Prediction vs Actual scatter plots  
- Residual and error distribution visualization  
- Time series prediction charts  

### ✅ Prediction Dashboard  
- Predict solar irradiance from new input data  
- Supports both manual entry and dataset-based prediction  
- Displays confidence level based on model’s R² score  

### ✅ Model Export  
- Download the trained **best model** as a `.pkl` file for reuse or deployment  

---

## 🧠 Machine Learning Models Used  

| **Category** | **Models** |
|---------------|------------|
| **Tree-based Ensembles** | LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Gradient Boosting |
| **Linear Models** | Ridge, Lasso, ElasticNet |
| **Others (Full Mode)** | AdaBoost, Decision Tree, KNN |

---

## 🧩 Tech Stack  

| **Component** | **Technology** |
|----------------|----------------|
| **Frontend / UI** | Streamlit |
| **Backend / ML Engine** | Python (scikit-learn, LightGBM, XGBoost, CatBoost) |
| **Visualization** | Matplotlib, Seaborn |
| **Data Handling** | Pandas, NumPy |
| **Model Export** | Pickle serialization (.pkl files) |

---

## ⚙️ Installation & Setup  

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Hybrid-ML-Pipeline-for-Solar-Irradiance-Prediction.git
   cd Hybrid-ML-Pipeline-for-Solar-Irradiance-Prediction
