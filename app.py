import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
from datetime import datetime
import time
import io
import pickle

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Page Configuration
st.set_page_config(
    page_title="Solar Irradiance Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #FFA500, #FF6B35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


class SolarIrradiancePredictor:
    """Advanced Solar Irradiance Prediction System"""
    
    def __init__(self, quick_mode=True):
        self.quick_mode = quick_mode
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.data = None
        self.raw_data = None
        self.feature_names = None
        self.best_model_name = None
        self.training_time = 0
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, uploaded_files):
        """Load and combine solar plant data from uploaded files"""
        dfs = []
        for i, uploaded_file in enumerate(uploaded_files, 1):
            try:
                df = pd.read_csv(uploaded_file)
                df['plant_id'] = i
                dfs.append(df)
            except Exception as e:
                st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        
        if not dfs:
            raise ValueError("No data loaded. Please upload valid CSV files.")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self.raw_data = combined_df.copy()
        return combined_df
    
    def get_eda_stats(self):
        """Get EDA statistics"""
        if self.raw_data is None:
            return None
        
        df = self.raw_data.copy()
        
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='mixed', dayfirst=True)
        
        stats = {
            'shape': df.shape,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing': df.isnull().sum(),
            'numeric_summary': df.select_dtypes(include=[np.number]).describe(),
            'dtypes': df.dtypes.value_counts()
        }
        
        if 'DATE_TIME' in df.columns:
            stats['date_range'] = (df['DATE_TIME'].min(), df['DATE_TIME'].max())
        
        return stats
    
    def create_eda_plots(self):
        """Create EDA visualizations"""
        if self.raw_data is None:
            return None
        
        df = self.raw_data.copy()
        
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='mixed', dayfirst=True)
        
        plots = {}
        
        # 1. Target Distribution
        if 'IRRADIATION' in df.columns:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            df['IRRADIATION'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(df['IRRADIATION'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["IRRADIATION"].mean():.2f}')
            ax1.axvline(df['IRRADIATION'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["IRRADIATION"].median():.2f}')
            ax1.set_xlabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title('Target Variable: Irradiation Distribution', fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            plots['target_dist'] = fig1
        
        # 2. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols_filtered = [col for col in numeric_cols if col not in ['PLANT_ID', 'SOURCE_KEY', 'plant_id']]
        
        if len(numeric_cols_filtered) > 0:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            corr_matrix = df[numeric_cols_filtered].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax2, square=True, linewidths=0.5)
            ax2.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=12)
            plots['correlation'] = fig2
        
        # 3. Time Series
        if 'DATE_TIME' in df.columns and 'IRRADIATION' in df.columns:
            df_sorted = df.sort_values('DATE_TIME')
            plot_size = min(5000, len(df_sorted))
            df_plot = df_sorted.tail(plot_size)
            
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            ax3.plot(df_plot['DATE_TIME'], df_plot['IRRADIATION'], linewidth=0.5, alpha=0.7, color='darkorange')
            ax3.fill_between(df_plot['DATE_TIME'], df_plot['IRRADIATION'], alpha=0.3, color='orange')
            ax3.set_xlabel('Date Time', fontweight='bold')
            ax3.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax3.set_title(f'Time Series: Irradiation Pattern (Last {plot_size} records)', fontweight='bold')
            ax3.grid(alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            plots['timeseries'] = fig3
        
        # 4. Hourly Pattern
        if 'DATE_TIME' in df.columns and 'IRRADIATION' in df.columns:
            df['hour'] = df['DATE_TIME'].dt.hour
            hourly_avg = df.groupby('hour')['IRRADIATION'].agg(['mean', 'std']).reset_index()
            
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            ax4.plot(hourly_avg['hour'], hourly_avg['mean'], marker='o', linewidth=2, color='darkblue', label='Mean')
            ax4.fill_between(hourly_avg['hour'], hourly_avg['mean'] - hourly_avg['std'], hourly_avg['mean'] + hourly_avg['std'], alpha=0.3, color='lightblue', label='±1 Std Dev')
            ax4.set_xlabel('Hour of Day', fontweight='bold')
            ax4.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax4.set_title('Hourly Irradiation Pattern', fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            ax4.set_xticks(range(0, 24, 2))
            plots['hourly'] = fig4
        
        # 5. Temperature vs Irradiation
        if 'AMBIENT_TEMPERATURE' in df.columns and 'IRRADIATION' in df.columns:
            sample = df.sample(min(5000, len(df)))
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            scatter = ax5.scatter(sample['AMBIENT_TEMPERATURE'], sample['IRRADIATION'], alpha=0.4, s=10, c=sample['IRRADIATION'], cmap='viridis')
            ax5.set_xlabel('Ambient Temperature (°C)', fontweight='bold')
            ax5.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax5.set_title('Temperature vs Irradiation Relationship', fontweight='bold')
            ax5.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label='Irradiance')
            plots['temp_vs_irr'] = fig5
        
        return plots
    
    def engineer_features(self, df):
        """Advanced feature engineering for solar data"""
        df = df.copy()
        
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='mixed', dayfirst=True)
            df = df.set_index('DATE_TIME')
        
        target_col = 'IRRADIATION'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")
        
        # Temporal features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_year'] = df.index.dayofyear
            df['month'] = df.index.month
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            df['quarter'] = df.index.quarter
            df['week_of_year'] = df.index.isocalendar().week
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        
        # Temperature features
        if 'AMBIENT_TEMPERATURE' in df.columns:
            df['ambient_temp_sq'] = df['AMBIENT_TEMPERATURE'] ** 2
            df['ambient_temp_cube'] = df['AMBIENT_TEMPERATURE'] ** 3
        
        if 'MODULE_TEMPERATURE' in df.columns:
            df['module_temp_sq'] = df['MODULE_TEMPERATURE'] ** 2
            df['module_temp_cube'] = df['MODULE_TEMPERATURE'] ** 3
        
        if 'AMBIENT_TEMPERATURE' in df.columns and 'MODULE_TEMPERATURE' in df.columns:
            df['temp_diff'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
            df['temp_ratio'] = df['MODULE_TEMPERATURE'] / (df['AMBIENT_TEMPERATURE'] + 1e-6)
            df['temp_product'] = df['MODULE_TEMPERATURE'] * df['AMBIENT_TEMPERATURE']
        
        # Lag features
        lags = [1, 2, 3] if self.quick_mode else [1, 2, 3, 6, 12, 24]
        for lag in lags:
            df[f'irr_lag_{lag}'] = df[target_col].shift(lag)
            if 'AMBIENT_TEMPERATURE' in df.columns:
                df[f'temp_lag_{lag}'] = df['AMBIENT_TEMPERATURE'].shift(lag)
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['PLANT_ID', 'SOURCE_KEY', 'plant_id']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        df = df[numeric_cols].copy()
        
        # Handle missing values
        df = df.dropna()
        df = df.rename(columns={target_col: 'target'})
        
        self.data = df
        return df
    
    def prepare_train_test(self, test_size=0.2):
        """Prepare time-series aware train-test split"""
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        self.feature_names = X.columns.tolist()
        
        split_idx = int(len(self.data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def create_models(self):
        """Create ensemble of ML models"""
        if self.quick_mode:
            models = {
                'LightGBM': LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1),
                'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, verbosity=0),
                'CatBoost': CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=0, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'Lasso': Lasso(alpha=1.0, random_state=42, max_iter=2000),
                'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
            }
        else:
            models = {
                'LightGBM': LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1),
                'XGBoost': XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=0),
                'CatBoost': CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05, verbose=0, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42),
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'Lasso': Lasso(alpha=1.0, random_state=42, max_iter=3000),
                'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=3000),
                'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeRegressor(max_depth=15, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
            }
        return models
    
    def evaluate_model(self, name, y_true, y_pred):
        """Calculate comprehensive metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        self.results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models in ensemble"""
        models = self.create_models()
        total_start = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            try:
                status_text.text(f"Training {name}...")
                start = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                elapsed = time.time() - start
                
                self.evaluate_model(name, y_test, y_pred)
                self.models[name] = model
                
                progress_bar.progress((i + 1) / len(models))
            except Exception as e:
                st.warning(f"Training failed for {name}: {str(e)}")
        
        self.training_time = time.time() - total_start
        
        results_df = pd.DataFrame(self.results).T.sort_values('RMSE')
        self.best_model_name = results_df.index[0]
        
        status_text.text(f"✅ Training complete! Best model: {self.best_model_name}")
        progress_bar.empty()
    
    def create_result_plots(self):
        """Create result visualizations"""
        if not self.results:
            return None
        
        results_df = pd.DataFrame(self.results).T.sort_values('RMSE')
        best_model = self.models[self.best_model_name]
        y_pred = best_model.predict(self.X_test)
        
        plots = {}
        
        # 1. RMSE Comparison
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
        results_df['RMSE'].sort_values().plot(kind='barh', ax=ax1, color=colors)
        ax1.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
        ax1.set_title('Model Performance: RMSE', fontweight='bold')
        ax1.grid(alpha=0.3, axis='x')
        plots['rmse'] = fig1
        
        # 2. R² Score Comparison
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(results_df))]
        results_df['R²'].sort_values(ascending=False).plot(kind='barh', ax=ax2, color=colors)
        ax2.set_xlabel('R² Score (Higher is Better)', fontweight='bold')
        ax2.set_title('Model Performance: R² Score', fontweight='bold')
        ax2.grid(alpha=0.3, axis='x')
        plots['r2'] = fig2
        
        # 3. Prediction vs Actual
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sample_size = min(2000, len(self.y_test))
        sample_idx = np.random.choice(len(self.y_test), sample_size, replace=False)
        scatter = ax3.scatter(self.y_test[sample_idx], y_pred[sample_idx], alpha=0.5, s=10, c=self.y_test[sample_idx], cmap='viridis')
        ax3.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Irradiance (W/m²)', fontweight='bold')
        ax3.set_ylabel('Predicted Irradiance (W/m²)', fontweight='bold')
        ax3.set_title(f'Predictions: {self.best_model_name}', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Actual Value')
        plots['pred_vs_actual'] = fig3
        
        # 4. Residual Plot
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        residuals = self.y_test - y_pred
        ax4.scatter(y_pred[sample_idx], residuals[sample_idx], alpha=0.5, s=10, c='coral')
        ax4.axhline(y=0, color='red', linestyle='--', lw=2)
        ax4.set_xlabel('Predicted Irradiance (W/m²)', fontweight='bold')
        ax4.set_ylabel('Residuals (W/m²)', fontweight='bold')
        ax4.set_title('Residual Analysis', fontweight='bold')
        ax4.grid(alpha=0.3)
        plots['residuals'] = fig4
        
        # 5. Time Series
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        plot_range = min(300, len(self.y_test))
        time_idx = range(plot_range)
        ax5.plot(time_idx, self.y_test[-plot_range:], label='Actual', linewidth=2, alpha=0.8)
        ax5.plot(time_idx, y_pred[-plot_range:], label='Predicted', linewidth=2, alpha=0.8)
        ax5.fill_between(time_idx, self.y_test[-plot_range:], y_pred[-plot_range:], alpha=0.2)
        ax5.set_xlabel('Time Steps', fontweight='bold')
        ax5.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
        ax5.set_title('Time Series: Actual vs Predicted', fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(alpha=0.3)
        plots['timeseries'] = fig5
        
        # 6. Error Distribution
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax6.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax6.axvline(x=residuals.mean(), color='green', linestyle='--', lw=2, label=f'Mean: {residuals.mean():.2f}')
        ax6.set_xlabel('Prediction Error (W/m²)', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('Error Distribution', fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3, axis='y')
        plots['error_dist'] = fig6
        
        return plots
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if not self.models:
            raise ValueError("No trained models available.")
        
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        for feat in self.feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        input_df = input_df[self.feature_names]
        input_scaled = self.scaler.transform(input_df)
        best_model = self.models[self.best_model_name]
        predictions = best_model.predict(input_scaled)
        
        return predictions, self.best_model_name


# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


# Main App
def main():
    st.markdown('<h1 class="main-header">☀️ Solar Irradiance Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML-powered solar irradiance forecasting with comprehensive EDA</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/sun.png", width=80)
        st.title("⚙️ Configuration")
        
        quick_mode = st.checkbox("Quick Mode", value=True, help="Faster training with slightly lower accuracy")
        test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
        
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.info("⏳ No data loaded")
        
        if st.session_state.trained:
            st.success("✅ Model Trained")
        else:
            st.info("⏳ Model not trained")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("This app uses ensemble ML models to predict solar irradiance based on weather sensor data.")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Data Upload", "🔍 EDA", "🤖 Model Training", "📊 Results", "🔮 Prediction"])
    
    # TAB 1: DATA UPLOAD
    with tab1:
        st.header("📂 Upload Solar Plant Data")
        st.markdown("Upload one or more CSV files containing solar plant weather sensor data.")
        
        uploaded_files = st.file_uploader("Choose CSV files", type=['csv'], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Load Data", type="primary"):
                with st.spinner("Loading data..."):
                    try:
                        predictor = SolarIrradiancePredictor(quick_mode=quick_mode)
                        df = predictor.load_data(uploaded_files)
                        st.session_state.predictor = predictor
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Successfully loaded {len(df):,} records from {len(uploaded_files)} file(s)")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Records", f"{len(df):,}")
                        with col2:
                            st.metric("Features", df.shape[1])
                        with col3:
                            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                        with col4:
                            st.metric("Files Loaded", len(uploaded_files))
                        
                        st.subheader("Data Preview")
                        st.dataframe(df.head(100), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
    
    # TAB 2: EDA
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data first in the 'Data Upload' tab.")
        else:
            predictor = st.session_state.predictor
            
            if st.button("Generate EDA Report", type="primary"):
                with st.spinner("Generating EDA visualizations..."):
                    try:
                        stats = predictor.get_eda_stats()
                        
                        # Basic Statistics
                        st.subheader("📊 Dataset Overview")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{stats['shape'][0]:,}")
                        with col2:
                            st.metric("Columns", stats['shape'][1])
                        with col3:
                            st.metric("Memory", f"{stats['memory_mb']:.2f} MB")
                        
                        if 'date_range' in stats:
                            st.info(f"📅 Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
                        
                        # Missing Values
                        st.subheader("🔍 Missing Values Analysis")
                        missing = stats['missing'][stats['missing'] > 0]
                        if len(missing) > 0:
                            missing_df = pd.DataFrame({
                                'Column': missing.index,
                                'Missing Count': missing.values,
                                'Percentage': (missing.values / stats['shape'][0] * 100).round(2)
                            })
                            st.dataframe(missing_df, use_container_width=True)
                        else:
                            st.success("✅ No missing values found!")
                        
                        # Statistical Summary
                        st.subheader("📈 Statistical Summary")
                        st.dataframe(stats['numeric_summary'].round(2), use_container_width=True)
                        
                        # Data Types
                        st.subheader("📋 Data Types Distribution")
                        st.bar_chart(stats['dtypes'])
                        
                        # Visualizations
                        st.subheader("📊 Visual Analysis")
                        plots = predictor.create_eda_plots()
                        
                        if plots:
                            # Target Distribution
                            if 'target_dist' in plots:
                                st.pyplot(plots['target_dist'])
                            
                            # Correlation Heatmap
                            if 'correlation' in plots:
                                st.pyplot(plots['correlation'])
                            
                            # Time Series
                            if 'timeseries' in plots:
                                st.pyplot(plots['timeseries'])
                            
                            # Two columns for smaller plots
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'hourly' in plots:
                                    st.pyplot(plots['hourly'])
                            with col2:
                                if 'temp_vs_irr' in plots:
                                    st.pyplot(plots['temp_vs_irr'])
                        
                        st.success("✅ EDA Complete!")
                        
                    except Exception as e:
                        st.error(f"Error generating EDA: {str(e)}")
    
    # TAB 3: MODEL TRAINING
    with tab3:
        st.header("🤖 Model Training")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data first in the 'Data Upload' tab.")
        else:
            st.markdown("### Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Mode:** {'⚡ Quick Mode' if quick_mode else '🎯 Precision Mode'}")
            with col2:
                st.info(f"**Test Size:** {test_size*100:.0f}%")
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        predictor = st.session_state.predictor
                        
                        # Feature Engineering
                        st.info("⚙️ Engineering features...")
                        df = predictor.engineer_features(predictor.raw_data)
                        st.success(f"✅ Created {df.shape[1]-1} features from {len(df):,} records")
                        
                        # Prepare train-test split
                        st.info("📊 Preparing train-test split...")
                        X_train, X_test, y_train, y_test = predictor.prepare_train_test(test_size)
                        st.success(f"✅ Training: {len(y_train):,} samples | Testing: {len(y_test):,} samples")
                        
                        # Train models
                        st.info("🤖 Training ensemble models...")
                        predictor.train_models(X_train, X_test, y_train, y_test)
                        
                        st.session_state.predictor = predictor
                        st.session_state.trained = True
                        
                        # Show results summary
                        results_df = pd.DataFrame(predictor.results).T.sort_values('RMSE')
                        
                        st.success(f"🏆 Training Complete! Best Model: **{predictor.best_model_name}**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        best_metrics = results_df.loc[predictor.best_model_name]
                        with col1:
                            st.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
                        with col2:
                            st.metric("MAE", f"{best_metrics['MAE']:.2f}")
                        with col3:
                            st.metric("R² Score", f"{best_metrics['R²']:.4f}")
                        with col4:
                            st.metric("MAPE", f"{best_metrics['MAPE']:.2f}%")
                        
                        st.markdown("### 📊 All Models Performance")
                        st.dataframe(results_df.round(4), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # TAB 4: RESULTS
    with tab4:
        st.header("📊 Model Results & Visualizations")
        
        if not st.session_state.trained:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
        else:
            predictor = st.session_state.predictor
            results_df = pd.DataFrame(predictor.results).T.sort_values('RMSE')
            
            # Performance Summary
            st.subheader("🏆 Best Model Summary")
            best_model_name = predictor.best_model_name
            best_metrics = results_df.loc[best_model_name]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Model", best_model_name)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMSE", f"{best_metrics['RMSE']:.2f} W/m²")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("R² Score", f"{best_metrics['R²']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("MAPE", f"{best_metrics['MAPE']:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Model Comparison Table
            st.subheader("📈 Model Comparison")
            
            # Add rank column
            results_display = results_df.copy()
            results_display.insert(0, 'Rank', range(1, len(results_display) + 1))
            
            # Color code the best model
            st.dataframe(
                results_display.style.highlight_max(subset=['R²'], color='lightgreen')
                                    .highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen'),
                use_container_width=True
            )
            
            # Visualizations
            st.subheader("📊 Performance Visualizations")
            
            with st.spinner("Generating plots..."):
                plots = predictor.create_result_plots()
                
                if plots:
                    # RMSE and R2 side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'rmse' in plots:
                            st.pyplot(plots['rmse'])
                    with col2:
                        if 'r2' in plots:
                            st.pyplot(plots['r2'])
                    
                    # Prediction vs Actual
                    if 'pred_vs_actual' in plots:
                        st.pyplot(plots['pred_vs_actual'])
                    
                    # Residuals and Error Distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'residuals' in plots:
                            st.pyplot(plots['residuals'])
                    with col2:
                        if 'error_dist' in plots:
                            st.pyplot(plots['error_dist'])
                    
                    # Time Series
                    if 'timeseries' in plots:
                        st.pyplot(plots['timeseries'])
            
            # Download Model
            st.subheader("💾 Export Model")
            if st.button("Download Best Model"):
                model_data = {
                    'model': predictor.models[best_model_name],
                    'scaler': predictor.scaler,
                    'feature_names': predictor.feature_names,
                    'model_name': best_model_name,
                    'metrics': predictor.results[best_model_name]
                }
                
                buffer = io.BytesIO()
                pickle.dump(model_data, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Model (.pkl)",
                    data=buffer,
                    file_name=f"solar_model_{best_model_name.replace(' ', '_')}.pkl",
                    mime="application/octet-stream"
                )
    
    # TAB 5: PREDICTION
    with tab5:
        st.header("🔮 Make Predictions")
        
        if not st.session_state.trained:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
        else:
            predictor = st.session_state.predictor
            
            st.markdown("### Input Features")
            st.info("Enter the values for prediction or use sample data from the dataset.")
            
            # Option to use sample or manual input
            input_mode = st.radio("Input Mode", ["Sample from Dataset", "Manual Input"])
            
            if input_mode == "Sample from Dataset":
                if st.button("Load Random Sample"):
                    sample = predictor.data.sample(1).drop('target', axis=1).iloc[0].to_dict()
                    st.session_state.sample_input = sample
                
                if 'sample_input' in st.session_state:
                    st.success("✅ Sample loaded! Edit values below if needed.")
                    
                    # Display sample in editable form
                    col1, col2, col3 = st.columns(3)
                    
                    input_data = {}
                    sample = st.session_state.sample_input
                    
                    # Key features for display
                    key_features = ['hour', 'month', 'day_of_year', 'AMBIENT_TEMPERATURE', 
                                  'MODULE_TEMPERATURE', 'temp_diff', 'is_daytime']
                    
                    display_features = [f for f in key_features if f in sample]
                    
                    for i, feat in enumerate(display_features):
                        with [col1, col2, col3][i % 3]:
                            if isinstance(sample[feat], (int, np.integer)):
                                input_data[feat] = st.number_input(feat, value=int(sample[feat]), key=f"input_{feat}")
                            else:
                                input_data[feat] = st.number_input(feat, value=float(sample[feat]), format="%.2f", key=f"input_{feat}")
                    
                    # Use all features from sample
                    for feat in sample:
                        if feat not in input_data:
                            input_data[feat] = sample[feat]
                    
                    if st.button("🔮 Predict", type="primary"):
                        with st.spinner("Making prediction..."):
                            try:
                                predictions, model_name = predictor.predict(input_data)
                                
                                st.success("✅ Prediction Complete!")
                                
                                # Display result prominently
                                st.markdown("### 📊 Prediction Result")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Predicted Irradiance", f"{predictions[0]:.2f} W/m²", 
                                            help="Predicted solar irradiance value")
                                with col2:
                                    st.metric("Model Used", model_name)
                                with col3:
                                    confidence = predictor.results[model_name]['R²'] * 100
                                    st.metric("Model Confidence", f"{confidence:.1f}%")
                                
                                # Show input summary
                                st.markdown("### 📝 Input Summary")
                                input_df = pd.DataFrame([{k: v for k, v in input_data.items() if k in display_features}])
                                st.dataframe(input_df, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
            
            else:  # Manual Input
                st.markdown("Enter values manually:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
                    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
                    day_of_year = st.number_input("Day of Year (1-365)", min_value=1, max_value=365, value=180)
                
                with col2:
                    ambient_temp = st.number_input("Ambient Temperature (°C)", value=25.0, format="%.2f")
                    module_temp = st.number_input("Module Temperature (°C)", value=35.0, format="%.2f")
                
                with col3:
                    is_weekend = st.selectbox("Is Weekend?", [0, 1])
                    is_daytime = st.selectbox("Is Daytime?", [0, 1])
                
                if st.button("🔮 Predict", type="primary"):
                    # Create input with required features
                    input_data = {
                        'hour': hour,
                        'month': month,
                        'day_of_year': day_of_year,
                        'AMBIENT_TEMPERATURE': ambient_temp,
                        'MODULE_TEMPERATURE': module_temp,
                        'is_weekend': is_weekend,
                        'is_daytime': is_daytime,
                        'temp_diff': module_temp - ambient_temp,
                    }
                    
                    # Add derived features
                    input_data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                    input_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
                    input_data['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
                    input_data['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)
                    input_data['ambient_temp_sq'] = ambient_temp ** 2
                    input_data['module_temp_sq'] = module_temp ** 2
                    
                    with st.spinner("Making prediction..."):
                        try:
                            predictions, model_name = predictor.predict(input_data)
                            
                            st.success("✅ Prediction Complete!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Predicted Irradiance", f"{predictions[0]:.2f} W/m²")
                            with col2:
                                st.metric("Model Used", model_name)
                            with col3:
                                confidence = predictor.results[model_name]['R²'] * 100
                                st.metric("Model Confidence", f"{confidence:.1f}%")
                            
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>☀️ Solar Irradiance Predictor | Built with Streamlit & Scikit-learn</p>
            <p>Powered by ensemble ML models for accurate solar energy forecasting</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()