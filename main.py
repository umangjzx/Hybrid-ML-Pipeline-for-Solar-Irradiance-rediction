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

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


class SolarIrradiancePredictor:
    """
    Advanced Solar Irradiance Prediction System with EDA
    
    A comprehensive machine learning framework for predicting solar irradiance
    using ensemble methods and advanced feature engineering.
    
    Parameters
    ----------
    quick_mode : bool, default=True
        If True, uses optimized settings for faster training.
        Set to False for maximum accuracy with longer training time.
    
    Attributes
    ----------
    models : dict
        Trained model instances
    results : dict
        Performance metrics for each model
    best_model_name : str
        Name of the best performing model
    """
    
    def __init__(self, quick_mode=True):
        self.quick_mode = quick_mode
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.data = None
        self.raw_data = None  # Store raw data for EDA
        self.feature_names = None
        self.best_model_name = None
        self.training_time = 0
        
        self._print_header()
    
    def _print_header(self):
        """Display system header"""
        print("\n" + "═" * 80)
        print("█▀▀ █▀█ █   ▄▀█ █▀█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ ▀█▀ █▀█ █▀█")
        print("▄█▄ █▄█ █▄▄ █▀█ █▀▄   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ ░█░ █▄█ █▀▄")
        print("═" * 80)
        print(f"Mode: {'⚡ QUICK' if self.quick_mode else '🎯 PRECISION'} | "
              f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("═" * 80 + "\n")
    
    def load_data(self, file_paths):
        """
        Load and combine solar plant data from multiple sources
        
        Parameters
        ----------
        file_paths : list of str
            Paths to CSV files containing solar plant data
            
        Returns
        -------
        pd.DataFrame
            Combined dataframe from all plants
        """
        print("📂 LOADING DATA")
        print("-" * 80)
        
        dfs = []
        for i, path in enumerate(file_paths, 1):
            try:
                df = pd.read_csv(path)
                df['plant_id'] = i
                dfs.append(df)
                print(f"  ✓ Plant {i}: {len(df):,} records | {df.shape[1]} columns")
            except FileNotFoundError:
                print(f"  ✗ Plant {i}: File not found - {path}")
            except Exception as e:
                print(f"  ✗ Plant {i}: Error - {str(e)[:50]}")
        
        if not dfs:
            raise ValueError("❌ No data loaded. Please check file paths.")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n  📊 Total records: {len(combined_df):,}")
        print(f"  📈 Total features: {combined_df.shape[1]}")
        print(f"  💾 Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Store raw data for EDA
        self.raw_data = combined_df.copy()
        
        return combined_df
    
    def perform_eda(self):
        """
        Comprehensive Exploratory Data Analysis
        
        Generates detailed statistical analysis and visualizations
        to understand data characteristics before modeling.
        """
        if self.raw_data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "═" * 80)
        print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
        print("═" * 80)
        
        df = self.raw_data.copy()
        
        # Parse datetime if exists
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='mixed', dayfirst=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. BASIC INFORMATION
        # ═══════════════════════════════════════════════════════════════════
        print("\n📊 1. DATASET OVERVIEW")
        print("-" * 80)
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Date Range: {df['DATE_TIME'].min()} to {df['DATE_TIME'].max()}" if 'DATE_TIME' in df.columns else "  No datetime column")
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. MISSING VALUES ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        print("\n🔍 2. MISSING VALUES ANALYSIS")
        print("-" * 80)
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        
        missing_cols = missing_df[missing_df['Missing_Count'] > 0]
        if len(missing_cols) > 0:
            print(missing_cols.to_string())
        else:
            print("  ✓ No missing values found!")
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. STATISTICAL SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        print("\n📈 3. STATISTICAL SUMMARY")
        print("-" * 80)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe().round(2).to_string())
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. DATA TYPES
        # ═══════════════════════════════════════════════════════════════════
        print("\n📋 4. DATA TYPES")
        print("-" * 80)
        print(df.dtypes.value_counts().to_string())
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. VISUALIZATIONS
        # ═══════════════════════════════════════════════════════════════════
        print("\n📊 5. GENERATING EDA VISUALIZATIONS...")
        print("-" * 80)
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Exploratory Data Analysis - Solar Irradiance Dataset', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # Create grid for subplots
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # --- Plot 1: Missing Values Heatmap ---
        ax1 = fig.add_subplot(gs[0, 0])
        if missing_df['Missing_Count'].sum() > 0:
            missing_matrix = df[numeric_cols].isnull().T
            sns.heatmap(missing_matrix.iloc[:, :100], cbar=True, cmap='RdYlGn_r', 
                       ax=ax1, yticklabels=True)
            ax1.set_title('Missing Values Pattern (First 100 rows)', fontweight='bold', fontsize=10)
            ax1.set_xlabel('Sample Index')
        else:
            ax1.text(0.5, 0.5, '✓ No Missing Values', 
                    ha='center', va='center', fontsize=14, color='green')
            ax1.set_title('Missing Values Check', fontweight='bold')
            ax1.axis('off')
        
        # --- Plot 2: Target Variable Distribution ---
        ax2 = fig.add_subplot(gs[0, 1])
        if 'IRRADIATION' in df.columns:
            df['IRRADIATION'].hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7, color='skyblue')
            ax2.axvline(df['IRRADIATION'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["IRRADIATION"].mean():.2f}')
            ax2.axvline(df['IRRADIATION'].median(), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {df["IRRADIATION"].median():.2f}')
            ax2.set_xlabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title('Target Variable: Irradiation Distribution', fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # --- Plot 3: Box Plot for Target Variable ---
        ax3 = fig.add_subplot(gs[0, 2])
        if 'IRRADIATION' in df.columns:
            box_data = df['IRRADIATION'].dropna()
            bp = ax3.boxplot([box_data], vert=True, patch_artist=True, 
                            labels=['Irradiation'])
            bp['boxes'][0].set_facecolor('lightblue')
            bp['medians'][0].set_color('red')
            bp['medians'][0].set_linewidth(2)
            
            # Add outlier statistics
            q1 = box_data.quantile(0.25)
            q3 = box_data.quantile(0.75)
            iqr = q3 - q1
            outliers = box_data[(box_data < q1 - 1.5*iqr) | (box_data > q3 + 1.5*iqr)]
            
            ax3.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax3.set_title(f'Outlier Detection\n{len(outliers)} outliers ({len(outliers)/len(box_data)*100:.1f}%)', 
                         fontweight='bold')
            ax3.grid(alpha=0.3, axis='y')
        
        # --- Plot 4: Temperature vs Irradiation Scatter ---
        ax4 = fig.add_subplot(gs[1, 0])
        if 'AMBIENT_TEMPERATURE' in df.columns and 'IRRADIATION' in df.columns:
            sample = df.sample(min(5000, len(df)))
            scatter = ax4.scatter(sample['AMBIENT_TEMPERATURE'], sample['IRRADIATION'], 
                                 alpha=0.4, s=10, c=sample['IRRADIATION'], cmap='viridis')
            ax4.set_xlabel('Ambient Temperature (°C)', fontweight='bold')
            ax4.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax4.set_title('Temperature vs Irradiation Relationship', fontweight='bold')
            ax4.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Irradiance')
        
        # --- Plot 5: Correlation Heatmap ---
        ax5 = fig.add_subplot(gs[1, 1:])
        numeric_cols_filtered = [col for col in numeric_cols 
                                if col not in ['PLANT_ID', 'SOURCE_KEY', 'plant_id']]
        if len(numeric_cols_filtered) > 0:
            corr_matrix = df[numeric_cols_filtered].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, ax=ax5, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax5.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=12)
        
        # --- Plot 6: Time Series Pattern (if datetime exists) ---
        ax6 = fig.add_subplot(gs[2, :])
        if 'DATE_TIME' in df.columns and 'IRRADIATION' in df.columns:
            df_sorted = df.sort_values('DATE_TIME')
            plot_size = min(5000, len(df_sorted))
            df_plot = df_sorted.tail(plot_size)
            
            ax6.plot(df_plot['DATE_TIME'], df_plot['IRRADIATION'], 
                    linewidth=0.5, alpha=0.7, color='darkorange')
            ax6.fill_between(df_plot['DATE_TIME'], df_plot['IRRADIATION'], 
                           alpha=0.3, color='orange')
            ax6.set_xlabel('Date Time', fontweight='bold')
            ax6.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax6.set_title(f'Time Series: Irradiation Pattern (Last {plot_size} records)', 
                         fontweight='bold')
            ax6.grid(alpha=0.3)
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        
        # --- Plot 7: Hourly Pattern ---
        ax7 = fig.add_subplot(gs[3, 0])
        if 'DATE_TIME' in df.columns and 'IRRADIATION' in df.columns:
            df['hour'] = df['DATE_TIME'].dt.hour
            hourly_avg = df.groupby('hour')['IRRADIATION'].agg(['mean', 'std']).reset_index()
            
            ax7.plot(hourly_avg['hour'], hourly_avg['mean'], 
                    marker='o', linewidth=2, color='darkblue', label='Mean')
            ax7.fill_between(hourly_avg['hour'], 
                           hourly_avg['mean'] - hourly_avg['std'],
                           hourly_avg['mean'] + hourly_avg['std'],
                           alpha=0.3, color='lightblue', label='±1 Std Dev')
            ax7.set_xlabel('Hour of Day', fontweight='bold')
            ax7.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
            ax7.set_title('Hourly Irradiation Pattern', fontweight='bold')
            ax7.legend()
            ax7.grid(alpha=0.3)
            ax7.set_xticks(range(0, 24, 2))
        
        # --- Plot 8: Monthly Pattern ---
        ax8 = fig.add_subplot(gs[3, 1])
        if 'DATE_TIME' in df.columns and 'IRRADIATION' in df.columns:
            df['month'] = df['DATE_TIME'].dt.month
            monthly_avg = df.groupby('month')['IRRADIATION'].mean().reset_index()
            
            colors = plt.cm.RdYlBu_r(monthly_avg['IRRADIATION'] / monthly_avg['IRRADIATION'].max())
            ax8.bar(monthly_avg['month'], monthly_avg['IRRADIATION'], 
                   color=colors, edgecolor='black', alpha=0.8)
            ax8.set_xlabel('Month', fontweight='bold')
            ax8.set_ylabel('Avg Solar Irradiance (W/m²)', fontweight='bold')
            ax8.set_title('Monthly Average Irradiation', fontweight='bold')
            ax8.grid(alpha=0.3, axis='y')
            ax8.set_xticks(range(1, 13))
        
        # --- Plot 9: Feature Distributions ---
        ax9 = fig.add_subplot(gs[3, 2])
        key_features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
        available_features = [f for f in key_features if f in df.columns]
        
        if available_features:
            for i, feat in enumerate(available_features[:3]):
                data = df[feat].dropna()
                ax9.hist(data, bins=30, alpha=0.5, label=feat, edgecolor='black')
            
            ax9.set_xlabel('Value', fontweight='bold')
            ax9.set_ylabel('Frequency', fontweight='bold')
            ax9.set_title('Key Feature Distributions', fontweight='bold')
            ax9.legend(loc='best')
            ax9.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # ═══════════════════════════════════════════════════════════════════
        # 6. KEY INSIGHTS
        # ═══════════════════════════════════════════════════════════════════
        print("\n💡 6. KEY INSIGHTS")
        print("-" * 80)
        
        if 'IRRADIATION' in df.columns:
            irr = df['IRRADIATION'].dropna()
            print(f"  Target Variable (IRRADIATION):")
            print(f"    • Range: {irr.min():.2f} - {irr.max():.2f} W/m²")
            print(f"    • Mean: {irr.mean():.2f} W/m²")
            print(f"    • Std Dev: {irr.std():.2f} W/m²")
            print(f"    • Skewness: {irr.skew():.3f}")
            print(f"    • Kurtosis: {irr.kurtosis():.3f}")
        
        if 'AMBIENT_TEMPERATURE' in df.columns:
            temp = df['AMBIENT_TEMPERATURE'].dropna()
            print(f"\n  Ambient Temperature:")
            print(f"    • Range: {temp.min():.2f} - {temp.max():.2f} °C")
            print(f"    • Mean: {temp.mean():.2f} °C")
        
        if 'MODULE_TEMPERATURE' in df.columns:
            mod_temp = df['MODULE_TEMPERATURE'].dropna()
            print(f"\n  Module Temperature:")
            print(f"    • Range: {mod_temp.min():.2f} - {mod_temp.max():.2f} °C")
            print(f"    • Mean: {mod_temp.mean():.2f} °C")
        
        # Correlation insights
        if 'IRRADIATION' in df.columns and len(numeric_cols_filtered) > 1:
            print(f"\n  Top Correlations with Target:")
            corr_with_target = df[numeric_cols_filtered].corr()['IRRADIATION'].sort_values(ascending=False)
            corr_with_target = corr_with_target[corr_with_target.index != 'IRRADIATION']
            for feat, corr_val in corr_with_target.head(5).items():
                print(f"    • {feat}: {corr_val:.3f}")
        
        print("\n" + "═" * 80)
        print("✅ EDA COMPLETE")
        print("═" * 80)
    
    def engineer_features(self, df):
        """
        Advanced feature engineering for solar data
        
        Creates temporal, cyclical, polynomial, and interaction features
        optimized for solar irradiance prediction.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw solar plant data
            
        Returns
        -------
        pd.DataFrame
            Enhanced dataframe with engineered features
        """
        print("\n⚙️  FEATURE ENGINEERING")
        print("-" * 80)
        
        df = df.copy()
        initial_features = df.shape[1]
        
        # Parse datetime
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='mixed', dayfirst=True)
            df = df.set_index('DATE_TIME')
            print("  ✓ Datetime index created")
        
        # Identify target
        target_col = 'IRRADIATION'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")
        
        # Extract temporal features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_year'] = df.index.dayofyear
            df['month'] = df.index.month
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            df['quarter'] = df.index.quarter
            df['week_of_year'] = df.index.isocalendar().week
            
            # Cyclical encoding (captures circular nature of time)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
            
            print("  ✓ Temporal features: 13 created")
        
        # Temperature features
        temp_features = 0
        if 'AMBIENT_TEMPERATURE' in df.columns:
            df['ambient_temp_sq'] = df['AMBIENT_TEMPERATURE'] ** 2
            df['ambient_temp_cube'] = df['AMBIENT_TEMPERATURE'] ** 3
            temp_features += 2
        
        if 'MODULE_TEMPERATURE' in df.columns:
            df['module_temp_sq'] = df['MODULE_TEMPERATURE'] ** 2
            df['module_temp_cube'] = df['MODULE_TEMPERATURE'] ** 3
            temp_features += 2
        
        # Temperature interactions
        if 'AMBIENT_TEMPERATURE' in df.columns and 'MODULE_TEMPERATURE' in df.columns:
            df['temp_diff'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
            df['temp_ratio'] = df['MODULE_TEMPERATURE'] / (df['AMBIENT_TEMPERATURE'] + 1e-6)
            df['temp_product'] = df['MODULE_TEMPERATURE'] * df['AMBIENT_TEMPERATURE']
            temp_features += 3
        
        if temp_features > 0:
            print(f"  ✓ Temperature features: {temp_features} created")
        
        # Lag features (time series patterns)
        lag_features = 0
        lags = [1, 2, 3] if self.quick_mode else [1, 2, 3, 6, 12, 24]
        
        for lag in lags:
            df[f'irr_lag_{lag}'] = df[target_col].shift(lag)
            lag_features += 1
            if 'AMBIENT_TEMPERATURE' in df.columns:
                df[f'temp_lag_{lag}'] = df['AMBIENT_TEMPERATURE'].shift(lag)
                lag_features += 1
        
        print(f"  ✓ Lag features: {lag_features} created")
        
        # Rolling statistics (capture trends)
        if not self.quick_mode:
            rolling_features = 0
            for window in [3, 6, 12, 24]:
                df[f'irr_mean_{window}'] = df[target_col].rolling(window).mean()
                df[f'irr_std_{window}'] = df[target_col].rolling(window).std()
                df[f'irr_min_{window}'] = df[target_col].rolling(window).min()
                df[f'irr_max_{window}'] = df[target_col].rolling(window).max()
                rolling_features += 4
            print(f"  ✓ Rolling features: {rolling_features} created")
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['PLANT_ID', 'SOURCE_KEY', 'plant_id']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        df = df[numeric_cols].copy()
        
        # Handle missing values
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        # Rename target
        df = df.rename(columns={target_col: 'target'})
        
        final_features = df.shape[1] - 1  # Exclude target
        print(f"\n  📊 Feature summary:")
        print(f"     Initial features: {initial_features}")
        print(f"     Final features: {final_features}")
        print(f"     Added: {final_features - initial_features}")
        print(f"     Rows dropped (NaN): {dropped_rows:,}")
        print(f"     Final dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
        
        self.data = df
        return df
    
    def prepare_train_test(self, test_size=0.2):
        """
        Prepare time-series aware train-test split
        
        Parameters
        ----------
        test_size : float, default=0.2
            Proportion of dataset for testing
            
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test) - scaled and ready for training
        """
        print("\n📊 PREPARING TRAIN-TEST SPLIT")
        print("-" * 80)
        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        self.feature_names = X.columns.tolist()
        
        # Time-based split (no shuffle - preserves temporal order)
        split_idx = int(len(self.data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Standardization
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  ✓ Training set: {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
        print(f"  ✓ Test set: {len(X_test):,} samples ({test_size*100:.0f}%)")
        print(f"  ✓ Features: {len(self.feature_names)}")
        print(f"  ✓ Scaling: StandardScaler applied")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def create_models(self):
        """Create ensemble of ML models"""
        if self.quick_mode:
            models = {
                'LightGBM': LGBMRegressor(
                    n_estimators=100, num_leaves=31, learning_rate=0.1,
                    n_jobs=-1, random_state=42, verbose=-1
                ),
                'XGBoost': XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    n_jobs=-1, random_state=42, verbosity=0
                ),
                'CatBoost': CatBoostRegressor(
                    iterations=100, depth=6, learning_rate=0.1,
                    verbose=0, random_state=42
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
                ),
                'Extra Trees': ExtraTreesRegressor(
                    n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                ),
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'Lasso': Lasso(alpha=1.0, random_state=42, max_iter=2000),
                'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
            }
        else:
            models = {
                'LightGBM': LGBMRegressor(
                    n_estimators=300, num_leaves=31, learning_rate=0.05,
                    n_jobs=-1, random_state=42, verbose=-1
                ),
                'XGBoost': XGBRegressor(
                    n_estimators=300, max_depth=8, learning_rate=0.05,
                    n_jobs=-1, random_state=42, verbosity=0
                ),
                'CatBoost': CatBoostRegressor(
                    iterations=300, depth=8, learning_rate=0.05,
                    verbose=0, random_state=42
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
                ),
                'Extra Trees': ExtraTreesRegressor(
                    n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42
                ),
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
        """
        Train all models in ensemble
        
        Parameters
        ----------
        X_train, X_test : array-like
            Training and test features
        y_train, y_test : array-like
            Training and test targets
        """
        print("\n🚀 TRAINING MODELS")
        print("=" * 80)
        print(f"{'Model':<20} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'MAPE':>7} | {'Time':>6}")
        print("-" * 80)
        
        models = self.create_models()
        total_start = time.time()
        
        for name, model in models.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                elapsed = time.time() - start
                
                self.evaluate_model(name, y_test, y_pred)
                self.models[name] = model
                
                metrics = self.results[name]
                print(f"{name:<20} | {metrics['RMSE']:>8.2f} | {metrics['MAE']:>8.2f} | "
                      f"{metrics['R²']:>8.4f} | {metrics['MAPE']:>6.2f}% | {elapsed:>5.1f}s")
                
            except Exception as e:
                print(f"{name:<20} | ✗ Training failed: {str(e)[:30]}")
        
        self.training_time = time.time() - total_start
        
        # Identify best model
        results_df = pd.DataFrame(self.results).T.sort_values('RMSE')
        self.best_model_name = results_df.index[0]
        
        print("=" * 80)
        print(f"✓ Trained {len(self.models)} models in {self.training_time:.1f}s")
        print(f"🏆 Best model: {self.best_model_name} "
              f"(RMSE: {results_df.loc[self.best_model_name, 'RMSE']:.2f})")
    
    def visualize_results(self, y_test, X_test):
        """
        Create comprehensive visualization dashboard
        
        Generates 8 plots showing model performance from multiple angles
        """
        if not self.results:
            print("⚠️  No results to visualize")
            return None
        
        print("\n📈 GENERATING VISUALIZATIONS")
        print("-" * 80)
        
        results_df = pd.DataFrame(self.results).T.sort_values('RMSE')
        best_model = self.models[self.best_model_name]
        y_pred = best_model.predict(X_test)
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Solar Irradiance Prediction - Comprehensive Analysis Dashboard', 
                     fontsize=16, fontweight='bold', y=0.995)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. RMSE Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
        results_df['RMSE'].sort_values().plot(kind='barh', ax=ax1, color=colors)
        ax1.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
        ax1.set_title('Model Performance: RMSE', fontweight='bold')
        ax1.grid(alpha=0.3, axis='x')
        
        # 2. R² Score Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(results_df))]
        results_df['R²'].sort_values(ascending=False).plot(kind='barh', ax=ax2, color=colors)
        ax2.set_xlabel('R² Score (Higher is Better)', fontweight='bold')
        ax2.set_title('Model Performance: R² Score', fontweight='bold')
        ax2.grid(alpha=0.3, axis='x')
        ax2.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent')
        ax2.legend()
        
        # 3. MAE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        colors = ['#2ecc71' if i == 0 else '#f39c12' for i in range(len(results_df))]
        results_df['MAE'].sort_values().plot(kind='barh', ax=ax3, color=colors)
        ax3.set_xlabel('MAE (Lower is Better)', fontweight='bold')
        ax3.set_title('Model Performance: MAE', fontweight='bold')
        ax3.grid(alpha=0.3, axis='x')
        
        # 4. Prediction vs Actual (Best Model)
        ax4 = fig.add_subplot(gs[1, 0])
        sample_size = min(2000, len(y_test))
        sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
        scatter = ax4.scatter(y_test[sample_idx], y_pred[sample_idx], 
                            alpha=0.5, s=10, c=y_test[sample_idx], cmap='viridis')
        ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax4.set_xlabel('Actual Irradiance (W/m²)', fontweight='bold')
        ax4.set_ylabel('Predicted Irradiance (W/m²)', fontweight='bold')
        ax4.set_title(f'Predictions: {self.best_model_name}', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Actual Value')
        
        # 5. Residual Plot
        ax5 = fig.add_subplot(gs[1, 1])
        residuals = y_test - y_pred
        ax5.scatter(y_pred[sample_idx], residuals[sample_idx], alpha=0.5, s=10, c='coral')
        ax5.axhline(y=0, color='red', linestyle='--', lw=2)
        ax5.set_xlabel('Predicted Irradiance (W/m²)', fontweight='bold')
        ax5.set_ylabel('Residuals (W/m²)', fontweight='bold')
        ax5.set_title('Residual Analysis', fontweight='bold')
        ax5.grid(alpha=0.3)
        
        # 6. Time Series Plot
        ax6 = fig.add_subplot(gs[1, 2])
        plot_range = min(300, len(y_test))
        time_idx = range(plot_range)
        ax6.plot(time_idx, y_test[-plot_range:], label='Actual', linewidth=2, alpha=0.8)
        ax6.plot(time_idx, y_pred[-plot_range:], label='Predicted', linewidth=2, alpha=0.8)
        ax6.fill_between(time_idx, y_test[-plot_range:], y_pred[-plot_range:], alpha=0.2)
        ax6.set_xlabel('Time Steps', fontweight='bold')
        ax6.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
        ax6.set_title('Time Series: Actual vs Predicted', fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(alpha=0.3)
        
        # 7. Error Distribution
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax7.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax7.axvline(x=residuals.mean(), color='green', linestyle='--', lw=2, 
                   label=f'Mean: {residuals.mean():.2f}')
        ax7.set_xlabel('Prediction Error (W/m²)', fontweight='bold')
        ax7.set_ylabel('Frequency', fontweight='bold')
        ax7.set_title('Error Distribution', fontweight='bold')
        ax7.legend()
        ax7.grid(alpha=0.3, axis='y')
        
        # 8. Top 5 Models Radar Comparison
        ax8 = fig.add_subplot(gs[2, 1:])
        top_5 = results_df.head(5)
        x = np.arange(len(top_5))
        width = 0.2
        
        rmse_norm = 1 - (top_5['RMSE'] / top_5['RMSE'].max())
        mae_norm = 1 - (top_5['MAE'] / top_5['MAE'].max())
        r2_norm = top_5['R²']
        mape_norm = 1 - (top_5['MAPE'] / 100)
        
        ax8.bar(x - 1.5*width, rmse_norm, width, label='RMSE (norm)', alpha=0.8, color='#3498db')
        ax8.bar(x - 0.5*width, mae_norm, width, label='MAE (norm)', alpha=0.8, color='#e74c3c')
        ax8.bar(x + 0.5*width, r2_norm, width, label='R²', alpha=0.8, color='#2ecc71')
        ax8.bar(x + 1.5*width, mape_norm, width, label='MAPE (norm)', alpha=0.8, color='#f39c12')
        
        ax8.set_xlabel('Models', fontweight='bold')
        ax8.set_ylabel('Normalized Score', fontweight='bold')
        ax8.set_title('Top 5 Models - Multi-Metric Comparison', fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(top_5.index, rotation=30, ha='right')
        ax8.legend(loc='best')
        ax8.grid(alpha=0.3, axis='y')
        ax8.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.show()
        
        print("  ✓ Dashboard generated successfully")
        
        return results_df
    
    def predict(self, input_data):
        """
        Make predictions on new data
        
        Parameters
        ----------
        input_data : dict or pd.DataFrame
            Input features for prediction
            
        Returns
        -------
        tuple
            (predictions, model_name) - Predicted values and model used
        """
        if not self.models:
            raise ValueError("❌ No trained models available. Run training first.")
        
        # Prepare input
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all features present
        for feat in self.feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        input_df = input_df[self.feature_names]
        
        # Scale and predict
        input_scaled = self.scaler.transform(input_df)
        best_model = self.models[self.best_model_name]
        predictions = best_model.predict(input_scaled)
        
        return predictions, self.best_model_name
    
    def run_complete_pipeline(self, file_paths, test_size=0.2, perform_eda=True):
        """
        Execute complete ML pipeline
        
        Parameters
        ----------
        file_paths : list of str
            Paths to data files
        test_size : float, default=0.2
            Test set proportion
        perform_eda : bool, default=True
            Whether to perform exploratory data analysis
            
        Returns
        -------
        pd.DataFrame
            Results summary for all models
        """
        pipeline_start = time.time()
        
        # Load data
        df = self.load_data(file_paths)
        
        # Perform EDA (optional)
        if perform_eda:
            self.perform_eda()
        
        # Feature engineering
        self.engineer_features(df)
        
        # Prepare train-test split
        X_train, X_test, y_train, y_test = self.prepare_train_test(test_size)
        
        # Train models
        self.train_models(X_train, X_test, y_train, y_test)
        
        # Visualize
        results_df = self.visualize_results(y_test, X_test)
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS SUMMARY")
        print("=" * 80)
        print(results_df.round(4))
        
        print("\n" + "=" * 80)
        print("🏆 CHAMPION MODEL")
        print("=" * 80)
        best_metrics = results_df.loc[self.best_model_name]
        print(f"  Model: {self.best_model_name}")
        print(f"  RMSE:  {best_metrics['RMSE']:.2f} W/m²")
        print(f"  MAE:   {best_metrics['MAE']:.2f} W/m²")
        print(f"  R²:    {best_metrics['R²']:.4f}")
        print(f"  MAPE:  {best_metrics['MAPE']:.2f}%")
        
        # Performance summary
        total_time = time.time() - pipeline_start
        print("\n" + "=" * 80)
        print("⚡ PERFORMANCE METRICS")
        print("=" * 80)
        print(f"  ⏱️  Total execution time: {total_time:.2f}s")
        print(f"  🤖 Models trained: {len(self.models)}")
        print(f"  📊 Data points processed: {len(self.data):,}")
        print(f"  🎯 Features engineered: {len(self.feature_names)}")
        print(f"  💾 Model size: {len(self.models)} ensemble members")
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE - READY FOR PRODUCTION")
        print("=" * 80 + "\n")
        
        return results_df
    
    def get_model_summary(self):
        """Generate a summary report of the trained system"""
        if not self.results:
            print("❌ No models trained yet.")
            return None
        
        results_df = pd.DataFrame(self.results).T.sort_values('RMSE')
        
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "MODEL PERFORMANCE SUMMARY" + " " * 33 + "║")
        print("╚" + "═" * 78 + "╝\n")
        
        print(f"{'Rank':<6} {'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'MAPE':<10}")
        print("-" * 80)
        
        for i, (name, row) in enumerate(results_df.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            print(f"{medal:<6} {name:<20} {row['RMSE']:<10.2f} {row['MAE']:<10.2f} "
                  f"{row['R²']:<10.4f} {row['MAPE']:<9.2f}%")
        
        return results_df
    
    def save_model(self, filepath='best_solar_model.pkl'):
        """
        Save the best model to disk
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        import pickle
        import os
        
        if not self.models:
            print("❌ No models to save.")
            return
        
        model_data = {
            'model': self.models[self.best_model_name],
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✅ Best model saved to: {filepath}")
        print(f"   Model: {self.best_model_name}")
        print(f"   Size: {os.path.getsize(filepath) / 1024:.2f} KB")


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def demo_prediction_interface(predictor):
    """Interactive prediction demonstration"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "PREDICTION INTERFACE" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    if predictor.data is None or len(predictor.models) == 0:
        print("❌ Model not trained. Please run the pipeline first.")
        return
    
    # Get sample from test data
    sample = predictor.data.iloc[-1].drop('target').to_dict()
    
    print("📋 Sample Input Features:")
    print("-" * 80)
    
    # Display key features
    key_features = ['hour', 'month', 'day_of_year', 'AMBIENT_TEMPERATURE', 
                   'MODULE_TEMPERATURE', 'temp_diff', 'is_daytime']
    
    for feat in key_features:
        if feat in sample:
            value = sample[feat]
            if isinstance(value, float):
                print(f"  • {feat:<25}: {value:>10.2f}")
            else:
                print(f"  • {feat:<25}: {value:>10}")
    
    print("\n🔮 Making Prediction...")
    print("-" * 80)
    
    try:
        predictions, model_name = predictor.predict(sample)
        
        print(f"\n✅ PREDICTION RESULT")
        print("=" * 80)
        print(f"  Predicted Solar Irradiance: {predictions[0]:.2f} W/m²")
        print(f"  Model Used: {model_name}")
        print(f"  Confidence: {predictor.results[model_name]['R²']*100:.1f}%")
        print("=" * 80)
        
        # Show prediction range
        actual = predictor.data.iloc[-1]['target']
        error = abs(predictions[0] - actual)
        print(f"\n📊 Validation (on last sample):")
        print(f"  Actual value: {actual:.2f} W/m²")
        print(f"  Prediction error: {error:.2f} W/m² ({(error/actual)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")


def main():
    """Main execution function"""
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    
    # Define file paths (UPDATE THESE WITH YOUR DATA PATHS)
    file_paths = [
        r"C:\Users\UMANG JAISWAL N\Downloads\archive (2)\Plant_1_Weather_Sensor_Data.csv",
        r"C:\Users\UMANG JAISWAL N\Downloads\archive (2)\Plant_2_Weather_Sensor_Data.csv"
    ]
    
    # Alternative: Use relative paths if data is in the same directory
    # file_paths = [
    #     "Plant_1_Weather_Sensor_Data.csv",
    #     "Plant_2_Weather_Sensor_Data.csv"
    # ]
    
    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZE & RUN PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    
    # Create predictor instance
    # Set quick_mode=True for faster training (recommended for demos)
    # Set quick_mode=False for maximum accuracy (takes longer)
    predictor = SolarIrradiancePredictor(quick_mode=True)
    
    try:
        # Run complete pipeline with EDA
        results = predictor.run_complete_pipeline(
            file_paths=file_paths,
            test_size=0.2,  # 20% data for testing
            perform_eda=True  # Set to False to skip EDA
        )
        
        # Show detailed model summary
        predictor.get_model_summary()
        
        # Demonstrate prediction interface
        demo_prediction_interface(predictor)
        
        # Optional: Save best model
        # predictor.save_model('solar_model.pkl')
        
    except FileNotFoundError as e:
        print("\n❌ ERROR: Data files not found!")
        print("=" * 80)
        print("Please update the file_paths variable with correct paths to your CSV files.")
        print("\nExpected files:")
        for path in file_paths:
            print(f"  • {path}")
        print("\nCurrent working directory:", os.getcwd())
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    main()