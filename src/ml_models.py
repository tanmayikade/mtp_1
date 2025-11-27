#!/usr/bin/env python3
"""
Eagle Ford Formation ML Model Training Pipeline
Trains Random Forest and XGBoost models for gamma ray prediction

Based on 2025 research findings and best practices:
- Ensemble methods (RF, XGBoost) achieve 89-89.6% correlation for well logs
- Hyperparameter optimization via Random Search and Bayesian methods
- Box-Cox normalization and Z-score outlier detection optimal
- Well-based splitting prevents data leakage
"""

import os
import sys
import logging
import traceback
import warnings
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
from scipy import stats
from scipy.stats import uniform, randint
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Memory management utilities
def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def force_cleanup():
    """Force garbage collection and return objects freed"""
    collected = gc.collect()
    return collected

def memory_monitor_decorator(func):
    """Decorator for automatic memory monitoring and cleanup"""
    def wrapper(*args, **kwargs):
        initial_memory = get_memory_usage()
        try:
            result = func(*args, **kwargs)
            collected = force_cleanup()
            final_memory = get_memory_usage()
            
            if hasattr(args[0], 'logger'):
                args[0].logger.debug(f"🧹 {func.__name__}: freed {collected} objects, "
                                   f"memory: {initial_memory:.1f}MB → {final_memory:.1f}MB")
            return result
        except Exception as e:
            force_cleanup()  # Cleanup on error
            raise e
    return wrapper

def clear_sklearn_cache():
    """Clear scikit-learn internal caches"""
    try:
        from sklearn.utils._testing import clear_cache
        clear_cache()
    except:
        pass

def clear_matplotlib_cache():
    """Clear matplotlib plots and cache - 2024 best practice order"""
    try:
        plt.clf()        # Clear current figure first
        plt.cla()        # Clear current axes  
        plt.close('all') # Then close all figures - prevents memory leaks
        # Clear matplotlib memory completely
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for Colab
    except:
        pass

def memory_limit_decorator(max_memory_mb=8000):
    """Decorator to check memory usage before function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_memory = get_memory_usage()
            if current_memory > max_memory_mb:
                if hasattr(args[0], 'logger'):
                    args[0].logger.warning(f"🔥 High memory usage detected: {current_memory:.1f}MB > {max_memory_mb}MB")
                    args[0].logger.warning("⚠️ Triggering aggressive cleanup...")
                
                # Aggressive cleanup
                clear_sklearn_cache()
                clear_matplotlib_cache()
                objects_freed = force_cleanup()
                
                new_memory = get_memory_usage()
                if hasattr(args[0], 'logger'):
                    args[0].logger.info(f"🧹 Emergency cleanup: {objects_freed} objects freed")
                    args[0].logger.info(f"💾 Memory: {current_memory:.1f}MB → {new_memory:.1f}MB")
                
                if new_memory > max_memory_mb * 0.9:  # Still high after cleanup
                    if hasattr(args[0], 'logger'):
                        args[0].logger.error(f"❌ Memory still high after cleanup: {new_memory:.1f}MB")
                    raise MemoryError(f"Memory usage too high: {new_memory:.1f}MB > {max_memory_mb}MB")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class EagleFordMLTrainer:
    """
    ML Model training pipeline for Eagle Ford well log prediction
    
    Implements Random Forest and XGBoost models with:
    - 2025 research-based hyperparameter optimization
    - Box-Cox normalization and outlier detection  
    - Cross-validation and robust evaluation
    - Feature importance analysis
    - Model interpretation with SHAP
    """
    
    def __init__(self, 
                 input_dir: str = "/Users/satan/projects/mtp_1/dataset/processed/features",
                 output_dir: str = "/Users/satan/projects/mtp_1/models",
                 log_level: str = "INFO",
                 run_name: Optional[str] = None,
                 target_column: str = "GR"):
        """
        Initialize ML training pipeline
        
        Args:
            input_dir: Directory containing feature-engineered data and splits
            output_dir: Directory for trained models and results
            log_level: Logging level
            run_name: Optional run name for organization
            target_column: Target variable to predict (default: GR)
        """
        # Setup run-based directory structure
        if run_name is None:
            run_name = f"ml_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_column = target_column
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Model configurations based on 2025 research
        self.model_configs = {
            'random_forest': {
                'model_class': RandomForestRegressor,
                'param_distributions': {
                    'n_estimators': randint(100, 1000),
                    'max_depth': randint(10, 50),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                    'bootstrap': [True, False],
                    'max_samples': uniform(0.7, 0.3)  # 0.7 to 1.0
                },
                'fixed_params': {
                    'random_state': 42,
                    'n_jobs': -1,
                    'oob_score': True
                }
            },
            'xgboost': {
                'model_class': xgb.XGBRegressor,
                'param_distributions': {
                    'n_estimators': randint(100, 2000),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
                    'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
                    'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
                    'min_child_weight': randint(1, 10),
                    'gamma': uniform(0, 0.5),
                    'reg_alpha': uniform(0, 1),
                    'reg_lambda': uniform(0, 1)
                },
                'fixed_params': {
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                }
            }
        }
        
        # Grid search parameter grids (more focused than random search)
        self.grid_params = {
            'random_forest': {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5],
                'bootstrap': [True, False]
            },
            'xgboost': {
                'n_estimators': [100, 500, 1000],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
        }
        
        # Training configuration
        self.config = {
            'cv_folds': 5,
            'n_iter_search': 100,  # Random search iterations
            'search_method': 'random',  # 'random' or 'grid' or 'both'
            'test_size': 0.2,
            'validation_size': 0.15,
            'random_state': 42,
            'scoring': 'r2',  # Primary metric
            'normalization': 'box_cox',  # Based on 2025 research
            'outlier_method': 'z_score',
            'outlier_threshold': 3.0,
            'min_cv_folds': 2,  # Minimum CV folds for small datasets
            'skip_preprocessing_for_test': False  # Option to skip preprocessing for testing
        }
        
        # Results storage
        self.results = {
            'preprocessing': {},
            'models': {},
            'evaluation': {},
            'feature_importance': {},
            'run_info': {
                'run_name': self.run_name,
                'timestamp': datetime.now().isoformat(),
                'target_column': self.target_column
            }
        }
        
        self.logger.info("Eagle Ford ML Trainer initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Target column: {self.target_column}")
        self.logger.info(f"Search method: {self.config['search_method']}")
        
    def set_search_method(self, method: str):
        """
        Set hyperparameter search method
        
        Args:
            method: 'random', 'grid', or 'both'
        """
        if method not in ['random', 'grid', 'both']:
            raise ValueError("Search method must be 'random', 'grid', or 'both'")
        
        self.config['search_method'] = method
        self.logger.info(f"Search method set to: {method}")
        
        if method == 'grid':
            total_combinations_rf = np.prod([len(v) for v in self.grid_params['random_forest'].values()])
            total_combinations_xgb = np.prod([len(v) for v in self.grid_params['xgboost'].values()])
            self.logger.info(f"Grid search combinations: RF={total_combinations_rf}, XGBoost={total_combinations_xgb}")
        elif method == 'random':
            self.logger.info(f"Random search iterations: {self.config['n_iter_search']}")
        elif method == 'both':
            self.logger.info("Will run both RandomizedSearchCV and GridSearchCV for comparison")
            
    def enable_test_mode(self):
        """
        Enable test mode with relaxed preprocessing for small datasets
        """
        self.config['skip_preprocessing_for_test'] = True
        self.config['outlier_threshold'] = 5.0  # More lenient outlier removal
        self.config['normalization'] = None  # Skip Box-Cox for testing
        self.config['n_iter_search'] = 3  # Faster search
        self.config['min_cv_folds'] = 2
        self.logger.info("🧪 Test mode enabled: Relaxed preprocessing, faster search")
        self.logger.info(f"  - Outlier threshold: {self.config['outlier_threshold']}")
        self.logger.info(f"  - Normalization: {self.config['normalization']}")
        self.logger.info(f"  - Search iterations: {self.config['n_iter_search']}")
        
    def setup_logging(self, log_level: str):
        """Setup logging system"""
        
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('EagleFordMLTrainer')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = log_dir / f"ml_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized - Level: {log_level}")
        
    def load_data_and_splits(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load feature-engineered data and train/validation/test splits
        
        Returns:
            Tuple of (feature_dataframe, splits_dict)
        """
        self.logger.info("Loading feature-engineered data and splits")
        
        try:
            # Convert input_dir to Path object if it's a string
            input_path = Path(self.input_dir) if isinstance(self.input_dir, str) else self.input_dir
            
            # Find the most recent feature engineering run or use specified input
            if input_path.name != "features":
                # Assume input_dir points to a specific run
                features_file = input_path / "master_dataset_features.csv"
                splits_file = input_path / "train_test_splits.json"
            else:
                # Look for most recent run
                run_dirs = [d for d in input_path.iterdir() if d.is_dir()]
                if not run_dirs:
                    raise FileNotFoundError(f"No feature engineering runs found in {input_path}")
                
                latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                features_file = latest_run / "master_dataset_features.csv"
                splits_file = latest_run / "train_test_splits.json"
                
                self.logger.info(f"Using latest feature engineering run: {latest_run.name}")
            
            # Load features
            if not features_file.exists():
                raise FileNotFoundError(f"Features file not found: {features_file}")
            
            df_features = pd.read_csv(features_file)
            self.logger.info(f"Loaded features: {len(df_features):,} records, {len(df_features.columns)} columns")
            
            # Load splits
            if not splits_file.exists():
                raise FileNotFoundError(f"Splits file not found: {splits_file}")
            
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            self.logger.info(f"Loaded splits: {len(splits['train'])} train, {len(splits['validation'])} val, {len(splits['test'])} test wells")
            
            # Validate target column
            if self.target_column not in df_features.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in features")
            
            # Store data info
            self.results['preprocessing']['total_records'] = len(df_features)
            self.results['preprocessing']['total_features'] = len(df_features.columns)
            self.results['preprocessing']['unique_wells'] = df_features['well_api'].nunique()
            
            return df_features, splits
            
        except Exception as e:
            error_msg = f"Failed to load data and splits: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise
            
    def apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply 2025 research-based preprocessing
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        self.logger.info("Applying advanced preprocessing based on 2025 research")
        
        df_processed = df.copy()
        
        try:
            # 1. Box-Cox transformation for target variable (2025 research finding)
            # Box-Cox is preferred over simple log transform because:
            # - Automatically finds optimal power parameter λ (λ=0 ≈ log transform, λ=1 ≈ no transform)
            # - More flexible than fixed log transform
            # - Includes standardization which improves model convergence
            # - Handles skewness in gamma ray data better than simple transforms
            if self.config['normalization'] == 'box_cox' and not self.config['skip_preprocessing_for_test']:
                target_values = df_processed[self.target_column].dropna()
                
                if (target_values > 0).all():
                    transformer = PowerTransformer(method='box-cox', standardize=True)
                    target_transformed = transformer.fit_transform(target_values.values.reshape(-1, 1))
                    
                    # Store transformer for later inverse transform during prediction
                    self.target_transformer = transformer
                    self.results['preprocessing']['box_cox_lambda'] = transformer.lambdas_[0]
                    
                    # Apply transformation
                    non_null_mask = df_processed[self.target_column].notna()
                    df_processed.loc[non_null_mask, f'{self.target_column}_transformed'] = target_transformed.flatten()
                    
                    self.logger.info(f"Applied Box-Cox transformation to {self.target_column} (λ={transformer.lambdas_[0]:.4f})")
                    if abs(transformer.lambdas_[0]) < 0.1:
                        self.logger.info("Note: λ ≈ 0 indicates transformation is approximately log transform")
                    self.results['preprocessing']['box_cox_applied'] = True
                else:
                    self.logger.warning(f"Box-Cox transformation skipped - {self.target_column} contains non-positive values")
                    self.logger.info("Fallback: Using original target values without transformation")
                    df_processed[f'{self.target_column}_transformed'] = df_processed[self.target_column]
                    self.results['preprocessing']['box_cox_applied'] = False
            else:
                if self.config['skip_preprocessing_for_test']:
                    self.logger.info("⚠️ Skipping Box-Cox transformation (test mode enabled)")
                else:
                    self.logger.info("⚠️ Skipping Box-Cox transformation (normalization disabled)")
                df_processed[f'{self.target_column}_transformed'] = df_processed[self.target_column]
                self.results['preprocessing']['box_cox_applied'] = False
            
            # 2. Z-score outlier detection (2025 research finding)
            if self.config['outlier_method'] == 'z_score':
                target_col = f'{self.target_column}_transformed' if f'{self.target_column}_transformed' in df_processed.columns else self.target_column
                
                z_scores = np.abs(stats.zscore(df_processed[target_col].dropna()))
                outlier_mask = z_scores > self.config['outlier_threshold']
                
                outliers_count = outlier_mask.sum()
                self.logger.info(f"Detected {outliers_count} outliers using Z-score method (threshold: {self.config['outlier_threshold']})")
                
                if outliers_count > 0:
                    # Remove outliers
                    outlier_indices = df_processed[target_col].dropna().index[outlier_mask]
                    df_processed = df_processed.drop(outlier_indices)
                    self.logger.info(f"Removed {outliers_count} outlier records")
                
                self.results['preprocessing']['outliers_removed'] = outliers_count
            
            # 3. Feature scaling (for features, not target)
            # NOTE: For tree-based models (RF, XGBoost), feature scaling is optional since they are scale-invariant
            # However, we apply StandardScaler for consistency and potential ensemble methods
            feature_cols = [col for col in df_processed.columns 
                          if col not in ['DEPTH', 'well_api', 'well_name', 'operator', 'filename', 
                                       self.target_column, f'{self.target_column}_transformed']]
            
            if feature_cols:
                # StandardScaler is preferred over MinMaxScaler for well log data
                # - Well logs have natural physical ranges that StandardScaler preserves better
                # - StandardScaler handles outliers more robustly than MinMaxScaler
                # - For tree-based models, scaling type doesn't significantly impact performance
                scaler = StandardScaler()
                df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
                
                # Store scaler for potential future use (model serving, etc.)
                self.feature_scaler = scaler
                self.feature_columns = feature_cols
                
                self.logger.info(f"Applied StandardScaler to {len(feature_cols)} features")
                self.logger.info("Note: Tree-based models are scale-invariant, but scaling aids in feature comparison")
                self.results['preprocessing']['features_scaled'] = len(feature_cols)
                self.results['preprocessing']['scaler_type'] = 'StandardScaler'
            
            self.logger.info(f"Preprocessing complete: {len(df_processed)} records remaining")
            return df_processed
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise
            
    def create_train_val_test_sets(self, df: pd.DataFrame, splits: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test sets based on well-based splits
        
        Args:
            df: Feature dataframe
            splits: Splits dictionary
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Creating train/validation/test sets")
        
        try:
            # Filter data by splits (well-based)
            train_df = df[df['well_api'].isin(splits['train'])].copy()
            val_df = df[df['well_api'].isin(splits['validation'])].copy()
            test_df = df[df['well_api'].isin(splits['test'])].copy()
            
            self.logger.info(f"Train set: {len(train_df):,} records from {len(splits['train'])} wells")
            self.logger.info(f"Validation set: {len(val_df):,} records from {len(splits['validation'])} wells")
            self.logger.info(f"Test set: {len(test_df):,} records from {len(splits['test'])} wells")
            
            # Store split statistics
            self.results['preprocessing']['splits'] = {
                'train_records': len(train_df),
                'val_records': len(val_df),
                'test_records': len(test_df),
                'train_wells': len(splits['train']),
                'val_wells': len(splits['validation']),
                'test_wells': len(splits['test'])
            }
            
            return train_df, val_df, test_df
            
        except Exception as e:
            error_msg = f"Failed to create train/val/test sets: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise
            
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and target vector y with smart missing value handling
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (features_X, target_y)
        """
        try:
            # Use transformed target if available
            target_col = f'{self.target_column}_transformed' if f'{self.target_column}_transformed' in df.columns else self.target_column
            
            # Exclude metadata and target columns
            exclude_cols = ['DEPTH', 'well_api', 'well_name', 'operator', 'filename', 
                          self.target_column, f'{self.target_column}_transformed']
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Prepare features and target
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Smart missing value handling for ML training
            self.logger.info(f"🔍 Data completeness analysis:")
            self.logger.info(f"  - Original samples: {len(X)}")
            self.logger.info(f"  - Features: {len(feature_cols)}")
            self.logger.info(f"  - Target completeness: {y.notna().sum()}/{len(y)} ({y.notna().mean()*100:.1f}%)")
            
            # Remove rows with invalid target
            valid_target_mask = y.notna()
            X = X[valid_target_mask]
            y = y[valid_target_mask]
            
            self.logger.info(f"  - After target filter: {len(X)} samples")
            
            if len(X) == 0:
                self.logger.error("❌ No samples with valid target values!")
                return pd.DataFrame(), pd.Series(dtype=float)
            
            # Analyze feature completeness
            feature_completeness = X.notna().mean()
            complete_features = feature_completeness[feature_completeness > 0.5]  # >50% complete
            
            self.logger.info(f"  - Features with >50% data: {len(complete_features)}/{len(feature_cols)}")
            
            if len(complete_features) == 0:
                self.logger.error("❌ No features with sufficient data completeness!")
                return pd.DataFrame(), pd.Series(dtype=float)
            
            # Use only features with reasonable completeness
            X_filtered = X[complete_features.index].copy()
            
            # Apply consistent imputation strategy from preprocessing pipeline
            # Following 2025 best practices and matching preprocessing approach
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import StandardScaler as MLScaler
            
            # Count missing before imputation
            missing_before = X_filtered.isna().sum().sum()
            self.logger.info(f"  - Missing values before imputation: {missing_before}")
            
            if missing_before > 0:
                # Use optimized strategy for ML training datasets (2025 best practices)
                self.logger.info("  - Applying feature-specific imputation (optimized for large datasets)")
                
                X_imputed = X_filtered.copy()
                
                # Categorize features for targeted imputation
                rolling_features = [col for col in X_filtered.columns if '_roll_' in col]
                gradient_features = [col for col in X_filtered.columns if any(x in col for x in ['gradient', '_diff_', '_pct_change', 'curvature'])]
                ratio_features = [col for col in X_filtered.columns if any(x in col for x in ['_ratio', '_product'])]
                base_features = [col for col in X_filtered.columns if col not in rolling_features + gradient_features + ratio_features]
                
                self.logger.info(f"    • Base features: {len(base_features)}")
                self.logger.info(f"    • Rolling features: {len(rolling_features)}")
                self.logger.info(f"    • Gradient features: {len(gradient_features)}")
                self.logger.info(f"    • Ratio features: {len(ratio_features)}")
                
                # 1. Rolling features: Forward/backward fill then interpolation
                if rolling_features:
                    for col in tqdm(rolling_features, desc="🔄 Rolling features", unit="feature"):
                        if X_imputed[col].isna().any():
                            # Forward fill, then backward fill, then interpolate
                            X_imputed[col] = X_imputed[col].fillna(method='ffill').fillna(method='bfill')
                            X_imputed[col] = X_imputed[col].interpolate(method='linear', limit_direction='both')
                            X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())
                
                # 2. Gradient features: Interpolation then forward/backward fill
                if gradient_features:
                    for col in tqdm(gradient_features, desc="📈 Gradient features", unit="feature"):
                        if X_imputed[col].isna().any():
                            X_imputed[col] = X_imputed[col].interpolate(method='linear', limit_direction='both')
                            X_imputed[col] = X_imputed[col].fillna(method='ffill').fillna(method='bfill')
                            X_imputed[col] = X_imputed[col].fillna(0.0)  # Gradient defaults to 0
                
                # 3. Remaining features: Use optimized well log imputation strategies (2025 best practices)
                remaining_features = ratio_features + [col for col in base_features if X_imputed[col].isna().any()]
                if len(remaining_features) > 0:
                    remaining_missing = X_imputed[remaining_features].isna().sum().sum()
                    if remaining_missing > 0:
                        n_samples = len(X_imputed)
                        n_features = len(remaining_features)
                        
                        self.logger.info(f"    🚀 Well log optimized imputation ({n_samples:,} samples, {n_features} features, {remaining_missing:,} missing)")
                        self.logger.info(f"    📊 Using fast vectorized methods optimized for geological data")
                        
                        # Use well log optimized imputation strategy
                        try:
                            with tqdm(total=4, desc="⚡ Fast well log imputation", unit="stage") as pbar:
                                # Stage 1: Depth-based interpolation for ratio features (geological continuity)
                                ratio_cols = [col for col in remaining_features if col in ratio_features]
                                if ratio_cols:
                                    for col in ratio_cols:
                                        # Linear interpolation - optimal for well log depth series
                                        X_imputed[col] = X_imputed[col].interpolate(
                                            method='linear', 
                                            limit_direction='both'
                                        )
                                        # Fill remaining edge cases with median
                                        X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())
                                    
                                    pbar.set_postfix_str(f"Interpolated {len(ratio_cols)} ratio features")
                                    pbar.update(1)
                                else:
                                    pbar.update(1)
                                
                                # Stage 2: Base feature imputation with geological constraints
                                base_cols = [col for col in remaining_features if col in base_features]
                                if base_cols:
                                    # Vectorized median imputation for base features
                                    medians = X_imputed[base_cols].median()
                                    X_imputed[base_cols] = X_imputed[base_cols].fillna(medians)
                                    
                                    pbar.set_postfix_str(f"Median filled {len(base_cols)} base features")
                                    pbar.update(1)
                                else:
                                    pbar.update(1)
                                
                                # Stage 3: Apply geological constraints to GR-related features
                                gr_related = [col for col in remaining_features if 'GR' in col.upper() or 'GAMMA' in col.upper()]
                                if gr_related:
                                    # Clip GR values to geological range (Eagle Ford: 10-250 API)
                                    for col in gr_related:
                                        X_imputed[col] = np.clip(X_imputed[col], 10, 250)
                                    
                                    pbar.set_postfix_str(f"Applied constraints to {len(gr_related)} GR features")
                                    pbar.update(1)
                                else:
                                    pbar.update(1)
                                
                                # Stage 4: Final validation and cleanup
                                final_missing = X_imputed[remaining_features].isna().sum().sum()
                                if final_missing > 0:
                                    # Emergency fallback: forward fill then median
                                    for col in remaining_features:
                                        X_imputed[col] = X_imputed[col].fillna(method='ffill').fillna(method='bfill')
                                        X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())
                                
                                pbar.set_postfix_str(f"Validation complete: {final_missing} remaining NaN")
                                pbar.update(1)
                            
                            self.logger.info(f"    ✅ Fast well log imputation: {n_features} features completed")
                            
                        except Exception as impute_error:
                            self.logger.warning(f"    ❌ Fast imputation failed, using vectorized median: {impute_error}")
                            # Ultimate fallback: vectorized median imputation
                            medians = X_imputed[remaining_features].median()
                            X_imputed[remaining_features] = X_imputed[remaining_features].fillna(medians)
                            
                            self.logger.info(f"    ✅ Fallback median imputation: {n_features} features")
                
                X_final = X_imputed
                
                # Verify imputation success
                final_missing = X_final.isna().sum().sum()
                self.logger.info(f"  ✅ Imputation complete: {missing_before:,} → {final_missing} missing values")
                
            else:
                X_final = X_filtered
                
            self.logger.info(f"✅ Final dataset: {X_final.shape[0]} samples, {X_final.shape[1]} features")
            
            # Verify no NaN values remain
            assert not X_final.isna().any().any(), "NaN values still present after preprocessing"
            assert not y.isna().any(), "NaN values in target after preprocessing"
            
            return X_final, y
            
        except Exception as e:
            error_msg = f"Failed to prepare features and target: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise
            
    @memory_monitor_decorator
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train and optimize a model using RandomizedSearchCV, GridSearchCV, or both
        MEMORY OPTIMIZED VERSION - includes automatic cleanup and monitoring
        
        Args:
            model_name: Model name ('random_forest' or 'xgboost')
            X_train: Training features
            y_train: Training target
            X_val: Validation features  
            y_val: Validation target
            
        Returns:
            Dictionary with trained model and results
        """
        search_method = self.config['search_method']
        self.logger.info(f"Training {model_name} model with {search_method} search optimization")
        
        # Memory monitoring at start
        initial_memory = get_memory_usage()
        self.logger.info(f"💾 Memory at training start: {initial_memory:.1f}MB")
        
        try:
            config = self.model_configs[model_name]
            
            # Clear any existing caches before training
            clear_sklearn_cache()
            clear_matplotlib_cache()
            force_cleanup()
            
            # Create base model
            base_model = config['model_class'](**config['fixed_params'])
            
            search_results = {}
            best_model = None
            best_score = -np.inf
            
            # 1. RandomizedSearchCV
            if search_method in ['random', 'both']:
                # Adjust CV folds for RandomizedSearchCV
                n_samples = len(X_train)
                cv_folds = min(self.config['cv_folds'], max(self.config['min_cv_folds'], n_samples))
                
                if cv_folds != self.config['cv_folds']:
                    self.logger.warning(f"Reduced RandomizedSearchCV folds from {self.config['cv_folds']} to {cv_folds} due to small dataset (n={n_samples})")
                
                # Progress tracking for RandomizedSearchCV
                total_fits = self.config['n_iter_search'] * cv_folds
                with tqdm(total=self.config['n_iter_search'], desc="🔍 RandomizedSearchCV", unit="iter") as pbar:
                    pbar.set_postfix_str(f"CV={cv_folds}, total fits={total_fits}")
                    
                    random_search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=config['param_distributions'],
                        n_iter=self.config['n_iter_search'],
                        cv=cv_folds,
                        scoring=self.config['scoring'],
                        n_jobs=-1,
                        random_state=self.config['random_state'],
                        verbose=0  # Disable verbose for clean progress bar
                    )
                    
                    random_search.fit(X_train, y_train)
                    pbar.update(self.config['n_iter_search'])
                    pbar.set_postfix_str(f"Best score: {random_search.best_score_:.4f}")
                
                search_results['random_search'] = {
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_,
                    'n_iterations': self.config['n_iter_search']
                }
                
                if random_search.best_score_ > best_score:
                    best_model = random_search.best_estimator_
                    best_score = random_search.best_score_
                    search_results['best_method'] = 'random_search'
                
                self.logger.info(f"RandomizedSearchCV best score: {random_search.best_score_:.4f}")
                
                # Memory cleanup after RandomizedSearchCV
                del random_search  # Free the search object
                objects_freed = force_cleanup()
                memory_after_random = get_memory_usage()
                self.logger.info(f"🧹 After RandomizedSearchCV: {objects_freed} objects freed, memory: {memory_after_random:.1f}MB")
            
            # 2. GridSearchCV
            if search_method in ['grid', 'both']:
                # Adjust CV folds for GridSearchCV  
                n_samples = len(X_train)
                cv_folds = min(self.config['cv_folds'], max(self.config['min_cv_folds'], n_samples))
                
                if cv_folds != self.config['cv_folds']:
                    self.logger.warning(f"Reduced GridSearchCV folds from {self.config['cv_folds']} to {cv_folds} due to small dataset (n={n_samples})")
                
                # Calculate total combinations for progress tracking
                param_grid = self.grid_params[model_name]
                total_combinations = 1
                for param_name, param_values in param_grid.items():
                    total_combinations *= len(param_values)
                
                total_fits = total_combinations * cv_folds
                
                with tqdm(total=total_combinations, desc="🎯 GridSearchCV", unit="combo") as pbar:
                    pbar.set_postfix_str(f"CV={cv_folds}, total fits={total_fits}")
                    
                    grid_search = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        cv=cv_folds,
                        scoring=self.config['scoring'],
                        n_jobs=-1,
                        verbose=0  # Disable verbose for clean progress bar
                    )
                    
                    grid_search.fit(X_train, y_train)
                    pbar.update(total_combinations)
                    pbar.set_postfix_str(f"Best score: {grid_search.best_score_:.4f}")
                
                search_results['grid_search'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'n_combinations': len(grid_search.cv_results_['params'])
                }
                
                if grid_search.best_score_ > best_score:
                    best_model = grid_search.best_estimator_
                    best_score = grid_search.best_score_
                    search_results['best_method'] = 'grid_search'
                
                self.logger.info(f"GridSearchCV best score: {grid_search.best_score_:.4f}")
                
                # Memory cleanup after GridSearchCV
                del grid_search  # Free the search object
                objects_freed = force_cleanup()
                memory_after_grid = get_memory_usage()
                self.logger.info(f"🧹 After GridSearchCV: {objects_freed} objects freed, memory: {memory_after_grid:.1f}MB")
            
            # If both methods used, log comparison
            if search_method == 'both':
                random_score = search_results['random_search']['best_score']
                grid_score = search_results['grid_search']['best_score']
                improvement = abs(grid_score - random_score)
                
                self.logger.info(f"Search method comparison:")
                self.logger.info(f"  RandomizedSearchCV: {random_score:.4f}")
                self.logger.info(f"  GridSearchCV: {grid_score:.4f}")
                self.logger.info(f"  Best method: {search_results['best_method']}")
                self.logger.info(f"  Score improvement: {improvement:.4f}")
            
            # Validate on validation set (features already aligned)
            if len(X_val) > 0:
                val_predictions = best_model.predict(X_val)
                val_r2 = r2_score(y_val, val_predictions)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
                val_mae = mean_absolute_error(y_val, val_predictions)
            else:
                self.logger.warning("⚠️ Validation set is empty, skipping validation metrics")
                val_predictions = np.array([])
                val_r2 = val_rmse = val_mae = 0.0
            
            # Cross-validation scores on full training set
            # Adjust CV folds for small datasets
            n_samples = len(X_train)
            cv_folds = min(self.config['cv_folds'], max(self.config['min_cv_folds'], n_samples))
            
            if cv_folds != self.config['cv_folds']:
                self.logger.warning(f"Reduced CV folds from {self.config['cv_folds']} to {cv_folds} due to small dataset (n={n_samples})")
            
            # Progress tracking for final cross-validation
            with tqdm(total=cv_folds, desc="✅ Final CV validation", unit="fold") as pbar:
                cv_scores = cross_val_score(best_model, X_train, y_train, 
                                          cv=cv_folds, 
                                          scoring=self.config['scoring'])
                pbar.update(cv_folds)
                pbar.set_postfix_str(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            results = {
                'model': best_model,
                'search_results': search_results,
                'best_score': best_score,
                'search_method_used': search_method,
                'validation_metrics': {
                    'r2': val_r2,
                    'rmse': val_rmse,
                    'mae': val_mae
                },
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                },
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            self.logger.info(f"{model_name} training complete:")
            self.logger.info(f"  Best CV score: {best_score:.4f}")
            self.logger.info(f"  Validation R²: {val_r2:.4f}")
            self.logger.info(f"  Validation RMSE: {val_rmse:.4f}")
            
            return results
            
        except Exception as e:
            error_msg = f"Model training failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise
            
    def evaluate_model(self, model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive model evaluation on test set
        
        Args:
            model: Trained model
            model_name: Model name
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Evaluating {model_name} model on test set")
        
        try:
            # Progress tracking for evaluation
            with tqdm(total=4, desc="📊 Model evaluation", unit="step") as pbar:
                # Predictions
                y_pred = model.predict(X_test)
                pbar.set_postfix_str(f"Predicted {len(X_test)} samples")
                pbar.update(1)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                pbar.set_postfix_str(f"R²={r2:.4f}, RMSE={rmse:.4f}")
                pbar.update(1)
            
            # Additional geological metrics
            # Percentage of predictions within reasonable geological range (10-250 API for GR)
            if self.target_column == 'GR':
                # If using transformed target, need to inverse transform for geological validation
                if hasattr(self, 'target_transformer'):
                    y_pred_original = self.target_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_test_original = self.target_transformer.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
                else:
                    y_pred_original = y_pred
                    y_test_original = y_test.values
                
                # Geological range validation
                valid_range_mask = (y_pred_original >= 10) & (y_pred_original <= 250)
                geological_validity = valid_range_mask.mean() * 100
                pbar.set_postfix_str(f"Geological validity: {geological_validity:.1f}%")
                pbar.update(1)
                
                # Calculate metrics on original scale
                r2_original = r2_score(y_test_original, y_pred_original)
                rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                mae_original = mean_absolute_error(y_test_original, y_pred_original)
            else:
                geological_validity = None
                r2_original = r2
                rmse_original = rmse
                mae_original = mae
                pbar.set_postfix_str("No geological validation")
                pbar.update(1)
            
            # Residual analysis
            residuals = y_test.values - y_pred
            pbar.set_postfix_str("Completed all metrics")
            pbar.update(1)
            
            evaluation = {
                'test_r2': r2,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2_original': r2_original,
                'test_rmse_original': rmse_original,
                'test_mae_original': mae_original,
                'geological_validity_pct': geological_validity,
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std(),
                'test_samples': len(y_test),
                'prediction_range': {
                    'min': y_pred.min(),
                    'max': y_pred.max(),
                    'mean': y_pred.mean(),
                    'std': y_pred.std()
                },
                'actual_range': {
                    'min': y_test.min(),
                    'max': y_test.max(),
                    'mean': y_test.mean(),
                    'std': y_test.std()
                }
            }
            
            self.logger.info(f"{model_name} test evaluation:")
            self.logger.info(f"  Test R²: {r2:.4f}")
            self.logger.info(f"  Test RMSE: {rmse:.4f}")
            self.logger.info(f"  Test MAE: {mae:.4f}")
            if geological_validity is not None:
                self.logger.info(f"  Geological validity: {geological_validity:.1f}%")
            
            return evaluation
            
        except Exception as e:
            error_msg = f"Model evaluation failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise
            
    def analyze_feature_importance(self, model, model_name: str, feature_names: List[str], 
                                 X_sample: Optional[pd.DataFrame] = None) -> Dict:
        """
        Comprehensive feature importance analysis using multiple methods including SHAP
        
        Args:
            model: Trained model
            model_name: Model name
            feature_names: List of feature names
            X_sample: Sample data for SHAP analysis (optional, uses subset if not provided)
            
        Returns:
            Feature importance analysis with SHAP values and visualizations
        """
        self.logger.info(f"Analyzing feature importance for {model_name}")
        
        try:
            importance_analysis = {}
            
            # Create visualization directory
            viz_dir = self.output_dir / model_name / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Model-specific feature importance
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (RF, XGBoost)
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                importance_analysis['model_importance'] = {
                    'values': importances.tolist(),
                    'feature_names': feature_names,
                    'ranking': [feature_names[i] for i in indices],
                    'top_10': [feature_names[i] for i in indices[:10]]
                }
                
                # Plot model-specific importance
                self._plot_feature_importance(importances, feature_names, model_name, viz_dir)
                
                self.logger.info(f"Top 5 important features for {model_name}:")
                for i in range(min(5, len(indices))):
                    idx = indices[i]
                    self.logger.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")
            
            # 2. SHAP analysis for model interpretability
            if X_sample is not None and len(X_sample) > 0:
                try:
                    # Use a subset for SHAP analysis (computational efficiency)
                    sample_size = min(1000, len(X_sample))
                    X_shap = X_sample.sample(n=sample_size, random_state=42)
                    
                    # Progress tracking for SHAP analysis
                    with tqdm(total=3, desc="🔬 SHAP analysis", unit="stage") as pbar:
                        # Create SHAP explainer based on model type
                        if model_name.lower() == 'random_forest':
                            # For Random Forest, use TreeExplainer
                            explainer = shap.TreeExplainer(model)
                            pbar.set_postfix_str("TreeExplainer (RF)")
                            pbar.update(1)
                            
                            shap_values = explainer.shap_values(X_shap)
                            pbar.set_postfix_str(f"Computed values for {sample_size} samples")
                            pbar.update(1)
                            
                        elif model_name.lower() == 'xgboost':
                            # For XGBoost, use TreeExplainer
                            explainer = shap.TreeExplainer(model)
                            pbar.set_postfix_str("TreeExplainer (XGB)")
                            pbar.update(1)
                            
                            shap_values = explainer.shap_values(X_shap)
                            pbar.set_postfix_str(f"Computed values for {sample_size} samples")
                            pbar.update(1)
                            
                        else:
                            # Fallback to general explainer
                            explainer = shap.Explainer(model, X_shap.sample(100, random_state=42))
                            pbar.set_postfix_str("General explainer")
                            pbar.update(1)
                            
                            shap_values = explainer(X_shap)
                            if hasattr(shap_values, 'values'):
                                shap_values = shap_values.values
                            pbar.set_postfix_str(f"Computed values for {sample_size} samples")
                            pbar.update(1)
                    
                    # Store SHAP results
                    importance_analysis['shap_analysis'] = {
                        'mean_absolute_values': np.mean(np.abs(shap_values), axis=0).tolist(),
                        'feature_names': feature_names,
                        'sample_size': sample_size
                    }
                    
                    # SHAP visualizations
                    self._create_shap_visualizations(shap_values, X_shap, feature_names, model_name, viz_dir)
                    pbar.set_postfix_str("Visualizations created")
                    pbar.update(1)
                    
                    self.logger.info(f"SHAP analysis completed for {model_name}")
                    
                except Exception as shap_error:
                    self.logger.warning(f"SHAP analysis failed for {model_name}: {shap_error}")
                    importance_analysis['shap_analysis'] = {'error': str(shap_error)}
            else:
                self.logger.info("Skipping SHAP analysis - no sample data provided")
                    
            return importance_analysis
            
        except Exception as e:
            error_msg = f"Feature importance analysis failed for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return {}
            
    def _plot_feature_importance(self, importances: np.ndarray, feature_names: List[str], 
                               model_name: str, viz_dir: Path):
        """Plot model-specific feature importance using matplotlib and seaborn"""
        
        try:
            # Get top 20 features for visualization
            indices = np.argsort(importances)[::-1]
            top_n = min(20, len(indices))
            top_indices = indices[:top_n]
            
            # Ensure proper imports
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("whitegrid")
            
            # Create dataframe for plotting
            plot_data = pd.DataFrame({
                'feature': [feature_names[i] for i in top_indices],
                'importance': importances[top_indices]
            })
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=plot_data, y='feature', x='importance', 
                       palette='viridis', orient='h', ax=ax)
            
            ax.set_title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            importance_path = viz_dir / f"{model_name}_feature_importance.png"
            fig.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Feature importance plot saved: {importance_path}")
            
        except Exception as viz_error:
            self.logger.warning(f"Feature importance visualization failed: {viz_error}")
            
    def _create_shap_visualizations(self, shap_values: np.ndarray, X_sample: pd.DataFrame,
                                  feature_names: List[str], model_name: str, viz_dir: Path):
        """Create comprehensive SHAP visualizations using matplotlib and seaborn"""
        
        try:
            # Import matplotlib here to avoid plt variable issues
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. SHAP Summary Plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                            show=False, max_display=20)
            plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
            summary_path = viz_dir / f"{model_name}_shap_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. SHAP Bar Plot (mean importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                            plot_type="bar", show=False, max_display=20)
            plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            bar_path = viz_dir / f"{model_name}_shap_bar.png"
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Feature correlation with SHAP values (custom plot)
            # Top 10 most important features by mean absolute SHAP value
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            top_features_idx = np.argsort(mean_shap)[::-1][:10]
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for i, feat_idx in enumerate(top_features_idx):
                if i >= 10:
                    break
                    
                feature_name = feature_names[feat_idx]
                
                # Scatter plot of feature value vs SHAP value
                axes[i].scatter(X_sample.iloc[:, feat_idx], shap_values[:, feat_idx], 
                              alpha=0.6, s=20, c='blue')
                axes[i].set_xlabel(f'{feature_name}')
                axes[i].set_ylabel('SHAP value')
                axes[i].set_title(f'{feature_name}\n(rank #{i+1})', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                
            plt.suptitle(f'Top 10 Feature-SHAP Relationships - {model_name}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            correlation_path = viz_dir / f"{model_name}_shap_correlation.png"
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. SHAP importance comparison with model importance
            if hasattr(self, 'feature_importances_'):
                plt.figure(figsize=(12, 6))
                
                # Compare top 15 features
                top_15_idx = np.argsort(mean_shap)[::-1][:15]
                model_imp_subset = self.feature_importances_[top_15_idx] if hasattr(self, 'feature_importances_') else np.zeros(15)
                shap_imp_subset = mean_shap[top_15_idx]
                
                x = np.arange(len(top_15_idx))
                width = 0.35
                
                plt.bar(x - width/2, model_imp_subset, width, label='Model Importance', alpha=0.8)
                plt.bar(x + width/2, shap_imp_subset, width, label='SHAP Importance', alpha=0.8)
                
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title(f'Model vs SHAP Feature Importance - {model_name}')
                plt.xticks(x, [feature_names[i] for i in top_15_idx], rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                
                comparison_path = viz_dir / f"{model_name}_importance_comparison.png"
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            # 5. Eagle Ford Geological Context SHAP Analysis
            self._create_geological_shap_analysis(shap_values, X_sample, feature_names, model_name, viz_dir)
            
            self.logger.info(f"SHAP visualizations created in {viz_dir}")
            
        except Exception as shap_viz_error:
            self.logger.warning(f"SHAP visualization creation failed: {shap_viz_error}")
            self.logger.debug(f"SHAP error details: {str(shap_viz_error)}", exc_info=True)
            
    def _create_geological_shap_analysis(self, shap_values: np.ndarray, X_sample: pd.DataFrame, 
                                       feature_names: List[str], model_name: str, viz_dir: Path):
        """Create Eagle Ford-specific geological SHAP analysis"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. Geological Feature Grouping Analysis
            geological_groups = {
                'Rolling Statistics': [f for f in feature_names if 'roll_' in f],
                'Geological Indicators': [f for f in feature_names if any(x in f for x in ['is_', 'gr_', 'in_eagle'])],
                'Gradient Features': [f for f in feature_names if any(x in f for x in ['gradient', 'diff', 'pct_change'])],
                'Depth Features': [f for f in feature_names if 'depth' in f],
                'Cross-Curve Features': [f for f in feature_names if any(x in f for x in ['ROP', 'TVD', 'VS', 'TEMP'])],
                'Curvature Features': [f for f in feature_names if 'curvature' in f]
            }
            
            # Calculate group-wise SHAP importance
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            group_importance = {}
            
            for group_name, group_features in geological_groups.items():
                group_indices = [i for i, fname in enumerate(feature_names) if fname in group_features]
                if group_indices:
                    group_importance[group_name] = np.sum(mean_shap[group_indices])
                else:
                    group_importance[group_name] = 0.0
            
            # Plot geological group importance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Group importance pie chart
            groups = list(group_importance.keys())
            importances = list(group_importance.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
            wedges, texts, autotexts = ax1.pie(importances, labels=groups, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            ax1.set_title(f'SHAP Importance by Geological Feature Group\n{model_name}', fontweight='bold')
            
            # Top features with geological annotation
            top_15_idx = np.argsort(mean_shap)[::-1][:15]
            top_features = [feature_names[i] for i in top_15_idx]
            top_shap = mean_shap[top_15_idx]
            
            # Color code by geological group
            feature_colors = []
            for feature in top_features:
                for group, group_features in geological_groups.items():
                    if feature in group_features:
                        if 'Rolling' in group:
                            feature_colors.append('#1f77b4')  # Blue
                        elif 'Geological' in group:
                            feature_colors.append('#ff7f0e')  # Orange
                        elif 'Gradient' in group:
                            feature_colors.append('#2ca02c')  # Green
                        elif 'Depth' in group:
                            feature_colors.append('#d62728')  # Red
                        elif 'Cross-Curve' in group:
                            feature_colors.append('#9467bd')  # Purple
                        else:
                            feature_colors.append('#8c564b')  # Brown
                        break
                else:
                    feature_colors.append('#7f7f7f')  # Gray for ungrouped
            
            bars = ax2.barh(range(len(top_features)), top_shap, color=feature_colors)
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features)
            ax2.set_xlabel('Mean |SHAP Value|')
            ax2.set_title(f'Top 15 Features by SHAP Importance\n{model_name}', fontweight='bold')
            ax2.invert_yaxis()
            
            # Add legend for colors
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='#1f77b4', label='Rolling Statistics'),
                plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', label='Geological Indicators'),
                plt.Rectangle((0,0),1,1, facecolor='#2ca02c', label='Gradient Features'),
                plt.Rectangle((0,0),1,1, facecolor='#d62728', label='Depth Features'),
                plt.Rectangle((0,0),1,1, facecolor='#9467bd', label='Cross-Curve Features'),
                plt.Rectangle((0,0),1,1, facecolor='#8c564b', label='Other')
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            plt.tight_layout()
            geological_path = viz_dir / f"{model_name}_shap_geological.png"
            plt.savefig(geological_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Lithology-specific SHAP Analysis
            # Find lithology indicator features
            lithology_features = {}
            for i, fname in enumerate(feature_names):
                if 'is_clean_sand' in fname:
                    lithology_features['Clean Sand'] = i
                elif 'is_shale' in fname:
                    lithology_features['Shale'] = i
                elif 'is_mixed_lithology' in fname:
                    lithology_features['Mixed Lithology'] = i
            
            if lithology_features:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create box plots for SHAP values by lithology
                lithology_data = []
                lithology_names = []
                
                for litho_name, feat_idx in lithology_features.items():
                    lithology_data.append(shap_values[:, feat_idx])
                    lithology_names.append(litho_name)
                
                bp = ax.boxplot(lithology_data, labels=lithology_names, patch_artist=True)
                
                # Color the boxes
                colors = ['lightblue', 'lightcoral', 'lightgreen']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f'SHAP Values Distribution by Lithology Indicators\n{model_name}', 
                           fontweight='bold', fontsize=14)
                ax.set_ylabel('SHAP Value')
                ax.set_xlabel('Lithology Type')
                ax.grid(True, alpha=0.3)
                
                # Add annotations with Eagle Ford geological context
                ax.text(0.02, 0.98, 'Eagle Ford Formation Context:\n• Clean Sand: 0-75 API\n• Shaly Sand: 75-150 API\n• Shale: 150-300 API', 
                       transform=ax.transAxes, verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                lithology_path = viz_dir / f"{model_name}_shap_lithology.png"
                plt.savefig(lithology_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. SHAP Dependence Plots for Top Geological Features
            top_geological_features = []
            for group_name, group_features in geological_groups.items():
                if group_name in ['Rolling Statistics', 'Geological Indicators']:
                    group_indices = [i for i, fname in enumerate(feature_names) if fname in group_features]
                    if group_indices:
                        group_shap = mean_shap[group_indices]
                        top_in_group = np.argmax(group_shap)
                        top_geological_features.append(group_indices[top_in_group])
            
            if len(top_geological_features) >= 2:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                for i, feat_idx in enumerate(top_geological_features[:2]):
                    feature_name = feature_names[feat_idx]
                    
                    # Scatter plot with color-coded SHAP values
                    scatter = axes[i].scatter(X_sample.iloc[:, feat_idx], shap_values[:, feat_idx], 
                                           c=X_sample.iloc[:, feat_idx], cmap='viridis', alpha=0.6)
                    axes[i].set_xlabel(f'{feature_name}')
                    axes[i].set_ylabel('SHAP value')
                    axes[i].set_title(f'SHAP Dependence: {feature_name}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add colorbar
                    plt.colorbar(scatter, ax=axes[i], label='Feature Value')
                
                plt.suptitle(f'SHAP Dependence Plots - Key Geological Features\n{model_name}', 
                           fontweight='bold', fontsize=14)
                plt.tight_layout()
                
                dependence_path = viz_dir / f"{model_name}_shap_dependence.png"
                plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Geological SHAP analysis completed for {model_name}")
            
        except Exception as geo_shap_error:
            self.logger.warning(f"Geological SHAP analysis failed: {geo_shap_error}")
            self.logger.debug(f"Geological SHAP error details: {str(geo_shap_error)}", exc_info=True)
    
    def create_learning_curves(self, model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Create learning curves using 2025 sklearn best practices
        
        Args:
            model: Trained model
            model_name: Model name
            X_train: Training features
            y_train: Training target
            
        Returns:
            Learning curves results
        """
        self.logger.info(f"Creating learning curves for {model_name}")
        
        try:
            import matplotlib.pyplot as plt
            from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
            import numpy as np
            
            # Create visualization directory
            viz_dir = self.output_dir / model_name / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Modern sklearn 2025 approach with comprehensive parameters
            common_params = {
                "train_sizes": np.linspace(0.1, 1.0, 10),
                "cv": ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
                "n_jobs": min(4, self.config.get('n_jobs', 4)),
                "line_kw": {"marker": "o", "markersize": 4},
                "std_display_style": "fill_between",
                "score_name": "R² Score",
                "scoring": "r2"
            }
            
            # Create learning curve display
            fig, ax = plt.subplots(figsize=(10, 6))
            display = LearningCurveDisplay.from_estimator(
                model, X_train, y_train, ax=ax, **common_params
            )
            
            # Enhance plot for Eagle Ford context
            ax.set_title(f'Learning Curves - {model_name}\nEagle Ford Gamma Ray Prediction', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Set Size (samples)', fontsize=12)
            ax.set_ylabel('R² Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
            
            # Add geological context annotation
            ax.text(0.02, 0.98, 'Eagle Ford Formation\nGamma Ray Log Prediction', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            
            plt.tight_layout()
            
            # Save learning curves
            learning_curve_path = viz_dir / f"{model_name}_learning_curves.png"
            fig.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Learning curves saved: {learning_curve_path}")
            
            return {
                'learning_curve_path': str(learning_curve_path),
                'train_sizes': display.train_sizes_,
                'train_scores': display.train_scores_,
                'test_scores': display.test_scores_
            }
            
        except Exception as lc_error:
            self.logger.warning(f"Learning curves creation failed: {lc_error}")
            self.logger.debug(f"Learning curves error details: {str(lc_error)}", exc_info=True)
            return {}
    
    def create_prediction_visualizations(self, model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Create Eagle Ford-specific prediction visualizations including depth plots
        
        Args:
            model: Trained model
            model_name: Model name  
            X_test: Test features
            y_test: Test target
            
        Returns:
            Prediction visualization results
        """
        self.logger.info(f"Creating Eagle Ford prediction visualizations for {model_name}")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            # Create visualization directory
            viz_dir = self.output_dir / model_name / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # 1. Predicted vs Actual (Eagle Ford context)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Predicted vs Actual scatter
            ax1.scatter(y_test, y_pred, alpha=0.6, s=20, c='steelblue')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Gamma Ray (API)', fontsize=12)
            ax1.set_ylabel('Predicted Gamma Ray (API)', fontsize=12)
            ax1.set_title(f'{model_name} - Predicted vs Actual\nEagle Ford Formation', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add R² annotation
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=11, fontweight='bold')
            
            # Plot 2: Residuals plot
            residuals = y_test - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, s=20, c='orange')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Gamma Ray (API)', fontsize=12)
            ax2.set_ylabel('Residuals (API)', fontsize=12)
            ax2.set_title('Residuals Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Histogram of residuals
            ax3.hist(residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.set_xlabel('Residuals (API)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add mean and std annotation
            ax3.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
            ax3.axvline(residuals.std(), color='blue', linestyle='--', linewidth=2, label=f'Std: {residuals.std():.2f}')
            ax3.axvline(-residuals.std(), color='blue', linestyle='--', linewidth=2)
            ax3.legend()
            
            # Plot 4: Gamma Ray range analysis (Eagle Ford specific)
            gr_ranges = {
                'Clean Sand': (0, 75),
                'Shaly Sand': (75, 150), 
                'Shale': (150, 300)
            }
            
            range_accuracy = {}
            for range_name, (min_val, max_val) in gr_ranges.items():
                mask = (y_test >= min_val) & (y_test < max_val)
                if mask.sum() > 0:
                    range_r2 = r2_score(y_test[mask], y_pred[mask])
                    range_accuracy[range_name] = range_r2
                    
            if range_accuracy:
                ranges = list(range_accuracy.keys())
                accuracies = list(range_accuracy.values())
                colors = ['lightgreen', 'gold', 'lightcoral'][:len(ranges)]
                
                bars = ax4.bar(ranges, accuracies, color=colors, alpha=0.7, edgecolor='black')
                ax4.set_ylabel('R² Score', fontsize=12)
                ax4.set_title('Accuracy by Gamma Ray Range\n(Eagle Ford Lithology)', fontsize=14, fontweight='bold')
                ax4.set_ylim(0, 1)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save prediction analysis
            pred_viz_path = viz_dir / f"{model_name}_prediction_analysis.png"
            fig.savefig(pred_viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 2. Depth plot (if depth information available)
            if 'DEPTH' in X_test.columns or 'depth_normalized' in X_test.columns:
                self._create_depth_plot(model_name, X_test, y_test, y_pred, viz_dir)
            
            self.logger.info(f"Prediction visualizations saved: {pred_viz_path}")
            
            return {
                'prediction_viz_path': str(pred_viz_path),
                'r2_score': float(r2),
                'residual_stats': {
                    'mean': float(residuals.mean()),
                    'std': float(residuals.std()),
                    'rmse': float(np.sqrt(np.mean(residuals**2)))
                },
                'range_accuracy': range_accuracy
            }
            
        except Exception as pred_error:
            self.logger.warning(f"Prediction visualizations creation failed: {pred_error}")
            self.logger.debug(f"Prediction error details: {str(pred_error)}", exc_info=True)
            return {}
    
    def _create_depth_plot(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, 
                          y_pred: np.ndarray, viz_dir: Path):
        """Create depth-based gamma ray log visualization"""
        try:
            import matplotlib.pyplot as plt
            
            # Get depth information
            if 'DEPTH' in X_test.columns:
                depth = X_test['DEPTH']
            elif 'depth_normalized' in X_test.columns:
                # Approximate depth from normalized values
                depth = X_test['depth_normalized'] * 1000 + 8000  # Approximate Eagle Ford depth range
            else:
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            
            # Plot 1: Actual vs Predicted by depth
            ax1.plot(y_test, depth, 'b-', linewidth=1.5, label='Actual GR', alpha=0.8)
            ax1.plot(y_pred, depth, 'r--', linewidth=1.5, label='Predicted GR', alpha=0.8)
            ax1.set_xlabel('Gamma Ray (API)', fontsize=12)
            ax1.set_ylabel('Depth (ft)', fontsize=12)
            ax1.set_title(f'{model_name}\nGamma Ray Log Prediction', fontsize=14, fontweight='bold')
            ax1.invert_yaxis()  # Typical well log convention
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add Eagle Ford formation window if applicable
            eagle_ford_top = 8000
            eagle_ford_bottom = 12000
            ax1.axhspan(eagle_ford_top, eagle_ford_bottom, alpha=0.2, color='yellow', 
                       label='Eagle Ford Window')
            
            # Plot 2: Prediction error by depth
            error = np.abs(y_test - y_pred)
            ax2.plot(error, depth, 'g-', linewidth=1.5, alpha=0.8)
            ax2.set_xlabel('Prediction Error (API)', fontsize=12)
            ax2.set_ylabel('Depth (ft)', fontsize=12)
            ax2.set_title('Prediction Error by Depth', fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save depth plot
            depth_plot_path = viz_dir / f"{model_name}_depth_analysis.png"
            fig.savefig(depth_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Depth plot saved: {depth_plot_path}")
            
        except Exception as depth_error:
            self.logger.debug(f"Depth plot creation failed: {depth_error}")
            
    def save_model_artifacts(self, model, model_name: str, results: Dict):
        """
        Save trained model and associated artifacts
        
        Args:
            model: Trained model
            model_name: Model name
            results: Training and evaluation results
        """
        self.logger.info(f"Saving {model_name} model artifacts")
        
        try:
            # Create model directory
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = model_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved: {model_path}")
            
            # Save results
            results_path = model_dir / f"{model_name}_results.json"
            with open(results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    from sklearn.base import BaseEstimator
                    import xgboost as xgb
                    from sklearn.ensemble import RandomForestRegressor
                    
                    # Check for model objects first (most specific)
                    if isinstance(obj, (BaseEstimator, xgb.XGBRegressor, RandomForestRegressor)):
                        return f"<{obj.__class__.__name__} model object - not serializable>"
                    elif hasattr(obj, '__module__') and ('sklearn' in obj.__module__ or 'xgboost' in obj.__module__):
                        return f"<{obj.__class__.__name__} model object - not serializable>"
                    elif str(type(obj).__name__) in ['RandomForestRegressor', 'XGBRegressor', 'Pipeline', 'GridSearchCV', 'RandomizedSearchCV']:
                        return f"<{obj.__class__.__name__} model object - not serializable>"
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    else:
                        return obj
                
                json.dump(convert_numpy(results), f, indent=2)
                
            self.logger.info(f"Results saved: {results_path}")
            
        except Exception as e:
            error_msg = f"Failed to save {model_name} artifacts: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
    def run_ml_pipeline(self) -> bool:
        """
        Run the complete ML training pipeline with memory optimization
        
        Returns:
            Success flag
        """
        start_time = datetime.now()
        self.logger.info("="*50)
        self.logger.info("EAGLE FORD ML TRAINING PIPELINE STARTED")
        self.logger.info("="*50)
        
        # Memory monitoring at pipeline start
        initial_pipeline_memory = get_memory_usage()
        self.logger.info(f"💾 Pipeline start memory: {initial_pipeline_memory:.1f}MB")
        
        try:
            # 1. Load data and splits
            self.logger.info("📂 Loading data and splits...")
            load_memory_before = get_memory_usage()
            df_features, splits = self.load_data_and_splits()
            load_memory_after = get_memory_usage()
            self.logger.info(f"💾 After data loading: {load_memory_after:.1f}MB (Δ: +{load_memory_after - load_memory_before:.1f}MB)")
            
            # 2. Apply preprocessing
            self.logger.info("🔧 Applying preprocessing...")
            preprocess_memory_before = get_memory_usage()
            df_processed = self.apply_preprocessing(df_features)
            
            # Clean up original features dataframe
            del df_features
            objects_freed = force_cleanup()
            preprocess_memory_after = get_memory_usage()
            self.logger.info(f"💾 After preprocessing: {preprocess_memory_after:.1f}MB (Δ: {preprocess_memory_after - preprocess_memory_before:.1f}MB)")
            self.logger.info(f"🧹 Freed {objects_freed} objects after original data cleanup")
            
            # 3. Create train/val/test sets
            self.logger.info("✂️ Creating train/val/test sets...")
            split_memory_before = get_memory_usage()
            train_df, val_df, test_df = self.create_train_val_test_sets(df_processed, splits)
            
            # Clean up processed dataframe after splitting
            del df_processed
            objects_freed = force_cleanup()
            split_memory_after = get_memory_usage()
            self.logger.info(f"💾 After data splitting: {split_memory_after:.1f}MB (Δ: {split_memory_after - split_memory_before:.1f}MB)")
            self.logger.info(f"🧹 Freed {objects_freed} objects after processed data cleanup")
            
            # 4. Prepare features and targets
            self.logger.info("🎯 Preparing features and targets...")
            features_memory_before = get_memory_usage()
            X_train, y_train = self.prepare_features_and_target(train_df)
            X_val, y_val = self.prepare_features_and_target(val_df)
            X_test, y_test = self.prepare_features_and_target(test_df)
            
            # Clean up dataframes after feature extraction
            del train_df, val_df, test_df
            objects_freed = force_cleanup()
            features_memory_after = get_memory_usage()
            self.logger.info(f"💾 After feature preparation: {features_memory_after:.1f}MB (Δ: {features_memory_after - features_memory_before:.1f}MB)")
            self.logger.info(f"🧹 Freed {objects_freed} objects after dataframe cleanup")
            
            # 5. Ensure feature consistency across all splits
            self.logger.debug(f"🔍 Pre-alignment feature counts: Train={X_train.shape[1]}, Val={X_val.shape[1]}, Test={X_test.shape[1]}")
            
            # Find common features across all splits
            common_features = set(X_train.columns)
            if len(X_val) > 0:
                common_features = common_features.intersection(set(X_val.columns))
            if len(X_test) > 0:
                common_features = common_features.intersection(set(X_test.columns))
            
            common_features = sorted(list(common_features))
            self.logger.info(f"🔧 Aligned to {len(common_features)} common features across all splits")
            
            # Apply feature alignment to all sets
            X_train = X_train[common_features]
            if len(X_val) > 0:
                X_val = X_val[common_features]
            if len(X_test) > 0:
                X_test = X_test[common_features]
            
            self.logger.info(f"Final dataset sizes:")
            self.logger.info(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
            self.logger.info(f"  Validation: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
            self.logger.info(f"  Test: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
            
            # Store feature names
            feature_names = list(X_train.columns)
            
            # 5. Train models with memory optimization
            models_to_train = ['random_forest', 'xgboost']
            
            for i, model_name in enumerate(models_to_train):
                self.logger.info(f"\n{'='*20} Training {model_name.upper()} ({i+1}/{len(models_to_train)}) {'='*20}")
                
                # Memory check before model training
                pre_model_memory = get_memory_usage()
                self.logger.info(f"💾 Memory before {model_name} training: {pre_model_memory:.1f}MB")
                
                # Train model
                self.logger.debug(f"🔧 Training {model_name} with features: {list(X_train.columns)}")
                self.logger.debug(f"📊 Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
                if len(X_val) > 0:
                    self.logger.debug(f"📊 Validation data shape: {X_val.shape}, Target shape: {y_val.shape}")
                else:
                    self.logger.debug("⚠️ Validation set is empty")
                
                training_results = self.train_model(model_name, X_train, y_train, X_val, y_val)
                
                # Evaluate on test set (features already aligned)
                evaluation_results = self.evaluate_model(
                    training_results['model'], model_name, X_test, y_test
                )
                
                # Learning curves analysis (2025 best practices)
                learning_curves_results = self.create_learning_curves(
                    training_results['model'], model_name, X_train, y_train
                )
                
                # Memory cleanup after each model to prevent accumulation
                clear_sklearn_cache()
                clear_matplotlib_cache()
                objects_freed = force_cleanup()
                post_model_memory = get_memory_usage()
                
                self.logger.info(f"💾 Memory after {model_name}: {post_model_memory:.1f}MB (Δ: {post_model_memory - pre_model_memory:.1f}MB)")
                self.logger.info(f"🧹 Cleaned {objects_freed} objects after {model_name} training")
                
                # Eagle Ford-specific prediction visualizations
                prediction_results = self.create_prediction_visualizations(
                    training_results['model'], model_name, X_test, y_test
                )
                
                # Feature importance analysis with SHAP
                importance_results = self.analyze_feature_importance(
                    training_results['model'], model_name, feature_names, X_train
                )
                
                # Combine all results
                combined_results = {
                    'training': training_results,
                    'learning_curves': learning_curves_results,
                    'predictions': prediction_results,
                    'evaluation': evaluation_results,
                    'feature_importance': importance_results
                }
                
                # Store in main results
                self.results['models'][model_name] = combined_results
                
                # Save model artifacts
                self.save_model_artifacts(training_results['model'], model_name, combined_results)
            
            # 6. Model comparison
            self.logger.info(f"\n{'='*20} MODEL COMPARISON {'='*20}")
            
            comparison = {}
            for model_name in models_to_train:
                model_results = self.results['models'][model_name]
                comparison[model_name] = {
                    'test_r2': model_results['evaluation']['test_r2'],
                    'test_rmse': model_results['evaluation']['test_rmse'],
                    'test_mae': model_results['evaluation']['test_mae'],
                    'cv_mean': model_results['training']['cv_scores']['mean'],
                    'cv_std': model_results['training']['cv_scores']['std']
                }
                
                self.logger.info(f"{model_name.upper()}:")
                self.logger.info(f"  Test R²: {comparison[model_name]['test_r2']:.4f}")
                self.logger.info(f"  Test RMSE: {comparison[model_name]['test_rmse']:.4f}")
                self.logger.info(f"  CV Score: {comparison[model_name]['cv_mean']:.4f} ± {comparison[model_name]['cv_std']:.4f}")
            
            # Determine best model
            best_model_name = max(comparison.keys(), key=lambda x: comparison[x]['test_r2'])
            self.logger.info(f"\nBest performing model: {best_model_name.upper()}")
            
            self.results['evaluation']['best_model'] = best_model_name
            self.results['evaluation']['comparison'] = comparison
            
            # 7. Save final results
            final_results_path = self.output_dir / "ml_training_results.json"
            with open(final_results_path, 'w') as f:
                def convert_numpy(obj):
                    from sklearn.base import BaseEstimator
                    import xgboost as xgb
                    from sklearn.ensemble import RandomForestRegressor
                    
                    # Check for model objects first (most specific)
                    if isinstance(obj, (BaseEstimator, xgb.XGBRegressor, RandomForestRegressor)):
                        return f"<{obj.__class__.__name__} model object - not serializable>"
                    elif hasattr(obj, '__module__') and ('sklearn' in obj.__module__ or 'xgboost' in obj.__module__):
                        return f"<{obj.__class__.__name__} model object - not serializable>"
                    elif str(type(obj).__name__) in ['RandomForestRegressor', 'XGBRegressor', 'Pipeline', 'GridSearchCV', 'RandomizedSearchCV']:
                        return f"<{obj.__class__.__name__} model object - not serializable>"
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    else:
                        return obj
                
                json.dump(convert_numpy(self.results), f, indent=2)
            
            # 8. Processing summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("="*50)
            self.logger.info("ML TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*50)
            self.logger.info(f"Duration: {duration}")
            self.logger.info(f"Models trained: {len(models_to_train)}")
            self.logger.info(f"Best model: {best_model_name}")
            self.logger.info(f"Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            error_msg = f"ML training pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"PIPELINE TRACEBACK:\n{traceback.format_exc()}")
            return False


def main():
    """Main execution with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Eagle Ford ML Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard ML training
  python ml_models.py
  
  # Test mode with specific feature run
  python ml_models.py --test --input-dir /path/to/features/run_name
  
  # Custom target column
  python ml_models.py --target-column RHOB
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with sample data')
    parser.add_argument('--run-name', type=str,
                       help='Custom run name for output organization')
    parser.add_argument('--input-dir', type=str,
                       help='Input directory containing feature data and splits')
    parser.add_argument('--output-dir', type=str,
                       default="/Users/satan/projects/mtp_1/models",
                       help='Base output directory for trained models')
    parser.add_argument('--target-column', type=str, default='GR',
                       help='Target column to predict (default: GR)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Test mode configurations
    if args.test:
        if not args.run_name:
            args.run_name = f"test_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not args.input_dir:
            args.input_dir = "/Users/satan/projects/mtp_1/tests/outputs/features"
        
        args.output_dir = "/Users/satan/projects/mtp_1/tests/outputs/models"
        print(f"🧪 Running in TEST mode")
        print(f"📁 Input: {args.input_dir}")
        print(f"📁 Output: {args.output_dir}")
    
    # Set default input directory if not provided
    if not args.input_dir:
        args.input_dir = "/Users/satan/projects/mtp_1/dataset/processed/features"
    
    # Initialize and run ML training
    ml_trainer = EagleFordMLTrainer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        log_level=args.log_level,
        run_name=args.run_name,
        target_column=args.target_column
    )
    
    success = ml_trainer.run_ml_pipeline()
    
    if success:
        print(f"\n✅ ML training pipeline completed successfully!")
        print(f"📁 Models saved to: {ml_trainer.output_dir}")
        print(f"🎯 Best model: {ml_trainer.results['evaluation']['best_model'].upper()}")
        print(f"📊 Results: ml_training_results.json")
        print(f"🚀 Ready for model deployment!")
    else:
        print(f"\n❌ ML training pipeline failed!")
        print(f"📋 Check logs in: {ml_trainer.output_dir}/logs/")
        sys.exit(1)


if __name__ == "__main__":
    main()