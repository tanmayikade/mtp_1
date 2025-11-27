#!/usr/bin/env python3
"""
Eagle Ford Formation Feature Engineering Pipeline
Converts clean master dataset to ML-ready features for training

Features Created:
1. Rolling window statistics (temporal context)
2. Gradient features (rate of change)
3. Depth-based features (spatial context)
4. Cross-curve relationships (multi-log wells)
5. Sequence features (LSTM preparation)
6. Geological indicators (formation-specific)
"""

import os
import sys
import logging
import traceback
import warnings
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
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

def memory_efficient_decorator(func):
    """Decorator for automatic memory cleanup"""
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

class EagleFordFeatureEngineer:
    """
    Comprehensive feature engineering for Eagle Ford well log ML models
    
    Supports different model types:
    - Tree-based models (RF, XGBoost): Rolling stats, ratios, geological features
    - Deep learning models (LSTM, Conv1D): Sequences, normalized features
    - Multi-log prediction: Cross-curve relationships, physics-based features
    """
    
    def __init__(self, 
                 input_file: str = "/Users/satan/projects/mtp_1/dataset/processed/master_dataset.csv",
                 output_dir: str = "/Users/satan/projects/mtp_1/dataset/processed",
                 log_level: str = "INFO",
                 run_name: Optional[str] = None):
        """
        Initialize feature engineering pipeline
        
        Args:
            input_file: Path to preprocessed master dataset
            output_dir: Directory for feature engineered outputs
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            run_name: Optional run name for output directory organization
        """
        # Setup run-based directory structure
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir / "features" / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set input file path (could be from preprocessing run)
        if isinstance(input_file, str) and not Path(input_file).is_absolute():
            # If relative path, assume it's from a preprocessing run
            self.input_file = self.base_output_dir / "preprocessed" / input_file
        else:
            self.input_file = Path(input_file)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Feature engineering configuration based on EDA insights
        self.config = {
            'rolling_windows': [5, 10, 20],  # Geological context windows (ft)
            'sequence_length': 50,  # For LSTM models (50 ft sequences)
            'sequence_step': 1,     # Sequence overlap
            'target_curves': ['GR'],  # Primary prediction targets
            'auxiliary_curves': ['RHOB', 'NPHI', 'RT', 'PE'],  # If available
            'depth_normalization': 'minmax',  # 'standard', 'minmax', 'robust'
            'outlier_detection': True,
            'geological_features': True,
            'cross_curve_features': True,
            'sequence_features': True
        }
        
        # Eagle Ford specific geological thresholds
        self.geological_thresholds = {
            'gr_shale_min': 80,      # API units - shale indicator
            'gr_sand_max': 50,       # API units - clean sand indicator  
            'rhob_shale_max': 2.4,   # g/cc - low density organics
            'nphi_shale_min': 20,    # % - high neutron clay/organics
            'rt_source_min': 10      # ohm-m - resistive source rock
        }
        
        # Statistics tracking
        self.stats = {
            'input_records': 0,
            'output_records': 0,
            'features_created': 0,
            'wells_processed': 0,
            'feature_summary': {},
            'processing_errors': []
        }
        
        self.logger.info("Eagle Ford Feature Engineer initialized")
        self.logger.info(f"Input file: {self.input_file}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def setup_logging(self, log_level: str):
        """Setup logging system"""
        
        # Setup logger
        self.logger = logging.getLogger('EagleFordFeatureEngineer')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
    def load_master_dataset(self) -> pd.DataFrame:
        """Load and validate master dataset"""
        
        self.logger.info(f"Loading master dataset from {self.input_file}")
        
        try:
            # Convert input_file to Path object if it's a string
            input_path = Path(self.input_file) if isinstance(self.input_file, str) else self.input_file
            
            if not input_path.exists():
                raise FileNotFoundError(f"Master dataset not found: {input_path}")
                
            df = pd.read_csv(input_path)
            
            self.logger.info(f"Loaded dataset: {len(df):,} records, {len(df.columns)} columns")
            
            # Validate required columns
            required_cols = ['DEPTH', 'well_api', 'GR']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Basic validation
            if len(df) == 0:
                raise ValueError("Empty dataset")
                
            unique_wells = df['well_api'].nunique()
            self.logger.info(f"Dataset contains {unique_wells} unique wells")
            
            self.stats['input_records'] = len(df)
            self.stats['wells_processed'] = unique_wells
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to load master dataset: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise
            
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'GR') -> pd.DataFrame:
        """
        Create rolling window statistical features
        
        Args:
            df: Well log dataframe (single well)
            target_col: Target curve for rolling features
            
        Returns:
            DataFrame with added rolling features
        """
        self.logger.debug(f"Creating rolling features for {target_col}")
        
        df_features = df.copy()
        features_added = 0
        
        try:
            if target_col not in df_features.columns:
                self.logger.warning(f"Target column {target_col} not found")
                return df_features
                
            for window in self.config['rolling_windows']:
                # Rolling statistics
                df_features[f'{target_col}_roll_mean_{window}'] = df_features[target_col].rolling(
                    window, center=True, min_periods=max(1, window//3)
                ).mean()
                
                df_features[f'{target_col}_roll_std_{window}'] = df_features[target_col].rolling(
                    window, center=True, min_periods=max(1, window//3)
                ).std()
                
                df_features[f'{target_col}_roll_min_{window}'] = df_features[target_col].rolling(
                    window, center=True, min_periods=max(1, window//3)
                ).min()
                
                df_features[f'{target_col}_roll_max_{window}'] = df_features[target_col].rolling(
                    window, center=True, min_periods=max(1, window//3)
                ).max()
                
                df_features[f'{target_col}_roll_range_{window}'] = (
                    df_features[f'{target_col}_roll_max_{window}'] - 
                    df_features[f'{target_col}_roll_min_{window}']
                )
                
                # Rolling percentiles
                df_features[f'{target_col}_roll_q25_{window}'] = df_features[target_col].rolling(
                    window, center=True, min_periods=max(1, window//3)
                ).quantile(0.25)
                
                df_features[f'{target_col}_roll_q75_{window}'] = df_features[target_col].rolling(
                    window, center=True, min_periods=max(1, window//3)
                ).quantile(0.75)
                
                features_added += 7
                
            self.logger.debug(f"Added {features_added} rolling features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Rolling features creation failed: {str(e)}")
            return df_features
            
    def create_gradient_features(self, df: pd.DataFrame, target_col: str = 'GR') -> pd.DataFrame:
        """
        Create gradient and rate-of-change features
        
        Args:
            df: Well log dataframe (single well)
            target_col: Target curve for gradient features
            
        Returns:
            DataFrame with added gradient features
        """
        self.logger.debug(f"Creating gradient features for {target_col}")
        
        df_features = df.copy()
        features_added = 0
        
        try:
            if target_col not in df_features.columns:
                return df_features
                
            # First derivative (gradient)
            df_features[f'{target_col}_gradient'] = np.gradient(df_features[target_col].fillna(method='ffill'))
            features_added += 1
            
            # Smoothed gradient
            smoothed = df_features[target_col].rolling(3, center=True).mean()
            df_features[f'{target_col}_gradient_smooth'] = np.gradient(smoothed.fillna(method='ffill'))
            features_added += 1
            
            # Second derivative (curvature)
            df_features[f'{target_col}_curvature'] = np.gradient(df_features[f'{target_col}_gradient'])
            features_added += 1
            
            # Rate of change over different intervals
            for lag in [1, 3, 5]:
                df_features[f'{target_col}_diff_{lag}'] = df_features[target_col].diff(lag)
                df_features[f'{target_col}_pct_change_{lag}'] = df_features[target_col].pct_change(lag)
                features_added += 2
                
            self.logger.debug(f"Added {features_added} gradient features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Gradient features creation failed: {str(e)}")
            return df_features
            
    def create_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create depth-based spatial features
        
        Args:
            df: Well log dataframe (single well)
            
        Returns:
            DataFrame with added depth features
        """
        self.logger.debug("Creating depth-based features")
        
        df_features = df.copy()
        features_added = 0
        
        try:
            # Normalized depth (0-1 within well)
            depth_min = df_features['DEPTH'].min()
            depth_max = df_features['DEPTH'].max()
            df_features['depth_normalized'] = (df_features['DEPTH'] - depth_min) / (depth_max - depth_min)
            features_added += 1
            
            # Scaled depth (thousands of feet)
            df_features['depth_scaled'] = df_features['DEPTH'] / 1000
            features_added += 1
            
            # Depth from top/bottom
            df_features['depth_from_top'] = df_features['DEPTH'] - depth_min
            df_features['depth_from_bottom'] = depth_max - df_features['DEPTH']
            features_added += 2
            
            # Sequence position
            df_features['sequence_position'] = np.arange(len(df_features))
            df_features['sequence_position_norm'] = df_features['sequence_position'] / len(df_features)
            features_added += 2
            
            # Eagle Ford specific depth zones (if known)
            # Typical Eagle Ford: 4000-12000 ft
            df_features['in_eagle_ford_window'] = (
                (df_features['DEPTH'] >= 4000) & (df_features['DEPTH'] <= 12000)
            ).astype(int)
            features_added += 1
            
            self.logger.debug(f"Added {features_added} depth features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Depth features creation failed: {str(e)}")
            return df_features
            
    def create_geological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Eagle Ford formation-specific geological features
        
        Args:
            df: Well log dataframe (single well)
            
        Returns:
            DataFrame with added geological features
        """
        self.logger.debug("Creating geological indicator features")
        
        df_features = df.copy()
        features_added = 0
        
        try:
            # GR-based lithology indicators
            if 'GR' in df_features.columns:
                thresholds = self.geological_thresholds
                
                # Shale indicator (high GR)
                df_features['is_shale'] = (df_features['GR'] >= thresholds['gr_shale_min']).astype(int)
                
                # Clean sand indicator (low GR)
                df_features['is_clean_sand'] = (df_features['GR'] <= thresholds['gr_sand_max']).astype(int)
                
                # Mixed lithology indicator
                df_features['is_mixed_lithology'] = (
                    (df_features['GR'] > thresholds['gr_sand_max']) & 
                    (df_features['GR'] < thresholds['gr_shale_min'])
                ).astype(int)
                
                # GR intensity categories
                df_features['gr_intensity'] = pd.cut(
                    df_features['GR'], 
                    bins=[0, 30, 50, 80, 150, 300], 
                    labels=['very_low', 'low', 'medium', 'high', 'very_high']
                ).astype(str)
                
                # One-hot encode categories
                intensity_dummies = pd.get_dummies(df_features['gr_intensity'], prefix='gr')
                df_features = pd.concat([df_features, intensity_dummies], axis=1)
                df_features.drop('gr_intensity', axis=1, inplace=True)
                
                features_added += 3 + len(intensity_dummies.columns)
                
            # Multi-curve geological indicators (if available)
            if 'RHOB' in df_features.columns:
                # Low density organics indicator
                df_features['is_organic_rich'] = (
                    df_features['RHOB'] <= thresholds['rhob_shale_max']
                ).astype(int)
                features_added += 1
                
            if 'NPHI' in df_features.columns:
                # High neutron clay/organics indicator
                df_features['is_clay_organic'] = (
                    df_features['NPHI'] >= thresholds['nphi_shale_min']
                ).astype(int)
                features_added += 1
                
            if 'RT' in df_features.columns:
                # Resistive source rock indicator
                df_features['is_resistive'] = (
                    df_features['RT'] >= thresholds['rt_source_min']
                ).astype(int)
                features_added += 1
                
            # Composite geological indicators
            available_curves = [col for col in ['GR', 'RHOB', 'NPHI', 'RT'] if col in df_features.columns]
            
            if len(available_curves) >= 2:
                # Source rock probability (simplified)
                source_indicators = []
                
                if 'GR' in available_curves:
                    source_indicators.append(df_features['is_shale'])
                if 'RHOB' in available_curves:
                    source_indicators.append(df_features['is_organic_rich'])
                if 'RT' in available_curves:
                    source_indicators.append(df_features['is_resistive'])
                    
                if source_indicators:
                    df_features['source_rock_probability'] = np.mean(source_indicators, axis=0)
                    features_added += 1
                    
            self.logger.debug(f"Added {features_added} geological features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Geological features creation failed: {str(e)}")
            return df_features
            
    def create_cross_curve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-curve relationship features for multi-log wells
        
        Args:
            df: Well log dataframe (single well)
            
        Returns:
            DataFrame with added cross-curve features
        """
        self.logger.debug("Creating cross-curve relationship features")
        
        df_features = df.copy()
        features_added = 0
        
        try:
            # Identify available log curves (exclude metadata columns)
            exclude_cols = ['DEPTH', 'well_api', 'well_name', 'operator', 'filename', 'sequence_position', 
                          'sequence_position_norm', 'depth_normalized', 'depth_scaled', 
                          'depth_from_top', 'depth_from_bottom', 'in_eagle_ford_window']
            
            log_curves = [col for col in df_features.columns if col not in exclude_cols and not col.startswith('gr_')]
            log_curves = [col for col in log_curves if not any(suffix in col for suffix in ['_roll_', '_gradient', '_diff_', '_pct_change', '_curvature', 'is_'])]
            
            if len(log_curves) < 2:
                self.logger.debug("Insufficient curves for cross-curve features")
                return df_features
                
            # Create ratios and differences
            for i, curve1 in enumerate(log_curves):
                for curve2 in log_curves[i+1:]:
                    if curve1 in df_features.columns and curve2 in df_features.columns:
                        
                        # Ratio features (avoid division by zero)
                        df_features[f'{curve1}_{curve2}_ratio'] = df_features[curve1] / (df_features[curve2] + 1e-8)
                        
                        # Difference features
                        df_features[f'{curve1}_{curve2}_diff'] = df_features[curve1] - df_features[curve2]
                        
                        # Product features
                        df_features[f'{curve1}_{curve2}_product'] = df_features[curve1] * df_features[curve2]
                        
                        features_added += 3
                        
            # Common petrophysical relationships (if curves available)
            if 'GR' in df_features.columns and 'RHOB' in df_features.columns:
                # Volumetric features
                df_features['gr_rhob_composite'] = df_features['GR'] / df_features['RHOB']
                features_added += 1
                
            if 'NPHI' in df_features.columns and 'RHOB' in df_features.columns:
                # Neutron-density porosity
                df_features['neutron_density_separation'] = df_features['NPHI'] - (2.65 - df_features['RHOB']) * 45
                features_added += 1
                
            if 'GR' in df_features.columns and 'RT' in df_features.columns:
                # Organic richness proxy
                df_features['organic_indicator'] = df_features['GR'] * np.log10(df_features['RT'] + 1)
                features_added += 1
                
            self.logger.debug(f"Added {features_added} cross-curve features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Cross-curve features creation failed: {str(e)}")
            return df_features
            
    def create_sequences_for_lstm(self, df: pd.DataFrame, target_col: str = 'GR') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            df: Well log dataframe (single well)
            target_col: Target column for prediction
            
        Returns:
            Tuple of (sequences, targets, depths)
        """
        self.logger.debug(f"Creating LSTM sequences for {target_col}")
        
        try:
            sequence_length = self.config['sequence_length']
            step_size = self.config['sequence_step']
            
            if len(df) < sequence_length * 2:
                self.logger.warning(f"Insufficient data for sequences: {len(df)} < {sequence_length * 2}")
                return np.array([]), np.array([]), np.array([])
                
            sequences = []
            targets = []
            depths = []
            
            # Select features for sequence input (exclude target)
            feature_cols = [col for col in df.columns if col not in [
                'DEPTH', 'well_api', 'well_name', 'operator', 'filename', target_col
            ] and df[col].notna().all()]
            
            if not feature_cols:
                self.logger.warning("No suitable features for sequences")
                return np.array([]), np.array([]), np.array([])
                
            for i in range(sequence_length, len(df) - sequence_length, step_size):
                try:
                    # Input sequence
                    seq_data = df.iloc[i-sequence_length:i][feature_cols].values
                    
                    # Target value
                    target = df.iloc[i][target_col]
                    
                    # Depth for reference
                    depth = df.iloc[i]['DEPTH']
                    
                    # Skip if any NaN values
                    try:
                        # Check for NaN in sequence data and target
                        seq_has_nan = np.isnan(seq_data).any() if seq_data.size > 0 else True
                        target_has_nan = np.isnan(target) if np.isscalar(target) else np.isnan(target).any()
                        
                        if not (seq_has_nan or target_has_nan):
                            sequences.append(seq_data)
                            targets.append(target)
                            depths.append(depth)
                    except Exception as seq_error:
                        self.logger.debug(f"Skipping sequence at index {i} due to NaN check error: {seq_error}")
                        continue
                        
                except Exception as loop_error:
                    self.logger.debug(f"Skipping sequence creation at index {i}: {loop_error}")
                    self.logger.debug(traceback.format_exc())
                    continue
                    
            sequences = np.array(sequences)
            targets = np.array(targets)
            depths = np.array(depths)
            
            self.logger.debug(f"Created {len(sequences)} sequences")
            return sequences, targets, depths
            
        except Exception as e:
            self.logger.error(f"Sequence creation failed: {str(e)}")
            return np.array([]), np.array([]), np.array([])
            
    def process_well(self, df_well: pd.DataFrame, well_api: str) -> pd.DataFrame:
        """
        Process features for a single well
        
        Args:
            df_well: Single well dataframe
            well_api: Well identifier
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.debug(f"Processing features for well {well_api}")
        
        try:
            # Start with original data
            df_features = df_well.copy()
            
            # Rolling window features
            if self.config.get('rolling_windows'):
                for target_col in self.config['target_curves']:
                    if target_col in df_features.columns:
                        df_features = self.create_rolling_features(df_features, target_col)
                        
            # Gradient features
            for target_col in self.config['target_curves']:
                if target_col in df_features.columns:
                    df_features = self.create_gradient_features(df_features, target_col)
                    
            # Depth features
            df_features = self.create_depth_features(df_features)
            
            # Geological features
            if self.config.get('geological_features'):
                df_features = self.create_geological_features(df_features)
                
            # Cross-curve features
            if self.config.get('cross_curve_features'):
                df_features = self.create_cross_curve_features(df_features)
                
            return df_features
            
        except Exception as e:
            error_msg = f"Feature processing failed for well {well_api}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
            self.stats['processing_errors'].append({
                'well_api': well_api,
                'error': error_msg
            })
            
            return df_well  # Return original if feature creation fails
            
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """
        Normalize features for ML training
        
        Args:
            df: Dataframe with features
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (normalized_dataframe, scaler_object)
        """
        self.logger.info(f"Normalizing features using {method} scaling")
        
        try:
            # Identify features to normalize (exclude metadata and categorical)
            exclude_cols = ['DEPTH', 'well_api', 'well_name', 'operator', 'filename'] + \
                          [col for col in df.columns if col.startswith('gr_') and df[col].dtype == 'uint8']
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
            df_normalized = df.copy()
            
            if feature_cols:
                df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
                self.logger.info(f"Normalized {len(feature_cols)} features")
            else:
                self.logger.warning("No features to normalize")
                
            return df_normalized, scaler
            
        except Exception as e:
            error_msg = f"Feature normalization failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return df, None
            
    def create_train_test_splits(self, feature_df: pd.DataFrame, 
                               train_ratio: float = 0.7, 
                               val_ratio: float = 0.15, 
                               test_ratio: float = 0.15) -> Dict[str, Any]:
        """
        Create well-based train/validation/test splits for ML models
        
        Args:
            feature_df: Feature-engineered dataset with all wells
            train_ratio: Training set proportion
            val_ratio: Validation set proportion  
            test_ratio: Test set proportion
            
        Returns:
            Dictionary with train/val/test well lists and statistics
        """
        self.logger.info("Creating train/validation/test splits by wells")
        
        try:
            # Get unique wells and their metadata
            wells = feature_df['well_api'].unique()
            
            # Group wells by operator for stratified splitting
            operator_wells = {}
            for well_api in wells:
                well_data = feature_df[feature_df['well_api'] == well_api]
                operator = well_data['operator'].iloc[0] if 'operator' in well_data.columns else 'Unknown'
                
                if operator not in operator_wells:
                    operator_wells[operator] = []
                operator_wells[operator].append(well_api)
                
            # Stratified split by operator
            np.random.seed(42)  # Reproducible splits
            train_wells, val_wells, test_wells = [], [], []
            
            # Global split logic - ignore operators for small datasets
            total_wells = len(wells)
            self.logger.info(f"🔍 Split configuration: train_ratio={train_ratio}, val_ratio={val_ratio}, test_ratio={test_ratio}")
            self.logger.info(f"📊 Total wells to split: {total_wells}")
            self.logger.debug(f"📋 Wells list: {wells}")
            
            if val_ratio > 0.0 and total_wells >= 3:
                self.logger.info("✅ Using global split (validation requested and sufficient wells)")
                # Respect validation ratio when explicitly configured
                np.random.shuffle(wells)
                
                n_train = max(1, int(total_wells * train_ratio))
                n_val = max(1, int(total_wells * val_ratio)) if val_ratio > 0 else 0
                n_test = total_wells - n_train - n_val
                
                if n_test < 1:
                    n_test = 1
                    n_val = max(0, total_wells - n_train - n_test)
                
                train_wells = wells[:n_train].tolist()
                val_wells = wells[n_train:n_train+n_val].tolist()
                test_wells = wells[n_train+n_val:].tolist()
                
                self.logger.info(f"✅ Global split: {n_train} train, {n_val} val, {n_test} test wells")
                self.logger.debug(f"📋 Train wells: {train_wells}")
                self.logger.debug(f"📋 Validation wells: {val_wells}")  
                self.logger.debug(f"📋 Test wells: {test_wells}")
                
            else:
                # Fallback: per-operator split for complex cases or no validation
                self.logger.info("⚠️ Using per-operator split (no validation requested or insufficient wells)")
                for operator, wells_list in operator_wells.items():
                    n_wells = len(wells_list)
                    self.logger.debug(f"Splitting {n_wells} wells for operator {operator}")
                    
                    if n_wells == 1:
                        train_wells.extend(wells_list)
                    elif n_wells == 2:
                        train_wells.append(wells_list[0])
                        test_wells.append(wells_list[1])
                    else:
                        np.random.shuffle(wells_list)
                        n_train = max(1, int(n_wells * train_ratio))
                        n_val = max(0, int(n_wells * val_ratio))
                        
                        train_wells.extend(wells_list[:n_train])
                        val_wells.extend(wells_list[n_train:n_train+n_val])
                        test_wells.extend(wells_list[n_train+n_val:])
                    
            # Create split statistics
            train_records = len(feature_df[feature_df['well_api'].isin(train_wells)])
            val_records = len(feature_df[feature_df['well_api'].isin(val_wells)])
            test_records = len(feature_df[feature_df['well_api'].isin(test_wells)])
            
            # Calculate feature completeness by split
            feature_cols = [col for col in feature_df.columns if col not in ['DEPTH', 'well_api', 'well_name', 'operator', 'filename']]
            
            train_completeness = feature_df[feature_df['well_api'].isin(train_wells)][feature_cols].notna().mean().mean() * 100
            val_completeness = feature_df[feature_df['well_api'].isin(val_wells)][feature_cols].notna().mean().mean() * 100 if val_wells else 0
            test_completeness = feature_df[feature_df['well_api'].isin(test_wells)][feature_cols].notna().mean().mean() * 100 if test_wells else 0
            
            splits = {
                'train': train_wells,
                'validation': val_wells,
                'test': test_wells,
                'statistics': {
                    'train_wells': len(train_wells),
                    'val_wells': len(val_wells),
                    'test_wells': len(test_wells),
                    'train_records': train_records,
                    'val_records': val_records,
                    'test_records': test_records,
                    'train_completeness_pct': train_completeness,
                    'val_completeness_pct': val_completeness,
                    'test_completeness_pct': test_completeness,
                    'operator_distribution': {op: len(wells) for op, wells in operator_wells.items()},
                    'split_ratios': {
                        'train_ratio_actual': train_records / len(feature_df),
                        'val_ratio_actual': val_records / len(feature_df),
                        'test_ratio_actual': test_records / len(feature_df)
                    }
                }
            }
            
            self.logger.info(f"Split created: {len(train_wells)} train, {len(val_wells)} val, {len(test_wells)} test wells")
            self.logger.info(f"Records: {train_records:,} train, {val_records:,} val, {test_records:,} test")
            self.logger.info(f"Feature completeness: Train {train_completeness:.1f}%, Val {val_completeness:.1f}%, Test {test_completeness:.1f}%")
            
            return splits
            
        except Exception as e:
            error_msg = f"Train/test split creation failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
            # Fallback: random split
            wells = list(feature_df['well_api'].unique())
            np.random.shuffle(wells)
            
            n_train = int(len(wells) * train_ratio)
            n_val = int(len(wells) * val_ratio)
            
            fallback_splits = {
                'train': wells[:n_train],
                'validation': wells[n_train:n_train+n_val],
                'test': wells[n_train+n_val:],
                'statistics': {
                    'error': 'Fallback random split used',
                    'train_wells': n_train,
                    'val_wells': n_val,
                    'test_wells': len(wells) - n_train - n_val
                }
            }
            
            self.logger.warning("Using fallback random split due to error")
            return fallback_splits
            
    def save_sequence_data(self, master_df: pd.DataFrame):
        """Save sequence data for LSTM training"""
        
        self.logger.info("Creating and saving sequence data for LSTM training")
        
        try:
            all_sequences = []
            all_targets = []
            all_metadata = []
            
            for well_api in master_df['well_api'].unique():
                df_well = master_df[master_df['well_api'] == well_api].copy()
                
                if len(df_well) < self.config['sequence_length'] * 2:
                    continue
                    
                sequences, targets, depths = self.create_sequences_for_lstm(df_well)
                
                if len(sequences) > 0:
                    all_sequences.append(sequences)
                    all_targets.append(targets)
                    
                    # Metadata for sequences
                    metadata = pd.DataFrame({
                        'sequence_idx': range(len(sequences)),
                        'well_api': well_api,
                        'depth': depths,
                        'well_name': df_well['well_name'].iloc[0],
                        'operator': df_well['operator'].iloc[0]
                    })
                    all_metadata.append(metadata)
                    
            if all_sequences:
                # Combine all sequences
                sequences_combined = np.concatenate(all_sequences, axis=0)
                targets_combined = np.concatenate(all_targets, axis=0)
                metadata_combined = pd.concat(all_metadata, ignore_index=True)
                
                # Save sequence data
                sequences_path = self.output_dir / "lstm_sequences.npz"
                np.savez_compressed(
                    sequences_path,
                    sequences=sequences_combined,
                    targets=targets_combined
                )
                
                # Save metadata
                metadata_path = self.output_dir / "lstm_sequence_metadata.csv"
                metadata_combined.to_csv(metadata_path, index=False)
                
                self.logger.info(f"Saved {len(sequences_combined)} sequences to {sequences_path}")
                
                # Update statistics
                self.stats['sequence_data'] = {
                    'total_sequences': len(sequences_combined),
                    'sequence_length': self.config['sequence_length'],
                    'feature_dimensions': sequences_combined.shape[2],
                    'wells_with_sequences': len(set(metadata_combined['well_api']))
                }
                
        except Exception as e:
            error_msg = f"Sequence data creation failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
    def run_feature_engineering(self) -> bool:
        """
        Run the complete feature engineering pipeline
        
        Returns:
            Success flag
        """
        start_time = datetime.now()
        self.logger.info("="*50)
        self.logger.info("EAGLE FORD FEATURE ENGINEERING STARTED")
        self.logger.info("="*50)
        
        try:
            # Load master dataset
            master_df = self.load_master_dataset()
            initial_memory = get_memory_usage()
            self.logger.info(f"💾 Initial memory usage: {initial_memory:.1f}MB")
            
            # Store original column info before processing
            original_cols = set(master_df.columns)
            
            # Process features well by well with CHUNKED PROCESSING
            self.logger.info("Processing features by well with memory optimization...")
            
            # Process in chunks to avoid memory accumulation
            unique_wells = master_df['well_api'].unique()
            chunk_size = 5  # Process 5 wells at a time
            feature_df_chunks = []
            
            # Use tqdm for progress tracking
            total_chunks = (len(unique_wells) + chunk_size - 1) // chunk_size
            
            for i in tqdm(range(0, len(unique_wells), chunk_size), 
                         desc="Processing well chunks", 
                         total=total_chunks,
                         unit="chunk"):
                
                chunk_wells = unique_wells[i:i + chunk_size]
                self.logger.info(f"🔄 Processing wells chunk {i//chunk_size + 1}/{total_chunks}: {len(chunk_wells)} wells")
                
                chunk_processed = []
                
                # Process individual wells in chunk with progress
                for well_api in tqdm(chunk_wells, desc=f"Chunk {i//chunk_size + 1} wells", leave=False):
                    df_well = master_df[master_df['well_api'] == well_api]
                    self.logger.debug(f"Processing well {well_api}: {len(df_well)} records")
                    
                    df_well_features = self.process_well(df_well, well_api)
                    chunk_processed.append(df_well_features)
                    
                    # Clean up individual well data immediately
                    del df_well
                    force_cleanup()
                
                # Combine chunk and clean up
                if chunk_processed:
                    chunk_df = pd.concat(chunk_processed, ignore_index=True)
                    feature_df_chunks.append(chunk_df)
                    
                    # Clean up chunk processing arrays
                    del chunk_processed
                    current_memory = get_memory_usage()
                    self.logger.info(f"💾 Memory after chunk {i//chunk_size + 1}: {current_memory:.1f}MB")
                    force_cleanup()
            
            # Clean up master dataset before final concat
            del master_df
            force_cleanup()
            
            # Final concatenation with memory monitoring
            self.logger.info("📋 Combining all processed chunks...")
            pre_concat_memory = get_memory_usage()
            feature_df = pd.concat(feature_df_chunks, ignore_index=True)
            
            # Clean up chunks immediately after concat
            del feature_df_chunks
            post_concat_memory = get_memory_usage()
            objects_freed = force_cleanup()
            final_memory = get_memory_usage()
            
            self.logger.info(f"💾 Memory during concat: {pre_concat_memory:.1f}MB → {post_concat_memory:.1f}MB → {final_memory:.1f}MB")
            self.logger.info(f"🧹 Objects freed after concat: {objects_freed}")
            
            # Feature summary using stored original columns
            new_cols = set(feature_df.columns) - original_cols
            
            self.stats['features_created'] = len(new_cols)
            self.stats['output_records'] = len(feature_df)
            
            self.logger.info(f"Created {len(new_cols)} new features")
            
            # Save feature-engineered dataset
            features_path = self.output_dir / "master_dataset_features.csv"
            feature_df.to_csv(features_path, index=False)
            self.logger.info(f"Feature dataset saved: {features_path}")
            
            # Create train/validation/test splits based on feature-engineered data
            self.logger.info("Creating train/validation/test splits...")
            
            # Get split ratios from config
            test_size = getattr(self.config, 'test_size', 0.2)
            validation_size = getattr(self.config, 'validation_size', 0.2)
            train_size = 1.0 - test_size - validation_size
            
            self.logger.debug(f"📊 Config split ratios: train={train_size}, val={validation_size}, test={test_size}")
            
            splits = self.create_train_test_splits(feature_df, 
                                                 train_ratio=train_size,
                                                 val_ratio=validation_size, 
                                                 test_ratio=test_size)
            
            # Save split information
            splits_path = self.output_dir / "train_test_splits.json"
            with open(splits_path, 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    else:
                        return obj
                        
                json.dump(convert_numpy(splits), f, indent=2)
            
            self.logger.info(f"Splits saved: {splits_path}")
            
            # Store split info in stats
            self.stats['splits'] = splits['statistics']
            
            # Create normalized versions for different model types with memory optimization
            self.logger.info("Creating normalized datasets with memory management...")
            
            # 1. Standard normalization for neural networks
            norm_memory_before = get_memory_usage()
            self.logger.info(f"💾 Memory before normalization: {norm_memory_before:.1f}MB")
            
            df_standard, scaler_standard = self.normalize_features(feature_df, 'standard')
            standard_path = self.output_dir / "master_dataset_features_normalized_standard.csv"
            df_standard.to_csv(standard_path, index=False)
            
            # Save scaler
            scaler_path = self.output_dir / "feature_scaler_standard.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_standard, f)
            
            # Clean up standard dataframe immediately
            del df_standard, scaler_standard
            standard_cleanup = force_cleanup()
            standard_memory = get_memory_usage()
            self.logger.info(f"🧹 After standard normalization cleanup: {standard_cleanup} objects freed, memory: {standard_memory:.1f}MB")
                
            # 2. MinMax normalization for some ML algorithms  
            df_minmax, scaler_minmax = self.normalize_features(feature_df, 'minmax')
            minmax_path = self.output_dir / "master_dataset_features_normalized_minmax.csv"
            df_minmax.to_csv(minmax_path, index=False)
            
            # Save scaler
            scaler_minmax_path = self.output_dir / "feature_scaler_minmax.pkl"
            with open(scaler_minmax_path, 'wb') as f:
                pickle.dump(scaler_minmax, f)
            
            # Clean up minmax dataframe immediately
            del df_minmax, scaler_minmax
            minmax_cleanup = force_cleanup()
            final_norm_memory = get_memory_usage()
            self.logger.info(f"🧹 After minmax normalization cleanup: {minmax_cleanup} objects freed, memory: {final_norm_memory:.1f}MB")
                
            self.logger.info("✅ Created normalized datasets with memory optimization")
            
            # Create sequence data for LSTM
            if self.config.get('sequence_features'):
                self.save_sequence_data(feature_df)
                
            # Feature summary report
            feature_summary = {
                'total_features': len(feature_df.columns),
                'original_features': len(original_cols),
                'engineered_features': len(new_cols),
                'feature_categories': {
                    'rolling': len([col for col in new_cols if '_roll_' in col]),
                    'gradient': len([col for col in new_cols if 'gradient' in col or 'diff_' in col or 'curvature' in col]),
                    'depth': len([col for col in new_cols if 'depth_' in col]),
                    'geological': len([col for col in new_cols if any(geo in col for geo in ['is_', 'gr_'])]),
                    'cross_curve': len([col for col in new_cols if '_ratio' in col or '_diff' in col or '_product' in col])
                },
                'wells_processed': len(feature_df['well_api'].unique()),
                'total_records': len(feature_df)
            }
            
            self.stats['feature_summary'] = feature_summary
            
            # Save feature engineering summary with run info
            summary_path = self.output_dir / "feature_engineering_summary.json"
            
            # Add run information to stats
            self.stats['run_info'] = {
                'run_name': self.run_name,
                'input_file': str(self.input_file),
                'output_dir': str(self.output_dir),
                'timestamp': datetime.now().isoformat(),
                'includes_splits': True
            }
            
            with open(summary_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
            # Processing summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("="*50)
            self.logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
            self.logger.info("="*50)
            self.logger.info(f"Duration: {duration}")
            self.logger.info(f"Features created: {feature_summary['engineered_features']}")
            self.logger.info(f"Output records: {feature_summary['total_records']:,}")
            self.logger.info(f"Feature-engineered dataset: {features_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"Feature engineering pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return False

def main():
    """Main execution with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Eagle Ford Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard feature engineering
  python feature_engineering.py
  
  # Test mode with specific run
  python feature_engineering.py --test --run-name test_features_001
  
  # Custom input from preprocessing run
  python feature_engineering.py --input-file preprocessed/run_20241122_143022/master_dataset.csv
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with sample data')
    parser.add_argument('--run-name', type=str,
                       help='Custom run name for output directory organization')
    parser.add_argument('--input-file', type=str,
                       default="/Users/satan/projects/mtp_1/dataset/processed/master_dataset.csv",
                       help='Path to preprocessed master dataset')
    parser.add_argument('--output-dir', type=str,
                       default="/Users/satan/projects/mtp_1/dataset/processed",
                       help='Base output directory for feature engineered results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Test mode configurations
    if args.test:
        if not args.run_name:
            args.run_name = f"test_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output_dir = "/Users/satan/projects/mtp_1/tests/outputs"
        
        # Only override input file if not explicitly provided
        if args.input_file == "/Users/satan/projects/mtp_1/dataset/processed/master_dataset.csv":
            args.input_file = "/Users/satan/projects/mtp_1/tests/outputs/preprocessed/master_dataset.csv"
        
        print(f"🧪 Running in TEST MODE")
        print(f"📁 Input: {args.input_file}")
        print(f"📁 Output: {args.output_dir}/features/{args.run_name}")
    
    # Initialize and run feature engineering
    feature_engineer = EagleFordFeatureEngineer(
        input_file=args.input_file,
        output_dir=args.output_dir,
        log_level=args.log_level,
        run_name=args.run_name
    )
    
    success = feature_engineer.run_feature_engineering()
    
    if success:
        print("\n✅ Feature engineering completed successfully!")
        print(f"📁 Outputs saved to: {feature_engineer.output_dir}")
        print("📄 Feature dataset: master_dataset_features.csv")
        print("🔢 Normalized datasets: *_normalized_*.csv")
        print("🧠 LSTM sequences: lstm_sequences.npz")
        print("🚀 Ready for ML model training!")
    else:
        print("\n❌ Feature engineering failed!")
        print("📋 Check logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()