#!/usr/bin/env python3
"""
Eagle Ford Formation Well Log Preprocessing Pipeline
Converts raw LAS files to clean, standardized master dataset ready for feature engineering

Based on comprehensive EDA findings:
- 30 LAS files with 98.6% average completeness
- 459,877 total records across all wells  
- Need 1.0 ft resampling, geological constraints 10-250 API
- Handle operator naming inconsistencies and missing values
"""

import os
import sys
import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import lasio
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class EagleFordPreprocessor:
    """
    Comprehensive preprocessing pipeline for Eagle Ford Formation well logs
    
    Pipeline Stages:
    1. LAS File Parsing & Standardization
    2. Quality Control & Outlier Detection  
    3. Depth Resampling & Interpolation
    4. Missing Value Imputation
    5. Geological Constraint Application
    6. Master Dataset Generation
    """
    
    def __init__(self, 
                 input_dir: str = "/Users/satan/projects/mtp_1/dataset/raw",
                 output_dir: str = "/Users/satan/projects/mtp_1/dataset/processed",
                 log_level: str = "INFO",
                 run_name: Optional[str] = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            input_dir: Directory containing raw LAS files
            output_dir: Base directory for processed outputs  
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            run_name: Optional run name for organizing outputs
        """
        self.input_dir = Path(input_dir)
        
        # Create run-based output directory structure
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.output_dir = Path(output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # EDA-informed configuration
        self.config = {
            'target_step_size': 1.0,  # EDA: 58% of files use 1.0 ft
            'null_values': [-999.25, -9999.0, -999.0, np.nan],  # EDA: Common null patterns
            'gr_constraints': {'min': 10.0, 'max': 250.0},  # EDA: Geological range
            'completeness_threshold': 95.0,  # EDA: 77% wells >99% complete
            'imputation_threshold': 20.0,  # EDA: Strategy for missing values
            'outlier_mad_threshold': 5.0,  # Statistical outlier detection
        }
        
        # Operator standardization from EDA
        self.operator_mapping = {
            'Chesapeake Energy': 'Chesapeake',
            'Chesapeake': 'Chesapeake', 
            'Anadarko': 'Anadarko',
            'Anadarko Petroleum Corporation': 'Anadarko',
            'Anadarko Petroleum Company': 'Anadarko',
            'Anadarko Petroeum Corporation': 'Anadarko',  # Typo found in data
            'Premier Directional Drilling': 'Anadarko',  # Service company for Anadarko
            'Sanchez OG': 'Sanchez',
            'Sanchez Oil & Gas': 'Sanchez',
            'Sanchez O&G Corp': 'Sanchez',
            'NEWFIELD EXPLORATION COMPANY': 'Newfield',
            'CHOYA OPERATING, LLC': 'Choya',
            'Choya Operating, LLC': 'Choya',
            'CHESAPEAKE OPERATING, INC.': 'Chesapeake'
        }
        
        # Curve name standardization from EDA
        self.curve_mapping = {
            'DEPT': 'DEPTH', 'DEPTH': 'DEPTH',
            'GR': 'GR', 'GR_1': 'GR', 'GR_RM': 'GR', 'GRGC': 'GR', 'GR_API': 'GR',
            'DEN': 'RHOB', 'RHOB': 'RHOB', 'ZDNC': 'RHOB', 'HDEN': 'RHOB',
            'NPHI': 'NPHI', 'CNCF': 'NPHI', 'NPRL': 'NPHI', 'NPRS': 'NPHI', 'NPRD': 'NPHI',
            'RT': 'RT', 'RES': 'RT', 'M1R6': 'RT', 'M1R1': 'RT', 'M1R2': 'RT', 'M1R3': 'RT',
            'PE': 'PE', 'PDPE': 'PE',
            'CAL': 'CALI', 'CALI': 'CALI',
            'SP': 'SP', 'SPCG': 'SP'
        }
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_records': 0,
            'wells_metadata': [],
            'processing_errors': [],
            'quality_flags': []
        }
        
        self.logger.info("Eagle Ford Preprocessor initialized")
        self.logger.info(f"Run name: {self.run_name}")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def setup_logging(self, log_level: str):
        """Setup comprehensive logging system"""
        
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('EagleFordPreprocessor')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        
    def safe_get_las_value(self, las_obj: Any, section: str, param: str) -> Any:
        """Safely extract values from LAS file sections"""
        try:
            if hasattr(las_obj, section):
                section_obj = getattr(las_obj, section)
                if hasattr(section_obj, param):
                    return getattr(section_obj, param).value
        except Exception as e:
            self.logger.debug(f"Could not extract {section}.{param}: {e}")
        return None
        
    def parse_las_file(self, filepath: Path) -> Tuple[Optional[Dict], Optional[pd.DataFrame], bool]:
        """
        Parse LAS file with comprehensive error handling
        
        Args:
            filepath: Path to LAS file
            
        Returns:
            Tuple of (metadata, dataframe, success_flag)
        """
        filename = filepath.name
        self.logger.info(f"Parsing {filename}")
        
        try:
            # Attempt parsing with different strategies
            las = None
            parse_errors = []
            
            # Strategy 1: Standard parsing
            try:
                las = lasio.read(filepath, ignore_header_errors=True)
                self.logger.debug(f"{filename}: Standard parsing successful")
            except Exception as e:
                parse_errors.append(f"Standard parsing: {str(e)}")
                
                # Strategy 2: Relaxed parsing
                try:
                    las = lasio.read(filepath, ignore_header_errors=True, null_policy='strict')
                    self.logger.debug(f"{filename}: Relaxed parsing successful")
                except Exception as e2:
                    parse_errors.append(f"Relaxed parsing: {str(e2)}")
                    
            if las is None:
                error_msg = f"All parsing strategies failed for {filename}: {'; '.join(parse_errors)}"
                self.logger.error(error_msg)
                self.stats['processing_errors'].append({
                    'file': filename,
                    'stage': 'parsing',
                    'error': error_msg
                })
                return None, None, False
            
            # Extract metadata using safe getter
            metadata = {
                'filename': filename,
                'api': self.safe_get_las_value(las, 'well', 'API'),
                'well_name': self.safe_get_las_value(las, 'well', 'WELL'),
                'company': self.safe_get_las_value(las, 'well', 'COMP'),
                'field': self.safe_get_las_value(las, 'well', 'FLD'),
                'county': self.safe_get_las_value(las, 'well', 'CNTY'),
                'state': self.safe_get_las_value(las, 'well', 'STAT'),
                'date': self.safe_get_las_value(las, 'well', 'DATE'),
                'start_depth': self.safe_get_las_value(las, 'well', 'STRT'),
                'stop_depth': self.safe_get_las_value(las, 'well', 'STOP'),
                'step': self.safe_get_las_value(las, 'well', 'STEP'),
                'null_value': self.safe_get_las_value(las, 'well', 'NULL'),
                'service_company': self.safe_get_las_value(las, 'well', 'SRVC'),
                'uwi': self.safe_get_las_value(las, 'well', 'UWI'),
            }
            
            # Standardize operator name
            raw_company = metadata.get('company', 'Unknown')
            metadata['company_standardized'] = self.operator_mapping.get(raw_company, raw_company)
            
            # Extract curve data
            try:
                df = las.df()
                if df.empty:
                    error_msg = f"No curve data in {filename}"
                    self.logger.warning(error_msg)
                    return metadata, None, False
                    
                # Reset index to make DEPTH a column
                df = df.reset_index()
                
                # Get original curve info
                original_curves = list(df.columns)
                metadata['original_curves'] = original_curves
                
                # Standardize column names with duplicate handling
                new_column_mapping = {}
                standardized_count = {}
                
                for col in df.columns:
                    if col in self.curve_mapping:
                        standard_name = self.curve_mapping[col]
                        
                        # Handle duplicates by adding suffix
                        if standard_name in standardized_count:
                            standardized_count[standard_name] += 1
                            final_name = f"{standard_name}_{standardized_count[standard_name]}"
                        else:
                            standardized_count[standard_name] = 0
                            final_name = standard_name
                            
                        new_column_mapping[col] = final_name
                    else:
                        # Keep original name if not in mapping
                        new_column_mapping[col] = col
                
                self.logger.debug(f"{filename}: Column mapping: {new_column_mapping}")
                df = df.rename(columns=new_column_mapping)
                standardized_curves = list(df.columns)
                metadata['standardized_curves'] = standardized_curves
                metadata['column_mapping'] = new_column_mapping
                
                # Basic data validation
                if len(df) == 0:
                    error_msg = f"Empty dataframe for {filename}"
                    self.logger.warning(error_msg)
                    return metadata, None, False
                    
                metadata['total_records'] = len(df)
                metadata['depth_range'] = f"{df.iloc[0, 0]:.1f} - {df.iloc[-1, 0]:.1f}"
                
                # Calculate actual step size
                if len(df) > 1:
                    depth_col = df.columns[0]  # First column should be depth
                    actual_step = df.iloc[1, 0] - df.iloc[0, 0]
                    metadata['actual_step'] = actual_step
                else:
                    metadata['actual_step'] = metadata.get('step', 1.0)
                
                self.logger.info(f"{filename}: Successfully parsed - {len(df)} records, {len(standardized_curves)} curves")
                
                return metadata, df, True
                
            except Exception as e:
                error_msg = f"Error extracting curve data from {filename}: {str(e)}"
                self.logger.error(error_msg)
                self.logger.debug(traceback.format_exc())
                self.stats['processing_errors'].append({
                    'file': filename,
                    'stage': 'curve_extraction',
                    'error': error_msg
                })
                return metadata, None, False
                
        except Exception as e:
            error_msg = f"Critical error parsing {filename}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            self.stats['processing_errors'].append({
                'file': filename,
                'stage': 'critical_parsing',
                'error': error_msg
            })
            return None, None, False
            
    def quality_control(self, df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Comprehensive quality control based on EDA findings
        
        Args:
            df: Well log dataframe
            metadata: Well metadata
            
        Returns:
            Tuple of (cleaned_dataframe, quality_report)
        """
        filename = metadata['filename']
        self.logger.debug(f"Quality control for {filename}")
        
        qc_report = {
            'filename': filename,
            'original_records': len(df),
            'flags': [],
            'statistics': {}
        }
        
        df_clean = df.copy()
        
        # Debug initial state
        self.logger.debug(f"QC START {filename}: shape={df_clean.shape}, columns={list(df_clean.columns)}")
        self.logger.debug(f"QC START {filename}: dtypes={df_clean.dtypes.to_dict()}")
        self.logger.debug(f"QC START {filename}: index_unique={df_clean.index.is_unique}")
        
        try:
            # Identify depth column (should be first column)
            depth_col = df_clean.columns[0]
            self.logger.debug(f"QC {filename}: depth_col='{depth_col}'")
            
            # 1. Check depth consistency
            depth_diff = df_clean[depth_col].diff()
            reversals = (depth_diff < 0).sum()
            if reversals > 0:
                qc_report['flags'].append(f"Depth reversals detected: {reversals}")
                self.logger.warning(f"{filename}: {reversals} depth reversals detected")
            
            # 2. Handle null values systematically
            null_values = self.config['null_values']
            self.logger.debug(f"QC {filename}: null_values to check={null_values}")
            
            for col in df_clean.columns:
                if col == depth_col:
                    continue
                    
                # Count various null representations
                total_nulls = 0
                self.logger.debug(f"QC {filename}: checking nulls in column '{col}', unique_values={len(df_clean[col].unique())}")
                
                for null_val in null_values:
                    try:
                        if pd.isna(null_val):
                            # Handle NaN comparisons - already handled by standard nulls
                            self.logger.debug(f"QC {filename}: skipping NaN null_val for {col}")
                            continue
                        else:
                            # Handle other null values
                            self.logger.debug(f"QC {filename}: checking {col} == {null_val}")
                            null_mask = (df_clean[col] == null_val)
                        
                        null_count = null_mask.sum()
                        if null_count > 0:
                            df_clean.loc[null_mask, col] = np.nan
                            total_nulls += null_count
                    except (TypeError, ValueError):
                        # Skip problematic null values
                        continue
                
                # Add standard null detection
                standard_nulls = df_clean[col].isna().sum() - total_nulls
                total_nulls += standard_nulls
                
                null_pct = (total_nulls / len(df_clean)) * 100
                qc_report['statistics'][f'{col}_null_pct'] = null_pct
                
                if null_pct > 10:
                    qc_report['flags'].append(f"{col}: {null_pct:.1f}% missing")
                    
            # 3. GR-specific quality checks (primary curve of interest)
            if 'GR' in df_clean.columns:
                gr_series = df_clean['GR'].copy()
                gr_valid = gr_series.dropna()
                
                if len(gr_valid) > 0:
                    # Statistical analysis
                    qc_report['statistics']['gr_min'] = float(gr_valid.min())
                    qc_report['statistics']['gr_max'] = float(gr_valid.max())
                    qc_report['statistics']['gr_mean'] = float(gr_valid.mean())
                    qc_report['statistics']['gr_std'] = float(gr_valid.std())
                    qc_report['statistics']['gr_completeness'] = (len(gr_valid) / len(df_clean)) * 100
                    
                    # Geological range check
                    gr_min_constraint = self.config['gr_constraints']['min']
                    gr_max_constraint = self.config['gr_constraints']['max']
                    
                    below_min = (gr_valid < gr_min_constraint).sum()
                    above_max = (gr_valid > gr_max_constraint).sum()
                    
                    if below_min > 0:
                        qc_report['flags'].append(f"GR below {gr_min_constraint}: {below_min} values")
                    if above_max > 0:
                        qc_report['flags'].append(f"GR above {gr_max_constraint}: {above_max} values")
                    
                    # Statistical outlier detection (MAD method)
                    median_gr = gr_valid.median()
                    mad_gr = (gr_valid - median_gr).abs().median()
                    outlier_threshold = median_gr + self.config['outlier_mad_threshold'] * mad_gr
                    outliers = (gr_valid > outlier_threshold).sum()
                    
                    qc_report['statistics']['gr_outliers_mad'] = outliers
                    
                    if outliers > len(gr_valid) * 0.01:  # More than 1% outliers
                        qc_report['flags'].append(f"Statistical outliers: {outliers} values ({outliers/len(gr_valid)*100:.1f}%)")
                        
                else:
                    qc_report['flags'].append("No valid GR data")
                    
            # 4. Record quality assessment
            qc_report['final_records'] = len(df_clean)
            qc_report['records_removed'] = qc_report['original_records'] - qc_report['final_records']
            
            # Overall quality score
            missing_penalty = sum([qc_report['statistics'].get(f'{col}_null_pct', 0) 
                                 for col in df_clean.columns if col != depth_col])
            quality_score = max(0, 100 - missing_penalty / len(df_clean.columns))
            qc_report['quality_score'] = quality_score
            
            self.logger.debug(f"{filename}: QC complete - Quality score: {quality_score:.1f}")
            
        except Exception as e:
            error_msg = f"Quality control failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"QC TRACEBACK for {filename}:\n{traceback.format_exc()}")
            
            # Debug data state
            self.logger.debug(f"QC DEBUG {filename}: df shape={df_clean.shape}, columns={list(df_clean.columns)}")
            self.logger.debug(f"QC DEBUG {filename}: df dtypes={df_clean.dtypes.to_dict()}")
            
            qc_report['flags'].append(f"QC Error: {str(e)}")
            
        return df_clean, qc_report
        
    def resample_to_common_depth(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Resample well log to common depth grid
        
        Args:
            df: Well log dataframe
            metadata: Well metadata
            
        Returns:
            Resampled dataframe
        """
        filename = metadata['filename']
        target_step = self.config['target_step_size']
        
        # Debug initial resampling state
        self.logger.debug(f"RESAMPLE START {filename}: shape={df.shape}, columns={list(df.columns)}")
        self.logger.debug(f"RESAMPLE START {filename}: index_unique={df.index.is_unique}")
        
        try:
            depth_col = df.columns[0]
            self.logger.debug(f"RESAMPLE {filename}: depth_col='{depth_col}'")
            
            # Define depth range
            depth_min = df[depth_col].min()
            depth_max = df[depth_col].max()
            
            # Create target depth array
            n_points = int((depth_max - depth_min) / target_step) + 1
            target_depths = np.linspace(depth_min, depth_max, n_points)
            
            # Initialize resampled dataframe
            resampled_df = pd.DataFrame({depth_col: target_depths})
            
            self.logger.debug(f"{filename}: Resampling from {len(df)} to {len(resampled_df)} records")
            
            # Interpolate each curve
            for col in df.columns:
                if col == depth_col:
                    continue
                    
                # Get valid data for interpolation
                valid_mask = df[col].notna()
                valid_data = df[valid_mask].copy()  # Make copy to avoid chained assignment
                
                # Sort by depth and remove duplicates, keeping first occurrence
                valid_data = valid_data.sort_values(depth_col).drop_duplicates(subset=[depth_col], keep='first')
                
                if len(valid_data) < 2:
                    # Not enough data for interpolation
                    resampled_df[col] = np.nan
                    self.logger.warning(f"{filename}: Insufficient data for {col} interpolation")
                    continue
                    
                try:
                    # Linear interpolation
                    resampled_df[col] = np.interp(
                        target_depths,
                        valid_data[depth_col].values,
                        valid_data[col].values,
                        left=np.nan,
                        right=np.nan
                    )
                    
                except Exception as e:
                    self.logger.warning(f"{filename}: Interpolation failed for {col}: {str(e)}")
                    resampled_df[col] = np.nan
                    
            self.logger.debug(f"{filename}: Resampling complete")
            return resampled_df
            
        except Exception as e:
            error_msg = f"Resampling failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"RESAMPLING TRACEBACK for {filename}:\n{traceback.format_exc()}")
            
            # Debug data state
            self.logger.debug(f"RESAMPLE DEBUG {filename}: input df shape={df.shape}, columns={list(df.columns)}")
            self.logger.debug(f"RESAMPLE DEBUG {filename}: depth_col='{depth_col}'")
            self.logger.debug(f"RESAMPLE DEBUG {filename}: df index unique={df.index.is_unique}")
            if not df.index.is_unique:
                self.logger.debug(f"RESAMPLE DEBUG {filename}: duplicate indices={df.index[df.index.duplicated()].tolist()[:10]}")
            
            return df  # Return original if resampling fails
            
    def impute_missing_values(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Strategic missing value imputation based on EDA findings
        
        Args:
            df: Well log dataframe  
            metadata: Well metadata
            
        Returns:
            Dataframe with imputed values
        """
        filename = metadata['filename']
        self.logger.debug(f"Imputing missing values for {filename}")
        
        df_imputed = df.copy()
        depth_col = df_imputed.columns[0]
        
        # Debug initial imputation state
        self.logger.debug(f"IMPUTE START {filename}: shape={df_imputed.shape}, columns={list(df_imputed.columns)}")
        self.logger.debug(f"IMPUTE START {filename}: index_unique={df_imputed.index.is_unique}")
        
        try:
            for col in df_imputed.columns:
                self.logger.debug(f"IMPUTE {filename}: processing column '{col}'")
                if col == depth_col:
                    continue
                    
                missing_mask = df_imputed[col].isna()
                missing_count = missing_mask.sum()
                
                # Handle potential Series result (shouldn't happen with unique columns now)
                if isinstance(missing_count, pd.Series):
                    missing_count = missing_count.iloc[0]
                    
                missing_pct = missing_count / len(df_imputed) * 100
                self.logger.debug(f"IMPUTE {filename}: column '{col}' has {missing_pct:.2f}% missing")
                
                if missing_pct == 0:
                    continue  # No missing values
                    
                elif missing_pct < 5:
                    # Small gaps: Linear interpolation
                    df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
                    self.logger.debug(f"{filename}: {col} - Linear interpolation ({missing_pct:.1f}% missing)")
                    
                elif missing_pct < self.config['imputation_threshold']:
                    # Medium gaps: KNN imputation using other curves
                    available_curves = [c for c in df_imputed.columns 
                                      if c != depth_col and c != col and 
                                      df_imputed[c].notna().sum() > len(df_imputed) * 0.5]
                    
                    if len(available_curves) >= 1:
                        try:
                            # Prepare data for KNN imputation
                            impute_cols = [col] + available_curves
                            impute_data = df_imputed[impute_cols].copy()
                            
                            # Only use rows with some non-null values
                            valid_rows = impute_data.notna().any(axis=1)
                            valid_count = valid_rows.sum()
                            
                            if valid_count > 10:
                                # Scale data
                                scaler = StandardScaler()
                                valid_data = impute_data[valid_rows]
                                scaled_data = scaler.fit_transform(valid_data.fillna(valid_data.median()))
                                
                                # KNN imputation
                                n_neighbors = min(5, valid_count//2)
                                imputer = KNNImputer(n_neighbors=n_neighbors)
                                imputed_scaled = imputer.fit_transform(scaled_data)
                                
                                # Inverse transform
                                imputed_data = scaler.inverse_transform(imputed_scaled)
                                
                                # Update only the target column
                                df_imputed.loc[valid_rows, col] = imputed_data[:, 0]
                                
                                self.logger.debug(f"{filename}: {col} - KNN imputation ({missing_pct:.1f}% missing)")
                            else:
                                # Fallback to median
                                median_val = df_imputed[col].median()
                                df_imputed[col].fillna(median_val, inplace=True)
                                self.logger.debug(f"{filename}: {col} - Median fill fallback ({missing_pct:.1f}% missing)")
                                
                        except Exception as e:
                            # Fallback to median on KNN failure
                            median_val = df_imputed[col].median()
                            df_imputed[col].fillna(median_val, inplace=True)
                            self.logger.warning(f"{filename}: {col} - KNN failed, using median: {str(e)}")
                    else:
                        # No suitable curves for KNN, use median
                        median_val = df_imputed[col].median()
                        df_imputed[col].fillna(median_val, inplace=True)
                        self.logger.debug(f"{filename}: {col} - Median fill ({missing_pct:.1f}% missing)")
                        
                else:
                    # High missing percentage: Consider exclusion but fill with median for now
                    median_val = df_imputed[col].median()
                    df_imputed[col].fillna(median_val, inplace=True)
                    self.logger.warning(f"{filename}: {col} - High missing ({missing_pct:.1f}%), filled with median")
                    
                    # Flag for potential exclusion
                    self.stats['quality_flags'].append({
                        'file': filename,
                        'curve': col,
                        'issue': 'high_missing',
                        'percentage': missing_pct
                    })
                    
        except Exception as e:
            error_msg = f"Imputation failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"IMPUTATION TRACEBACK for {filename}:\n{traceback.format_exc()}")
            
            # Debug data state  
            self.logger.debug(f"IMPUTE DEBUG {filename}: df shape={df_imputed.shape}, columns={list(df_imputed.columns)}")
            self.logger.debug(f"IMPUTE DEBUG {filename}: current col='{col}'")
            if 'missing_pct' in locals():
                self.logger.debug(f"IMPUTE DEBUG {filename}: missing_pct={missing_pct}")
            return df  # Return original if imputation fails
            
        return df_imputed
        
    def apply_geological_constraints(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Apply geological constraints based on Eagle Ford formation characteristics
        
        Args:
            df: Well log dataframe
            metadata: Well metadata
            
        Returns:
            Constrained dataframe
        """
        filename = metadata['filename']
        self.logger.debug(f"Applying geological constraints for {filename}")
        
        df_constrained = df.copy()
        depth_col = df_constrained.columns[0]
        
        try:
            # GR constraints (primary focus)
            if 'GR' in df_constrained.columns:
                gr_min = self.config['gr_constraints']['min']
                gr_max = self.config['gr_constraints']['max']
                
                original_count = df_constrained['GR'].notna().sum()
                
                # Apply constraints
                below_min = df_constrained['GR'] < gr_min
                above_max = df_constrained['GR'] > gr_max
                
                outlier_count = below_min.sum() + above_max.sum()
                
                if outlier_count > 0:
                    self.logger.warning(f"{filename}: Constraining {outlier_count} GR values outside {gr_min}-{gr_max} API range")
                    
                    # Clip values to geological range
                    df_constrained['GR'] = df_constrained['GR'].clip(lower=gr_min, upper=gr_max)
                    
                final_count = df_constrained['GR'].notna().sum()
                self.logger.debug(f"{filename}: GR constraint applied - {original_count} -> {final_count} valid values")
                
            # Additional constraints for other curves if present
            if 'RHOB' in df_constrained.columns:
                # Typical density range for Eagle Ford: 2.0-2.8 g/cc
                df_constrained['RHOB'] = df_constrained['RHOB'].clip(lower=1.5, upper=3.0)
                
            if 'NPHI' in df_constrained.columns:
                # Neutron porosity: 0-60%
                df_constrained['NPHI'] = df_constrained['NPHI'].clip(lower=0, upper=60)
                
            if 'PE' in df_constrained.columns:
                # Photoelectric factor: 0-10 barns/electron
                df_constrained['PE'] = df_constrained['PE'].clip(lower=0, upper=10)
                
        except Exception as e:
            error_msg = f"Geological constraints failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return df  # Return original if constraining fails
            
        return df_constrained
        
    def process_single_well(self, filepath: Path) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """
        Process a single well through the complete pipeline
        
        Args:
            filepath: Path to LAS file
            
        Returns:
            Tuple of (metadata, processed_dataframe) or None if failed
        """
        filename = filepath.name
        
        try:
            # Stage 1: Parse LAS file
            metadata, df, parse_success = self.parse_las_file(filepath)
            
            if not parse_success or df is None:
                self.stats['files_failed'] += 1
                return None
                
            # Stage 2: Quality control
            self.logger.debug(f"PROCESS {filename}: starting QC, input shape={df.shape}")
            df_qc, qc_report = self.quality_control(df, metadata)
            self.logger.debug(f"PROCESS {filename}: QC complete, output shape={df_qc.shape}")
            metadata['qc_report'] = qc_report
            
            # Stage 3: Depth resampling
            self.logger.debug(f"PROCESS {filename}: starting resampling, input shape={df_qc.shape}")
            df_resampled = self.resample_to_common_depth(df_qc, metadata)
            self.logger.debug(f"PROCESS {filename}: resampling complete, output shape={df_resampled.shape}")
            
            # Stage 4: Missing value imputation
            self.logger.debug(f"PROCESS {filename}: starting imputation, input shape={df_resampled.shape}")
            df_imputed = self.impute_missing_values(df_resampled, metadata)
            self.logger.debug(f"PROCESS {filename}: imputation complete, output shape={df_imputed.shape}")
            
            # Stage 5: Geological constraints
            self.logger.debug(f"PROCESS {filename}: starting constraints, input shape={df_imputed.shape}")
            df_final = self.apply_geological_constraints(df_imputed, metadata)
            self.logger.debug(f"PROCESS {filename}: constraints complete, output shape={df_final.shape}")
            
            # Update metadata with final statistics
            metadata['processed_records'] = len(df_final)
            metadata['processing_timestamp'] = datetime.now().isoformat()
            
            # Add final data completeness
            depth_col = df_final.columns[0]
            for col in df_final.columns:
                if col != depth_col:
                    completeness = (df_final[col].notna().sum() / len(df_final)) * 100
                    metadata[f'{col}_final_completeness'] = completeness
                    
            self.stats['files_processed'] += 1
            self.stats['total_records'] += len(df_final)
            self.stats['wells_metadata'].append(metadata)
            
            self.logger.info(f"✓ {filename}: Processing complete - {len(df_final)} records")
            
            return metadata, df_final
            
        except Exception as e:
            error_msg = f"Processing failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"PROCESS TRACEBACK for {filename}:\n{traceback.format_exc()}")
            
            self.stats['files_failed'] += 1
            self.stats['processing_errors'].append({
                'file': filename,
                'stage': 'overall_processing',
                'error': error_msg
            })
            
            return None
            
    # Note: Train/test split functionality moved to feature_engineering.py
    # This ensures splits are based on final feature-engineered dataset
            
    def run_pipeline(self) -> bool:
        """
        Run the complete preprocessing pipeline
        
        Returns:
            Success flag
        """
        start_time = datetime.now()
        self.logger.info("="*50)
        self.logger.info("EAGLE FORD PREPROCESSING PIPELINE STARTED")
        self.logger.info("="*50)
        
        try:
            # Find all LAS files
            las_files = list(self.input_dir.glob("*.las"))
            
            if not las_files:
                self.logger.error(f"No LAS files found in {self.input_dir}")
                return False
                
            self.logger.info(f"Found {len(las_files)} LAS files to process")
            
            # Process each file
            all_wells_data = []
            
            for i, filepath in enumerate(las_files, 1):
                self.logger.info(f"Processing {i}/{len(las_files)}: {filepath.name}")
                
                result = self.process_single_well(filepath)
                if result:
                    metadata, df_processed = result
                    
                    # Add well identification
                    df_processed['well_api'] = metadata.get('api', filepath.stem)
                    df_processed['well_name'] = metadata.get('well_name', 'Unknown')
                    df_processed['operator'] = metadata.get('company_standardized', 'Unknown')
                    df_processed['filename'] = filepath.name
                    
                    all_wells_data.append(df_processed)
                    
            if not all_wells_data:
                self.logger.error("No wells successfully processed")
                return False
                
            # Combine all wells into master dataset
            self.logger.info("Creating master dataset...")
            self.logger.debug(f"MASTER: combining {len(all_wells_data)} wells")
            
            for i, well_df in enumerate(all_wells_data):
                self.logger.debug(f"MASTER well {i}: shape={well_df.shape}, index_unique={well_df.index.is_unique}")
                if not well_df.index.is_unique:
                    dup_count = well_df.index.duplicated().sum()
                    self.logger.debug(f"MASTER well {i}: {dup_count} duplicate indices")
            
            # Ensure all dataframes have consistent columns and reset index
            normalized_data = []
            for df in all_wells_data:
                df_reset = df.reset_index(drop=True)
                normalized_data.append(df_reset)
            
            self.logger.debug(f"MASTER: concatenating {len(normalized_data)} normalized dataframes")
            master_df = pd.concat(normalized_data, ignore_index=True, sort=False)
            self.logger.debug(f"MASTER: result shape={master_df.shape}, index_unique={master_df.index.is_unique}")
            
            # Ensure consistent column ordering
            depth_col = [col for col in master_df.columns if 'DEPTH' in col.upper()][0]
            id_cols = ['well_api', 'well_name', 'operator', 'filename']
            log_cols = [col for col in master_df.columns if col not in id_cols + [depth_col]]
            
            column_order = [depth_col] + id_cols + sorted(log_cols)
            master_df = master_df[column_order]
            
            # Save outputs
            self.logger.info("Saving outputs...")
            
            # Master dataset
            master_csv_path = self.output_dir / "master_dataset.csv"
            master_df.to_csv(master_csv_path, index=False)
            self.logger.info(f"Master dataset saved: {master_csv_path}")
            self.logger.info("Note: Train/test splits will be created in feature engineering pipeline")
            
            # Metadata - helper function to convert numpy types for JSON serialization
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
                    
            metadata_path = self.output_dir / "wells_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(convert_numpy(self.stats), f, indent=2)
            self.logger.info(f"Metadata saved: {metadata_path}")
            
            # Processing summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            summary = {
                'processing_start': start_time.isoformat(),
                'processing_end': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'files_found': len(las_files),
                'files_processed': self.stats['files_processed'],
                'files_failed': self.stats['files_failed'],
                'success_rate': (self.stats['files_processed'] / len(las_files)) * 100,
                'total_records': len(master_df),
                'unique_wells': len(master_df['well_api'].unique()),
                'operators': list(master_df['operator'].unique()),
                'available_curves': [col for col in master_df.columns if col not in ['DEPTH', 'well_api', 'well_name', 'operator', 'filename']],
                'splits_note': 'Train/test splits created in feature engineering pipeline'
            }
            
            summary_path = self.output_dir / "processing_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info("="*50)
            self.logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*50)
            self.logger.info(f"Duration: {duration}")
            self.logger.info(f"Success rate: {summary['success_rate']:.1f}%")
            self.logger.info(f"Total records: {summary['total_records']:,}")
            self.logger.info(f"Unique wells: {summary['unique_wells']}")
            self.logger.info(f"Master dataset: {master_csv_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"PIPELINE TRACEBACK:\n{traceback.format_exc()}")
            
            # Debug master dataset creation state
            if 'all_wells_data' in locals():
                self.logger.debug(f"PIPELINE DEBUG: all_wells_data length={len(all_wells_data)}")
                for i, df in enumerate(all_wells_data):
                    self.logger.debug(f"PIPELINE DEBUG: well {i} shape={df.shape}, columns={list(df.columns)}, index_unique={df.index.is_unique}")
            
            return False

if __name__ == "__main__":
    """
    Main execution
    Usage: python preprocessing_pipeline.py [--test] [--run-name NAME]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Eagle Ford Preprocessing Pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode with sample data')
    parser.add_argument('--run-name', type=str, help='Custom run name for output organization')
    parser.add_argument('--input-dir', type=str, help='Custom input directory')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set directories based on mode
    if args.test:
        input_dir = args.input_dir or "/Users/satan/projects/mtp_1/tests/data"
        output_dir = args.output_dir or "/Users/satan/projects/mtp_1/tests/outputs"
        run_name = args.run_name or "test_sample"
        print("🧪 Running in TEST mode with sample data")
    else:
        input_dir = args.input_dir or "/Users/satan/projects/mtp_1/dataset/raw"
        output_dir = args.output_dir or "/Users/satan/projects/mtp_1/dataset/processed"
        run_name = args.run_name
        print("🚀 Running in PRODUCTION mode with full dataset")
    
    # Initialize and run pipeline
    preprocessor = EagleFordPreprocessor(
        input_dir=input_dir,
        output_dir=output_dir,
        log_level=args.log_level,
        run_name=run_name
    )
    
    success = preprocessor.run_pipeline()
    
    if success:
        print(f"\n✅ Preprocessing pipeline completed successfully!")
        print(f"📁 Outputs saved to: {preprocessor.output_dir}")
        print(f"📄 Master dataset: {preprocessor.output_dir}/master_dataset.csv")
        print(f"🔄 Next step: Run feature engineering pipeline")
    else:
        print(f"\n❌ Preprocessing pipeline failed!")
        print(f"📋 Check logs in: {preprocessor.output_dir}/logs/")
        sys.exit(1)