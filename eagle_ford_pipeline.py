#!/usr/bin/env python3
"""
Eagle Ford ML Pipeline - Complete End-to-End Processing (FIXED VERSION)
From Raw LAS Files → Preprocessing → Feature Engineering → Model Training

✅ PLATFORM COMPATIBILITY:
- ✅ Kaggle Notebooks
- ✅ Google Colab  
- ✅ VSCode (Mac M2 Air 8GB RAM)
- ✅ General Python environments

Author: Eagle Ford ML Team
Date: November 2025
Version: 2.0 (Fixed)
"""

import sys
import os
import json
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
def safe_import(module_name, package_name=None, install_cmd=None):
    """Safely import a module with automatic installation if missing"""
    try:
        if package_name:
            return __import__(package_name, fromlist=[module_name])
        else:
            return __import__(module_name)
    except ImportError:
        if install_cmd:
            try:
                print(f"📦 Installing {module_name}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_cmd])
                if package_name:
                    return __import__(package_name, fromlist=[module_name])
                else:
                    return __import__(module_name)
            except Exception as e:
                print(f"❌ Failed to install {module_name}: {e}")
                return None
        else:
            print(f"⚠️  {module_name} not available")
            return None

# Import required packages with auto-installation
try:
    import psutil
except ImportError:
    psutil = safe_import('psutil', install_cmd='psutil')

# Environment detection with enhanced platform support
def detect_environment():
    """Enhanced environment detection with platform-specific optimizations"""
    env_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_gb': 8.0,  # Default fallback
        'is_kaggle': False,
        'is_colab': False,
        'is_jupyter': False,
        'has_gpu': False,
        'gpu_available_frameworks': []
    }
    
    # Memory detection with fallback
    try:
        if psutil:
            env_info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except:
        # Fallback memory estimation
        try:
            import resource
            mem_limit = resource.getrlimit(resource.RLIMIT_AS)[0]
            if mem_limit != resource.RLIM_INFINITY:
                env_info['memory_gb'] = round(mem_limit / (1024**3), 1)
        except:
            pass
    
    # Environment type detection
    env_info['is_kaggle'] = any([
        'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
        'KAGGLE_DATA_PROXY_TOKEN' in os.environ,
        '/kaggle/' in os.getcwd()
    ])
    
    env_info['is_colab'] = any([
        'COLAB_GPU' in os.environ,
        'COLAB_TPU_ADDR' in os.environ,
        '/content/' in os.getcwd(),
        'google.colab' in str(globals().get('get_ipython', lambda: ''))
    ])
    
    env_info['is_jupyter'] = any([
        'JPY_PARENT_PID' in os.environ,
        'JUPYTER_RUNTIME_DIR' in os.environ,
        hasattr(__builtins__, '__IPYTHON__')
    ])
    
    # GPU detection with multiple framework support
    gpu_frameworks = []
    
    # PyTorch GPU detection
    try:
        torch = safe_import('torch', install_cmd='torch')
        if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
            env_info['has_gpu'] = True
            gpu_frameworks.append('pytorch')
            env_info['gpu_name'] = torch.cuda.get_device_name(0)
            env_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        pass
    
    # TensorFlow GPU detection  
    try:
        tf = safe_import('tensorflow', install_cmd='tensorflow')
        if tf and len(tf.config.list_physical_devices('GPU')) > 0:
            env_info['has_gpu'] = True
            gpu_frameworks.append('tensorflow')
    except:
        pass
    
    env_info['gpu_available_frameworks'] = gpu_frameworks
    
    # Environment type classification
    if env_info['is_kaggle']:
        env_info['env_type'] = 'kaggle'
    elif env_info['is_colab']:
        env_info['env_type'] = 'colab'
    elif env_info['platform'] == 'Darwin' and 'arm' in env_info['machine'].lower():
        env_info['env_type'] = 'mac_m2'
    elif env_info['is_jupyter']:
        env_info['env_type'] = 'jupyter'
    else:
        env_info['env_type'] = 'general'
    
    return env_info

# Enhanced Configuration with Platform Compatibility
class EagleFordConfig:
    """
    Enhanced configuration with full platform compatibility
    Supports Kaggle, Colab, Mac M2, and general environments
    """
    
    def __init__(self, env_info: Dict = None):
        self.env_info = env_info or detect_environment()
        
        # =================================================================
        # GLOBAL PIPELINE SETTINGS
        # =================================================================
        self.GLOBAL = {
            'run_mode': 'auto',
            'target_column': 'GR',
            'formation': 'eagle_ford',
            'random_state': 42,
            'n_jobs': min(self.env_info['cpu_count'] - 1, 8) if self.env_info['cpu_count'] > 1 else 1,
            'max_memory_usage': 0.7 if self.env_info['memory_gb'] <= 8 else 0.85,
            'output_compression': 'gzip' if self.env_info['memory_gb'] <= 8 else None,
            'enable_gpu': self.env_info.get('has_gpu', False),
            'chunk_processing': self.env_info['memory_gb'] <= 8,
        }
        
        # =================================================================
        # PLATFORM-SPECIFIC OPTIMIZATIONS (Enhanced)
        # =================================================================
        self.apply_platform_optimizations()
        
        # =================================================================
        # PROCESSING CONFIGURATION
        # =================================================================
        self.PROCESSING = {
            # Preprocessing settings that match actual EagleFordPreprocessor
            'input_pattern': '*.las',
            'target_step_size': 1.0,  # Matches preprocessing_pipeline.py config
            'gr_constraints': {'min': 10.0, 'max': 250.0},  # Matches preprocessing config
            'completeness_threshold': 95.0,
            'imputation_threshold': 20.0,
            'outlier_mad_threshold': 5.0,
            
            # Feature engineering settings that match EagleFordFeatureEngineer
            'rolling_windows': [5, 10, 20],  # Matches feature_engineering.py config
            'sequence_length': 50,
            'target_curves': ['GR'],
            'geological_features': True,
            'cross_curve_features': True,
            'depth_normalization': 'minmax',
            
            # ML settings that match EagleFordMLTrainer
            'cv_folds': 3 if self.env_info['memory_gb'] <= 8 else 5,
            'n_iter_search': 10 if self.env_info['memory_gb'] <= 8 else 100,
            'search_method': 'random',
            'normalization': 'box_cox',
            'outlier_threshold': 3.0,
        }
        
        # =================================================================
        # LOGGING CONFIGURATION
        # =================================================================
        self.LOGGING = {
            'level': 'INFO',
            'enable_progress_bars': True,
            'log_to_file': True,
            'log_to_console': True,
            'detailed_timing': True,
        }
        
    def apply_platform_optimizations(self):
        """Apply platform-specific optimizations"""
        
        if self.env_info['env_type'] == 'kaggle':
            # Kaggle optimizations
            self.GLOBAL.update({
                'n_jobs': -1,
                'max_memory_usage': 0.9,
                'chunk_processing': False,
            })
            self.PROCESSING.update({
                'cv_folds': 5,
                'n_iter_search': 50,
                'normalization': 'box_cox',
            })
            
        elif self.env_info['env_type'] == 'colab':
            # Colab optimizations
            self.GLOBAL.update({
                'n_jobs': -1,
                'max_memory_usage': 0.85,
                'chunk_processing': self.env_info['memory_gb'] <= 12,
            })
            self.PROCESSING.update({
                'cv_folds': 5,
                'n_iter_search': 30,
            })
            
        elif self.env_info['env_type'] == 'mac_m2':
            # Mac M2 optimizations
            self.GLOBAL.update({
                'n_jobs': min(8, self.env_info['cpu_count']),
                'max_memory_usage': 0.7,
                'chunk_processing': True,
            })
            self.PROCESSING.update({
                'cv_folds': 3,
                'n_iter_search': 15,
                'normalization': None,  # Skip Box-Cox for speed
                'outlier_threshold': 5.0,
            })

# Enhanced Pipeline Class with Fixed Imports and Methods
class EagleFordPipeline:
    """
    Fixed Eagle Ford ML Pipeline with proper class imports and method calls
    """
    
    def __init__(self, config: EagleFordConfig = None, input_dir: str = None, output_dir: str = None):
        self.config = config or EagleFordConfig()
        
        # Set up directories with platform-aware defaults
        if input_dir:
            self.input_dir = Path(input_dir)
        else:
            # Platform-aware default input directories
            if self.config.env_info['env_type'] == 'kaggle':
                self.input_dir = Path('/kaggle/input')
            elif self.config.env_info['env_type'] == 'colab':
                self.input_dir = Path('/content/dataset/raw')
            else:
                self.input_dir = Path.cwd() / "dataset" / "raw"
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Platform-aware default output directories
            if self.config.env_info['env_type'] in ['kaggle', 'colab']:
                self.output_dir = Path('/tmp/eagle_ford_output')
            else:
                self.output_dir = Path.cwd() / "eagle_ford_output"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize pipeline components with fixed imports
        self.setup_pipeline_components()
        
        # Store pipeline state
        self.pipeline_state = {
            'preprocessing_complete': False,
            'feature_engineering_complete': False,
            'model_training_complete': False,
            'preprocessing_output': None,
            'feature_engineering_output': None,
            'model_training_output': None,
        }
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 EAGLE FORD ML PIPELINE INITIALIZED (FIXED VERSION)")
        self.logger.info("=" * 80)
        self.logger.info(f"Environment: {self.config.env_info['env_type'].upper()}")
        self.logger.info(f"Platform: {self.config.env_info['platform']} {self.config.env_info['machine']}")
        self.logger.info(f"Python: {self.config.env_info['python_version']}")
        self.logger.info(f"Memory: {self.config.env_info['memory_gb']}GB")
        self.logger.info(f"CPU Cores: {self.config.env_info['cpu_count']}")
        self.logger.info(f"GPU Available: {self.config.env_info.get('has_gpu', False)}")
        if self.config.env_info.get('gpu_available_frameworks'):
            self.logger.info(f"GPU Frameworks: {', '.join(self.config.env_info['gpu_available_frameworks'])}")
        self.logger.info(f"Input Directory: {self.input_dir}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info("")
    
    def setup_logging(self):
        """Setup enhanced logging system"""
        log_filename = f"eagle_ford_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = self.logs_dir / log_filename
        
        # Create logger
        self.logger = logging.getLogger('EagleFordPipeline')
        self.logger.setLevel(getattr(logging, self.config.LOGGING['level']))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        if self.config.LOGGING['log_to_file']:
            file_handler = logging.FileHandler(log_filepath)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if self.config.LOGGING['log_to_console']:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        self.log_file = log_filepath
    
    def setup_pipeline_components(self):
        """Initialize pipeline components with FIXED imports and error handling"""
        
        # Add current directory to Python path for imports
        current_dir = Path(__file__).parent
        code_dir = current_dir / "code" / "src"
        
        if code_dir.exists():
            sys.path.insert(0, str(code_dir))
        else:
            # Try alternative paths
            alternative_paths = [
                current_dir / "src",
                current_dir,
                Path.cwd() / "code" / "src",
                Path.cwd() / "src"
            ]
            for alt_path in alternative_paths:
                if alt_path.exists():
                    sys.path.insert(0, str(alt_path))
                    break
        
        try:
            # FIXED IMPORTS - Use correct class names
            self.logger.info("📦 Importing pipeline components...")
            
            # Import preprocessing with correct class name
            try:
                from preprocessing_pipeline import EagleFordPreprocessor
                self.logger.info("✅ Successfully imported EagleFordPreprocessor")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import EagleFordPreprocessor: {e}")
                raise
            
            # Import feature engineering with correct class name  
            try:
                from feature_engineering import EagleFordFeatureEngineer
                self.logger.info("✅ Successfully imported EagleFordFeatureEngineer")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import EagleFordFeatureEngineer: {e}")
                raise
            
            # Import ML models
            try:
                from ml_models import EagleFordMLTrainer
                self.logger.info("✅ Successfully imported EagleFordMLTrainer")
            except ImportError as e:
                self.logger.error(f"❌ Failed to import EagleFordMLTrainer: {e}")
                raise
            
            # Initialize components with proper directory structure
            self.preprocessor = EagleFordPreprocessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir / "preprocessing"),
                log_level=self.config.LOGGING['level']
            )
            
            self.feature_engineer = EagleFordFeatureEngineer(
                input_file=str(self.output_dir / "preprocessing" / "master_dataset.csv"),
                output_dir=str(self.output_dir),
                log_level=self.config.LOGGING['level']
            )
            
            self.ml_trainer = EagleFordMLTrainer(
                input_dir=str(self.output_dir / "features"),
                output_dir=str(self.output_dir / "models"),
                log_level=self.config.LOGGING['level']
            )
            
            self.logger.info("✅ All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline components: {e}")
            self.logger.error("💡 Make sure all component files are available and properly structured")
            raise
    
    def run_preprocessing(self, force_rerun: bool = False) -> bool:
        """Run preprocessing stage with FIXED method calls"""
        if self.pipeline_state['preprocessing_complete'] and not force_rerun:
            self.logger.info("⏭️  Preprocessing already completed, skipping...")
            return True
        
        self.logger.info("🔧 STAGE 1: PREPROCESSING")
        self.logger.info("-" * 50)
        
        try:
            # Apply configuration updates that match the actual class attributes
            # Note: Only update attributes that actually exist in EagleFordPreprocessor
            if hasattr(self.preprocessor, 'config'):
                config_updates = {
                    'target_step_size': self.config.PROCESSING['target_step_size'],
                    'gr_constraints': self.config.PROCESSING['gr_constraints'],
                    'completeness_threshold': self.config.PROCESSING['completeness_threshold'],
                    'imputation_threshold': self.config.PROCESSING['imputation_threshold'],
                    'outlier_mad_threshold': self.config.PROCESSING['outlier_mad_threshold'],
                }
                # Only update existing config keys
                for key, value in config_updates.items():
                    if key in self.preprocessor.config:
                        self.preprocessor.config[key] = value
            
            # FIXED METHOD CALL - Use correct method name
            start_time = datetime.now()
            result = self.preprocessor.run_pipeline()  # Correct method name
            end_time = datetime.now()
            
            if result:
                self.pipeline_state['preprocessing_complete'] = True
                self.pipeline_state['preprocessing_output'] = self.output_dir / "preprocessing"
                
                duration = (end_time - start_time).total_seconds()
                self.logger.info(f"✅ Preprocessing completed in {duration:.2f} seconds")
                self.logger.info(f"📁 Output saved to: {self.pipeline_state['preprocessing_output']}")
                return True
            else:
                self.logger.error("❌ Preprocessing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Preprocessing error: {str(e)}")
            if self.config.LOGGING['level'] == 'DEBUG':
                import traceback
                self.logger.debug(traceback.format_exc())
            return False
    
    def run_feature_engineering(self, force_rerun: bool = False) -> bool:
        """Run feature engineering stage with FIXED method calls"""
        if not self.pipeline_state['preprocessing_complete']:
            self.logger.error("❌ Preprocessing must be completed before feature engineering")
            return False
        
        if self.pipeline_state['feature_engineering_complete'] and not force_rerun:
            self.logger.info("⏭️  Feature engineering already completed, skipping...")
            return True
        
        self.logger.info("🎛️  STAGE 2: FEATURE ENGINEERING")
        self.logger.info("-" * 50)
        
        try:
            # Update input file path to match preprocessing output
            preprocessed_file = self.pipeline_state['preprocessing_output'] / "master_dataset.csv"
            if preprocessed_file.exists():
                self.feature_engineer.input_file = preprocessed_file
            
            # Apply configuration updates that match the actual class attributes
            if hasattr(self.feature_engineer, 'config'):
                config_updates = {
                    'rolling_windows': self.config.PROCESSING['rolling_windows'],
                    'sequence_length': self.config.PROCESSING['sequence_length'],
                    'target_curves': self.config.PROCESSING['target_curves'],
                    'geological_features': self.config.PROCESSING['geological_features'],
                    'cross_curve_features': self.config.PROCESSING['cross_curve_features'],
                    'depth_normalization': self.config.PROCESSING['depth_normalization'],
                }
                # Only update existing config keys
                for key, value in config_updates.items():
                    if key in self.feature_engineer.config:
                        self.feature_engineer.config[key] = value
            
            # FIXED METHOD CALL - Use correct method name
            start_time = datetime.now()
            result = self.feature_engineer.run_feature_engineering()  # Correct method name
            end_time = datetime.now()
            
            if result:
                self.pipeline_state['feature_engineering_complete'] = True
                self.pipeline_state['feature_engineering_output'] = self.feature_engineer.output_dir
                
                duration = (end_time - start_time).total_seconds()
                self.logger.info(f"✅ Feature engineering completed in {duration:.2f} seconds")
                self.logger.info(f"📁 Output saved to: {self.pipeline_state['feature_engineering_output']}")
                return True
            else:
                self.logger.error("❌ Feature engineering failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Feature engineering error: {str(e)}")
            if self.config.LOGGING['level'] == 'DEBUG':
                import traceback
                self.logger.debug(traceback.format_exc())
            return False
    
    def run_model_training(self, force_rerun: bool = False) -> bool:
        """Run model training stage with FIXED method calls and proper error handling"""
        if not self.pipeline_state['feature_engineering_complete']:
            self.logger.error("❌ Feature engineering must be completed before model training")
            return False
        
        if self.pipeline_state['model_training_complete'] and not force_rerun:
            self.logger.info("⏭️  Model training already completed, skipping...")
            return True
        
        self.logger.info("🤖 STAGE 3: MODEL TRAINING")
        self.logger.info("-" * 50)
        
        try:
            # Update ML trainer input directory
            self.ml_trainer.input_dir = self.pipeline_state['feature_engineering_output']
            
            # Apply configuration updates that match the actual class attributes
            if hasattr(self.ml_trainer, 'config'):
                config_updates = {
                    'search_method': self.config.PROCESSING['search_method'],
                    'n_iter_search': self.config.PROCESSING['n_iter_search'],
                    'cv_folds': self.config.PROCESSING['cv_folds'],
                    'normalization': self.config.PROCESSING['normalization'],
                    'outlier_threshold': self.config.PROCESSING['outlier_threshold'],
                }
                # Only update existing config keys
                for key, value in config_updates.items():
                    if key in self.ml_trainer.config:
                        self.ml_trainer.config[key] = value
            
            # Apply platform-specific optimizations
            if self.config.env_info['env_type'] == 'mac_m2':
                self.ml_trainer.enable_test_mode()
            elif self.config.env_info['memory_gb'] <= 8:
                if hasattr(self.ml_trainer, 'enable_test_mode'):
                    self.ml_trainer.enable_test_mode()
            
            # Set search method if supported
            if hasattr(self.ml_trainer, 'set_search_method'):
                self.ml_trainer.set_search_method(self.config.PROCESSING['search_method'])
            
            # FIXED METHOD CALL - Use correct method name
            start_time = datetime.now()
            result = self.ml_trainer.run_ml_pipeline()  # Correct method name
            end_time = datetime.now()
            
            if result:
                self.pipeline_state['model_training_complete'] = True
                self.pipeline_state['model_training_output'] = self.ml_trainer.output_dir
                
                duration = (end_time - start_time).total_seconds()
                self.logger.info(f"✅ Model training completed in {duration:.2f} seconds")
                
                # Get best model info if available
                if hasattr(self.ml_trainer, 'results') and 'evaluation' in self.ml_trainer.results:
                    best_model = self.ml_trainer.results.get('evaluation', {}).get('best_model', 'Unknown')
                    self.logger.info(f"🏆 Best model: {best_model.upper()}")
                
                self.logger.info(f"📁 Models saved to: {self.pipeline_state['model_training_output']}")
                return True
            else:
                self.logger.error("❌ Model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Model training error: {str(e)}")
            if self.config.LOGGING['level'] == 'DEBUG':
                import traceback
                self.logger.debug(traceback.format_exc())
            return False
    
    def run_complete_pipeline(self, force_rerun: bool = False) -> bool:
        """Run the complete pipeline from start to finish"""
        start_time = datetime.now()
        
        self.logger.info("🚀 STARTING COMPLETE EAGLE FORD ML PIPELINE")
        self.logger.info("=" * 80)
        
        # Save configuration
        config_file = self.output_dir / "pipeline_config.json"
        self.save_config_safe(config_file)
        self.logger.info(f"💾 Pipeline configuration saved: {config_file}")
        self.logger.info("")
        
        # Check input directory and files
        try:
            if not self.input_dir.exists():
                self.logger.error(f"❌ Input directory not found: {self.input_dir}")
                return False
            
            las_files = list(self.input_dir.glob("*.las"))
            if not las_files:
                self.logger.error(f"❌ No LAS files found in {self.input_dir}")
                return False
            
            self.logger.info(f"📁 Found {len(las_files)} LAS files to process")
            self.logger.info("")
            
        except Exception as e:
            self.logger.error(f"❌ Error checking input files: {e}")
            return False
        
        # Run pipeline stages
        success = True
        
        # Stage 1: Preprocessing
        if not self.run_preprocessing(force_rerun):
            success = False
        
        # Stage 2: Feature Engineering
        if success and not self.run_feature_engineering(force_rerun):
            success = False
        
        # Stage 3: Model Training
        if success and not self.run_model_training(force_rerun):
            success = False
        
        # Pipeline completion
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        if success:
            self.logger.info("=" * 80)
            self.logger.info("🎊 PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Total execution time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
            self.logger.info(f"🖥️  Environment: {self.config.env_info['env_type'].upper()}")
            self.logger.info(f"💾 Memory used: {self.config.env_info['memory_gb']}GB")
            self.logger.info(f"📁 All outputs saved to: {self.output_dir}")
            self.logger.info(f"📋 Log file: {self.log_file}")
            self.logger.info("")
            self.logger.info("📊 Pipeline Results Summary:")
            self.logger.info(f"   ✅ Preprocessing: {self.pipeline_state['preprocessing_complete']}")
            self.logger.info(f"   ✅ Feature Engineering: {self.pipeline_state['feature_engineering_complete']}")
            self.logger.info(f"   ✅ Model Training: {self.pipeline_state['model_training_complete']}")
            
            # Save pipeline state
            self.save_pipeline_state()
            
            return True
        else:
            self.logger.error("=" * 80)
            self.logger.error("❌ PIPELINE FAILED!")
            self.logger.error("=" * 80)
            self.logger.error(f"⏱️  Execution time before failure: {total_duration:.2f} seconds")
            self.logger.error(f"📋 Check log file for details: {self.log_file}")
            return False
    
    def save_config_safe(self, config_file: Path):
        """Safely save configuration with error handling"""
        try:
            config_dict = {
                'global': self.config.GLOBAL,
                'processing': self.config.PROCESSING,
                'logging': self.config.LOGGING,
                'environment': self.config.env_info,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save config file: {e}")
    
    def save_pipeline_state(self):
        """Save pipeline state with error handling"""
        try:
            state_file = self.output_dir / "pipeline_state.json"
            state_dict = {
                **self.pipeline_state,
                'total_duration': (datetime.now() - self.pipeline_state.get('start_time', datetime.now())).total_seconds() if 'start_time' in self.pipeline_state else 0,
                'environment': self.config.env_info['env_type'],
                'completed_at': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save pipeline state: {e}")

# Enhanced main function with better error handling
def main():
    """Enhanced main execution with comprehensive error handling"""
    import argparse
    
    # Print environment info
    env_info = detect_environment()
    print("\n" + "=" * 80)
    print("🚀 EAGLE FORD ML PIPELINE - FIXED VERSION")
    print("=" * 80)
    print(f"🖥️  Environment: {env_info['env_type'].upper()}")
    print(f"🐍 Python: {env_info['python_version']}")
    print(f"💾 Memory: {env_info['memory_gb']}GB")
    print(f"⚡ GPU: {'Available' if env_info.get('has_gpu') else 'Not available'}")
    if env_info.get('gpu_available_frameworks'):
        print(f"🎯 GPU Frameworks: {', '.join(env_info['gpu_available_frameworks'])}")
    print("")
    
    parser = argparse.ArgumentParser(
        description="Eagle Ford ML Pipeline - Fixed Version for All Platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and run complete pipeline
  python eagle_ford_pipeline_fixed.py
  
  # Kaggle/Colab with custom paths
  python eagle_ford_pipeline_fixed.py --input-dir /kaggle/input --output-dir /kaggle/working
  
  # Mac M2 with test mode
  python eagle_ford_pipeline_fixed.py --mode test --input-dir ./dataset/raw
  
  # Run specific stage only
  python eagle_ford_pipeline_fixed.py --stage preprocessing
        """
    )
    
    parser.add_argument("--input-dir", type=str, help="Input directory containing LAS files")
    parser.add_argument("--output-dir", type=str, help="Output directory for pipeline results")
    parser.add_argument("--mode", choices=['auto', 'test', 'production'], default='auto',
                       help="Pipeline execution mode")
    parser.add_argument("--force-rerun", action='store_true', 
                       help="Force rerun of all stages even if completed")
    parser.add_argument("--stage", choices=['preprocessing', 'features', 'models', 'all'], 
                       default='all', help="Run specific pipeline stage")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    try:
        # Create enhanced configuration
        config = EagleFordConfig(env_info)
        config.GLOBAL['run_mode'] = args.mode
        config.LOGGING['level'] = args.log_level
        
        # Platform-specific defaults if not provided
        if not args.input_dir:
            if env_info['env_type'] == 'kaggle':
                args.input_dir = '/kaggle/input'
            elif env_info['env_type'] == 'colab':
                args.input_dir = '/content/dataset/raw'
            else:
                args.input_dir = str(Path.cwd() / "dataset" / "raw")
        
        if not args.output_dir:
            if env_info['env_type'] in ['kaggle', 'colab']:
                args.output_dir = '/tmp/eagle_ford_output'
            else:
                args.output_dir = str(Path.cwd() / "eagle_ford_output")
        
        print(f"📁 Input Directory: {args.input_dir}")
        print(f"📁 Output Directory: {args.output_dir}")
        print(f"🎯 Execution Mode: {args.mode.upper()}")
        print(f"📋 Log Level: {args.log_level}")
        print("")
        
        # Create and run pipeline
        pipeline = EagleFordPipeline(
            config=config,
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
        # Run requested stage
        if args.stage == 'preprocessing':
            success = pipeline.run_preprocessing(args.force_rerun)
        elif args.stage == 'features':
            success = pipeline.run_feature_engineering(args.force_rerun)
        elif args.stage == 'models':
            success = pipeline.run_model_training(args.force_rerun)
        else:  # all
            success = pipeline.run_complete_pipeline(args.force_rerun)
        
        if success:
            print("\n✅ Pipeline completed successfully!")
            print(f"📁 Check outputs in: {args.output_dir}")
        else:
            print("\n❌ Pipeline failed!")
            print(f"📋 Check logs for details")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure all required files are present:")
        print("   - code/src/preprocessing_pipeline.py")
        print("   - code/src/feature_engineering.py") 
        print("   - code/src/ml_models.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()