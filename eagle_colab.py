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
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Memory management utilities
def force_garbage_collection():
    """Force garbage collection and return memory freed"""
    initial_objects = len(gc.get_objects())
    gc.collect()
    final_objects = len(gc.get_objects())
    return initial_objects - final_objects

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def memory_cleanup_decorator(func):
    """Decorator to automatically clean up memory after function execution"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Force garbage collection
            objects_freed = force_garbage_collection()
            if hasattr(args[0], 'logger'):
                args[0].logger.debug(f"🧹 Memory cleanup: {objects_freed} objects freed")
            return result
        except Exception as e:
            # Cleanup even on error
            force_garbage_collection()
            raise e
    return wrapper

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
        print("Memory estimation failed. Using default fallback.")
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

class ColabResourceMonitor:
      """Advanced resource monitoring with automatic cache clearing for Colab"""

      def __init__(self, logger):
          self.logger = logger
          self.setup_monitoring()

          # Thresholds for automatic cleanup (80% of limits)
          self.ram_threshold = 0.80  # 80% of available RAM
          self.gpu_threshold = 0.80  # 80% of GPU memory  
          self.disk_threshold = 0.80  # 80% of disk space

      def setup_monitoring(self):
          """Setup resource monitoring tools"""
          try:
              import psutil
              self.psutil = psutil
          except ImportError:
              self.psutil = safe_import('psutil', install_cmd='psutil')

          try:
              import pynvml
              pynvml.nvmlInit()
              self.pynvml = pynvml
              self.gpu_available = True
          except:
              try:
                  import subprocess
                  result = subprocess.run(['nvidia-smi'], capture_output=True)
                  self.gpu_available = result.returncode == 0
              except:
                  self.gpu_available = False
              self.pynvml = None

      def get_system_resources(self):
          """Get current system resource usage"""
          resources = {}

          # System RAM
          if self.psutil:
              memory = self.psutil.virtual_memory()
              resources.update({
                  'ram_used_gb': round(memory.used / (1024**3), 2),
                  'ram_total_gb': round(memory.total / (1024**3), 2),
                  'ram_percent': memory.percent,
                  'ram_available_gb': round(memory.available / (1024**3), 2)
              })

          # GPU Memory
          if self.gpu_available and self.pynvml:
              try:
                  handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                  gpu_memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                  resources.update({
                      'gpu_used_gb': round(gpu_memory.used / (1024**3), 2),
                      'gpu_total_gb': round(gpu_memory.total / (1024**3), 2),
                      'gpu_percent': round((gpu_memory.used / gpu_memory.total) * 100, 1),
                      'gpu_free_gb': round(gpu_memory.free / (1024**3), 2)
                  })
              except:
                  resources.update({'gpu_error': 'Unable to read GPU memory'})

          # Disk Space
          if self.psutil:
              disk = self.psutil.disk_usage('/')
              resources.update({
                  'disk_used_gb': round(disk.used / (1024**3), 2),
                  'disk_total_gb': round(disk.total / (1024**3), 2),
                  'disk_percent': round((disk.used / disk.total) * 100, 1),
                  'disk_free_gb': round(disk.free / (1024**3), 2)
              })

          return resources

      def check_and_cleanup_if_needed(self, stage_name=""):
          """Check resources and cleanup if approaching limits"""
          resources = self.get_system_resources()
          cleanup_performed = []

          # Check RAM
          if resources.get('ram_percent', 0) > (self.ram_threshold * 100):
              self.logger.warning(f"🔥 RAM usage high: {resources['ram_percent']:.1f}% - Performing cleanup")
              cleanup_performed.extend(self.cleanup_system_memory())

          # Check GPU
          if resources.get('gpu_percent', 0) > (self.gpu_threshold * 100):
              self.logger.warning(f"🔥 GPU memory high: {resources['gpu_percent']:.1f}% - Performing cleanup")
              cleanup_performed.extend(self.cleanup_gpu_memory())

          # Check Disk
          if resources.get('disk_percent', 0) > (self.disk_threshold * 100):
              self.logger.warning(f"🔥 Disk usage high: {resources['disk_percent']:.1f}% - Performing cleanup")
              cleanup_performed.extend(self.cleanup_disk_space())

          if cleanup_performed:
              self.logger.info(f"🧹 Cleanup completed for {stage_name}: {', '.join(cleanup_performed)}")

          return resources, cleanup_performed

      def cleanup_system_memory(self):
          """Aggressive system RAM cleanup based on 2024 best practices"""
          cleanup_actions = []
          initial_memory = get_memory_usage()
          
          try:
              # 1. Force garbage collection multiple times
              for i in range(3):
                  collected = gc.collect()
                  if collected > 0:
                      cleanup_actions.append(f"GC cycle {i+1}: {collected} objects")
              
              # 2. Clear pandas cached data if available
              try:
                  import pandas as pd
                  # Clear internal pandas caches
                  pd._testing.clear_cache()
                  cleanup_actions.append("Pandas cache cleared")
              except:
                  pass
                  
              # 3. Clear matplotlib memory if used - 2024 best practice order
              try:
                  import matplotlib.pyplot as plt
                  plt.clf()        # Clear current figure first
                  plt.cla()        # Clear current axes  
                  plt.close('all') # Then close all figures - prevents memory leaks
                  cleanup_actions.append("Matplotlib plots cleared (2024 best practice)")
              except:
                  pass
              
              # 4. Clear IPython history in Colab
              try:
                  from IPython import get_ipython
                  ip = get_ipython()
                  if ip:
                      ip.reset(new_session=False)
                      cleanup_actions.append("IPython history cleared")
              except:
                  pass
              
              # 5. Optimize memory fragmentation 
              try:
                  import ctypes
                  libc = ctypes.CDLL("libc.so.6")
                  libc.malloc_trim(0)
                  cleanup_actions.append("Memory fragmentation optimized")
              except:
                  pass
              
              final_memory = get_memory_usage()
              memory_freed = initial_memory - final_memory
              if memory_freed > 0:
                  cleanup_actions.append(f"Total memory freed: {memory_freed:.1f}MB")
                  
          except Exception as e:
              self.logger.warning(f"Memory cleanup failed: {e}")
          
          return cleanup_actions

      def cleanup_gpu_memory(self):
          """Clean up GPU memory"""
          cleanup_actions = []

          try:
              # PyTorch cleanup
              import torch
              if torch.cuda.is_available():
                  torch.cuda.empty_cache()
                  torch.cuda.synchronize()
                  cleanup_actions.append("PyTorch GPU cache cleared")
          except ImportError:
              pass
          except Exception as e:
              self.logger.warning(f"PyTorch GPU cleanup failed: {e}")

          try:
              # TensorFlow cleanup
              import tensorflow as tf
              tf.keras.backend.clear_session()
              cleanup_actions.append("TensorFlow session cleared")
          except ImportError:
              pass
          except Exception as e:
              self.logger.warning(f"TensorFlow cleanup failed: {e}")

          return cleanup_actions

      def cleanup_disk_space(self):
          """Clean up disk space"""
          cleanup_actions = []

          try:
              import shutil
              import tempfile

              # Clear temp files
              temp_dir = Path(tempfile.gettempdir())
              temp_files = list(temp_dir.glob("*"))
              if len(temp_files) > 100:  # Only if many temp files
                  for temp_file in temp_files[:50]:  # Clear oldest 50
                      try:
                          if temp_file.is_file():
                              temp_file.unlink()
                          elif temp_file.is_dir():
                              shutil.rmtree(temp_file, ignore_errors=True)
                      except:
                          continue
                  cleanup_actions.append("Temp files cleared")
          except Exception as e:
              self.logger.warning(f"Disk cleanup failed: {e}")

          return cleanup_actions

      def log_resources(self, stage_name="", event="monitor"):
          """Log current resource usage"""
          resources = self.get_system_resources()

          log_msg = f"📊 [{stage_name}] Resources: "
          log_parts = []

          if 'ram_percent' in resources:
              log_parts.append(f"RAM: {resources['ram_used_gb']:.1f}GB/{resources['ram_total_gb']:.1f}GB ({resources['ram_percent']:.1f}%)")

          if 'gpu_percent' in resources:
              log_parts.append(f"GPU: {resources['gpu_used_gb']:.1f}GB/{resources['gpu_total_gb']:.1f}GB ({resources['gpu_percent']:.1f}%)")

          if 'disk_percent' in resources:
              log_parts.append(f"Disk: {resources['disk_used_gb']:.1f}GB/{resources['disk_total_gb']:.1f}GB ({resources['disk_percent']:.1f}%)")

          self.logger.info(log_msg + " | ".join(log_parts))
          return resources

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
        
        # =================================================================
        # APPLY PLATFORM-SPECIFIC OPTIMIZATIONS (After all configs defined)
        # =================================================================
        self.apply_platform_optimizations()
        
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

    def load_config_file(self, filepath: str):
        """Load and apply configuration from JSON file"""
        config_path = Path(filepath)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Store additional config sections
        self.PATHS = config_dict.get('paths', {})
        self.TEST_SPECIFIC = config_dict.get('test_specific', {})
        self.PIPELINE_INFO = config_dict.get('pipeline_info', {})
        
        # Update configurations with deep merge
        self.GLOBAL.update(config_dict.get('global', {}))
        self.LOGGING.update(config_dict.get('logging', {}))
        
        # Map config sections to processing
        if 'preprocessing' in config_dict:
            self.PROCESSING.update(config_dict['preprocessing'])
        if 'feature_engineering' in config_dict:
            self.PROCESSING.update(config_dict['feature_engineering'])
        if 'machine_learning' in config_dict:
            self.PROCESSING.update(config_dict['machine_learning'])
        
        print(f"✅ Configuration loaded: {self.PIPELINE_INFO.get('name', 'Custom Config')}")
        if self.PIPELINE_INFO.get('description'):
            print(f"📝 {self.PIPELINE_INFO['description']}")
        
        return config_dict
    
    def save_config(self, filepath: str):
        """Save current configuration to JSON file"""
        config_dict = {
            'pipeline_info': getattr(self, 'PIPELINE_INFO', {}),
            'global': self.GLOBAL,
            'logging': self.LOGGING,
            'processing': self.PROCESSING,
            'paths': getattr(self, 'PATHS', {}),
            'test_specific': getattr(self, 'TEST_SPECIFIC', {}),
            'environment_info': self.env_info,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

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
        
        # Setup stage-specific logging  
        self.stage_loggers = self.setup_stage_loggers()
        
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
    
    def setup_stage_loggers(self) -> Dict:
        """Setup separate debug loggers for each pipeline stage with comprehensive logging"""
        stage_loggers = {}
        
        # Create unique run ID for correlation across logs
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        stages = {
            'preprocessing': 'EagleFord-Preprocessing',
            'feature_engineering': 'EagleFord-FeatureEng', 
            'model_training': 'EagleFord-MLTraining',
            'validation': 'EagleFord-Validation'
        }
        
        # Create master log aggregator
        master_log_file = self.logs_dir / f"pipeline_master_{self.run_id}.log"
        master_handler = logging.FileHandler(master_log_file)
        master_formatter = logging.Formatter(
            f'%(asctime)s - [RUN:{self.run_id}] - %(name)s - %(levelname)s - %(message)s'
        )
        master_handler.setFormatter(master_formatter)
        
        for stage_name, logger_name in stages.items():
            # Create stage-specific log file
            stage_log_file = self.logs_dir / f"{stage_name}_{self.run_id}.log"
            
            # Create stage logger
            stage_logger = logging.getLogger(logger_name)
            stage_logger.setLevel(logging.DEBUG)  # Always DEBUG for stage loggers
            
            # Clear existing handlers
            stage_logger.handlers.clear()
            
            # Create detailed file handler for stage
            stage_file_handler = logging.FileHandler(stage_log_file)
            stage_formatter = logging.Formatter(
                f'%(asctime)s - [RUN:{self.run_id}] - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
            )
            stage_file_handler.setFormatter(stage_formatter)
            stage_logger.addHandler(stage_file_handler)
            
            # Add to master log aggregator
            stage_logger.addHandler(master_handler)
            
            # Add console handler if enabled globally
            if self.config.LOGGING.get('log_to_console', True):
                stage_console_handler = logging.StreamHandler()
                stage_console_formatter = logging.Formatter(
                    f'[{stage_name.upper()}] %(levelname)s - %(message)s'
                )
                stage_console_handler.setFormatter(stage_console_formatter)
                stage_console_handler.setLevel(logging.INFO)  # Less verbose on console
                stage_logger.addHandler(stage_console_handler)
            
            stage_loggers[stage_name] = {
                'logger': stage_logger,
                'log_file': stage_log_file,
                'run_id': self.run_id
            }
            
            self.logger.info(f"📋 Stage logger created: {stage_name} -> {stage_log_file}")
        
        # Create performance and resource monitoring logger
        perf_log_file = self.logs_dir / f"performance_{self.run_id}.log"
        perf_logger = logging.getLogger('EagleFord-Performance')
        perf_logger.setLevel(logging.INFO)
        perf_logger.handlers.clear()
        
        perf_handler = logging.FileHandler(perf_log_file)
        perf_formatter = logging.Formatter(
            f'%(asctime)s - [RUN:{self.run_id}] - %(name)s - %(levelname)s - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        
        stage_loggers['performance'] = {
            'logger': perf_logger,
            'log_file': perf_log_file,
            'run_id': self.run_id
        }
        
        self.logger.info(f"📋 Master log aggregator: {master_log_file}")
        self.logger.info(f"📊 Performance logger: {perf_log_file}")
        self.logger.info(f"🆔 Pipeline Run ID: {self.run_id}")
        
        return stage_loggers
    
    def log_performance_metrics(self, stage_name: str, metrics: Dict):
        """Log performance metrics for a pipeline stage"""
        try:
            perf_logger = self.stage_loggers['performance']['logger']
            perf_logger.info(f"STAGE:{stage_name} - {json.dumps(metrics, default=str)}")
        except Exception as e:
            self.logger.warning(f"Failed to log performance metrics: {e}")
    
    def get_system_metrics(self) -> Dict:
        """Get current system resource usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory_used_gb': round((memory.total - memory.available) / (1024**3), 2),
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'available_memory_gb': round(memory.available / (1024**3), 2)
            }
        except:
            return {
                'memory_used_gb': 'unknown',
                'memory_percent': 'unknown', 
                'cpu_percent': 'unknown',
                'available_memory_gb': 'unknown'
            }
    
    def stage_timer(self, stage_name: str):
        """Context manager for timing pipeline stages"""
        class StageTimer:
            def __init__(self, pipeline, stage_name):
                self.pipeline = pipeline
                self.stage_name = stage_name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = datetime.now()
                start_metrics = self.pipeline.get_system_metrics()
                start_metrics.update({
                    'stage': self.stage_name,
                    'event': 'stage_start',
                    'timestamp': self.start_time.isoformat()
                })
                self.pipeline.log_performance_metrics(self.stage_name, start_metrics)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = datetime.now()
                duration = (end_time - self.start_time).total_seconds()
                end_metrics = self.pipeline.get_system_metrics()
                end_metrics.update({
                    'stage': self.stage_name,
                    'event': 'stage_complete',
                    'duration_seconds': duration,
                    'start_time': self.start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'success': exc_type is None
                })
                if exc_type is not None:
                    end_metrics['error_type'] = str(exc_type.__name__)
                    end_metrics['error_message'] = str(exc_val)
                    
                self.pipeline.log_performance_metrics(self.stage_name, end_metrics)
                
        return StageTimer(self, stage_name)
    
    def setup_pipeline_components(self):
        """Initialize pipeline components with FIXED imports and error handling"""
        
        # Colab-specific path search
        search_paths = [
            Path("/content/code/src"),
            Path("/content/src"),
            Path("/content")
        ]
        
        # Find correct module path
        module_path_found = False
        for search_path in search_paths:
            if search_path.exists():
                required_files = ["preprocessing_pipeline.py", "feature_engineering.py", "ml_models.py"]
                if all((search_path / file).exists() for file in required_files):
                    if str(search_path) not in sys.path:
                        sys.path.insert(0, str(search_path))
                    self.logger.info(f"✅ Found modules at: {search_path}")
                    module_path_found = True
                    break

        if not module_path_found:
            self.logger.error("❌ Required modules not found")
            raise ImportError("Required pipeline modules not found")

        try:
            # Initialize resource monitor with memory limit
            self.resource_monitor = ColabResourceMonitor(self.logger)

            # Import components
            self.logger.info("📦 Importing pipeline components...")
            from preprocessing_pipeline import EagleFordPreprocessor
            from feature_engineering import EagleFordFeatureEngineer
            from ml_models import EagleFordMLTrainer
            self.logger.info("✅ All imports successful")

            # Initialize components with production-scale optimizations
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
            
            # Apply production configuration optimizations
            self._configure_for_production()

            self.logger.info("✅ All components initialized")

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _configure_for_production(self):
        """Configure pipeline for production-scale full run with GPU acceleration"""
        self.logger.info("⚙️ Applying production configuration...")
        
        # GPU Configuration
        if self.config.GLOBAL.get('enable_gpu', False):
            self.logger.info("🚀 Enabling GPU acceleration for production run")
            
            # Update XGBoost to use GPU
            if hasattr(self.config, 'PROCESSING') and 'xgboost' in self.config.PROCESSING:
                self.config.PROCESSING['xgboost']['tree_method'] = 'gpu_hist'
                self.config.PROCESSING['xgboost']['gpu_id'] = 0
                self.logger.info("✅ XGBoost configured for GPU acceleration")
        
        # Production scale settings
        self.config.GLOBAL['run_mode'] = 'production'
        self.config.GLOBAL['chunk_processing'] = True
        self.config.GLOBAL['max_memory_usage'] = 0.8
        
        # Optimize processing settings for full dataset
        if hasattr(self.config, 'PROCESSING'):
            # Reduce memory pressure with chunked processing
            self.config.PROCESSING['batch_size'] = 20  # Increased for full run
            self.config.PROCESSING['chunk_processing'] = True
            self.config.PROCESSING['chunk_size'] = 10000
            
            # More lenient thresholds for production data
            self.config.PROCESSING['completeness_threshold'] = 80.0
            self.config.PROCESSING['imputation_threshold'] = 35.0
            
            # Enable all features for production
            self.config.PROCESSING['save_intermediate_files'] = True
            self.config.PROCESSING['save_processing_report'] = True
            self.config.PROCESSING['save_quality_metrics'] = True
            
        self.logger.info("✅ Production configuration applied")

    @memory_cleanup_decorator
    def run_preprocessing(self, force_rerun: bool = False) -> bool:
        """Run preprocessing stage with FIXED method calls and enhanced logging"""
        stage_logger = self.stage_loggers['preprocessing']['logger']
        
        if self.pipeline_state['preprocessing_complete'] and not force_rerun:
            self.logger.info("⏭️  Preprocessing already completed, skipping...")
            stage_logger.info("Preprocessing stage skipped - already completed")
            return True
        
        self.logger.info("🔧 STAGE 1: PREPROCESSING")
        self.logger.info("-" * 50)
        
        with self.stage_timer('preprocessing'):
            stage_logger.info("="*60)
            stage_logger.info("EAGLE FORD PREPROCESSING STAGE STARTED")
            stage_logger.info("="*60)
            stage_logger.debug(f"Input directory: {self.input_dir}")
            stage_logger.debug(f"Output directory: {self.output_dir}")
            stage_logger.debug(f"Force rerun: {force_rerun}")
            stage_logger.debug(f"Configuration: {self.config.PROCESSING}")
            # Monitor resources at stage start
            self.resource_monitor.log_resources('preprocessing', 'start')
            
            try:
                # Apply configuration updates that match the actual class attributes
                stage_logger.info("Applying configuration updates to preprocessor")
                stage_logger.debug(f"Preprocessor attributes: {dir(self.preprocessor)}")
                
                if hasattr(self.preprocessor, 'config'):
                    stage_logger.debug(f"Preprocessor config before update: {self.preprocessor.config}")
                    config_updates = {
                        'target_step_size': self.config.PROCESSING['target_step_size'],
                        'gr_constraints': self.config.PROCESSING['gr_constraints'],
                        'completeness_threshold': self.config.PROCESSING['completeness_threshold'],
                        'imputation_threshold': self.config.PROCESSING['imputation_threshold'],
                        'outlier_mad_threshold': self.config.PROCESSING['outlier_mad_threshold'],
                    }
                    
                    # Only update existing config keys
                    updated_keys = []
                    for key, value in config_updates.items():
                        if key in self.preprocessor.config:
                            old_value = self.preprocessor.config[key]
                            self.preprocessor.config[key] = value
                            updated_keys.append(f"{key}: {old_value} -> {value}")
                            stage_logger.debug(f"Updated config {key}: {old_value} -> {value}")
                    
                    stage_logger.info(f"Configuration updates applied: {len(updated_keys)} parameters updated")
                    stage_logger.debug(f"Updated parameters: {updated_keys}")
                else:
                    stage_logger.warning("Preprocessor has no config attribute")
                
                # FIXED METHOD CALL - Use correct method name
                stage_logger.info("Starting preprocessing pipeline execution")
                start_time = datetime.now()
                stage_logger.debug(f"Pipeline start time: {start_time}")
                
                result = False  # Initialize result variable
                try:
                    # Force memory cleanup before processing
                    memory_before = get_memory_usage()
                    self.resource_monitor.check_and_cleanup_if_needed('preprocessing_start')
                    stage_logger.info(f"💾 Memory before preprocessing: {memory_before:.1f}MB")
                    
                    result = self.preprocessor.run_pipeline()  # Correct method name
                    stage_logger.debug(f"Preprocessor run_pipeline() returned: {result}")
                    
                    # Immediate memory cleanup after preprocessing
                    # Don't delete result here as we need it for the if statement
                    objects_freed = force_garbage_collection()
                    memory_after = get_memory_usage()
                    stage_logger.info(f"🧹 Post-processing cleanup: {objects_freed} objects freed")
                    stage_logger.info(f"💾 Memory after preprocessing: {memory_after:.1f}MB (freed: {memory_before - memory_after:.1f}MB)")
                    
                except Exception as run_error:
                    stage_logger.error(f"Error during preprocessor execution: {run_error}")
                    stage_logger.debug("Preprocessing execution failed", exc_info=True)
                    # Cleanup on error
                    force_garbage_collection()
                    raise
                
                end_time = datetime.now()
                stage_logger.debug(f"Pipeline end time: {end_time}")
                
                if result:
                    self.pipeline_state['preprocessing_complete'] = True
                    self.pipeline_state['preprocessing_output'] = self.preprocessor.output_dir
                    stage_logger.info(f"Preprocessing completed successfully")
                    stage_logger.debug(f"Preprocessing output directory: {self.preprocessor.output_dir}")
                    
                    # Update feature engineer path to use actual preprocessing output
                    actual_master_file = Path(self.preprocessor.output_dir) / "master_dataset.csv"
                    stage_logger.debug(f"Checking for master dataset file at: {actual_master_file}")
                    
                    if actual_master_file.exists():
                        old_path = self.feature_engineer.input_file
                        self.feature_engineer.input_file = str(actual_master_file)
                        stage_logger.info(f"Updated feature engineer input path: {old_path} -> {actual_master_file}")
                        self.logger.info(f"📝 Updated feature engineer input path: {actual_master_file}")
                    else:
                        stage_logger.warning(f"Master dataset file not found at expected location: {actual_master_file}")
                    
                    duration = (end_time - start_time).total_seconds()
                    stage_logger.info(f"Preprocessing duration: {duration:.2f} seconds")
                    stage_logger.info("="*60)
                    stage_logger.info("PREPROCESSING STAGE COMPLETED SUCCESSFULLY")
                    stage_logger.info("="*60)
                    
                    self.logger.info(f"✅ Preprocessing completed in {duration:.2f} seconds")
                    self.logger.info(f"📁 Output saved to: {self.pipeline_state['preprocessing_output']}")
                    return True
                else:
                    stage_logger.error("Preprocessing pipeline returned failure status")
                    self.logger.error("❌ Preprocessing failed")
                    return False
                    
            except Exception as e:
                stage_logger.error(f"CRITICAL ERROR in preprocessing stage: {str(e)}")
                stage_logger.error("Full traceback:", exc_info=True)
                
                self.logger.error(f"❌ Preprocessing error: {str(e)}")
                if self.config.LOGGING['level'] == 'DEBUG':
                    import traceback
                    self.logger.debug(traceback.format_exc())
                return False
    
    @memory_cleanup_decorator
    def run_feature_engineering(self, force_rerun: bool = False) -> bool:
        """Run feature engineering stage with FIXED method calls and enhanced logging"""
        stage_logger = self.stage_loggers['feature_engineering']['logger']
        
        if not self.pipeline_state['preprocessing_complete']:
            self.logger.error("❌ Preprocessing must be completed before feature engineering")
            stage_logger.error("Preprocessing not complete - cannot proceed")
            return False
        
        if self.pipeline_state['feature_engineering_complete'] and not force_rerun:
            self.logger.info("⏭️  Feature engineering already completed, skipping...")
            stage_logger.info("Feature engineering stage skipped - already completed")
            return True
        
        self.logger.info("🎛️  STAGE 2: FEATURE ENGINEERING")
        self.logger.info("-" * 50)
        
        with self.stage_timer('feature_engineering'):
            stage_logger.info("="*60)
            stage_logger.info("EAGLE FORD FEATURE ENGINEERING STAGE STARTED")
            stage_logger.info("="*60)

            # Monitor resources and cleanup if needed
            self.resource_monitor.check_and_cleanup_if_needed('feature_engineering')
            self.resource_monitor.log_resources('feature_engineering', 'start')
            
            try:
                # Update input file path to match preprocessing output
                preprocessed_file = Path(self.pipeline_state['preprocessing_output']) / "master_dataset.csv"
                stage_logger.debug(f"Looking for preprocessed file at: {preprocessed_file}")
                
                if preprocessed_file.exists():
                    old_path = self.feature_engineer.input_file
                    self.feature_engineer.input_file = str(preprocessed_file)
                    stage_logger.info(f"Updated input file path: {old_path} -> {preprocessed_file}")
                else:
                    stage_logger.error(f"Preprocessed file not found: {preprocessed_file}")
                    self.logger.error(f"❌ Preprocessed file not found: {preprocessed_file}")
                    return False
                
                # Apply configuration updates that match the actual class attributes
                stage_logger.info("Applying configuration updates to feature engineer")
                stage_logger.debug(f"Feature engineer attributes: {dir(self.feature_engineer)}")
                
                if hasattr(self.feature_engineer, 'config'):
                    stage_logger.debug(f"Feature engineer config before update: {self.feature_engineer.config}")
                    config_updates = {
                        'rolling_windows': self.config.PROCESSING['rolling_windows'],
                        'sequence_length': self.config.PROCESSING['sequence_length'],
                        'target_curves': self.config.PROCESSING['target_curves'],
                        'geological_features': self.config.PROCESSING['geological_features'],
                        'cross_curve_features': self.config.PROCESSING['cross_curve_features'],
                        'depth_normalization': self.config.PROCESSING['depth_normalization'],
                    }
                    
                    # Only update existing config keys
                    updated_keys = []
                    for key, value in config_updates.items():
                        if key in self.feature_engineer.config:
                            old_value = self.feature_engineer.config[key]
                            self.feature_engineer.config[key] = value
                            updated_keys.append(f"{key}: {old_value} -> {value}")
                            stage_logger.debug(f"Updated config {key}: {old_value} -> {value}")
                    
                    stage_logger.info(f"Configuration updates applied: {len(updated_keys)} parameters updated")
                    stage_logger.debug(f"Updated parameters: {updated_keys}")
                else:
                    stage_logger.warning("Feature engineer has no config attribute")
                
                # FIXED METHOD CALL - Use correct method name
                stage_logger.info("Starting feature engineering pipeline execution")
                start_time = datetime.now()
                stage_logger.debug(f"Feature engineering start time: {start_time}")
                
                result = False  # Initialize result variable
                try:
                    # CRITICAL: Force aggressive memory cleanup before feature engineering
                    memory_before = get_memory_usage()
                    stage_logger.info(f"💾 Memory before feature engineering: {memory_before:.1f}MB")
                    
                    # This is where crashes happen - aggressive cleanup
                    if memory_before > 8000:  # If using > 8GB
                        stage_logger.warning(f"🔥 High memory usage detected: {memory_before:.1f}MB - forcing cleanup")
                        cleanup_actions = self.resource_monitor.cleanup_system_memory()
                        stage_logger.info(f"🧹 Emergency cleanup: {cleanup_actions}")
                    
                    result = self.feature_engineer.run_feature_engineering()  # Correct method name
                    stage_logger.debug(f"Feature engineer run_feature_engineering() returned: {result}")
                    
                    # CRITICAL: Immediate cleanup after feature engineering to prevent accumulation
                    # Don't delete result here as we need it for the if statement
                    objects_freed = force_garbage_collection()
                    memory_after = get_memory_usage()
                    stage_logger.info(f"🧹 Post-feature engineering cleanup: {objects_freed} objects freed")
                    stage_logger.info(f"💾 Memory after feature engineering: {memory_after:.1f}MB (freed: {memory_before - memory_after:.1f}MB)")
                    
                except Exception as run_error:
                    stage_logger.error(f"Error during feature engineering execution: {run_error}")
                    stage_logger.debug("Feature engineering execution failed", exc_info=True)
                    # CRITICAL: Cleanup on error to prevent memory leaks
                    force_garbage_collection()
                    raise
                
                end_time = datetime.now()
                stage_logger.debug(f"Feature engineering end time: {end_time}")
                
                if result:
                    self.pipeline_state['feature_engineering_complete'] = True
                    self.pipeline_state['feature_engineering_output'] = self.feature_engineer.output_dir
                    
                    duration = (end_time - start_time).total_seconds()
                    stage_logger.info(f"Feature engineering completed successfully")
                    stage_logger.debug(f"Feature engineering output directory: {self.feature_engineer.output_dir}")
                    stage_logger.info(f"Feature engineering duration: {duration:.2f} seconds")
                    stage_logger.info("="*60)
                    stage_logger.info("FEATURE ENGINEERING STAGE COMPLETED SUCCESSFULLY")
                    stage_logger.info("="*60)
                    
                    self.logger.info(f"✅ Feature engineering completed in {duration:.2f} seconds")
                    self.logger.info(f"📁 Output saved to: {self.pipeline_state['feature_engineering_output']}")
                    return True
                else:
                    stage_logger.error("Feature engineering pipeline returned failure status")
                    self.logger.error("❌ Feature engineering failed")
                    return False
                    
            except Exception as e:
                stage_logger.error(f"CRITICAL ERROR in feature engineering stage: {str(e)}")
                stage_logger.error("Full traceback:", exc_info=True)
                
                self.logger.error(f"❌ Feature engineering error: {str(e)}")
                if self.config.LOGGING['level'] == 'DEBUG':
                    import traceback
                    self.logger.debug(traceback.format_exc())
                return False
    
    @memory_cleanup_decorator
    def run_model_training(self, force_rerun: bool = False) -> bool:
        """Run model training stage with FIXED method calls and enhanced logging"""
        stage_logger = self.stage_loggers['model_training']['logger']
        
        if not self.pipeline_state['feature_engineering_complete']:
            self.logger.error("❌ Feature engineering must be completed before model training")
            stage_logger.error("Feature engineering not complete - cannot proceed")
            return False
        
        if self.pipeline_state['model_training_complete'] and not force_rerun:
            self.logger.info("⏭️  Model training already completed, skipping...")
            stage_logger.info("Model training stage skipped - already completed")
            return True
        
        self.logger.info("🤖 STAGE 3: MODEL TRAINING")
        self.logger.info("-" * 50)
        
        with self.stage_timer('model_training'):
            stage_logger.info("="*60)
            stage_logger.info("EAGLE FORD MODEL TRAINING STAGE STARTED")
            stage_logger.info("="*60)
            self.resource_monitor.check_and_cleanup_if_needed('model_training')
            self.resource_monitor.log_resources('model_training', 'start')
            
            try:
                # Update ML trainer input directory
                feature_output_dir = Path(self.pipeline_state['feature_engineering_output'])
                stage_logger.debug(f"Setting ML trainer input to: {feature_output_dir}")
                
                old_input = self.ml_trainer.input_dir
                self.ml_trainer.input_dir = str(feature_output_dir)
                stage_logger.info(f"Updated ML trainer input directory: {old_input} -> {feature_output_dir}")
                
                # Apply configuration updates that match the actual class attributes
                stage_logger.info("Applying configuration updates to ML trainer")
                stage_logger.debug(f"ML trainer attributes: {dir(self.ml_trainer)}")
                
                if hasattr(self.ml_trainer, 'config'):
                    stage_logger.debug(f"ML trainer config before update: {self.ml_trainer.config}")
                    config_updates = {
                        'search_method': self.config.PROCESSING['search_method'],
                        'n_iter_search': self.config.PROCESSING['n_iter_search'],
                        'cv_folds': self.config.PROCESSING['cv_folds'],
                        'normalization': self.config.PROCESSING['normalization'],
                        'outlier_threshold': self.config.PROCESSING['outlier_threshold'],
                    }
                    
                    # Only update existing config keys
                    updated_keys = []
                    for key, value in config_updates.items():
                        if key in self.ml_trainer.config:
                            old_value = self.ml_trainer.config[key]
                            self.ml_trainer.config[key] = value
                            updated_keys.append(f"{key}: {old_value} -> {value}")
                            stage_logger.debug(f"Updated config {key}: {old_value} -> {value}")
                    
                    stage_logger.info(f"Configuration updates applied: {len(updated_keys)} parameters updated")
                    stage_logger.debug(f"Updated parameters: {updated_keys}")
                else:
                    stage_logger.warning("ML trainer has no config attribute")
                
                # Apply platform-specific optimizations
                stage_logger.info("Applying platform-specific optimizations")
                if self.config.env_info['env_type'] == 'mac_m2':
                    if hasattr(self.ml_trainer, 'enable_test_mode'):
                        self.ml_trainer.enable_test_mode()
                        stage_logger.info("Enabled test mode for Mac M2 platform")
                elif self.config.env_info['memory_gb'] <= 8:
                    if hasattr(self.ml_trainer, 'enable_test_mode'):
                        self.ml_trainer.enable_test_mode()
                        stage_logger.info("Enabled test mode for low memory system")
                
                # Set search method if supported
                if hasattr(self.ml_trainer, 'set_search_method'):
                    self.ml_trainer.set_search_method(self.config.PROCESSING['search_method'])
                    stage_logger.debug(f"Set search method to: {self.config.PROCESSING['search_method']}")
                
                # FIXED METHOD CALL - Use correct method name
                stage_logger.info("Starting ML training pipeline execution")
                start_time = datetime.now()
                stage_logger.debug(f"ML training start time: {start_time}")
                
                result = False  # Initialize result variable
                try:
                    result = self.ml_trainer.run_ml_pipeline()  # Correct method name
                    stage_logger.debug(f"ML trainer run_ml_pipeline() returned: {result}")
                except Exception as run_error:
                    stage_logger.error(f"Error during ML training execution: {run_error}")
                    stage_logger.debug("ML training execution failed", exc_info=True)
                    raise
                
                end_time = datetime.now()
                stage_logger.debug(f"ML training end time: {end_time}")
                
                if result:
                    self.pipeline_state['model_training_complete'] = True
                    self.pipeline_state['model_training_output'] = self.ml_trainer.output_dir
                    
                    duration = (end_time - start_time).total_seconds()
                    stage_logger.info(f"Model training completed successfully")
                    stage_logger.debug(f"Model training output directory: {self.ml_trainer.output_dir}")
                    
                    # Get best model info if available
                    if hasattr(self.ml_trainer, 'results') and 'evaluation' in self.ml_trainer.results:
                        best_model = self.ml_trainer.results.get('evaluation', {}).get('best_model', 'Unknown')
                        stage_logger.info(f"Best performing model: {best_model}")
                    
                    stage_logger.info(f"Model training duration: {duration:.2f} seconds")
                    stage_logger.info("="*60)
                    stage_logger.info("MODEL TRAINING STAGE COMPLETED SUCCESSFULLY")
                    stage_logger.info("="*60)
                    
                    self.logger.info(f"✅ Model training completed in {duration:.2f} seconds")
                    if hasattr(self.ml_trainer, 'results') and 'evaluation' in self.ml_trainer.results:
                        best_model = self.ml_trainer.results.get('evaluation', {}).get('best_model', 'Unknown')
                        self.logger.info(f"🏆 Best model: {best_model.upper()}")
                    self.logger.info(f"📁 Models saved to: {self.pipeline_state['model_training_output']}")
                    return True
                else:
                    stage_logger.error("Model training pipeline returned failure status")
                    self.logger.error("❌ Model training failed")
                    return False
                    
            except Exception as e:
                stage_logger.error(f"CRITICAL ERROR in model training stage: {str(e)}")
                stage_logger.error("Full traceback:", exc_info=True)
                
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
    import sys # Import sys for argument parsing

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
    parser.add_argument("--config", type=str, help="Configuration file path (JSON)")
    parser.add_argument("--mode", choices=['auto', 'test', 'production'], default='auto',
                       help="Pipeline execution mode")
    parser.add_argument("--force-rerun", action='store_true',
                       help="Force rerun of all stages even if completed")
    parser.add_argument("--stage", choices=['preprocessing', 'features', 'models', 'all'],
                       default='all', help="Run specific pipeline stage")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Logging level")

    # Fix: parse only known arguments or arguments explicitly provided
    args, unknown = parser.parse_known_args()

    try:
        # Create enhanced configuration
        config = EagleFordConfig(env_info)
        config.GLOBAL['run_mode'] = args.mode
        config.LOGGING['level'] = args.log_level
        
        # Enable production-scale settings for full runs
        if args.mode == 'production' or (args.mode == 'auto' and env_info.get('has_gpu')):
            print("🚀 Enabling production configuration with GPU acceleration")
            config.GLOBAL['enable_gpu'] = True
            config.GLOBAL['run_mode'] = 'production'
            config.GLOBAL['chunk_processing'] = True
            config.GLOBAL['max_memory_usage'] = 0.85
            
            # Production XGBoost settings
            if 'xgboost' in config.PROCESSING:
                config.PROCESSING['xgboost']['tree_method'] = 'gpu_hist'
                config.PROCESSING['xgboost']['gpu_id'] = 0
                print("✅ GPU acceleration enabled for XGBoost")

        # Load custom configuration if provided
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                config.load_config_file(str(config_path))
                print(f"📄 Loaded configuration from: {config_path}")

                # Override paths from config if not specified in args
                if hasattr(config, 'PATHS'):
                    if not args.input_dir and config.PATHS.get('input_dir'):
                        args.input_dir = config.PATHS['input_dir']
                    if not args.output_dir and config.PATHS.get('output_dir'):
                        args.output_dir = config.PATHS['output_dir']
            else:
                print(f"❌ Configuration file not found: {config_path}")
                return 1

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