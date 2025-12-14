"""Configuration management for Cybersecurity World Model."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from cybersecurity_world_model.exceptions import ConfigurationError


class Config:
    """Configuration manager with YAML file and environment variable support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self._config: Dict[str, Any] = {}
        self._load_defaults()
        
        if config_path:
            self.load_from_file(config_path)
        
        self._load_from_environment()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self._config = {
            'model': {
                'latent_dim': 256,
                'feature_dim': 256,
                'action_dim': 50,
                'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
            },
            'training': {
                'batch_size': 4,
                'sequence_length': 100,
                'epochs': 10,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'checkpoint_dir': 'checkpoints',
                'save_interval': 10,
            },
            'defense': {
                'warning_thresholds': {
                    'high_confidence': 0.85,
                    'medium_confidence': 0.65,
                    'imminent_threat': 0.9,
                    'anomaly_threshold': 0.8,
                },
                'forecast_horizon_hours': 24,
            },
            'logging': {
                'level': 'INFO',
                'log_file': None,
                'log_dir': 'logs',
            },
            'integrations': {
                'siem': {
                    'enabled': False,
                    'endpoint': None,
                },
                'edr': {
                    'enabled': False,
                    'endpoint': None,
                },
                'cloud': {
                    'enabled': False,
                    'endpoint': None,
                },
            },
        }
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        try:
            path = Path(config_path)
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._merge_config(self._config, file_config)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Model settings
        if os.environ.get('CWM_MODEL_DEVICE'):
            self._config['model']['device'] = os.environ['CWM_MODEL_DEVICE']
        
        if os.environ.get('CWM_MODEL_LATENT_DIM'):
            self._config['model']['latent_dim'] = int(os.environ['CWM_MODEL_LATENT_DIM'])
        
        # Training settings
        if os.environ.get('CWM_TRAINING_BATCH_SIZE'):
            self._config['training']['batch_size'] = int(os.environ['CWM_TRAINING_BATCH_SIZE'])
        
        if os.environ.get('CWM_TRAINING_LEARNING_RATE'):
            self._config['training']['learning_rate'] = float(os.environ['CWM_TRAINING_LEARNING_RATE'])
        
        # Logging
        if os.environ.get('CWM_LOG_LEVEL'):
            self._config['logging']['level'] = os.environ['CWM_LOG_LEVEL']
        
        # Integrations
        if os.environ.get('CWM_SIEM_ENDPOINT'):
            self._config['integrations']['siem']['endpoint'] = os.environ['CWM_SIEM_ENDPOINT']
            self._config['integrations']['siem']['enabled'] = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'model.latent_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'model.latent_dim')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def save(self, config_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

