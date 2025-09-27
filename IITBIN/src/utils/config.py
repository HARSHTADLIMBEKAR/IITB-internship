"""
Configuration management utilities.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf


class Config:
    """
    Configuration class for managing experiment settings.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        self._validate_config()
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    def update(self, other: Dict[str, Any]):
        """Update configuration with other dictionary."""
        self._config.update(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def _validate_config(self):
        """Validate configuration."""
        required_sections = ['data', 'model', 'training']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def save(self, filepath: str):
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from file with optional overrides.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional configuration overrides
        
    Returns:
        Config instance
    """
    config = Config.load(config_path)
    
    if overrides:
        config.update(overrides)
    
    return config


def merge_configs(*configs: Config) -> Config:
    """
    Merge multiple configurations.
    
    Args:
        *configs: Configuration instances to merge
        
    Returns:
        Merged configuration
    """
    merged_dict = {}
    
    for config in configs:
        merged_dict.update(config.to_dict())
    
    return Config(merged_dict)


def create_experiment_config(
    base_config_path: str,
    experiment_name: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Create experiment configuration.
    
    Args:
        base_config_path: Path to base configuration
        experiment_name: Name of experiment
        overrides: Optional configuration overrides
        
    Returns:
        Experiment configuration
    """
    config = load_config(base_config_path, overrides)
    
    # Add experiment metadata
    config['experiment'] = {
        'name': experiment_name,
        'timestamp': str(Path().cwd()),
        'config_path': base_config_path
    }
    
    return config
