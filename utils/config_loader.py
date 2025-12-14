"""Configuration loader for YAML settings."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    global _config
    
    if config_path is None:
        # Default to config/settings.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)
        
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)
        
    return _config


def get_config() -> Dict[str, Any]:
    """
    Get loaded configuration.
    
    Returns:
        Configuration dictionary
        
    Raises:
        RuntimeError: If configuration not loaded
    """
    global _config
    
    if _config is None:
        # Auto-load on first access
        _config = load_config()
        
    return _config


def get_region_name(region_code: str) -> str:
    """
    Convert region code to full name.
    
    Args:
        region_code: 'dc' or 'va'
        
    Returns:
        Full region name
    """
    config = get_config()
    mapping = config.get("region_mapping", {})
    return mapping.get(region_code.lower(), region_code)

