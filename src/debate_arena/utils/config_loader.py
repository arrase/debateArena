import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: The configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    # If path is relative, make it absolute relative to the project root
    # assuming this is running from the project root or we can find it
    if not os.path.isabs(config_path):
        # Fallback for now: assume running from root
        pass
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing configuration file: {e}")
