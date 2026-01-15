import os
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing configuration file: {e}")
