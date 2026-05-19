"""
Utility modules for Relighty.

Provides:
- split.py: Dataset splitting utility
- config_loader.py: Centralized configuration management
"""

from .split import split_dataset
from .config_loader import get_config, get_data_path, get_checkpoint_path, get_split_path, get_project_root

__all__ = [
    "split_dataset",
    "get_config",
    "get_data_path",
    "get_checkpoint_path",
    "get_split_path",
    "get_project_root",
]