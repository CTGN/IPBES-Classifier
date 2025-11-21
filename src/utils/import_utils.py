"""
Import utilities for IPBES-Classifier
Handles robust imports that work regardless of execution context
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

def get_project_root() -> Path:
    """Get the project root directory reliably."""
    # Try multiple strategies to find the project root
    current_file = Path(__file__).resolve()
    
    # Strategy 1: Look for the project root from current file location
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists() or (parent / "README.md").exists():
            return parent
    
    # Strategy 2: Use current working directory if it has project files
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() or (cwd / "README.md").exists():
        return cwd
    
    # Strategy 3: Look for environment variable
    env_root = os.getenv("IPBES_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    
    # Strategy 4: Default fallback
    return current_file.parent.parent.parent

def add_src_to_path() -> None:
    """Add the src directory to Python path if not already there."""
    project_root = get_project_root()
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Also add project root to path for absolute imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def safe_import(module_name: str, fallback: Optional[Any] = None) -> Any:
    """
    Safely import a module with fallback handling.
    
    Args:
        module_name: Name of the module to import
        fallback: Fallback value if import fails
        
    Returns:
        Imported module or fallback value
    """
    try:
        # First try to import normally
        module = __import__(module_name)
        return module
    except ImportError:
        # If that fails, try adding src to path and importing again
        try:
            add_src_to_path()
            module = __import__(module_name)
            return module
        except ImportError:
            if fallback is not None:
                return fallback
            raise ImportError(f"Could not import {module_name} even after path adjustment")

def get_config() -> dict:
    """
    Get configuration reliably.

    This function now returns the YAML-based CONFIG that's backed by ConfigManager.
    The CONFIG object is backward-compatible and can be used as a dictionary.

    Returns:
        Configuration dictionary (backed by YAML ConfigManager)
    """
    # Try multiple import strategies to get the YAML-based CONFIG
    try:
        # Strategy 1: Direct import (preferred - uses YAML config system)
        from src.config import CONFIG
        return CONFIG
    except ImportError:
        try:
            # Strategy 2: Add src to path and import
            add_src_to_path()
            from src.config import CONFIG
            return CONFIG
        except ImportError:
            try:
                # Strategy 3: Relative import
                from ..config import CONFIG
                return CONFIG
            except (ImportError, ValueError):
                # Strategy 4: Fallback - create minimal config only if all imports fail
                # This should rarely happen now that we use YAML config system
                import warnings
                warnings.warn("Could not load YAML config system, using fallback minimal config")
                project_root = get_project_root()
                return {
                    "seed": 42,
                    "num_labels": 3,
                    "project_root": str(project_root),
                    "data_dir": str(project_root / "data"),
                    "results_dir": str(project_root / "results"),
                    "models_dir": str(project_root / "results" / "models"),
                    "final_model_dir": str(project_root / "results" / "final_model"),
                    "ray_results_dir": str(project_root / "results" / "ray_results"),
                    "test_preds_dir": str(project_root / "results" / "test_preds"),
                    "metrics_dir": str(project_root / "results" / "metrics"),
                    "plot_dir": str(project_root / "plots"),
                    "raw_data_dir": str(project_root / "data" / "Raw"),
                    "corpus_dir": str(project_root / "data" / "Raw" / "Corpus"),
                    "positives_dir": str(project_root / "data" / "Raw" / "Positives"),
                    "folds_dir": str(project_root / "data" / "folds"),
                    "cleaned_dataset_path": str(project_root / "data" / "cleaned_dataset.csv"),
                    "corpus_output_dir": str(project_root / "data" / "corpus"),
                    "checkpoints_dir": str(project_root / "outputs" / "checkpoints"),
                    "pyalex_email": os.getenv("PYALEX_EMAIL", ""),
                    "pyalex_max_retries": 1,
                    "pyalex_retry_backoff_factor": 0.1,
                    "environment": os.getenv("IPBES_ENV", "development"),
                }

# Initialize path when module is imported
add_src_to_path()
