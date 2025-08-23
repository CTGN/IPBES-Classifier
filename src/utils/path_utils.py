"""
Path utilities for IPBES-Classifier
Handles path resolution and configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from src.config import CONFIG

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.absolute()

def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path string, handling environment variables and relative paths.
    
    Args:
        path_str: Path string that may contain environment variables
        base_dir: Base directory for relative paths (defaults to project root)
        
    Returns:
        Resolved Path object
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    # Replace environment variables
    resolved_path = os.path.expandvars(path_str)
    
    # Convert to Path and resolve
    path = Path(resolved_path)
    
    # If it's a relative path, make it relative to base_dir
    if not path.is_absolute():
        path = base_dir / path
    
    return path.resolve()

def get_config_paths() -> Dict[str, Path]:
    """
    Get all configured paths as Path objects.
    
    Returns:
        Dictionary mapping path names to Path objects
    """
    paths = {}
    project_root = get_project_root()
    
    # Base paths
    paths['project_root'] = project_root
    paths['data_dir'] = project_root / 'data'
    paths['results_dir'] = project_root / 'results'
    
    # Data subdirectories
    paths['raw_data_dir'] = project_root / 'data' / 'Raw'
    paths['corpus_dir'] = project_root / 'data' / 'Raw' / 'Corpus'
    paths['positives_dir'] = project_root / 'data' / 'Raw' / 'Positives'
    paths['folds_dir'] = project_root / 'data' / 'folds'
    paths['corpus_output_dir'] = project_root / 'data' / 'corpus'
    
    # Results subdirectories
    paths['models_dir'] = project_root / 'results' / 'models'
    paths['final_model_dir'] = project_root / 'results' / 'final_model'
    paths['ray_results_dir'] = project_root / 'results' / 'ray_results'
    paths['test_preds_dir'] = project_root / 'results' / 'test_preds'
    paths['metrics_dir'] = project_root / 'results' / 'metrics'
    paths['archives_dir'] = project_root / 'results' / 'archives'
    
    # Output directories
    paths['plots_dir'] = project_root / 'plots'
    paths['checkpoints_dir'] = project_root / 'outputs' / 'checkpoints'
    
    # Specific files
    paths['cleaned_dataset'] = project_root / 'data' / 'cleaned_dataset.csv'
    
    return paths

def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
    """
    path.mkdir(parents=True, exist_ok=True)

def ensure_directories(paths: Dict[str, Path]) -> None:
    """
    Ensure all directories in the paths dictionary exist.
    
    Args:
        paths: Dictionary of path names to Path objects
    """
    for name, path in paths.items():
        if path.suffix == '':  # Directory path
            ensure_directory(path)
        else:  # File path, ensure parent directory exists
            ensure_directory(path.parent)

def get_fold_paths(fold_idx: int, run_idx: int, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get paths for a specific fold and run.
    
    Args:
        fold_idx: Fold index
        run_idx: Run index
        base_dir: Base directory for folds (defaults to configured folds_dir)
        
    Returns:
        Dictionary with train, dev, and test fold paths
    """
    if base_dir is None:
        base_dir = Path(CONFIG['folds_dir'])
    
    return {
        'train': base_dir / f'train{fold_idx}_run-{run_idx}.csv',
        'dev': base_dir / f'dev{fold_idx}_run-{run_idx}.csv',
        'test': base_dir / f'test{fold_idx}_run-{run_idx}.csv'
    }

def get_model_output_paths(model_name: str, loss_type: str, fold: int, with_title: bool = False) -> Dict[str, Path]:
    """
    Get output paths for a specific model configuration.
    
    Args:
        model_name: Name of the model
        loss_type: Type of loss function
        fold: Fold index
        with_title: Whether title was used in training
        
    Returns:
        Dictionary with various output paths
    """
    base_results = Path(CONFIG['results_dir'])
    
    # Clean model name for file paths
    clean_model_name = model_name.replace('/', '_')
    
    # Title suffix
    title_suffix = '_with_title' if with_title else ''
    
    return {
        'model': base_results / 'models' / f'{clean_model_name}_{loss_type}{title_suffix}',
        'final_model': Path(CONFIG['final_model_dir']) / f'best_model_cross_val_{loss_type}_{clean_model_name}_fold-{fold+1}',
        'test_preds': base_results / 'test_preds' / f'{clean_model_name}_{loss_type}{title_suffix}_fold_{fold}',
        'metrics': base_results / 'metrics' / f'{clean_model_name}_{loss_type}{title_suffix}_metrics.csv'
    }

def load_paths_config(config_file: str = "configs/paths.yaml") -> Dict[str, Any]:
    """
    Load paths configuration from YAML file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = get_project_root() / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# Initialize paths when module is imported
PATHS = get_config_paths()
ensure_directories(PATHS)
