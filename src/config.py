"""
Configuration module for IPBES Classifier
Provides backward-compatible CONFIG dict that's backed by the YAML-based ConfigManager
"""

import logging
from pathlib import Path
from src.utils.config_manager import get_config, ConfigManager

logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class ConfigDict:
    """
    Dictionary-like wrapper around ConfigManager for backward compatibility.
    Allows existing code to access config using CONFIG["key"] syntax.
    """

    def __init__(self, config_manager: ConfigManager):
        self._manager = config_manager
        self._key_mapping = self._create_key_mapping()

    def _create_key_mapping(self) -> dict:
        """
        Create a mapping from old flat keys to new nested keys.
        This ensures backward compatibility with existing code.
        """
        return {
            # General settings
            "seed": "seed",
            "num_labels": "num_labels",
            "environment": "environment",
            "project_root": "project_root",

            # Paths
            "data_dir": "paths.data_dir",
            "results_dir": "paths.results_dir",
            "models_dir": "paths.models_dir",
            "final_model_dir": "paths.final_model_dir",
            "ray_results_dir": "paths.ray_results_dir",
            "test_preds_dir": "paths.test_preds_dir",
            "metrics_dir": "paths.metrics_dir",
            "plot_dir": "paths.plots_dir",
            "raw_data_dir": "paths.raw_data_dir",
            "corpus_dir": "paths.corpus_dir",
            "positives_dir": "paths.positives_dir",
            "folds_dir": "paths.folds_dir",
            "cleaned_dataset_path": "paths.cleaned_dataset_path",
            "corpus_output_dir": "paths.corpus_output_dir",
            "checkpoints_dir": "paths.checkpoints_dir",

            # Training args
            "default_training_args": "model.training",

            # API config
            "pyalex_email": "api.pyalex.email",
            "pyalex_max_retries": "api.pyalex.max_retries",
            "pyalex_retry_backoff_factor": "api.pyalex.retry_backoff_factor",

            # Dataset identifiers
            "ipbes_ias_dataset": "datasets.ipbes_ias",
            "ipbes_sua_dataset": "datasets.ipbes_sua",
            "ipbes_va_dataset": "datasets.ipbes_va",

            # Output file names
            "results_csv": "output_files.results_csv",
            "multi_labels_metrics_csv": "output_files.multi_labels_metrics_csv",
            "test_folds_csv": "output_files.test_folds_csv",
        }

    def _resolve_key(self, key: str) -> str:
        """Resolve old key to new nested key."""
        return self._key_mapping.get(key, key)

    def __getitem__(self, key: str):
        """Get config value using dictionary syntax."""
        resolved_key = self._resolve_key(key)
        value = self._manager.get(resolved_key)

        if value is None:
            # Try to get from manager directly in case it's a new-style key
            try:
                value = self._manager[key]
            except KeyError:
                raise KeyError(f"Config key not found: {key}")

        return value

    def __setitem__(self, key: str, value):
        """Set config value (updates the underlying config)."""
        logger.warning(f"Setting config value at runtime: {key}={value}")
        # For now, we don't support setting nested keys directly
        self._manager.update({key: value})

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        resolved_key = self._resolve_key(key)
        return self._manager.get(resolved_key) is not None

    def get(self, key: str, default=None):
        """Get config value with default."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Get all keys (returns old-style flat keys for compatibility)."""
        return self._key_mapping.keys()

    def values(self):
        """Get all values."""
        return [self[key] for key in self.keys()]

    def items(self):
        """Get all key-value pairs."""
        return [(key, self[key]) for key in self.keys()]

    def __repr__(self):
        return f"ConfigDict(environment='{self._manager.environment}')"


# Initialize the global config manager
try:
    _config_manager = get_config()
    _config_manager.project_root = PROJECT_ROOT

    # Create backward-compatible CONFIG dict
    CONFIG = ConfigDict(_config_manager)

    # Also expose the manager directly for new code
    config_manager = _config_manager

    logger.info(f"Configuration loaded successfully (environment: {_config_manager.environment})")

except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    # Fall back to minimal config to prevent complete failure
    CONFIG = {
        "seed": 42,
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(PROJECT_ROOT / "data"),
        "results_dir": str(PROJECT_ROOT / "results"),
    }
    config_manager = None


def ensure_directories():
    """
    Create necessary directories if they don't exist.
    This function is kept for backward compatibility but directories
    are now created automatically by ConfigManager.
    """
    if config_manager:
        # Directories are already created by ConfigManager._ensure_directories()
        logger.debug("Directories already ensured by ConfigManager")
    else:
        # Fallback for if config_manager failed to load
        Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
        Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)


def get_config_manager() -> ConfigManager:
    """
    Get the ConfigManager instance.
    Use this in new code instead of accessing CONFIG directly.

    Returns:
        ConfigManager instance

    Example:
        >>> config = get_config_manager()
        >>> learning_rate = config.get('model.training.learning_rate')
    """
    return _config_manager


def reload_config():
    """Reload configuration from YAML files."""
    global _config_manager, CONFIG, config_manager

    _config_manager = get_config(reload=True)
    _config_manager.project_root = PROJECT_ROOT
    CONFIG = ConfigDict(_config_manager)
    config_manager = _config_manager

    logger.info("Configuration reloaded")


# Export commonly used items
__all__ = ['CONFIG', 'config_manager', 'get_config_manager', 'ensure_directories', 'reload_config', 'PROJECT_ROOT']
