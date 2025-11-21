"""
ConfigManager for IPBES-Classifier
Handles loading and managing configuration from YAML files with environment-specific overrides
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager that loads and merges YAML configuration files.

    Features:
    - Loads main config.yaml
    - Applies environment-specific overrides from environment.yaml
    - Resolves environment variables in config values
    - Provides dot-notation access to nested config values
    - Automatically resolves paths relative to project root
    """

    def __init__(self, config_dir: Optional[Path] = None, env: Optional[str] = None):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Path to config directory (defaults to project_root/configs)
            env: Environment name (defaults to IPBES_ENV environment variable or 'development')
        """
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_dir = config_dir or (self.project_root / "configs")
        self.env = env or os.getenv('IPBES_ENV', 'development')
        self._config = {}
        self._load_config()

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {filepath}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
            return {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        Override values take precedence over base values.
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _resolve_env_vars(self, value: Any) -> Any:
        """
        Recursively resolve environment variables in config values.
        Supports ${VAR_NAME:-default_value} syntax.
        """
        if isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]
        elif isinstance(value, str):
            # Handle ${VAR:-default} syntax
            if value.startswith('${') and value.endswith('}'):
                var_expr = value[2:-1]  # Remove ${ and }

                if ':-' in var_expr:
                    var_name, default = var_expr.split(':-', 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(var_expr, value)
            return value
        else:
            return value

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve all paths relative to project root.
        Converts relative paths in 'paths' section to absolute paths.
        """
        if 'paths' not in config:
            return config

        resolved_config = deepcopy(config)
        paths = resolved_config['paths']

        for key, value in paths.items():
            if isinstance(value, str) and not value.startswith('/'):
                # Convert relative path to absolute
                paths[key] = str(self.project_root / value)

        return resolved_config

    def _load_config(self):
        """Load and merge all configuration files."""
        # Load main config
        main_config_path = self.config_dir / 'config.yaml'
        self._config = self._load_yaml(main_config_path)

        if not self._config:
            logger.error(f"Failed to load main config from {main_config_path}")
            raise FileNotFoundError(f"Main config file not found: {main_config_path}")

        # Load environment-specific overrides
        env_config_path = self.config_dir / 'environment.yaml'
        env_configs = self._load_yaml(env_config_path)

        if env_configs and self.env in env_configs:
            logger.info(f"Applying {self.env} environment overrides")
            self._config = self._deep_merge(self._config, env_configs[self.env])
        else:
            logger.warning(f"No environment config found for '{self.env}'")

        # Resolve environment variables
        self._config = self._resolve_env_vars(self._config)

        # Resolve paths relative to project root
        self._config = self._resolve_paths(self._config)

        # Ensure directories exist
        self._ensure_directories()

        logger.info(f"Configuration loaded successfully (environment: {self.env})")

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        if 'paths' not in self._config:
            return

        paths_to_create = [
            'data_dir', 'results_dir', 'plots_dir', 'logs_dir',
            'raw_data_dir', 'corpus_dir', 'positives_dir', 'folds_dir',
            'corpus_output_dir', 'models_dir', 'final_model_dir',
            'ray_results_dir', 'test_preds_dir', 'metrics_dir',
            'archives_dir', 'checkpoints_dir'
        ]

        for path_key in paths_to_create:
            if path_key in self._config['paths']:
                path = Path(self._config['paths'][path_key])
                path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key.
        Supports dot notation for nested keys (e.g., 'model.training.learning_rate').

        Args:
            key: Config key (supports dot notation)
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """
        Get config value by key (dictionary-style access).

        Args:
            key: Config key

        Returns:
            Config value

        Raises:
            KeyError: If key not found
        """
        if '.' in key:
            # Support dot notation
            value = self.get(key)
            if value is None:
                raise KeyError(f"Config key not found: {key}")
            return value
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None

    def keys(self):
        """Get all top-level config keys."""
        return self._config.keys()

    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary."""
        return deepcopy(self._config)

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """
        self._config = self._deep_merge(self._config, updates)

    def reload(self):
        """Reload configuration from files."""
        self._config = {}
        self._load_config()

    @property
    def environment(self) -> str:
        """Get current environment."""
        return self.env

    @property
    def project_root(self) -> Path:
        """Get project root path."""
        return self._project_root

    @project_root.setter
    def project_root(self, value: Path):
        """Set project root path."""
        self._project_root = Path(value)

    def __repr__(self) -> str:
        return f"ConfigManager(env='{self.env}', config_dir='{self.config_dir}')"


# Global config instance
_config_instance: Optional[ConfigManager] = None


def get_config(reload: bool = False) -> ConfigManager:
    """
    Get the global ConfigManager instance.

    Args:
        reload: Whether to reload the configuration

    Returns:
        ConfigManager instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = ConfigManager()

    return _config_instance


def reset_config():
    """Reset the global config instance."""
    global _config_instance
    _config_instance = None


if __name__ == "__main__":
    # Test configuration loading
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        config = get_config()
        print(f"\nConfiguration loaded successfully!")
        print(f"Environment: {config.environment}")
        print(f"Project root: {config.project_root}")
        print(f"\nSample values:")
        print(f"  Seed: {config.get('seed')}")
        print(f"  Num labels: {config.get('num_labels')}")
        print(f"  Data dir: {config.get('paths.data_dir')}")
        print(f"  Learning rate: {config.get('model.training.learning_rate')}")
        print(f"  PyAlex email: {config.get('api.pyalex.email')}")
        print(f"\nLabel names: {config.get('datasets.label_names')}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
