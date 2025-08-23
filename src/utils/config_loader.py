"""
Configuration loader for IPBES-Classifier
Handles loading and merging of configuration files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_environment_config(env: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment-specific configuration.
    
    Args:
        env: Environment name (development, production, testing)
        
    Returns:
        Environment configuration dictionary
    """
    if env is None:
        env = os.getenv('IPBES_ENV', 'development')
    
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'environment.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {config_path}")
    
    config = load_yaml_config(config_path)
    
    # Get environment-specific config
    env_config = config.get(env, {})
    
    # Merge with common config
    common_config = config.get('common', {})
    
    # Environment-specific config takes precedence
    merged_config = {**common_config, **env_config}
    
    return merged_config

def resolve_environment_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variables in configuration values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with resolved environment variables
    """
    resolved_config = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            resolved_config[key] = resolve_environment_variables(value)
        elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            # Extract environment variable name and default value
            env_var = value[2:-1]  # Remove ${ and }
            
            if ':-' in env_var:
                var_name, default = env_var.split(':-', 1)
                resolved_config[key] = os.getenv(var_name, default)
            else:
                resolved_config[key] = os.getenv(env_var, value)
        else:
            resolved_config[key] = value
    
    return resolved_config

def get_config(env: Optional[str] = None, config_files: Optional[list] = None) -> Dict[str, Any]:
    """
    Get complete configuration with environment-specific overrides.
    
    Args:
        env: Environment name
        config_files: Additional configuration files to load
        
    Returns:
        Complete configuration dictionary
    """
    # Load base environment config
    config = load_environment_config(env)
    
    # Load additional config files if specified
    if config_files:
        for config_file in config_files:
            if Path(config_file).exists():
                additional_config = load_yaml_config(config_file)
                config.update(additional_config)
    
    # Resolve environment variables
    config = resolve_environment_variables(config)
    
    return config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'data_dir',
        'results_dir',
        'seed',
        'pyalex.email'
    ]
    
    for field in required_fields:
        if '.' in field:
            # Nested field
            keys = field.split('.')
            value = config
            for key in keys:
                if key not in value:
                    print(f"Missing required field: {field}")
                    return False
                value = value[key]
        else:
            if field not in config:
                print(f"Missing required field: {field}")
                return False
    
    return True

# Convenience function for getting config
def get_ipbes_config(env: Optional[str] = None) -> Dict[str, Any]:
    """
    Get IPBES-Classifier configuration.
    
    Args:
        env: Environment name
        
    Returns:
        Configuration dictionary
    """
    return get_config(env)

if __name__ == "__main__":
    # Test configuration loading
    config = get_ipbes_config()
    print("Configuration loaded successfully!")
    print(f"Environment: {os.getenv('IPBES_ENV', 'development')}")
    print(f"Data directory: {config.get('data_dir')}")
    print(f"Results directory: {config.get('results_dir')}")
