# IPBES Classifier Configuration System

This directory contains the YAML-based configuration system for the IPBES Classifier project.

## Overview

The configuration system has been migrated from a hardcoded Python dictionary to a flexible YAML-based system that supports:
- Environment-specific settings (development, production, testing)
- Environment variable resolution
- Automatic path resolution
- Backward compatibility with existing code
- Easy updates without code changes

## Configuration Files

### 1. `config.yaml` (Main Configuration)

The main configuration file that contains all default settings:
- General settings (seed, num_labels)
- Path configurations
- API settings
- Model configurations
- Training parameters
- Data processing settings
- HPO (Hyperparameter Optimization) settings
- Logging configuration

**Example:**
```yaml
seed: 42
num_labels: 3
paths:
  data_dir: data
  results_dir: results
model:
  training:
    learning_rate: 2e-5
    num_train_epochs: 10
```

### 2. `environment.yaml` (Environment Overrides)

Contains environment-specific overrides that modify the main config based on the environment:
- `development`: Fast training, verbose logging, fewer HPO trials
- `production`: Optimized settings, more retries, production paths
- `testing`: Minimal epochs, debug logging, small batches

**Example:**
```yaml
development:
  model:
    training:
      num_train_epochs: 5  # Override to 5 instead of 10
  hpo:
    n_trials: 5  # Fewer trials for faster development
```

### 3. `paths.yaml` (DEPRECATED)

This file has been deprecated. All path configurations are now in `config.yaml`.

## Usage

### Basic Usage (Backward Compatible)

Existing code continues to work without changes:

```python
from src.config import CONFIG

# Access config as dictionary
seed = CONFIG["seed"]
data_dir = CONFIG["data_dir"]
learning_rate = CONFIG["default_training_args"]["learning_rate"]
```

### New Usage (Recommended)

For new code, use the ConfigManager directly:

```python
from src.config import get_config_manager

config = get_config_manager()

# Access using dot notation
seed = config.get('seed')
data_dir = config.get('paths.data_dir')
learning_rate = config.get('model.training.learning_rate')

# With defaults
batch_size = config.get('model.training.per_device_train_batch_size', 16)
```

### Accessing Nested Configuration

The ConfigManager supports dot notation for nested keys:

```python
config = get_config_manager()

# These are equivalent:
learning_rate = config.get('model.training.learning_rate')
# vs
learning_rate = config['model']['training']['learning_rate']
```

## Environment Variables

### Setting the Environment

Set the `IPBES_ENV` environment variable to switch environments:

```bash
# Development (default)
export IPBES_ENV=development

# Production
export IPBES_ENV=production

# Testing
export IPBES_ENV=testing
```

### Required Environment Variables

**`PYALEX_EMAIL`** (Required): Your email for API requests

```bash
export PYALEX_EMAIL="your.email@example.com"
```

### Optional Environment Variables

- `IPBES_DATA_DIR`: Override data directory (production only)
- `IPBES_RESULTS_DIR`: Override results directory (production only)
- `IPBES_PLOTS_DIR`: Override plots directory (production only)
- `IPBES_PROJECT_ROOT`: Override project root detection

## Environment Variable Syntax in YAML

Use `${VAR_NAME:-default_value}` syntax in YAML files:

```yaml
api:
  pyalex:
    email: ${PYALEX_EMAIL:-}  # Empty default, must be set by user

paths:
  data_dir: ${IPBES_DATA_DIR:-data}  # Defaults to 'data' if not set
```

## Configuration Hierarchy

Configurations are merged in this order (later overrides earlier):

1. **Main config** (`config.yaml`) - Base configuration
2. **Environment config** (`environment.yaml`) - Environment-specific overrides
3. **Environment variables** - Runtime overrides via `${VAR:-default}` syntax

## Path Resolution

All paths in the `paths` section are automatically resolved relative to the project root:

```yaml
paths:
  data_dir: data  # Becomes /absolute/path/to/project/data
```

Absolute paths (starting with `/`) are left unchanged:

```yaml
paths:
  data_dir: /var/data/ipbes  # Stays /var/data/ipbes
```

## Adding New Configuration

### 1. Add to `config.yaml`

```yaml
new_feature:
  enabled: true
  threshold: 0.5
```

### 2. Access in Code (New Style)

```python
config = get_config_manager()
if config.get('new_feature.enabled'):
    threshold = config.get('new_feature.threshold')
```

### 3. Add Backward Compatibility (if needed)

If existing code uses `CONFIG["new_feature_enabled"]`, add mapping in `src/config.py`:

```python
# In ConfigDict._create_key_mapping()
return {
    ...
    "new_feature_enabled": "new_feature.enabled",
    ...
}
```

## Environment-Specific Overrides

To add environment-specific behavior:

### 1. Define in `environment.yaml`

```yaml
development:
  new_feature:
    enabled: false  # Disable in development

production:
  new_feature:
    enabled: true
    threshold: 0.8  # Stricter in production
```

### 2. Use in Code

```python
config = get_config_manager()
print(f"Environment: {config.environment}")
enabled = config.get('new_feature.enabled')
# Returns false in development, true in production
```

## Migration Guide

### For Existing Code

No changes needed! The backward-compatible `CONFIG` dict works as before:

```python
from src.config import CONFIG
seed = CONFIG["seed"]  # Still works!
```

### For New Code

Use the ConfigManager for better type safety and dot notation:

```python
from src.config import get_config_manager

config = get_config_manager()
seed = config.get('seed')
```

### Converting Old Code to New Style

**Before:**
```python
from src.config import CONFIG

data_dir = CONFIG["data_dir"]
learning_rate = CONFIG["default_training_args"]["learning_rate"]
```

**After:**
```python
from src.config import get_config_manager

config = get_config_manager()
data_dir = config.get('paths.data_dir')
learning_rate = config.get('model.training.learning_rate')
```

## Troubleshooting

### Config Not Loading

1. Check that `configs/config.yaml` exists
2. Verify YAML syntax (no tabs, proper indentation)
3. Check logs for error messages

### Environment Variable Not Resolved

1. Ensure you're using the correct syntax: `${VAR_NAME:-default}`
2. Check that environment variable is exported: `echo $PYALEX_EMAIL`
3. Restart your shell/IDE after setting environment variables

### Path Not Found

1. Verify the path exists in `config.yaml` under the `paths` section
2. Check that directories are created (ConfigManager creates them automatically)
3. Verify project root is detected correctly:
   ```python
   from src.config import PROJECT_ROOT
   print(PROJECT_ROOT)
   ```

### Backward Compatibility Issues

If old code breaks:

1. Check that key mapping exists in `ConfigDict._create_key_mapping()`
2. Verify the new nested key path is correct
3. Add missing mappings if needed

## Testing Configuration

### Test Config Loading

```bash
python3 -m src.utils.config_manager
```

### Test in Python

```python
from src.config import CONFIG, get_config_manager

# Test backward compatibility
print("Seed:", CONFIG["seed"])
print("Data dir:", CONFIG["data_dir"])

# Test new style
config = get_config_manager()
print("Environment:", config.environment)
print("Learning rate:", config.get('model.training.learning_rate'))
```

### Test Different Environments

```bash
# Test development
IPBES_ENV=development python3 -m src.utils.config_manager

# Test production
IPBES_ENV=production python3 -m src.utils.config_manager

# Test testing
IPBES_ENV=testing python3 -m src.utils.config_manager
```

## Best Practices

1. **Use environment variables for secrets**: Never hardcode emails, API keys, or passwords in YAML files
2. **Keep defaults in `config.yaml`**: Only put overrides in `environment.yaml`
3. **Use dot notation in new code**: `config.get('paths.data_dir')` is clearer than nested dict access
4. **Document new settings**: Add comments in YAML files explaining what each setting does
5. **Test environment-specific settings**: Verify your code works in all environments
6. **Don't modify config at runtime**: Configuration should be read-only after initialization

## Examples

### Example 1: Training Script

```python
from src.config import get_config_manager

config = get_config_manager()

# Get training parameters
learning_rate = config.get('model.training.learning_rate')
num_epochs = config.get('model.training.num_train_epochs')
batch_size = config.get('model.training.per_device_train_batch_size')

# Get paths
model_dir = config.get('paths.models_dir')
data_dir = config.get('paths.data_dir')

print(f"Training with lr={learning_rate}, epochs={num_epochs}")
```

### Example 2: Data Processing

```python
from src.config import CONFIG  # Backward compatible

# Get data processing settings
with_title = CONFIG.get('data_processing', {}).get('with_title', True)
batch_size = CONFIG.get('data_processing', {}).get('batch_size', 100)

# Get paths
folds_dir = CONFIG['folds_dir']
corpus_dir = CONFIG['corpus_dir']
```

### Example 3: API Client

```python
from src.config import get_config_manager
import os

config = get_config_manager()

# Get API settings (email from environment variable)
email = config.get('api.pyalex.email')
if not email:
    raise ValueError("PYALEX_EMAIL environment variable must be set")

max_retries = config.get('api.pyalex.max_retries')
backoff_factor = config.get('api.pyalex.retry_backoff_factor')
```

## Support

For issues or questions about the configuration system:
1. Check this README
2. Review `src/utils/config_manager.py` documentation
3. Open an issue on GitHub
