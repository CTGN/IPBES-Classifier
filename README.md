# IPBES-Classifier

A machine learning project focused on classifying scientific publications related to IPBES (Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem Services).

## Overview

This project aims to classify scientific publications into IPBES-relevant categories and identify "super-positives" - highly relevant papers for biodiversity assessment. It supports multi-label classification across three main domains:

- **IAS**: Invasive Alien Species
- **SUA**: Sustainable Use of Animals  
- **VA**: Values Assessment

## Features

- **Multi-Model Support**: Biomedical BERT variants, general-purpose models, and traditional ML approaches
- **Advanced Training Pipeline**: Cross-validation, hyperparameter optimization, early stopping
- **Ensemble Learning**: Combines predictions from multiple models
- **Comprehensive Evaluation**: Rich metrics and visualization tools
- **Configurable Architecture**: Centralized configuration management

## Configuration System

The project now uses a centralized configuration system to eliminate hardcoded paths and make deployment easier across different environments.

### Configuration Files

- **`src/config.py`**: Core configuration with path resolution
- **`configs/paths.yaml`**: Path-specific configurations
- **`configs/environment.yaml`**: Environment-specific settings
- **`configs/train.yaml`**: Training configuration
- **`configs/hpo.yaml`**: Hyperparameter optimization settings

### Environment Support

The system supports multiple deployment environments:

- **Development**: Default local development setup
- **Production**: Production deployment with environment variable overrides
- **Testing**: Testing environment with separate data directories

### Environment Variables

Key environment variables can be set to override default paths:

```bash
export IPBES_ENV=production
export IPBES_DATA_DIR=/data/ipbes
export IPBES_RESULTS_DIR=/results/ipbes
export IPBES_PLOTS_DIR=/plots/ipbes
export PYALEX_EMAIL=your-email@domain.com
```

### Using Configuration

```python
from src.utils.config_loader import get_ipbes_config

# Get configuration for current environment
config = get_ipbes_config()

# Get configuration for specific environment
prod_config = get_ipbes_config('production')

# Access configuration values
data_dir = config['data_dir']
results_dir = config['results_dir']
```

## Project Structure

```
IPBES-Classifier/
├── src/
│   ├── config.py              # Core configuration
│   ├── data_pipeline/         # Data processing pipeline
│   │   └── ipbes/            # IPBES-specific data handling
│   ├── models/                # Model implementations
│   │   └── ipbes/            # IPBES-specific models
│   └── utils/                 # Utility functions
│       ├── path_utils.py      # Path management
│       └── config_loader.py   # Configuration loading
├── configs/                   # Configuration files
├── scripts/                   # Shell scripts
├── experiments/               # Jupyter notebooks
└── results/                   # Model outputs and metrics
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CTGN/IPBES-Classifier.git
cd IPBES-Classifier
```

2. Install dependencies:
```bash
uv sync
```

3. Verify installation:
```bash
uv run python -c "import torch; print('Installation successful!')"
```

## Usage

### Training a Model

```bash
# Hyperparameter optimization
uv run src/models/ipbes/hpo.py \
  --config configs/hpo.yaml \
  --fold 0 \
  --run 0 \
  --n_trials 25 \
  --hpo_metric "eval_roc_auc" \
  -m "bert-base-uncased" \
  --loss "BCE" \
  -t

# Training with best HPO config
uv run src/models/ipbes/train.py \
  --config configs/train.yaml \
  --hp_config configs/best_hpo.yaml \
  --fold 0 \
  --run 0 \
  -m "bert-base-uncased" \
  -bm "eval_roc_auc" \
  --loss "BCE" \
  -t
```

### Running the Full Pipeline

```bash
# Launch the complete IPBES pipeline from pre-processing to testing
./scripts/launch_ipbes_pipeline.sh
```

### Data Preprocessing

```bash
# Preprocess IPBES data
uv run src/data_pipeline/ipbes/preprocess_ipbes.py \
  --balanced \
  --balance_coeff 5 \
  --n_folds 5 \
  --n_runs 2 \
  --seed 42
```

## Model Performance

TODO