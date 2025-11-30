# IPBES Classifier

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A production-ready multi-label text classifier for identifying IPBES-relevant scientific publications in biodiversity research

The IPBES Classifier automatically categorizes scientific publications across three critical biodiversity assessment domains, enabling researchers to efficiently discover relevant literature from millions of publications.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training & Evaluation](#training--evaluation)
- [Model Performance](#model-performance)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Implementation Details](#implementation-details)
- [Contributing](#contributing)
- [License & Citation](#license--citation)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is IPBES?

The [Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem Services (IPBES)](https://www.ipbes.net/) is an independent intergovernmental body that assesses the state of biodiversity and ecosystem services globally. IPBES assessments synthesize evidence from thousands of scientific publications to inform policy decisions.

### The Problem

Identifying relevant publications from the vast scientific literature (millions of papers) is a critical challenge for IPBES assessments. Manual review is time-consuming, inconsistent, and difficult to scale.

### The Solution

This multi-label classifier automatically predicts the relevance of scientific publications for three IPBES assessment domains:

- **IAS**: Invasive Alien Species Assessment
- **SUA**: Sustainable Use Assessment
- **VA**: Values Assessment

Each publication receives independent binary predictions for all three labels, allowing papers to be relevant to multiple assessments.

### Why It Matters

- **Efficiency**: Automates literature discovery, reducing manual screening time
- **Coverage**: Processes millions of publications to find relevant research
- **Consistency**: Provides reproducible, objective relevance predictions
- **Scalability**: Easily adapts to new assessment domains or updated datasets

---

## Key Features

- **Multi-Label Classification**: Independent predictions for 3 assessment domains (IAS, SUA, VA)
- **State-of-the-Art Models**: Fine-tunes transformer models (BERT, BioBERT, BiomedBERT, RoBERTa)
- **Hyperparameter Optimization**: Automated HPO using Ray Tune with HyperOpt search algorithm
- **Stratified Cross-Validation**: Multi-label stratified k-fold splitting maintains label distributions
- **Ensemble Methods**: Simple averaging of predictions across multiple models
- **Custom Loss Functions**: Binary Cross-Entropy with per-label pos-weights, and Focal Loss
- **Threshold Optimization**: Per-label thresholds optimized on validation set (no data leakage)
- **Comprehensive Metrics**: F1, Precision, Recall, ROC-AUC, Average Precision, MCC, Kappa, NDCG
- **Reproducible Pipeline**: Multi-level seeding and deterministic operations
- **Class Imbalance Handling**: Optimized per-label positive class weights
- **Production Ready**: Complete pipeline from data preprocessing to inference

---

## Architecture

### Data Flow

```
┌─────────────┐
│  Raw Data   │ (IPBES assessments + corpus)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Preprocessing  │ Clean text, create folds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CV Folds      │ Stratified multi-label splits
└────┬───────┬────┘
     │       │
     ▼       ▼
┌─────┐   ┌──────────┐
│ HPO │──>│ Training │ Fine-tune transformers
└─────┘   └────┬─────┘
               │
               ▼
          ┌────────────┐
          │ Evaluation │ Metrics + predictions
          └─────┬──────┘
                │
                ▼
          ┌──────────┐
          │ Ensemble │ Average model scores
          └──────────┘
```

### Directory Structure

```
IPBES-Classifier/
├── configs/                  # YAML configuration files
│   ├── config.yaml          # Main configuration
│   ├── hpo.yaml             # HPO settings
│   ├── train.yaml           # Training settings
│   ├── ensemble.yaml        # Ensemble model list
│   └── best_hpo.yaml        # Auto-generated best params
├── scripts/                  # Orchestration scripts
│   └── launch_ipbes_pipeline.sh  # End-to-end training
├── src/
│   ├── config.py            # Configuration manager
│   ├── data_pipeline/
│   │   └── ipbes/
│   │       ├── preprocess_ipbes.py  # Dataset creation + CV folds
│   │       ├── create_ipbes_raw.py  # Data fetching
│   │       └── fetch.py             # Metadata enrichment
│   └── models/
│       └── ipbes/
│           ├── model_init.py    # CustomTrainer + loss functions
│           ├── hpo.py           # Hyperparameter optimization
│           ├── train.py         # Model training
│           └── ensemble.py      # Ensemble predictions
├── data/                     # Cleaned dataset + CV folds
├── results/                  # Models, metrics, predictions
│   ├── final_model/         # Trained checkpoints
│   ├── metrics/             # Performance metrics
│   └── test_preds/          # Per-fold predictions
└── plots/                    # Visualizations
```

---

## Installation

### System Requirements

- **Python**: 3.11 or higher
- **GPU**: CUDA-compatible GPU recommended (training)
- **Memory**: 16GB+ RAM recommended
- **Storage**: ~5GB for dependencies, ~10GB for models and data

### Install Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) as the package manager for fast, reliable dependency management.

1. **Clone the repository**
   ```bash
   git clone https://github.com/CTGN/IPBES-Classifier.git
   cd IPBES-Classifier
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Verify installation**
   ```bash
   uv run python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"
   ```

### Download Dataset and Pre-trained Models

The dataset and trained model weights are publicly hosted in an S3 bucket: [ipbes-classifier (public)](https://ipbes-classifier.s3.text-analytics.ch/)

**Download everything (Linux/macOS):**

```bash
# Create directories
mkdir -p data results/final_model

# Download pre-trained model checkpoints → results/final_model/
wget -r -np -nH --cut-dirs=1 -R "index.html*" -e robots=off \
  -P results/final_model \
  https://ipbes-classifier.s3.text-analytics.ch/checkpoints/

# Download cleaned dataset + CV folds → data/
wget -r -np -nH --cut-dirs=1 -R "index.html*" -e robots=off \
  -P data \
  https://ipbes-classifier.s3.text-analytics.ch/dataset/
```

After downloading:
- `results/final_model/` contains trained checkpoints for all folds and models
- `data/` contains `cleaned_dataset.csv` and fold indices

---

## Quick Start

### For Researchers: Using Pre-trained Models

Load a trained model and make predictions on new publications:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a trained model (example: BERT fold-0)
model_dir = "results/final_model/best_model_cross_val_BCE_bert-base-uncased_fold-0"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Example publication text
title = "Impact of invasive plant species on biodiversity"
abstract = """This study examines how invasive alien plant species affect
native biodiversity in temperate ecosystems. We found significant negative
impacts on species richness and community composition."""

# Prepare input (title + abstract, as used during training)
text = f"{title} {abstract}"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Get predictions
with torch.no_grad():
    logits = model(**inputs).logits
    scores = torch.sigmoid(logits).numpy().squeeze()

# Interpret results (3 scores for IAS, SUA, VA)
labels = ["IAS", "SUA", "VA"]
for label, score in zip(labels, scores):
    print(f"{label}: {score:.4f} ({'Relevant' if score > 0.5 else 'Not Relevant'})")

# Output:
# IAS: 0.9234 (Relevant)
# SUA: 0.3421 (Not Relevant)
# VA: 0.4102 (Not Relevant)
```

**Note**: Use the optimal thresholds from validation results for better predictions (default 0.5 shown above).

---

## Dataset

### Overview

The dataset contains **7,166 scientific publications** with multi-label annotations:

| Label | Description | Count | Percentage |
|-------|-------------|-------|------------|
| **IAS** | Invasive Alien Species Assessment | ~558 | 7.8% |
| **SUA** | Sustainable Use Assessment | ~315 | 4.4% |
| **VA** | Values Assessment | ~328 | 4.6% |

**Data Format**: CSV with columns:
- `doi`: Digital Object Identifier
- `title`: Publication title
- `abstract`: Publication abstract
- `IAS`, `SUA`, `VA`: Binary labels (0 or 1)
- Metadata: `Publication Year`, `author`, `topics`, etc.

### Cross-Validation Splits

- **5-fold stratified cross-validation**
- **Multi-label stratification**: Maintains label distributions across folds
- **Split proportions**: 70% train, 15% validation, 15% test per fold
- **Fold files**: Stored in `data/folds/` as index CSVs

### Building the Dataset (Optional)

You can regenerate the dataset from raw sources:

```bash
uv run src/data_pipeline/ipbes/preprocess_ipbes.py \
  -nf 5 \      # Number of folds
  -nr 1 \      # Number of runs
  -fm          # Fill missing metadata via CrossRef API
```

This produces:
- `data/cleaned_dataset.csv`: Cleaned publications
- `data/folds/`: Train/dev/test indices for each fold and run

**Environment variable** (for API access):
```bash
export PYALEX_EMAIL=your-email@domain.com
```

---

## Training & Evaluation

### Training Workflow

The training pipeline consists of three stages:

1. **Hyperparameter Optimization (HPO)** → Finds best hyperparameters
2. **Training** → Trains model with optimized parameters
3. **Ensemble** → Combines predictions from multiple models

### 1. Hyperparameter Optimization

Optimizes learning rate, weight decay, and per-label class weights using Ray Tune:

```bash
uv run src/models/ipbes/hpo.py \
  --config configs/hpo.yaml \
  --fold 0 \
  --run 0 \
  --n_trials 30 \
  --hpo_metric eval_AP_weighted \
  --model_name google-bert/bert-base-uncased \
  --loss BCE \
  --with_title
```

**Key Parameters**:
- `--fold`: Cross-validation fold (0-4)
- `--run`: Run index for multiple repetitions
- `--n_trials`: Number of HPO trials (default: 30)
- `--hpo_metric`: Metric to optimize (`eval_AP_weighted`, `eval_f1_macro`, `eval_kappa_weighted`)
- `--model_name`: HuggingFace model identifier
- `--loss`: Loss function (`BCE` or `focal`)
- `--with_title` / `-t`: Include publication title in input
- `--with_keywords` / `-k`: Include keywords (experimental)

**Output**: Best hyperparameters saved to `configs/best_hpo.yaml`

### 2. Model Training

Trains the model using the best hyperparameters from HPO:

```bash
uv run src/models/ipbes/train.py \
  --config configs/train.yaml \
  --hp_config configs/best_hpo.yaml \
  --fold 0 \
  --run 0 \
  -m google-bert/bert-base-uncased \
  -bm eval_AP_weighted \
  --loss BCE \
  -t
```

**Key Parameters**:
- `--hp_config`: Path to best hyperparameters file
- `-m` / `--model_name`: Transformer model to fine-tune
- `-bm` / `--best_metric`: Metric for early stopping
- `--loss`: Loss function (must match HPO)

**Supported Models**:
- `google-bert/bert-base-uncased`
- `dmis-lab/biobert-v1.1`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
- `FacebookAI/roberta-base`

**Output**:
- Trained checkpoint → `results/final_model/best_model_cross_val_{loss}_{model}_fold-{fold}/`
- Metrics → `results/metrics/results.csv`
- Predictions → `results/test_preds/bert/fold-{fold}_...csv`

### 3. Ensemble

Averages prediction scores from multiple models:

```bash
uv run src/models/ipbes/ensemble.py \
  --config configs/ensemble.yaml \
  --fold 0 \
  --run 0 \
  --loss BCE \
  -t
```

Models to ensemble are specified in `configs/ensemble.yaml`.

**Output**: Ensemble predictions → `results/test_preds/bert/fold-{fold}_Ensemble_...csv`

### End-to-End Pipeline

Train all models across all folds with a single script:

```bash
./scripts/launch_ipbes_pipeline.sh
```

This script:
1. Runs HPO for each model (BERT, BioBERT, BiomedBERT, RoBERTa)
2. Trains each model with best hyperparameters
3. Creates ensemble predictions
4. Repeats for all 5 folds

---

## Model Performance

Cross-validation results on the test set (5-fold average):

| Model | F1 (macro) | Precision (macro) | Recall (macro) | ROC-AUC (macro) | AP (weighted) |
|-------|------------|-------------------|----------------|-----------------|---------------|
| **BERT** (base) | 0.854 | 0.879 | 0.853 | 0.950 | 0.937 |
| **BioBERT** v1.1 | 0.866 | 0.894 | 0.858 | 0.973 | 0.963 |
| **BiomedBERT** (abstract) | 0.862 | 0.898 | 0.851 | 0.976 | 0.961 |
| **BiomedBERT** (fulltext) | 0.873 | 0.868 | 0.894 | 0.971 | 0.963 |
| **RoBERTa** (base) | 0.864 | 0.876 | 0.868 | 0.963 | 0.946 |
| **Ensemble** (all models) | 0.755 | 0.705 | 0.873 | 0.940 | 0.905 |

**Per-Label Performance** (BiomedBERT-abstract example):

| Label | F1 | Precision | Recall | ROC-AUC | AP |
|-------|-----|-----------|--------|---------|-----|
| IAS | 0.956 | 0.946 | 0.966 | 0.991 | 0.992 |
| SUA | 0.851 | 0.783 | 0.930 | 0.974 | 0.934 |
| VA | 0.780 | 0.964 | 0.655 | 0.964 | 0.934 |

**Key Insights**:
- **Best overall**: BiomedBERT-fulltext (F1: 0.873, AP: 0.963)
- **Best ROC-AUC**: BiomedBERT-abstract (0.976)
- **IAS detection**: Excellent performance (F1: 0.95+, ROC-AUC: 0.99)
- **SUA/VA detection**: More challenging due to class imbalance (F1: 0.78-0.85)
- **Ensemble**: High recall but lower precision (aggressive predictions)

**Note**: Ensemble performance varies by threshold; shown results use model-specific optimal thresholds.

---

## Outputs

All outputs are organized in the `results/` directory:

### 1. Trained Models (`results/final_model/`)

Saved HuggingFace checkpoints for each fold and model:

```
results/final_model/
├── best_model_cross_val_BCE_bert-base-uncased_fold-0/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ...
├── best_model_cross_val_BCE_bert-base-uncased_fold-1/
└── ...
```

**Total**: 5 folds × 5 models = 25 checkpoint directories

**Usage**: Load with `AutoModelForSequenceClassification.from_pretrained(model_dir)`

### 2. Metrics (`results/metrics/`)

**`results.csv`**: Aggregated metrics for all experiments

Columns include:
- `model_name`, `loss_type`, `fold`, `run`
- Per-label metrics: `f1_IAS`, `precision_IAS`, `recall_IAS`, etc.
- Aggregated metrics: `f1_macro`, `f1_micro`, `f1_weighted`
- Advanced metrics: `ROC-AUC`, `AP`, `MCC`, `NDCG`, `kappa`
- Confusion matrix elements: `TP`, `FP`, `TN`, `FN` (per label)

### 3. Predictions (`results/test_preds/`)

Per-fold prediction files with scores and binary predictions:

```
results/test_preds/bert/
├── fold-0_bert-base_BCE_with_title_run-0_opt_neg-100.csv
├── fold-0_Ensemble_BCE_with_title_run-0_opt_neg-100.csv
└── ...
```

**Columns** (per prediction file):
- Labels: `IAS_label`, `SUA_label`, `VA_label` (ground truth)
- Predictions: `IAS_pred`, `SUA_pred`, `VA_pred` (binary, using optimal thresholds)
- Scores: `IAS_score`, `SUA_score`, `VA_score` (sigmoid probabilities)
- Metadata: `fold`, `title`

### 4. Visualizations (`plots/`)

Generated plots and figures:

- **Loss Evolution**: Training and validation loss curves (`plots/Loss Evolutions/`)
- **ROC Curves**: Per-label ROC curves with AUC scores
- **Hyperparameter Plots**: HPO trial visualizations (`plots/hyperparams/`)

### 5. HPO Results (`results/ray_results/`)

Ray Tune trial logs and checkpoints:
- Full trial history
- Best trial checkpoints
- TensorBoard event files

---

## Configuration

### Configuration System

The project uses a hierarchical YAML-based configuration system managed by `src/config.py`.

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Main configuration (paths, defaults, API settings) |
| `configs/hpo.yaml` | HPO-specific settings (search space, scheduler) |
| `configs/train.yaml` | Training hyperparameters (batch size, epochs, etc.) |
| `configs/best_hpo.yaml` | **Auto-generated** best hyperparameters from HPO |
| `configs/ensemble.yaml` | List of models to ensemble |
| `configs/environment.yaml` | Environment-specific overrides (optional) |

### Key Configuration Sections

**Paths** (`configs/config.yaml`):
```yaml
paths:
  data_dir: data
  results_dir: results
  final_model_dir: results/final_model
  metrics_dir: results/metrics
  test_preds_dir: results/test_preds
  plots_dir: plots
```

**Training Arguments** (`configs/train.yaml`):
```yaml
model:
  training:
    learning_rate: 2e-5
    num_train_epochs: 10
    per_device_train_batch_size: 35
    gradient_accumulation_steps: 4
    warmup_steps: 500
    weight_decay: 0.01
    evaluation_strategy: epoch
    save_strategy: epoch
    load_best_model_at_end: true
    metric_for_best_model: eval_AP_weighted
```

**HPO Settings** (`configs/hpo.yaml`):
```yaml
hpo:
  n_trials: 30
  metric: eval_AP_weighted
  mode: max
  scheduler:
    type: ASHA
    max_t: 10
    grace_period: 1
    reduction_factor: 2
```

### Environment Variables

Optional environment variables:

```bash
# For data fetching (PyAlex API)
export PYALEX_EMAIL=your-email@domain.com

# Environment mode (dev/prod/test)
export IPBES_ENV=prod
```

---

## Advanced Usage

### Custom Model Training

Train with a custom transformer model:

```python
from src.models.ipbes.train import main

# Train with custom settings
main(
    config_path="configs/train.yaml",
    hp_config_path="configs/best_hpo.yaml",
    model_name="allenai/scibert_scivocab_uncased",
    fold=0,
    run=0,
    loss="BCE",
    with_title=True,
    best_metric="eval_f1_macro"
)
```

### Modifying Hyperparameter Search Space

Edit the search space in `src/models/ipbes/hpo.py`:

```python
# For BCE loss
search_space = {
    "learning_rate": tune.loguniform(1e-5, 5e-5),
    "weight_decay": tune.loguniform(1e-6, 1e-4),
    "pos_weight_ias": tune.uniform(0.5, 2.0),
    "pos_weight_sua": tune.uniform(1.5, 4.0),
    "pos_weight_va": tune.uniform(1.5, 4.0),
}
```

### Adding New Models to Ensemble

Edit `configs/ensemble.yaml`:

```yaml
models:
  - google-bert/bert-base-uncased
  - dmis-lab/biobert-v1.1
  - allenai/scibert_scivocab_uncased  # Add new model
```

### Adjusting Cross-Validation Strategy

Modify fold creation in `src/data_pipeline/ipbes/preprocess_ipbes.py`:

```python
# Change number of folds
n_folds = 10  # Default: 5

# Change split proportions
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
```

### Custom Loss Functions

Implement new loss in `src/models/ipbes/model_init.py`:

```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Your custom loss implementation
        pass
```

---

## Implementation Details

### Data Leakage Prevention

**Critical Design**: Optimal thresholds are computed ONLY on the validation set, never on the test set.

```python
# src/models/ipbes/train.py
optimal_thresholds, _ = compute_optimal_thresholds(
    val_true_labels, val_scores,  # Validation set only!
    metric='f1'
)

# Apply thresholds to test set
test_preds = (test_scores >= optimal_thresholds).astype(int)
```

This prevents information leakage from test data into the model evaluation.

### Reproducibility

Multi-level seeding ensures reproducible results:

```python
# Python, NumPy, PyTorch seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transformers library
set_seed(42)

# Training arguments
TrainingArguments(seed=42, data_seed=42, ...)
```

### Class Imbalance Handling

Per-label positive class weights optimized via HPO:

```python
# Example best hyperparameters
pos_weight_ias: 0.584  # IAS is relatively balanced
pos_weight_sua: 3.139  # SUA is more imbalanced (fewer positives)
pos_weight_va: 2.445   # VA is more imbalanced
```

These weights are applied in the BCE loss:

```python
loss = F.binary_cross_entropy_with_logits(
    logits, labels,
    pos_weight=torch.tensor([pos_weight_ias, pos_weight_sua, pos_weight_va])
)
```

### Multi-Label Stratification

Uses `iterstrat.MultilabelStratifiedKFold` to maintain label distributions:

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(mskf.split(X, y)):
    # Each fold maintains similar label distributions
    pass
```

This is crucial for multi-label problems where labels are imbalanced.

### Memory Optimization

- **Gradient Accumulation**: Effective batch size = `per_device_batch_size × gradient_accumulation_steps`
- **Lazy Loading**: Datasets loaded per-fold, then cleared
- **CUDA Cache Clearing**: `torch.cuda.empty_cache()` after each fold
- **FP16 Training**: Mixed precision training (optional)

### Hyperparameter Optimization Details

**Algorithm**: HyperOpt (Tree-structured Parzen Estimator)
- Bayesian optimization for efficient search
- Builds probabilistic model of objective function

**Scheduler**: ASHA (Asynchronous Successive Halving Algorithm)
- Early stopping of unpromising trials
- Aggressive pruning saves computation

**Metrics**: Configurable optimization target
- `eval_AP_weighted`: Weighted average precision (default)
- `eval_f1_macro`: Macro F1 score
- `eval_kappa_weighted`: Weighted Cohen's Kappa

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**: Follow existing code style and conventions
4. **Test your changes**: Ensure existing functionality still works
5. **Commit your changes**: Use clear, descriptive commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**: Describe your changes and motivation

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/IPBES-Classifier.git
cd IPBES-Classifier

# Install dependencies
uv sync

# Make changes and test
uv run python -m pytest  # If tests exist
```

### Code Structure for Developers

- **`src/config.py`**: Configuration management and directory setup
- **`src/data_pipeline/`**: Data fetching, preprocessing, and fold creation
- **`src/models/ipbes/`**: Model training, HPO, and evaluation
- **`src/utils/`**: Utility functions (metrics, plotting, etc.)

---

## License & Citation

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this code or the IPBES Classifier in your research, please cite:

```bibtex
@software{ipbes_classifier,
  title = {IPBES Classifier: Multi-Label Text Classification for Biodiversity Research},
  author = {Your Name/Organization},
  year = {2025},
  url = {https://github.com/CTGN/IPBES-Classifier},
  version = {0.1.0}
}
```

### Contact

For questions, issues, or collaboration opportunities:
- **GitHub Issues**: [https://github.com/CTGN/IPBES-Classifier/issues](https://github.com/CTGN/IPBES-Classifier/issues)
- **Email**: [Your contact email]

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in `configs/train.yaml`:
   ```yaml
   per_device_train_batch_size: 16  # Default: 35
   ```
2. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 8  # Default: 4
   ```
3. Enable gradient checkpointing:
   ```yaml
   gradient_checkpointing: true
   ```

#### Import Errors

**Symptom**: `ModuleNotFoundError` or import issues

**Solutions**:
1. Ensure you're using `uv run`:
   ```bash
   uv run python src/models/ipbes/train.py ...
   ```
2. Verify dependencies are installed:
   ```bash
   uv sync
   ```
3. Check Python version (requires 3.11+):
   ```bash
   python --version
   ```

#### Ray Tune Errors

**Symptom**: HPO crashes or Ray errors

**Solutions**:
1. Check Ray installation:
   ```bash
   uv run python -c "import ray; print(ray.__version__)"
   ```
2. Clear Ray temporary files:
   ```bash
   rm -rf ~/ray_results
   ```
3. Reduce number of parallel trials:
   ```yaml
   hpo:
     num_samples: 10  # Reduce from 30
   ```

#### Low Performance / Poor Metrics

**Possible Causes**:
1. **Insufficient HPO trials**: Increase `n_trials` to 50+
2. **Suboptimal thresholds**: Use validation-optimized thresholds (not 0.5)
3. **Data leakage**: Ensure thresholds computed on validation set only
4. **Class imbalance**: Verify pos_weights are being optimized

#### Memory Leaks During Training

**Solution**: Clear CUDA cache between folds:
```python
import torch
torch.cuda.empty_cache()
```

#### Slow Training

**Solutions**:
1. Enable mixed precision training (FP16):
   ```yaml
   fp16: true
   ```
2. Use larger batch size with gradient accumulation
3. Enable DataLoader workers:
   ```yaml
   dataloader_num_workers: 4
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/CTGN/IPBES-Classifier/issues)
2. **Open a new issue**: Provide error messages, configuration, and system info
3. **Consult documentation**: Review this README and inline code comments

---

**Built with Transformers, Ray Tune, and PyTorch for biodiversity research**
