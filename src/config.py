import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

CONFIG = {
    "seed": 42,
    
    # Directory paths
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(PROJECT_ROOT / "data"),
    "results_dir": str(PROJECT_ROOT / "results"),
    "models_dir": str(PROJECT_ROOT / "results" / "models"),
    "final_model_dir": str(PROJECT_ROOT / "results" / "final_model"),
    "ray_results_dir": str(PROJECT_ROOT / "results" / "ray_results"),
    "test_preds_dir": str(PROJECT_ROOT / "results" / "test_preds"),
    "metrics_dir": str(PROJECT_ROOT / "results" / "metrics"),
    "plots_dir": str(PROJECT_ROOT / "plots"),
    
    # Data subdirectories
    "raw_data_dir": str(PROJECT_ROOT / "data" / "Raw"),
    "corpus_dir": str(PROJECT_ROOT / "data" / "Raw" / "Corpus"),
    "positives_dir": str(PROJECT_ROOT / "data" / "Raw" / "Positives"),
    "folds_dir": str(PROJECT_ROOT / "data" / "folds"),
    "cleaned_dataset_path": str(PROJECT_ROOT / "data" / "cleaned_dataset.csv"),
    "corpus_output_dir": str(PROJECT_ROOT / "data" / "corpus"),
    
    # Model configuration
    "num_labels": 1,
    "default_training_args": {
        "save_total_limit": 1,
        "learning_rate": None,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "fp16": False,
        "logging_strategy": "epoch",
        "report_to": "tensorboard",
    },
    
    # API configuration
    "pyalex_email": "leandre.catogni@hesge.ch",
    "pyalex_max_retries": 1,
    "pyalex_retry_backoff_factor": 0.1,
    
    # File paths for specific datasets
    "ipbes_ias_dataset": "IPBES IAS_2352922",
    "ipbes_sua_dataset": "IPBES SUA_2344805", 
    "ipbes_va_dataset": "IPBES VA_2345372",
    
    # Output file names
    "results_csv": "results.csv",
    "multi_labels_metrics_csv": "multi_labels_metrics.csv",
    "test_folds_csv": "test_folds.csv",
    
    # Environment-specific overrides
    "environment": os.getenv("IPBES_ENV", "development"),
}

# Environment-specific overrides
if CONFIG["environment"] == "production":
    # Production paths can be different
    pass
elif CONFIG["environment"] == "development":
    # Development paths
    pass

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        CONFIG["data_dir"],
        CONFIG["results_dir"],
        CONFIG["models_dir"],
        CONFIG["final_model_dir"],
        CONFIG["ray_results_dir"],
        CONFIG["test_preds_dir"],
        CONFIG["metrics_dir"],
        CONFIG["plots_dir"],
        CONFIG["raw_data_dir"],
        CONFIG["corpus_dir"],
        CONFIG["positives_dir"],
        CONFIG["folds_dir"],
        CONFIG["corpus_output_dir"],
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories
ensure_directories()
