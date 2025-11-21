import os
import numpy as np
from ray import tune
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import random
import gc
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset, load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,multilabel_confusion_matrix
import evaluate
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from sklearn.metrics import average_precision_score,matthews_corrcoef,ndcg_score,cohen_kappa_score,roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import sys

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    

# Use robust import system for reproducibility
from src.utils.import_utils import get_config
from src.config import *


logger = logging.getLogger(__name__)

def map_name(model_name):
    if model_name == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract":
        return "BiomedBERT-abs"
    elif model_name == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext":
        return "BiomedBERT-abs-ft"
    elif model_name == "FacebookAI/roberta-base":
        return "roberta-base"
    elif model_name == "dmis-lab/biobert-v1.1":
        return "biobert-v1"
    elif model_name == "google-bert/bert-base-uncased":
        return "bert-base"
    else:
        return model_name

def save_dataframe(metric_df, path=None, file_name="results.csv"):
    if path is None:
        path = CONFIG['metrics_dir']
        if metric_df is not None:
            metric_df.to_csv(os.path.join(path, file_name),index=False)
            logger.info(f"Metrics stored successfully at {os.path.join(path, file_name)}")
        else:
            raise ValueError("result_metrics is None. Consider running the model before storing metrics.")

def detailed_metrics(predictions: np.ndarray, labels: np.ndarray, scores=None, label_names=['IAS','SUA','VA']) -> Dict[str, float]:
    """Compute and display detailed metrics including confusion matrix.
    
    Args:
        predictions: Binary predictions array. Shape (n_samples,) for binary or (n_samples, n_labels) for multi-label
        labels: True labels array. Shape (n_samples,) for binary or (n_samples, n_labels) for multi-label  
        scores: Prediction scores/probabilities (optional). Same shape as predictions
        label_names: List of label names for multi-label case (e.g., ["IAS", "SUA", "VA"])
        
    Returns:
        Dictionary containing computed metrics
    """
    # Convert to numpy arrays and ensure correct shape
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    # Determine if this is multi-label (2D arrays) or binary (1D arrays)
    is_multilabel = len(predictions.shape) > 1 and predictions.shape[1] > 1
    
    if is_multilabel:
        return _compute_multilabel_metrics(predictions, labels, scores, label_names)
    else:
        return _compute_binary_metrics(predictions, labels, scores)


def _compute_binary_metrics(predictions: np.ndarray, labels: np.ndarray, scores=None) -> Dict[str, float]:
    """Compute metrics for binary classification."""
    # Ensure 1D arrays for binary case
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    if len(labels.shape) > 1:
        labels = labels.flatten()
        
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot()
    plt.savefig(os.path.join(CONFIG["plot_dir"], "confusion_matrix.png"))
    plt.close()

    metrics = {
        **(evaluate.load("f1").compute(predictions=predictions, references=labels) or {}),
        **(evaluate.load("recall").compute(predictions=predictions, references=labels) or {}),
        **(evaluate.load("precision").compute(predictions=predictions, references=labels) or {}),
        **(evaluate.load("accuracy").compute(predictions=predictions, references=labels) or {}),
        "roc_auc": roc_auc_score(labels, scores) if scores is not None else 0.0,
        "AP": average_precision_score(labels, scores) if scores is not None else 0.0,
        "MCC": matthews_corrcoef(labels, predictions),
        "NDCG": ndcg_score(np.asarray(labels).reshape(1, -1), np.asarray(scores).reshape(1, -1)) if scores is not None else 0.0,
        "kappa": cohen_kappa_score(labels, predictions),
        'TN': tn, 'FP': fp, 'FN': fn, "TP": tp
    }
    
    logger.info(f"Metrics: {metrics}")
    return metrics


def _compute_multilabel_metrics(predictions: np.ndarray, labels: np.ndarray, scores=None, label_names=None) -> Dict[str, float]:
    """Compute metrics for multi-label classification."""
    n_labels = predictions.shape[1]
    
    # Set default label names if not provided
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(n_labels)]
    elif len(label_names) != n_labels:
        logger.warning(f"Number of label names ({len(label_names)}) doesn't match number of labels ({n_labels}). Using default names.")
        label_names = [f"Label_{i}" for i in range(n_labels)]
    
    logger.info(f"Computing multi-label metrics for {n_labels} labels: {label_names}")
    logger.info(f"Labels shape: {labels.shape}, Predictions shape: {predictions.shape}")
    
    # Compute multi-label confusion matrices
    mcm = multilabel_confusion_matrix(labels, predictions)
    
    # Plot multi-label confusion matrix (one subplot per label)
    fig, axes = plt.subplots(1, n_labels, figsize=(6*n_labels, 6))
    if n_labels == 1:
        axes = [axes]
    
    for i, (cm, label_name) in enumerate(zip(mcm, label_names)):
        disp = ConfusionMatrixDisplay(cm, display_labels=[f"Not {label_name}", label_name])
        disp.plot(ax=axes[i])
        axes[i].set_title(f"Confusion Matrix - {label_name}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["plot_dir"], "multilabel_confusion_matrix.png"))
    plt.close()
    
    # Compute per-label metrics
    f1_per_label = f1_score(labels, predictions, average=None, zero_division=0)
    recall_per_label = recall_score(labels, predictions, average=None, zero_division=0)
    precision_per_label = precision_score(labels, predictions, average=None, zero_division=0)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Add per-label metrics
    for i, label_name in enumerate(label_names):
        tn, fp, fn, tp = mcm[i].ravel()
        metrics.update({
            f"f1_{label_name}": float(f1_per_label[i]),
            f"recall_{label_name}": float(recall_per_label[i]),
            f"precision_{label_name}": float(precision_per_label[i]),
            f"TN_{label_name}": int(tn),
            f"FP_{label_name}": int(fp),
            f"FN_{label_name}": int(fn),
            f"TP_{label_name}": int(tp)
        })
    
    # Add aggregate metrics
    metrics.update({
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(labels, predictions, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(labels, predictions, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "MCC": float(matthews_corrcoef(labels.flatten(), predictions.flatten())),
        "kappa": float(cohen_kappa_score(labels.flatten(), predictions.flatten()))
    })
    
    # Add score-based metrics if scores are provided
    if scores is not None:
        scores = np.asarray(scores)
        try:
            # Compute per-label NDCG and average
            ndcg_per_label = []
            for i in range(labels.shape[1]):
                try:
                    # For each label, treat it as a ranking problem per sample
                    ndcg = ndcg_score(labels[:, i].reshape(-1, 1), scores[:, i].reshape(-1, 1))
                    ndcg_per_label.append(ndcg)
                except ValueError:
                    ndcg_per_label.append(0.0)

            metrics.update({
                "roc_auc_macro": float(roc_auc_score(labels, scores, average="macro")),
                "roc_auc_micro": float(roc_auc_score(labels, scores, average="micro")),
                "roc_auc_weighted": float(roc_auc_score(labels, scores, average="weighted")),
                "AP_macro": float(average_precision_score(labels, scores, average="macro")),
                "AP_micro": float(average_precision_score(labels, scores, average="micro")),
                "AP_weighted": float(average_precision_score(labels, scores, average="weighted")),
                "NDCG": float(np.mean(ndcg_per_label)) if ndcg_per_label else 0.0
            })
            
            # Add per-label ROC-AUC and AP
            for i, label_name in enumerate(label_names):
                try:
                    metrics[f"roc_auc_{label_name}"] = float(roc_auc_score(labels[:, i], scores[:, i]))
                    metrics[f"AP_{label_name}"] = float(average_precision_score(labels[:, i], scores[:, i]))
                except ValueError as e:
                    logger.warning(f"Could not compute ROC-AUC/AP for {label_name}: {e}")
                    metrics[f"roc_auc_{label_name}"] = 0.0
                    metrics[f"AP_{label_name}"] = 0.0
                    
        except ValueError as e:
            logger.warning(f"Could not compute score-based metrics: {e}")
            for metric_name in ["roc_auc_macro", "roc_auc_micro", "roc_auc_weighted", 
                               "AP_macro", "AP_micro", "AP_weighted", "NDCG"]:
                metrics[metric_name] = 0.0
    
    logger.info(f"Multi-label metrics computed: {len(metrics)} metrics")
    return metrics

def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds set to {seed}")

def load_datasets(processed: bool = True) -> Tuple[Dataset, Dataset]:
    """Load training, validation, and test datasets from CSV files."""
    train_path = CONFIG["data_paths"]["processed_train"] if processed else CONFIG["data_paths"]["raw_train"]
    train_ds = load_dataset("csv", data_files=train_path, split="train")
    test_ds = load_dataset("csv", data_files=CONFIG["data_paths"]["test"], split="train")
    logger.info(f"Loaded datasets. Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    return train_ds, test_ds

#TODO : check python doc to see what are "*" and "**"
def tokenize_datasets(
    *datasets: Dataset, tokenizer, with_title: bool, with_keywords=False
) -> Tuple[Dataset, ...]:
    """Tokenize one, two, or three datasets."""
    def tokenization(batch: Dict) -> Dict:
        if with_title:
            if with_keywords:
                sep_tok = tokenizer.sep_token or "[SEP]"
                combined = [t + sep_tok + k
                            for t, k in zip(batch["title"], batch["Keywords"])]

                return tokenizer(
                    combined,
                    batch["abstract"],  
                    truncation=True,
                    return_attention_mask=True,
                )
            else:
                return tokenizer(batch["title"], batch["abstract"], truncation=True, max_length=512)
        else:
            if with_keywords:
                return tokenizer(batch["abstract"], batch["Keywords"], truncation=True, max_length=512)
            else:
                return tokenizer(batch["abstract"], truncation=True, max_length=512)

    tokenized_datasets = tuple(
        ds.map(tokenization, batched=True, batch_size=1000, num_proc=os.cpu_count()) for ds in datasets
    )
    logger.info(f"{len(datasets)} datasets tokenized successfully")
    return tokenized_datasets

def compute_optimal_thresholds(y_true, y_scores, metric="f1", multi_label=False):
    """
    Compute optimal thresholds for classification based on a metric.
    This function should be called ONCE on the validation set after training.

    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall', 'kappa')
        multi_label: Whether this is multi-label classification

    Returns:
        Optimal threshold(s) - float for single-label, array for multi-label
    """
    if not multi_label:
        # Single label case
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        metric_scores = []

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            try:
                if metric == "f1":
                    score = f1_score(y_true, y_pred, zero_division=0)
                elif metric == "accuracy":
                    score = accuracy_score(y_true, y_pred)
                elif metric == "precision":
                    score = precision_score(y_true, y_pred, zero_division=0)
                elif metric == "recall":
                    score = recall_score(y_true, y_pred, zero_division=0)
                elif metric == "kappa":
                    score = cohen_kappa_score(y_true, y_pred)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            except ValueError:
                score = 0
            metric_scores.append(score)

        optimal_idx = np.argmax(metric_scores)
        optimal_threshold = thresholds[optimal_idx]

        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.4f}")
        return optimal_threshold

    else:
        # Multi-label case - compute optimal threshold for each label
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        n_labels = y_true.shape[1]
        optimal_thresholds = []

        for label_idx in range(n_labels):
            y_true_label = y_true[:, label_idx]
            y_scores_label = y_scores[:, label_idx]

            # Compute ROC curve for this label
            fpr, tpr, thresholds = roc_curve(y_true_label, y_scores_label)

            # Find optimal threshold based on the specified metric
            metric_scores = []
            for thresh in thresholds:
                y_pred_label = (y_scores_label >= thresh).astype(int)
                try:
                    if metric == "f1":
                        score = f1_score(y_true_label, y_pred_label, zero_division=0)
                    elif metric == "accuracy":
                        score = accuracy_score(y_true_label, y_pred_label)
                    elif metric == "precision":
                        score = precision_score(y_true_label, y_pred_label, zero_division=0)
                    elif metric == "recall":
                        score = recall_score(y_true_label, y_pred_label, zero_division=0)
                    elif metric == "kappa":
                        score = cohen_kappa_score(y_true_label, y_pred_label)
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
                except ValueError:
                    score = 0
                metric_scores.append(score)

            optimal_idx = np.argmax(metric_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_thresholds.append(optimal_threshold)

        optimal_thresholds = np.array(optimal_thresholds)
        logger.info(f"Optimal thresholds for {metric} (per label): {optimal_thresholds}")
        return optimal_thresholds

def plot_roc_curve(y_true, y_scores, logger, plot_dir, data_type=None, metric="eval_f1",store_plot=True,multi_label=False, return_thresholds=True):
    """
    Plot ROC curve and optionally return optimal thresholds.

    Args:
        y_true: True labels
        y_scores: Prediction scores
        logger: Logger instance
        plot_dir: Directory to save plots
        data_type: Type of data ('val', 'test', etc.) for filename
        metric: Metric to optimize for threshold selection
        store_plot: Whether to save the plot
        multi_label: Whether this is multi-label classification
        return_thresholds: Whether to compute and return optimal thresholds (set to False for test set!)

    Returns:
        Optimal threshold(s) if return_thresholds=True, None otherwise
    """

    if not multi_label:
        # Single label case
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute optimal threshold only if requested (for validation set)
        if return_thresholds:
            optimal_threshold = compute_optimal_thresholds(y_true, y_scores, metric=metric, multi_label=False)
        else:
            optimal_threshold = 0.5  # Default for plotting only

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(fpr[::10])
        ax2.set_xticklabels([f'{t:.2f}' for t in thresholds[::10]], rotation=45, fontsize=8)
        ax2.set_xlabel('Thresholds')

        filename = f"roc_curve_{data_type}.png" if data_type else "roc_curve.png"
        if store_plot:
            plt.savefig(os.path.join(plot_dir, filename))
            plt.close()
            logger.info(f"ROC curve saved to {os.path.join(plot_dir, filename)}")
        else:
            plt.show()
            logger.info("ROC curve displayed")

        return optimal_threshold if return_thresholds else None

    else:
        # Multi-label case
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        n_labels = y_true.shape[1]

        # Compute optimal thresholds only if requested (for validation set)
        if return_thresholds:
            optimal_thresholds = compute_optimal_thresholds(y_true, y_scores, metric=metric, multi_label=True)
        else:
            optimal_thresholds = np.array([0.5] * n_labels)  # Default for plotting only
        
        # Create subplots for each label
        fig, axes = plt.subplots(1, n_labels, figsize=(6*n_labels, 6))
        if n_labels == 1:
            axes = [axes]
        
        colors = ['darkorange', 'green', 'red', 'purple', 'brown', 'pink']

        for label_idx in range(n_labels):
            y_true_label = y_true[:, label_idx]
            y_scores_label = y_scores[:, label_idx]

            # Compute ROC curve for this label
            fpr, tpr, thresholds = roc_curve(y_true_label, y_scores_label)
            roc_auc = auc(fpr, tpr)

            # Get the optimal threshold for this label (already computed above)
            optimal_threshold = optimal_thresholds[label_idx]

            # Plot ROC curve for this label
            ax = axes[label_idx]
            color = colors[label_idx % len(colors)]
            ax.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - Label {label_idx}\n(Optimal threshold: {optimal_threshold:.3f})')
            ax.legend(loc="lower right")
            
            # Add threshold information
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            if len(fpr) > 10:
                step = len(fpr) // 10
                ax2.set_xticks(fpr[::step])
                ax2.set_xticklabels([f'{t:.2f}' for t in thresholds[::step]], rotation=45, fontsize=8)
            ax2.set_xlabel('Thresholds')
        
        plt.tight_layout()
        
        filename = f"roc_curve_multilabel_{data_type}.png" if data_type else "roc_curve_multilabel.png"
        if store_plot:
            plt.savefig(os.path.join(plot_dir, filename))
            plt.close()
            logger.info(f"Multi-label ROC curves saved to {os.path.join(plot_dir, filename)}")
            logger.info(f"Optimal thresholds: {optimal_thresholds}")
        else:
            plt.show()
            logger.info("Multi-label ROC curves displayed")
            if return_thresholds:
                logger.info(f"Optimal thresholds: {optimal_thresholds}")

        return optimal_thresholds if return_thresholds else None

def plot_precision_recall_curve(y_true, y_scores,plot_dir,data_type=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
        if data_type is not None:
            plt.savefig(os.path.join(plot_dir, "precision_recall_curve"+data_type+".png"))
            plt.close()
            logger.info(f"Precision-Recall curve saved to {plot_dir}/precision_recall_curve"+data_type+".png")
        else:
            plt.savefig(os.path.join(plot_dir, "precision_recall_curve.png"))
            plt.close()
            logger.info(f"Precision-Recall curve saved to {plot_dir}/precision_recall_curve.png")
    return avg_precision

def visualize_ray_tune_results(analysis, logger, plot_dir=None, metric="eval_recall", mode="max"):
    if plot_dir is None:
        plot_dir = CONFIG['plot_dir']
    """
    Create visualizations of Ray Tune hyperparameter search results.
    
    Args:
        experiment_path: Path to the Ray Tune experiment directory
        metric: Metric to optimize (default: "eval_f1")
        mode: Optimization mode ("max" or "min")
    """
    
    # Load experiment data
    df = analysis.dataframe()
    
    # Get best configuration
    best_trial = analysis.get_best_trial(metric=metric, mode=mode)
    best_config = best_trial.config
    
    # Create plots directory
    os.makedirs(os.path.join(plot_dir, "hyperparams"), exist_ok=True)
    
    # For BCE loss (pos_weight parameters)
    # Check for per-label pos_weights (new format)
    if "pos_weight_ias" in df.columns:
        # Create separate plots for each label
        labels_config = [
            ("pos_weight_ias", "IAS"),
            ("pos_weight_sua", "SUA"),
            ("pos_weight_va", "VA")
        ]

        for param_name, label_name in labels_config:
            if param_name in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(df[param_name], df[metric], alpha=0.7)
                if param_name in best_config:
                    plt.axvline(x=best_config[param_name], color='r', linestyle='--',
                               label=f"Best {param_name}: {best_config[param_name]:.2f}")
                plt.xlabel(f"{param_name}")
                plt.ylabel(f"{metric}")
                plt.title(f"Effect of {param_name} ({label_name}) on {metric}")
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(plot_dir, "hyperparams", f"{param_name}_effect.png"))
                plt.close()
                logger.info(f"{param_name} effect plot saved")

    # For backward compatibility with old single pos_weight format
    elif "pos_weight" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df["pos_weight"], df[metric], alpha=0.7)
        if "pos_weight" in best_config:
            plt.axvline(x=best_config["pos_weight"], color='r', linestyle='--',
                       label=f"Best pos_weight: {best_config['pos_weight']:.2f}")
        plt.xlabel("pos_weight")
        plt.ylabel(f"{metric}")
        plt.title(f"Effect of pos_weight on {metric}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "hyperparams", "pos_weight_effect.png"))
        plt.close()
        logger.info(f"Pos weight effect plot saved")
    
    # For focal loss (alpha and gamma parameters)
    if "alpha" in df.columns and "gamma" in df.columns:
        # 2D scatter plot with colorbar for metric
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df["alpha"], df["gamma"], c=df[metric], 
                             cmap="viridis", s=100, alpha=0.7)
        
        if "alpha" in best_config and "gamma" in best_config:
            plt.scatter([best_config["alpha"]], [best_config["gamma"]], 
                      color='red', s=200, marker='*', 
                      label=f"Best (α={best_config['alpha']:.2f}, γ={best_config['gamma']:.2f})")
            
        plt.colorbar(scatter, label=metric)
        plt.xlabel("Alpha (α)")
        plt.ylabel("Gamma (γ)")
        plt.title(f"Effect of Focal Loss Parameters on {metric}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "hyperparams", "focal_params_effect.png"))
        plt.close()
        logger.info(f"Focal loss parameters effect plot saved")
        
        # 3D surface plot for better visualization
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create a grid for the surface plot
        unique_alphas = sorted(df["alpha"].unique())
        unique_gammas = sorted(df["gamma"].unique())
        
        if len(unique_alphas) > 1 and len(unique_gammas) > 1:
            # Only create surface plot if we have multiple values for both parameters
            X, Y = np.meshgrid(unique_alphas, unique_gammas)
            Z = np.zeros((len(unique_gammas), len(unique_alphas)))
            
            # Fill the grid with metric values
            for i, gamma in enumerate(unique_gammas):
                for j, alpha in enumerate(unique_alphas):
                    subset = df[(df["alpha"] == alpha) & (df["gamma"] == gamma)]
                    if not subset.empty:
                        Z[i, j] = subset[metric].mean()
            
            # Create 3D surface plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
            
            # Mark the best point
            if "alpha" in best_config and "gamma" in best_config:
                best_alpha_idx = unique_alphas.index(best_config["alpha"]) if best_config["alpha"] in unique_alphas else 0
                best_gamma_idx = unique_gammas.index(best_config["gamma"]) if best_config["gamma"] in unique_gammas else 0
                best_z = Z[best_gamma_idx, best_alpha_idx]
                ax.scatter([best_config["alpha"]], [best_config["gamma"]], [best_z], 
                          color='red', s=200, marker='*')
            
            ax.set_xlabel("Alpha (α)")
            ax.set_ylabel("Gamma (γ)")
            ax.set_zlabel(metric)
            ax.set_title(f"3D Surface of Focal Loss Parameters vs {metric}")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=metric)
            plt.savefig(os.path.join(plot_dir, "hyperparams", "focal_params_surface.png"))
            plt.close()
            logger.info(f"3D surface plot for focal loss parameters saved")
    
    # Plot training curves for the best trial if available
    if "training_iteration" in df.columns and metric in df.columns:
        best_trial_df = df[df["trial_id"] == best_trial.trial_id]
        if len(best_trial_df) > 1:  # Only plot if we have multiple iterations
            plt.figure(figsize=(10, 6))
            plt.plot(best_trial_df["training_iteration"], best_trial_df[metric], 
                    marker='o', linestyle='-', linewidth=2)
            plt.xlabel("Training Iteration")
            plt.ylabel(metric)
            plt.title(f"{metric} Progress for Best Trial")
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "hyperparams", "best_trial_progress.png"))
            plt.close()
            logger.info(f"Best trial progress plot saved")
    
    # Plot parallel coordinates for all parameters
    try:
        from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

        if isinstance(analysis, ExperimentAnalysis):
            ax = None
            try:
                import pandas as pd
                from pandas.plotting import parallel_coordinates

                # Convert analysis dataframe to a format suitable for parallel coordinates
                df = analysis.dataframe()
                if not df.empty:
                    param_columns = [col for col in df.columns if col.startswith("config/")]
                    if param_columns:
                        df_params = df[param_columns + [metric]].dropna()
                        df_params = df_params.rename(columns=lambda x: x.replace("config/", ""))
                        df_params["trial_id"] = df["trial_id"]
                        
                        # Normalize metric for better visualization
                        df_params[metric] = (df_params[metric] - df_params[metric].min()) / (df_params[metric].max() - df_params[metric].min())
                        
                        plt.figure(figsize=(12, 6))
                        parallel_coordinates(df_params, class_column="trial_id", colormap=plt.cm.viridis)
                        plt.title(f"Parallel Coordinates Plot for {metric}")
                        plt.xlabel("Parameters")
                        plt.ylabel("Normalized Metric")
                        plt.grid(True)
                        plt.savefig(os.path.join(plot_dir, "hyperparams", "parallel_coordinates.png"))
                        plt.close()
                        logger.info(f"Parallel coordinates plot saved")
                    else:
                        logger.warning("No parameter columns found for parallel coordinates plot")
                else:
                    logger.warning("Analysis dataframe is empty, cannot create parallel coordinates plot")
                if ax:
                    plt.title(f"Parallel Coordinates Plot for {metric}")
                    plt.savefig(os.path.join(plot_dir, "hyperparams", "parallel_coordinates.png"))
                    plt.close()
                    logger.info(f"Parallel coordinates plot saved")
            except Exception as e:
                logger.warning(f"Could not create parallel coordinates plot: {e}")
    except ImportError:
        logger.warning("Could not import ExperimentAnalysis for parallel coordinates plot")

def plot_trial_performance(analysis,logger,plot_dir, metric="eval_recall",file_name="trials_comparison.png"):
    """
    Plot performance across different trials.
    
    Args:
        experiment_path: Path to the Ray Tune experiment directory
        metric: Metric to visualize
    """
    
    # Load experiment data
    df = analysis.dataframe()
    logger.info(f"analysis dataframe : {df.head()}")
    
    if "trial_id" in df.columns and metric in df.columns:
        # Get final results for each trial
        trial_final = df.groupby("trial_id")[metric].last().reset_index()
        trial_final = trial_final.sort_values("trial_id")
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(trial_final)), trial_final[metric],"o", alpha=0.7)
        plt.xlabel("Trial Number")
        plt.ylabel(metric)
        plt.title(f"Final {metric} Score by Trial")
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(plot_dir, "hyperparams", file_name))
        plt.close()
        logger.info(f"Trial comparison plot saved")

def clear_cuda_cache():
    """Clear CUDA cache and log memory usage."""
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Cleared CUDA cache. Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")