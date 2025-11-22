
import os
from typing import Dict, List, Tuple, Optional

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from ray.tune.schedulers import ASHAScheduler
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from sklearn.metrics import cohen_kappa_score,ndcg_score,matthews_corrcoef,average_precision_score,roc_auc_score,fbeta_score,recall_score,f1_score,accuracy_score,precision_score,roc_curve,auc
from torchvision.ops import sigmoid_focal_loss
from src.utils.utils import *
import logging
from src.config import *
from src.utils.utils import compute_optimal_thresholds

logger = logging.getLogger(__name__)


def multi_label_compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute evaluation metrics from model predictions.
    This function is used DURING TRAINING for validation set evaluation.
    It computes optimal thresholds on the validation set.
    """

    logits, labels = eval_pred
    scores = 1 / (1 + np.exp(-logits.squeeze()))
    
    optimal_thresholds = compute_optimal_thresholds(labels, scores, metric="f1", multi_label=True)

    predictions = np.array([scores[:, label_idx] >= optimal_thresholds[label_idx]
                           for label_idx in range(labels.shape[1])]).T.astype(int)

    # Compute metrics with multiple averaging strategies
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    recall_weighted = recall_score(labels, predictions, average="weighted", zero_division=0)
    precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    precision_weighted = precision_score(labels, predictions, average="weighted", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    
    # Compute per-label NDCG and average for multi-label
    ndcg_value = 0.0
    if scores is not None:
        ndcg_per_label = []
        for i in range(labels.shape[1]):
            try:
                ndcg = ndcg_score(labels[:, i].reshape(-1, 1), scores[:, i].reshape(-1, 1))
                ndcg_per_label.append(ndcg)
            except ValueError:
                ndcg_per_label.append(0.0)
        ndcg_value = float(np.mean(ndcg_per_label)) if ndcg_per_label else 0.0

    # Create metrics dict
    metrics = {
        # F1 scores with different averaging strategies
        "f1": f1_weighted,  # Default now uses weighted
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "f2_macro": fbeta_score(labels, predictions, beta=2, zero_division=0, average="macro"),
        "f2_weighted": fbeta_score(labels, predictions, beta=2, zero_division=0, average="weighted"),

        # Other metrics
        "accuracy": accuracy,
        "precision": precision_weighted,  # Default now uses weighted
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall": recall_weighted,  # Default now uses weighted
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,

        # AUC and AP scores
        "roc_auc_macro": roc_auc_score(labels, scores, average="macro") if scores is not None else 0.0,
        "roc_auc_micro": roc_auc_score(labels, scores, average="micro") if scores is not None else 0.0,
        "roc_auc_weighted": roc_auc_score(labels, scores, average="weighted") if scores is not None else 0.0,
        "AP_macro": average_precision_score(labels, scores, average="macro") if scores is not None else 0.0,
        "AP_weighted": average_precision_score(labels, scores, average="weighted") if scores is not None else 0.0,
        "NDCG": ndcg_value,

        # Store optimal thresholds for later use (as array for programmatic access)
        "optim_thresholds": optimal_thresholds,
    }

    # Log individual thresholds as scalars for TensorBoard
    for i, threshold in enumerate(optimal_thresholds):
        metrics[f"optim_threshold_{i}"] = float(threshold)

    return metrics


class LossPlottingCallback(TrainerCallback):
    """Callback to log and plot training and validation losses."""
    def __init__(self):
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1] if state.log_history else {}
        logger.debug(f"log history: {state.log_history}")
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.eval_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.xticks(range(len(self.train_losses)))
        plt.savefig(os.path.join(CONFIG["plot_dir"], "loss_curve.png"))
        plt.close()
        logger.info(f"Loss plot saved to {CONFIG['plot_dir']}/loss_curve.png")


class CustomTrainingArguments(TrainingArguments):
    #TODO : make multi label a mandatory arg
    def __init__(self,loss_type: str,multi_label: Optional[bool] = False,pos_weight: Optional[float | List[float]] = None,alpha: Optional[float] = None,gamma: Optional[float] = None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.loss_type = loss_type
        self.multi_label=multi_label

        if loss_type == "BCE":
            if pos_weight is None:
                # Default pos_weight based on actual class distribution (IAS, SUA, VA)
                # Calculated as num_negative / num_positive for each class
                self.pos_weight = [1.1158, 2.7395, 2.5960]
                logger.info(f"Positive weight parameter set to default (in TrainingArguments) -> pos_weight={self.pos_weight}")
            elif isinstance(pos_weight, list):
                # Use provided list of pos_weights (one per label)
                assert len(pos_weight) == 3, "pos_weight list must have 3 values for [IAS, SUA, VA]"
                self.pos_weight = pos_weight
                logger.info(f"pos_weight value in TrainingArguments (per-label): {self.pos_weight}")
            else:
                # Single value: replicate for all labels
                self.pos_weight = [pos_weight for _ in range(3)]
                logger.info(f"pos_weight value in TrainingArguments (uniform): {self.pos_weight}")
        elif loss_type == "focal":
            self.alpha = alpha if alpha is not None else 0.25  # Default alpha
            self.gamma = gamma if gamma is not None else 2.0  # Default gamma
            logger.info(f"Alpha and gamma values before training (in TrainingArguments): -> alpha={self.alpha} and gamma={self.gamma}")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        

class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs: bool = False,num_items_in_batch=None):
        logger.info(f"Inputs keys : {inputs.keys()}")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self.args.multi_label:
            logits = outputs.logits
        else:
            logits = outputs.logits
        
        if self.args.loss_type == "BCE":
            pos_weight=torch.tensor(self.args.pos_weight,device=self.model.device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(logits, labels.float())
        elif self.args.loss_type == "focal":
            loss = sigmoid_focal_loss(logits, labels.float(), alpha=self.args.alpha, gamma=self.args.gamma, reduction="mean")
        
        return (loss, outputs) if return_outputs else loss
    
class LearningRateCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.scheduler = None  # Stores the scheduler reference

        def on_step_end(self, args, state, control, **kwargs):
            # Capture the scheduler after each step
            self.scheduler = kwargs.get("scheduler", self.scheduler)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and self.scheduler is not None:
                # Get the latest learning rate from the scheduler
                logs["learning_rate"] = self.scheduler.get_last_lr()[0]
        def on_train_begin(self, args, state, control, **kwargs):
            logger.debug(f"STATE at beginning of training: {state}")
            return super().on_train_begin(args, state, control, **kwargs)
        def on_train_end(self, args, state, control, **kwargs):
            logger.debug(f"STATE at ending of training: {state}")
            return super().on_train_end(args, state, control, **kwargs)


class EpochMetricsCallback(TrainerCallback):
    """Callback to track metrics across epochs for learning curve plotting."""
    def __init__(self):
        super().__init__()
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self.eval_metrics: Dict[str, List[float]] = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation at the end of each epoch."""
        if metrics is not None:
            epoch = int(state.epoch) if state.epoch is not None else len(self.epochs)

            # Only record metrics if this is a new epoch
            if epoch not in self.epochs:
                self.epochs.append(epoch)

                # Store eval loss
                if "eval_loss" in metrics:
                    self.eval_losses.append(metrics["eval_loss"])

                # Store other eval metrics
                for key, value in metrics.items():
                    if key.startswith("eval_") and key != "eval_loss" and isinstance(value, (int, float)):
                        if key not in self.eval_metrics:
                            self.eval_metrics[key] = []
                        self.eval_metrics[key].append(value)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs - capture training loss."""
        if logs is not None and "loss" in logs and "epoch" in logs:
            # Only record training loss once per epoch (not every logging step)
            current_epoch = int(logs["epoch"])
            if len(self.train_losses) < current_epoch:
                self.train_losses.append(logs["loss"])


def plot_learning_curves(
    metrics_callback: EpochMetricsCallback,
    plot_dir: str,
    trial_name: str,
    primary_metric: str = "f1_weighted"
) -> None:
    """
    Plot learning curves showing training/validation loss and key metrics across epochs.

    Args:
        metrics_callback: The EpochMetricsCallback instance containing tracked metrics
        plot_dir: Directory to save plots
        trial_name: Name of the trial for plot filename
        primary_metric: The primary metric to plot (e.g., 'f1_weighted', 'accuracy')
    """
    os.makedirs(plot_dir, exist_ok=True)

    epochs = metrics_callback.epochs
    if not epochs:
        logger.warning("No epoch data available for plotting learning curves")
        return

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Training and Validation Loss
    if metrics_callback.train_losses:
        ax1.plot(range(1, len(metrics_callback.train_losses) + 1),
                metrics_callback.train_losses,
                marker='o',
                label='Training Loss',
                linewidth=2)

    if metrics_callback.eval_losses:
        ax1.plot(epochs,
                metrics_callback.eval_losses,
                marker='s',
                label='Validation Loss',
                linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Key Evaluation Metrics
    metric_key = f"eval_{primary_metric}"
    if metric_key in metrics_callback.eval_metrics:
        ax2.plot(epochs,
                metrics_callback.eval_metrics[metric_key],
                marker='o',
                label=primary_metric.replace('_', ' ').title(),
                linewidth=2,
                color='green')

    # Also plot F1 macro if different from primary metric
    if primary_metric != "f1_macro" and "eval_f1_macro" in metrics_callback.eval_metrics:
        ax2.plot(epochs,
                metrics_callback.eval_metrics["eval_f1_macro"],
                marker='s',
                label='F1 Macro',
                linewidth=2,
                color='blue')

    # Also plot ROC AUC if available
    if "eval_roc_auc_weighted" in metrics_callback.eval_metrics:
        ax2.plot(epochs,
                metrics_callback.eval_metrics["eval_roc_auc_weighted"],
                marker='^',
                label='ROC AUC Weighted',
                linewidth=2,
                color='orange')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Metrics Across Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(plot_dir, f"learning_curves_{trial_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Learning curves saved to {plot_path}")

    # Log summary statistics
    if metrics_callback.eval_losses:
        logger.info(f"Final validation loss: {metrics_callback.eval_losses[-1]:.4f}")
        logger.info(f"Best validation loss: {min(metrics_callback.eval_losses):.4f} at epoch {epochs[metrics_callback.eval_losses.index(min(metrics_callback.eval_losses))]}")

    if metric_key in metrics_callback.eval_metrics:
        final_score = metrics_callback.eval_metrics[metric_key][-1]
        best_score = max(metrics_callback.eval_metrics[metric_key])
        best_epoch = epochs[metrics_callback.eval_metrics[metric_key].index(best_score)]
        logger.info(f"Final {primary_metric}: {final_score:.4f}")
        logger.info(f"Best {primary_metric}: {best_score:.4f} at epoch {best_epoch}")

