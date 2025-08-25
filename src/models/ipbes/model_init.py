
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

logger = logging.getLogger(__name__)


def multi_label_compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics from model predictions."""
   
    logits, labels = eval_pred
    scores = 1 / (1 + np.exp(-logits.squeeze()))  # Sigmoid
    optimal_thresholds = plot_roc_curve(labels, scores, logger=logger, plot_dir=CONFIG["plot_dir"], data_type="val",metric="f1",multi_label=True)
    predictions = np.array([scores[:, label_idx] >= optimal_thresholds[label_idx] for label_idx in range(labels.shape[1])]).T.astype(int)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0) or {}
    recall = recall_score(labels, predictions, average="macro", zero_division=0) or {}
    accuracy = accuracy_score(labels, predictions) or {}
    precision = precision_score(labels, predictions, average="macro", zero_division=0) or {}
    
    
    return {"f1": f1,
            "f2": fbeta_score(labels, predictions, beta=2, zero_division=0,average="macro"),
            "roc_auc" : roc_auc_score(labels,scores,average="macro") if scores is not None else {},
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "optim_thresholds": optimal_thresholds,
            "AP":average_precision_score(labels,scores,average="macro") if scores is not None else {},
            "NDCG":ndcg_score(np.asarray(labels).reshape(1, -1),scores.reshape(1, -1)) if scores is not None else {}
            }


class LossPlottingCallback(TrainerCallback):
    """Callback to log and plot training and validation losses."""
    def __init__(self):
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1] if state.log_history else {}
        print("log history :",state.log_history)
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
    def __init__(self,loss_type: str,multi_label: Optional[bool] = False,pos_weight: Optional[float] = None,alpha: Optional[float] = None,gamma: Optional[float] = None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.loss_type = loss_type
        self.multi_label=multi_label

        if loss_type == "BCE":
            if pos_weight is None:
                # Calculate default pos_weight if not provided
                self.pos_weight = torch.tensor([5.0, 5.0, 5.0]) 
                logger.info(f"Positive weight parameter set to default (in TrainingArguments) -> pos_weight={self.pos_weight}")
            else:
                #TODO : Make this dynamic in case  we want to add labels
                self.pos_weight =[pos_weight for _ in range(3)]
                logger.info(f"pos_weight value in TrainingArguments : {self.pos_weight}")
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
            print("STATE at beggining of training : ",state)
            return super().on_train_begin(args, state, control, **kwargs)
        def on_train_end(self, args, state, control, **kwargs):
            print("STATE at ending of training  : ",state)
            return super().on_train_end(args, state, control, **kwargs)
        
