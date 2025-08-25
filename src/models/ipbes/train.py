import logging
import os
from typing import *
import argparse
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from ray.tune.schedulers import ASHAScheduler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import transformers
import sys
import datasets
from time import perf_counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import yaml
import pandas as pd

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Use robust import system for reproducibility
from src.utils.import_utils import get_config

# Get configuration reliably
CONFIG = get_config()

from src.models.ipbes.HPO_callbacks import CleanupCallback
from src.utils import *
from src.models.ipbes.model_init import *

from src.config import *

logger = logging.getLogger(__name__)

def set_reproducibility(seed):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Randomness sources seeded with {seed} for reproducibility.")

    set_random_seeds(CONFIG["seed"])
    set_seed(CONFIG["seed"])

#TODO : Add some variables to the config file and lnk them to here from the config (ex: Early Stopping patience)
#TODO : Make the paths reproducible


def parse_args():
    parser = argparse.ArgumentParser(description="Run HPO for our BERT classifier")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (e.g. configs/hpo.yaml)"
    )
    parser.add_argument(
        "--hp_config",
        type=str,
        required=True,
        help="Path to the YAML hyperparameters config file (e.g. configs/hpo.yaml)"
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        help="Which CV fold to run (overrides config)"
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        help="Which CV fold to run (overrides config)"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        default=[0,1,2],
        help="CUDA_VISIBLE_DEVICES string (e.g. '0,1')"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-on",
        "--nb_opt_negs",
        type=int,
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        help="Type of loss to use during training"
    )
    parser.add_argument(
        "-t",
        "--with_title",
        action="store_true",
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-k",
        "--with_keywords",
        action="store_true",
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-bm",
        "--best_metric",
        type=str,
        help="Type of loss to use during training"
    )
    
    return parser.parse_args()

def train(cfg,hp_cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True)
    
            train_split = load_dataset("csv", data_files=f"{CONFIG['folds_dir']}/train{cfg['fold']}_run-{cfg['run']}.csv",split="train")
        dev_split = load_dataset("csv", data_files=f"{CONFIG['folds_dir']}/dev{cfg['fold']}_run-{cfg['fold']}.csv",split="train")
        test_split = load_dataset("csv", data_files=f"{CONFIG['folds_dir']}/test{cfg['fold']}_run-{cfg['run']}.csv",split="train")

    logger.info(f"train split size : {len(train_split)}")
    logger.info(f"dev split size : {len(dev_split)}")
    logger.info(f"test split size : {len(test_split)}")
    
    
    tokenized_train,tokenized_dev, tokenized_test = tokenize_datasets(train_split,dev_split,test_split, tokenizer=tokenizer,with_title=cfg['with_title'],with_keywords=cfg['with_keywords'])

    
    #TODO : Check Julien's article about how to implement that (ask him about the threholding optimization)
    logger.info(f"Final training...")
    start_time=perf_counter()
    model=AutoModelForSequenceClassification.from_pretrained(cfg['model_name'], num_labels=CONFIG["num_labels"])
    #model.gradient_checkpointing_enable()

    batch_size=25

    # Set up training arguments
    training_args = CustomTrainingArguments(
            output_dir=CONFIG['models_dir'],
            seed=CONFIG["seed"],
            data_seed=CONFIG["seed"],
            **CONFIG["default_training_args"],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            loss_type=cfg['loss_type'],
            metric_for_best_model=cfg['best_metric'],
            load_best_model_at_end=True,
            save_strategy= "steps",
            eval_strategy="steps",
        )
    
    training_args.pos_weight = hp_cfg["pos_weight"] if cfg['loss_type'] == "BCE" else None
    training_args.alpha = hp_cfg["alpha"] if cfg['loss_type'] == "focal" else None
    training_args.gamma = hp_cfg["gamma"] if cfg['loss_type'] == "focal" else None
    training_args.learning_rate = hp_cfg["learning_rate"]
    training_args.num_train_epochs = 10
    

    class CustomEarlyStoppingCallback(EarlyStoppingCallback):
        def on_train_end(self, args, state, control, **kwargs):
            if state.best_model_checkpoint:
                logger.info(f"Early stopping triggered. Best model checkpoint saved at: {state.best_model_checkpoint}")
            else:
                logger.info("Early stopping triggered, but no best model checkpoint was saved.")

    early_stopping_callback = CustomEarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
    )

    logger.info(f"training size : {len(train_split)}")
    logger.info(f"dev size : {len(dev_split)}")
    logger.info(f"test size : {len(test_split)}")

    metrics = trainer.train().metrics

    eval_results_dev=trainer.evaluate()

    log_history = trainer.state.log_history
    train_logs = [log for log in log_history if 'loss' in log and 'epoch' in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log and 'epoch' in log]

    if train_logs and eval_logs:
        train_epochs = [log['epoch'] for log in train_logs]
        train_loss = [log['loss'] for log in train_logs]
        eval_epochs = [log['epoch'] for log in eval_logs]
        eval_loss = [log['eval_loss'] for log in eval_logs]

        plt.figure(figsize=(12, 6))
        plt.plot(train_epochs, train_loss, 'o-', label='Training Loss')
        plt.plot(eval_epochs, eval_loss, 'o-', label='Validation Loss')
        plt.title(f"Loss Evolution\nModel: {map_name(cfg['model_name'])}, Loss: {cfg['loss_type']}, Fold: {cfg['fold']}, Run: {cfg['run']}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plot_filename = f"loss_evolution_{map_name(cfg['model_name'])}_loss-{cfg['loss_type']}_fold-{cfg['fold']}_run-{cfg['run']}.png"
        plot_path = os.path.join(CONFIG["plot_dir"], "Loss Evolutions",plot_filename)
        plt.savefig(plot_path)
        logger.info(f"Loss evolution plot saved at {plot_path}")
        plt.close()

    end_time_train=perf_counter()
    logger.info(f"Training time : {end_time_train-start_time}")

    eval_results_test = trainer.evaluate(tokenized_test)

    end_time_val=perf_counter()
    logger.info(f"Evaluation time : {end_time_val-end_time_train}")
    logger.info(f"Evaluation results on test set: {eval_results_test}")
    # Number of optimizer updates performed:

    n_updates = trainer.state.global_step
    logger.info(f"Total updates (optimizer steps): {n_updates}")
    n_steps = metrics["train_steps_per_second"] * metrics["train_runtime"]
    avg_step_time = metrics["train_runtime"] / n_steps
    logger.info(f"Avg time / step: {avg_step_time:.3f}s")
    

            final_model_path = os.path.join(CONFIG['final_model_dir'], "best_model_cross_val_"+str(cfg['loss_type'])+"_" +str(map_name(cfg['model_name'])) + "_fold-"+str(cfg['fold']+1))
    
    trainer.save_model(final_model_path)
    logger.info(f"Best model saved to {final_model_path}")

    results=[]
    logger.info(f"On test Set (with threshold 0.5) : ")
    # Compute detailed metrics
    predictions = trainer.predict(tokenized_test)
    
    scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
    preds = (scores > 0.5).astype(int)
    logger.info(f"Raw predictions shape: {predictions.predictions.shape}")
    logger.info(f"Raw predictions: {predictions.predictions[:10]}")  # First 10 predictions
    logger.info(f"Scores shape: {scores.shape}")
    logger.info(f"Scores: {scores[:10]}")  # First 10 scores
    logger.info(f"Preds shape: {preds.shape}")
    logger.info(f"Preds: {preds[:10]}")  # First 10 predictions
    logger.info(f"Test labels shape: {np.asarray(test_split['labels']).shape}")
    logger.info(f"Test labels: {test_split['labels'][:10]}")  # First 10 labels
    logger.info(f"Test split: {test_split}")

    logger.info(f"Unique values in predictions: {np.unique(preds)}")
    logger.info(f"Unique values in labels: {np.unique(test_split['labels'])}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(test_split['labels'], preds)}")
    res1=detailed_metrics(preds, test_split["labels"],scores=scores)


    plot_roc_curve(test_split["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
    plot_precision_recall_curve(test_split["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

    #! The following seems weird. we are talking about decision here. View it like a ranking problem. take a perspective for usage
    threshold = eval_results_dev["eval_optim_threshold"]
    logger.info(f"\nOn test Set (New optimal threshold of {threshold} according to the dev set): ")
    preds = (scores > threshold).astype(int)
    res2=detailed_metrics(preds, test_split["labels"],scores=scores)
    plot_precision_recall_curve(test_split["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

    logger.info(f"Results for fold {cfg['fold']+1} with optim_threshold learned from dev set : {res2}")

    clear_cuda_cache()

            result_metrics_path=os.path.join(CONFIG['metrics_dir'], "results.csv")

    if os.path.isfile(result_metrics_path):
        result_metrics=pd.read_csv(result_metrics_path)
    else:
        result_metrics=pd.DataFrame(columns=["model_name", "loss_type","fold", "run", "with_title", "with_keywords","nb_added_negs"])

    #We update the results dataframe
    result_metrics = pd.concat([
        result_metrics,
        pd.DataFrame([{
            "model_name": map_name(cfg["model_name"]),
            "loss_type": cfg['loss_type'],
            "fold": cfg['fold']+1,
            "run": cfg['run']+1, 
            "with_title": cfg['with_title'],
            "with_keywords":cfg['with_keywords'],
            "nb_added_negs": cfg['nb_optional_negs'],
            "num_trials": hp_cfg['num_trials'],
            "hpo_metric": hp_cfg['hpo_metric'],
            **res2
        }])
    ], ignore_index=True)
    
    save_dataframe(result_metrics)

    fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"prediction":preds,'score':scores,"fold":[cfg['fold'] for _ in range(len(preds))],"title":test_split['title'] })
        test_preds_path=os.path.join(CONFIG['test_preds_dir'], f"fold_{cfg['fold']}_{map_name(os.path.basename(cfg['model_name']))}_{cfg['loss_type']}{'_with_title' if cfg['with_title'] else ''}{'_with_keywords' if cfg['with_keywords'] else ''}_run-{cfg['run']}_opt_neg-{cfg['nb_optional_negs']}.csv")
    
    fold_preds_df.to_csv(test_preds_path)

    return scores

def main():
    args = parse_args()
    set_reproducibility(CONFIG["seed"])

    logger.info(args)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.hp_config is not None:
        with open(args.hp_config, "r") as f:
            hp_cfg = yaml.safe_load(f)
    else:
        hp_cfg=cfg["default_hyperparameters"]

    logger.info(cfg)
    if args.nb_opt_negs is not None:
        cfg["nb_optional_negs"] = args.nb_opt_negs
    if args.model_name is not None:
        cfg["model_name"] = args.model_name
    if args.with_title is not None:
        cfg["with_title"] = args.with_title
    if args.with_keywords is not None:
        cfg["with_keywords"] = args.with_keywords
    if args.loss is not None:
        cfg["loss_type"] = args.loss
    if args.best_metric is not None:
        cfg['best_metric']=args.best_metric
    if args.fold is not None:
        cfg['fold']=args.fold
    if args.run is not None:
        cfg['run']=args.run


    train(cfg,hp_cfg)
    

if __name__ == "__main__":
    main()