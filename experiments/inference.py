#!/usr/bin/env python3
"""
Inference script for IPBES models using the instantiation module.

This script demonstrates how to use trained IPBES models to make predictions
on new texts using the instantiation.py module.
"""
import argparse
from typing import *
import numpy as np
import torch
import sys
import os
import argparse
import logging
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.utils import map_name


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IPBESPredictor():
    """This class object englobes all the functionalities related to the inference pipeline,
    ie. loading the models weights, loading the input data, using the model's weights to predict the output"""
    
    def __init__(
        self,
        weights_path,
        model_name,
        loss_type="BCE",
        with_title="False",
        with_keywords="False",
        threshold=0.5
        ) -> None:

        self.weights_path=weights_path
        self.model_name=model_name
        self.loss_type=loss_type
        self.with_title=with_title
        self.with_keywords=with_keywords
        self.threshold=threshold

        if not os.path.exists(self.weights_parent_dir):
            raise FileNotFoundError(f"Checkpoints parent directory does not exist : {self.weights_parent_dir}")
        
        #Path of every fold for the given model config
        self.model_paths  = [
            os.path.join(self.weights_parent_dir, dirname)
            for dirname in os.listdir(self.weights_parent_dir)
            if dirname.startswith( "best_model_cross_val_"+str(self.loss_type)+"_" +str(map_name(self.model_name))) and os.path.isdir(os.path.join(self.weights_parent_dir, dirname))
        ]
        
        self._load_model()

    def _load_model(self):
        """
        Loads models for each fold, for the given configuration. initializes the data data collators.
        """

        if not self.model_paths:
            raise FileNotFoundError(f"No model checkpoints found in directory: {self.weights_parent_dir}")
            
        try:
            try:
                self.tokenizer_per_fold = [AutoTokenizer.from_pretrained(model_path) for model_path in self.model_paths]
            except:
                logger.warning("Tokenizer not found in model path, using default BERT tokenizer")
                self.tokenizer_per_fold = [AutoTokenizer.from_pretrained("google-bert/bert-base-uncased") for _ in range(len(self.model_paths)) ]
            
            #List of model object per fold
            self.models_per_fold = [AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3
            ) for model_path in self.model_paths]

            # Move models to device and set to eval mode
            for model in self.models_per_fold:
                model.to(self.device)
                model.eval()
            
            #! We never use that !
            #Set the data collator for each tokenizer per fold
            self.data_collators = [DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True
            ) for tokenizer in self.tokenizer_per_fold]
            
            logger.info(f"Successfully loaded {len(self.model_paths)} models from: {self.weights_parent_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict_score(
            self,
            abstract: str,
            title: Optional[str] = None,
            keywords: Optional[str] = None
        ) -> float:
            """Predict score using ensemble of all fold models"""
            fold_scores = []
            
            for (model, tokenizer) in zip(self.models_per_fold, self.tokenizer_per_folds):
                # Tokenize with specific fold tokenizer
                tokens = self._tokenize_text_with_tokenizer(abstract, title, keywords, tokenizer)
                
                with torch.no_grad():
                    outputs = model(**tokens)
                    #The following will be an tensor of dimension (1,3) ?
                    #TODO : Double check that
                    logits = outputs.logits.squeeze()
                    score = torch.sigmoid(logits).cpu().item()
                    fold_scores.append(score)
            
            # Return folds average adjust the dimension based on the answer of the above question
            return np.mean(fold_scores)

    def _tokenize_text_with_tokenizer(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[str] = None,
        tokenizer: AutoTokenizer = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with specific tokenizer"""
        if tokenizer is None:
            tokenizer = self.tokenizer_per_fold[0]
            
        if self.with_title and self.with_keywords:
            if title is None or keywords is None:
                raise ValueError("Model requires both title and keywords, but one or both are missing")
            
            sep_tok = tokenizer.sep_token or "[SEP]"
            combined = title + sep_tok + keywords
            
            tokens = tokenizer(
                combined,
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        elif self.with_title:
            if title is None:
                raise ValueError("Model requires title, but it's missing")
                
            tokens = tokenizer(
                title,
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        elif self.with_keywords:
            if keywords is None:
                raise ValueError("Model requires keywords, but they're missing")
                
            tokens = tokenizer(
                abstract,
                keywords,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        else:
            tokens = tokenizer(
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
        
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        return tokens

    def evaluate_text(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[str] = None,
        return_binary: bool = False
    ) -> Dict[str, Any]:
        score = self.predict_score(abstract, title, keywords)
        
        result = {
            "abstract": abstract,
            "score": score
        }
        
        if title is not None:
            result["title"] = title
        if keywords is not None:
            result["keywords"] = keywords
            
        if return_binary:
            result["prediction"] = int(score > self.threshold)
            
        return result

def load_predictor(
    model_name: str,
    loss_type: str = "BCE",
    with_title: bool = True,
    with_keywords: bool = False,
    device: Optional[str] = None,
    weights_parent_dir: str = "results/final_model",
    threshold: float = 0.5
) -> IPBESPredictor:
    return IPBESPredictor(
        model_name=model_name,
        loss_type=loss_type,
        with_title=with_title,
        with_keywords=with_keywords,
        device=device,
        weights_parent_dir=weights_parent_dir,
        threshold=threshold
    )

def main():
    pass

if __name__ == "__main__":
    main()
