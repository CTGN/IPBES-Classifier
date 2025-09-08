## IPBES-Classifier

Multi-label classifier for identifying IPBES-relevant scientific publications. The model predicts relevance for three assessment domains:

- IAS: Invasive Alien Species
- SUA: Sustainable Use of Animals
- VA: Values Assessment

It fine-tunes Transformer models (e.g., BERT, BioBERT, RoBERTa) and supports cross-validation, hyperparameter optimization (HPO), and simple ensemble averaging.

## Project layout

```
IPBES-Classifier/
├── configs/                  # YAML configs for HPO, train, ensemble, env
├── scripts/                  # Orchestration scripts
├── src/
│   ├── config.py            # Main configuration and directory creation
│   ├── data_pipeline/
│   │   └── ipbes/
│   │       ├── preprocess_ipbes.py  # Build cleaned dataset + CV folds
│   │       └── ...
│   └── models/
│       └── ipbes/
│           ├── hpo.py       # Hyperparameter optimization
│           ├── train.py     # Training using best HPO params
│           └── ensemble.py  # Simple ensemble of model scores
├── results/                  # Metrics, test predictions, final_model/
├── data/                     # Cleaned dataset and CV folds
└── plots/                    # Generated plots
```

Notes:
- Paths and defaults are driven by `src/config.py`. On import, it creates the necessary directories under the project root (`data/`, `results/`, `plots/`, ...).
- `configs/environment.yaml` exists but most codepaths use `src/config.py` directly for paths and seeds.

## Installation

1) Clone and enter the project
```bash
git clone https://github.com/CTGN/IPBES-Classifier.git
cd IPBES-Classifier
```

2) Install dependencies (Python >= 3.11)
```bash
uv sync
```

3) Quick check
```bash
uv run python -c "import torch; print('Torch OK')"
```

## Get data and model weights (public S3)

If you want to run evaluation or inference right away, download the prepared dataset and final checkpoints from the public S3 bucket. Replace `<your-bucket>` with the actual bucket name.

Prerequisite (macOS):
```bash
brew install awscli
```

Download the dataset into `IPBES-Classifier/data`:
```bash
aws s3 sync s3://<your-bucket>/dataset ./data --no-sign-request
```

Download final model checkpoints into `IPBES-Classifier/results/final_model`:
```bash
aws s3 sync s3://<your-bucket>/checkpoints ./results/final_model --no-sign-request
```

Expected structure after download:
```
data/
├── cleaned_dataset.csv
└── folds/
    ├── train0_run-0.csv
    ├── dev0_run-0.csv
    ├── test0_run-0.csv
    └── ... (for each fold/run)

results/
└── final_model/
    └── best_model_cross_val_<loss>_<model>_fold-<k>/
```

Tip: If the bucket is public, `--no-sign-request` avoids requiring AWS credentials.

## Build the dataset yourself (optional)

You can also generate the cleaned dataset and CV folds locally. This is intended for maintainers; it pulls/uses raw sources configured in `src/data_pipeline/ipbes/`.

```bash
uv run src/data_pipeline/ipbes/preprocess_ipbes.py \
  -nf 5 -nr 1 \
  -fm              # optionally fill missing metadata via CrossRef
```

This will produce `data/cleaned_dataset.csv` and fold index CSVs under `data/folds/`.

## Train and evaluate

HPO (writes best params to `configs/best_hpo.yaml`):
```bash
uv run src/models/ipbes/hpo.py \
  --config configs/hpo.yaml \
  --fold 0 \
  --run 0 \
  --n_trials 20 \
  --hpo_metric eval_AP \
  --model_name google-bert/bert-base-uncased \
  --loss BCE \
  --with_title
```

Training using the best HPO params:
```bash
uv run src/models/ipbes/train.py \
  --config configs/train.yaml \
  --hp_config configs/best_hpo.yaml \
  --fold 0 \
  --run 0 \
  -m google-bert/bert-base-uncased \
  -bm eval_AP \
  --loss BCE \
  -t
```

Ensemble (averages model scores listed in `configs/ensemble.yaml`):
```bash
uv run src/models/ipbes/ensemble.py \
  --config configs/ensemble.yaml \
  --fold 0 \
  --run 0 \
  --loss BCE \
  -t
```

End-to-end (preprocess → HPO → train → ensemble) for a set of models:
```bash
./scripts/launch_ipbes_pipeline.sh
```

## Outputs

- `results/metrics/results.csv`: Aggregated metrics per model/fold/run.
- `results/test preds/bert/*.csv`: Per-example scores and predictions per fold.
- `results/final_model/best_model_cross_val_<...>/`: Saved Hugging Face checkpoints for the best model per fold.
- `plots/`: Training/eval curves and HPO visuals.

## Inference with a saved checkpoint

You can load a saved checkpoint directory from `results/final_model/` with Hugging Face Transformers:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

model_dir = "results/final_model/best_model_cross_val_BCE_bert-base-uncased_fold-1"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

text = "This paper studies invasive alien species management in urban ecosystems."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
scores = torch.sigmoid(logits).numpy().squeeze()  # 3 scores for [IAS, SUA, VA]
print(scores)
```

## Configuration

- Primary configuration lives in `src/config.py` and is imported across the codebase. It:
  - defines project paths (data/results/plots) relative to the repo root,
  - sets defaults (seed, training args), and
  - creates needed directories on import.
- `IPBES_ENV` is read but paths are not overridden elsewhere; default local layout is recommended.

Optional environment variables you might care about:
```bash
export PYALEX_EMAIL=your-email@domain.com   # only relevant if you use data fetching utils
export CUDA_VISIBLE_DEVICES=0               # pick GPU
```

## Notes and tips

- Batch size is fixed to 100 in code paths for stability across GPUs.
- Metrics commonly used: `eval_AP`, `eval_kappa`, etc. Pick one via `--hpo_metric` and `-bm`.
- Model names can be any HF identifier (e.g., `google-bert/bert-base-uncased`, `dmis-lab/biobert-v1.1`, `FacebookAI/roberta-base`).

## License

See repository license (if provided) or contact the authors.