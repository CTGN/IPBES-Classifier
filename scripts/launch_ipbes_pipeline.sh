#!/usr/bin/env bash
#set -euo pipefail
IFS=$'\n\t'

#TODO : Add argument parsing for number of runs, folds, models, losses, etc.
# Models to iterate over

MODELS=(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    "FacebookAI/roberta-base"
    "dmis-lab/biobert-v1.1"
    "google-bert/bert-base-uncased"
)

NUM_RUNS=1
NUM_FOLDS=5


for (( run=0; run<NUM_RUNS; run++ )); do
echo "--> Run #${run}"

for (( fold=0; fold<NUM_FOLDS; fold++ )); do
echo "------> Fold: ${fold}"
    for loss in BCE; do
    echo "----> Loss function: ${loss}"

    for model in "${MODELS[@]}"; do
        echo "--------> Model: ${model}"
        #HPO
        uv run src/models/ipbes/hpo.py \
        --config configs/hpo.yaml \
        --fold "${fold}" \
        --run "${run}" \
        --n_trials 20 \
        --hpo_metric "eval_AP_weighted" \
        -m "${model}" \
        --loss "${loss}" \
        -t

        # Finbal training
        uv run src/models/ipbes/train.py \
        --config configs/train.yaml \
        --hp_config configs/best_hpo.yaml \
        --fold "${fold}" \
        --run "${run}" \
        -m "${model}" \
        -bm "eval_AP_weighted" \
        --loss "${loss}" \
        -t
    done

    # Ensemble step for this fold/run
    echo "--------> Ensemble for fold ${fold}, run ${run}"
    uv run src/models/ipbes/ensemble.py \
        --config configs/ensemble.yaml \
        --fold "${fold}" \
        --run "${run}" \
        --loss "${loss}" \
        -t
    done
done
done

# clean up folds
#echo "==> Cleaning up folds directory"
#rm -r "./data/ipbes/folds/"*
