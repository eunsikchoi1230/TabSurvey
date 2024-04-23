#!/bin/bash

CONFIG="config/cmc_screening.yml"

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

declare -A MODELS_ENV
declare -A MODELS_OPTIMIZE
declare -A MODELS_N_TRIALS
declare -A MODELS_N_STARTUP

MODELS_ENV["TabNet"]=$TORCH_ENV
MODELS_OPTIMIZE["TabNet"]=true
MODELS_N_TRIALS["TabNet"]=30
MODELS_N_STARTUP["TabNet"]=10


# conda init bash
eval "$(conda shell.bash hook)"

for model in "${!MODELS_ENV[@]}"; do
  printf "\n\n----------------------------------------------------------------------------\n"
  printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS_ENV[$model]}"

  conda activate "${MODELS_ENV[$model]}"

  if ${MODELS_OPTIMIZE[$model]}; then
    python train.py --config $CONFIG --model_name "$model" --problem_transformation None --n_trials "${MODELS_N_TRIALS[$model]}" --n_startup_trials "${MODELS_N_STARTUP[$model]}" --optimize_hyperparameters
  else
    python train.py --config $CONFIG --model_name "$model" --problem_transformation None --n_trials "${MODELS_N_TRIALS[$model]}" --n_startup_trials "${MODELS_N_STARTUP[$model]}"
  fi
  conda deactivate

done
