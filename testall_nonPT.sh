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

MODELS_ENV["MLKNN"]=$SKLEARN_ENV
MODELS_OPTIMIZE["MLKNN"]=true
MODELS_N_TRIALS["MLKNN"]=30
MODELS_N_STARTUP["MLKNN"]=10

MODELS_ENV["RandomForest"]=$SKLEARN_ENV
MODELS_OPTIMIZE["RandomForest"]=true
MODELS_N_TRIALS["RandomForest"]=30
MODELS_N_STARTUP["RandomForest"]=10

MODELS_ENV["CatBoost"]=$GBDT_ENV
MODELS_OPTIMIZE["CatBoost"]=true
MODELS_N_TRIALS["CatBoost"]=5
MODELS_N_STARTUP["CatBoost"]=0

# MODELS_ENV["FFN"]=$TORCH_ENV
# MODELS_OPTIMIZE["FFN"]=true
# MODELS_N_TRIALS["FFN"]=5
# MODELS_N_STARTUP["FFN"]=0

# MODELS_ENV["TabNet"]=$TORCH_ENV
# MODELS_OPTIMIZE["TabNet"]=true
# MODELS_N_TRIALS["TabNet"]=5
# MODELS_N_STARTUP["TabNet"]=0


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
