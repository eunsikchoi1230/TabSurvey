#!/bin/bash

N_TRIALS=2
# EPOCHS=3

CONFIG="config/cmc_screening.yml"

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# PROBLEM_TRANSFORMATION=("BinaryRelevance" "ClassifierChain" "LabelPowerset")
PROBLEM_TRANSFORMATION=("BinaryRelevance")


declare -A PT_MODELS_ENV
declare -A PT_MODELS_OPTIMIZE
declare -A PT_MODELS_N_TRIALS
declare -A PT_MODELS_N_STARTUP

PT_MODELS_ENV["NaiveBayes"]="$SKLEARN_ENV"
PT_MODELS_OPTIMIZE["NaiveBayes"]=true
PT_MODELS_N_TRIALS["NaiveBayes"]=1
PT_MODELS_N_STARTUP["NaiveBayes"]=0

PT_MODELS_ENV["XGBoost"]="$SKLEARN_ENV"
PT_MODELS_OPTIMIZE["XGBoost"]=true
PT_MODELS_N_TRIALS["XGBoost"]=5
PT_MODELS_N_STARTUP["XGBoost"]=2



declare -A MODELS_ENV
declare -A MODELS_OPTIMIZE
declare -A MODELS_N_TRIALS
declare -A MODELS_N_STARTUP

MODELS_ENV["TabNet"]=$TORCH_ENV
MODELS_OPTIMIZE["TabNet"]=true
MODELS_N_TRIALS["TabNet"]=3
MODELS_N_STARTUP["TabNet"]=0




# conda init bash
eval "$(conda shell.bash hook)"

for problem_transformation in "${PROBLEM_TRANSFORMATION[@]}"; do

  for model in "${!PT_MODELS_ENV[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s %s with %s in env %s\n\n' "$problem_transformation " "$model" "$config" "${PT_MODELS_ENV[$model]}"

    conda activate "${PT_MODELS_ENV[$model]}"

    if ${PT_MODELS_OPTIMIZE[$model]}; then
      python train.py --config $CONFIG --model_name "$model" --problem_transformation "$problem_transformation" --n_trials "${PT_MODELS_N_TRIALS[$model]}" --n_startup_trials "${PT_MODELS_N_STARTUP[$model]}" --optimize_hyperparameters
    else
      python train.py --config $CONFIG --model_name "$model" --problem_transformation "$problem_transformation" --n_trials "${PT_MODELS_N_TRIALS[$model]}" --n_startup_trials "${PT_MODELS_N_STARTUP[$model]}"
    fi

    conda deactivate

  done

done


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
