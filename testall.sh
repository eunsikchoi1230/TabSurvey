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

# (Environment, Optimize Hyperparameters, n_trials, n_startup_trials)
declare -A PT_MODELS
PT_MODELS=( 
        ["NaiveBayes"]=($SKLEARN_ENV false 0 0)
        #  ["KNN"]=($SKLEARN_ENV true 30 10)
        #  ["RandomForest"]=($SKLEARN_ENV true 30 10)
         ["XGBoost"]=($GBDT_ENV true 30 10)
        #  ["CatBoost"]=($GBDT_ENV true 30 10)
          )

declare -A MODELS
MODELS=( 
        # ["RandomForest"]=($SKLEARN_ENV true 30 10)
         ["TabNet"]=($TORCH_ENV true 3 0)
          )



# conda init bash
eval "$(conda shell.bash hook)"

for problem_transformation in "${PROBLEM_TRANSFORMATION[@]}"; do

  for model in "${!PT_MODELS[@][0]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s %s with %s in env %s\n\n' "$problem_transformation " "$model" "$config" "${PT_MODELS[$model][0]}"

    conda activate "${PT_MODELS[$model][0]}"

    if ${PT_MODELS[$model][1]}; then
      python train.py --config $CONFIG --model_name "$model" --problem_transformation "$problem_transformation" --n_trials "${PT_MODELS[$model][2]}" --n_startup_trials "${PT_MODELS[$model][3]}" --optimize_hyperparameters
    else
      python train.py --config $CONFIG --model_name "$model" --problem_transformation "$problem_transformation" --n_trials "${PT_MODELS[$model][2]}" --n_startup_trials "${PT_MODELS[$model][3]}"

    conda deactivate

  done

done


for model in "${!MODELS[@]}"; do
  printf "\n\n----------------------------------------------------------------------------\n"
  printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS[$model]}"

  conda activate "${MODELS[$model]}"

  if ${PT_MODELS[$model][1]}; then
    python train.py --config $CONFIG --model_name "$model" --problem_transformation None --n_trials "${PT_MODELS[$model][2]}" --n_startup_trials "${PT_MODELS[$model][3]}" --optimize_hyperparameters
  else
    python train.py --config $CONFIG --model_name "$model" --problem_transformation None --n_trials "${PT_MODELS[$model][2]}" --n_startup_trials "${PT_MODELS[$model][3]}"

  conda deactivate

done
