#!/bin/bash

CONFIG="config/cmc_screening.yml"

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

PROBLEM_TRANSFORMATION=("BinaryRelevance" "ClassifierChain" "LabelPowerset")

declare -A PT_MODELS_ENV
declare -A PT_MODELS_OPTIMIZE
declare -A PT_MODELS_N_TRIALS
declare -A PT_MODELS_N_STARTUP

# PT_MODELS_ENV["NaiveBayes"]="$SKLEARN_ENV"
# PT_MODELS_OPTIMIZE["NaiveBayes"]=true
# PT_MODELS_N_TRIALS["NaiveBayes"]=1
# PT_MODELS_N_STARTUP["NaiveBayes"]=0

# PT_MODELS_ENV["LogisticRegression"]="$SKLEARN_ENV"
# PT_MODELS_OPTIMIZE["LogisticRegression"]=true
# PT_MODELS_N_TRIALS["LogisticRegression"]=1
# PT_MODELS_N_STARTUP["LogisticRegression"]=0

# PT_MODELS_ENV["KNN"]="$SKLEARN_ENV"
# PT_MODELS_OPTIMIZE["KNN"]=true
# PT_MODELS_N_TRIALS["KNN"]=30
# PT_MODELS_N_STARTUP["KNN"]=10

# PT_MODELS_ENV["RandomForest"]="$SKLEARN_ENV"
# PT_MODELS_OPTIMIZE["RandomForest"]=true
# PT_MODELS_N_TRIALS["RandomForest"]=30
# PT_MODELS_N_STARTUP["RandomForest"]=10

# PT_MODELS_ENV["XGBoost"]="$SKLEARN_ENV"
# PT_MODELS_OPTIMIZE["XGBoost"]=true
# PT_MODELS_N_TRIALS["XGBoost"]=30
# PT_MODELS_N_STARTUP["XGBoost"]=10 

PT_MODELS_ENV["CatBoost"]="$GBDT_ENV"
PT_MODELS_OPTIMIZE["CatBoost"]=true
PT_MODELS_N_TRIALS["CatBoost"]=5
PT_MODELS_N_STARTUP["CatBoost"]=0 


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