import logging
import sys

import optuna

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    valid_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "multi-label_classification":
        kf = MultilabelStratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective " + args.objective + " is not yet implemented.")

    for i, (train_index, valid_index) in enumerate(kf.split(X, y)):

        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_valid, y_valid)
        train_timer.end()

        # Validate model
        valid_timer.start()
        curr_model.predict(X_valid)
        valid_timer.end()

        # Save model weights and the truth/prediction pairs for traceability in output/model_name/dataset/(predictions or models)
        curr_model.save_model_and_predictions(y_valid, i)

        if save_model:
            # Save the loss history to a file in output/model_name/dataset/logging
            save_loss_to_file(args, loss_history, "cross_validation_train_loss", extension=i)
            save_loss_to_file(args, val_loss_history, "cross_validation_val_loss", extension=i)

        # Compute scores on the output
        sc.eval(y_valid, curr_model.predictions, curr_model.prediction_probabilities)

        print(sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", valid_timer.get_average_time())

        # Save the all statistics to a file in output/model_name/dataset/results.txt
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), valid_timer.get_average_time(),
                             model.params)

    return sc, (train_timer.get_average_time(), valid_timer.get_average_time())

def split_data(X, y, args):
    if args.objective == "multi-label_classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    else:
        raise NotImplementedError("Objective " + args.objective + " is not yet implemented.")

    return X_train, y_train, X_test, y_test

def test_model(model, X_train, y_train, X_test, y_test, args, save_model=False): # Need to check code
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    curr_model = model.clone()

    # Train model
    train_timer.start()
    loss_history, test_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)
    train_timer.end()

    # Test model
    test_timer.start()
    curr_model.predict(X_test)
    test_timer.end()

    # Save model weights and the truth/prediction pairs for traceability in output/model_name/dataset/(predictions or models)
    curr_model.save_model_and_predictions(y_test)

    if save_model:
        # Save the loss history to a file in output/model_name/dataset/logging 
        save_loss_to_file(args, loss_history, "testing_train_loss", extension="testing")
        save_loss_to_file(args, test_loss_history, "testing_test_loss", extension="testing")

    sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

    print(sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file in output/model_name/dataset/results.txt
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)
    
    if args.feature_importance:
        curr_model.feature_importance(X_test, y_test)

    return sc, (train_timer.get_average_time(), test_timer.get_average_time())


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args)

        # Log tested hyperparameters to file in output/model_name/dataset/hp_log.txt
        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)
    X_train, y_train, X_test, y_test = split_data(X, y, args)

    model_name = str2model(args)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout)) 
    study_name = args.problem_transformation + "_" + args.model_name + "_" + args.dataset
    storage_name = "sqlite:///optuna/{}.db".format(study_name)
    
    # Start from scratch if the study already exists
    all_studies = optuna.study.get_all_study_summaries(storage=storage_name)
    if any(study.study_name == study_name for study in all_studies):
        optuna.delete_study(study_name=study_name, storage=storage_name)

    # Optimize hyperparameters
    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=args.n_startup_trials, seed=args.seed))
    study.optimize(Objective(args, model_name, X_test, y_test), n_trials=args.n_trials)
    print("\n" + "=" * 20)
    print("Best parameters:", study.best_trial.params)
    print("=" * 20 + "\n")

    # Run best trial again with all training data and save the model
    model = model_name(study.best_trial.params, args)
    test_model(model, X_train, y_train, X_test, y_test, args, save_model=True)


def main_once(args):
    print("Train model with given hyperparameters")

    # Commented for test
    X, y = load_data(args)
    X_train, y_train, X_test, y_test = split_data(X, y, args) 

    model_name = str2model(args) # Get the model class

    parameters = args.parameters[args.dataset][args.problem_transformation][args.model_name] # Dictionary of hyperparameters
    model = model_name(parameters, args) # Create model with given hyperparameters

    sc, time = test_model(model, X_train, y_train, X_test, y_test, args) # Train and test the model

    print(sc.get_results())
    print(time)


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print(arguments)

    if arguments.optimize_hyperparameters:
        main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser()
        arguments = parser.parse_args()
        main_once(arguments)
