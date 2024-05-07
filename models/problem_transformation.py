# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.multioutput import ClassifierChain as ClassifierChainMethod
from skmultilearn.problem_transform import BinaryRelevance as BinaryRelevanceMethod
from skmultilearn.problem_transform import ClassifierChain as ClassifierChainMethod
from skmultilearn.problem_transform import LabelPowerset as LabelPowersetMethod

from models.basemodel import BaseModel

import numpy as np


def get_base_model(params, args, objective):

    if args.model_name == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    
    elif args.model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000)

    elif args.model_name == "SVM":
        from sklearn.svm import SVC
        return SVC(probability=True, kernel="linear", C=params["C"])

    elif args.model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])

    elif args.model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], n_jobs=-1)
    
    elif args.model_name == "XGBoost":
        from xgboost import XGBClassifier

        if objective == "classification":
            params["objective"] = "multi:softprob"
            params["num_class"] = 2 ** args.num_classes # For label-powerset, Need to fix for RAkEL or other methods
            params["eval_metric"] = "mlogloss"
        elif objective == "binary":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "auc"
        else:
            raise NotImplementedError("XGBoost Objective \"" + objective + "\" not yet implemented")

        return XGBClassifier(**params)

    elif args.model_name == "CatBoost":
        from catboost import CatBoostClassifier

        params["iterations"] = args.epochs
        params["od_type"] = "Iter"
        params["od_wait"] = args.early_stopping_rounds
        params["verbose"] = args.logging_period
        params["train_dir"] = "output/CatBoost/" + args.dataset + "/catboost_info"

        if args.use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = [args.gpu_ids]

        params["cat_features"] = args.cat_idx

        return CatBoostClassifier(**params)

    else:
        raise NotImplementedError("Model \"" + model + "\" not yet implemented")


def get_base_model_params(trial, args):

    if args.model_name == "NaiveBayes":
        return dict()
    
    elif args.model_name == "LogisticRegression":
        return dict()

    elif args.model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-10, 1e10, log=True)
        }
    
    elif args.model_name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 40)
        }
    
    elif args.model_name == "RandomForest":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "n_estimators": trial.suggest_int("n_estimators", 5, 100)
        }
    
    elif args.model_name == "XGBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
            "n_estimators": trial.suggest_int("n_estimators", 200, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 6),
            "gamma": trial.suggest_float("gamma", 0.01, 0.5),
            "subsample": trial.suggest_float("subsample", 0.5, 0.99),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.99),
        }

    elif args.model_name == "CatBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30),
        }

    else:
        raise NotImplementedError("Model \"" + model + "\" not yet implemented")


class BaseModelProblemTransformation(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return super().predict_proba(X).toarray()

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = get_base_model_params(trial, args)
        return params


class BinaryRelevance(BaseModelProblemTransformation):

    def __init__(self, params, args):
        super().__init__(params, args)

        base_model = get_base_model(self.params, args, "binary")
        self.model = BinaryRelevanceMethod(classifier=base_model, require_dense=[True, True], cat_idx=args.cat_idx)


class ClassifierChain(BaseModelProblemTransformation):

    def __init__(self, params, args):
        super().__init__(params, args)

        base_model = get_base_model(self.params, args, "binary")
        np.random.seed(args.seed)
        self.model = ClassifierChainMethod(classifier=base_model, require_dense=[True, True], order=np.random.permutation(args.num_classes), cat_idx=args.cat_idx)


class LabelPowerset(BaseModelProblemTransformation):
        
    def __init__(self, params, args):
        super().__init__(params, args)

        base_model = get_base_model(self.params, args, "classification")
        self.model = LabelPowersetMethod(classifier=base_model, require_dense=[True, True], cat_idx=args.cat_idx)
