from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

from models.basemodel import BaseModel


def get_base_model(params, args, objective):

    if args.model_name == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    
    elif args.model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000)

    elif args.model_name == "SVM":
        from sklearn.svm import SVC
        return SVC(probability=True, kernel=params["kernel"])

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
            "kernel": trial.suggest_categorical("kernel", ["linear"])
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


    def fit(self, X, y, X_val=None, y_val=None):

        if self.args.model_name == "CatBoost" and self.args.cat_idx:
            X = X.astype('object')
            X_val = X_val.astype('object')
            X[:, self.args.cat_idx] = X[:, self.args.cat_idx].astype('int')
            X_val[:, self.args.cat_idx] = X_val[:, self.args.cat_idx].astype('int')

        super().fit(X, y, X_val, y_val)

        return [], []

    def predict(self, X):

        if self.args.model_name == "CatBoost" and self.args.cat_idx:
            X = X.astype('object')
            X[:, self.args.cat_idx] = X[:, self.args.cat_idx].astype('int')

        return super().predict(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = get_base_model_params(trial, args)
        return params


'''
    Binary Relevance - Binary Relevance method for multi-label classification
'''


class BinaryRelevance(BaseModelProblemTransformation):

    def __init__(self, params, args):
        super().__init__(params, args)

        base_model = get_base_model(self.params, args, "binary")
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)


'''
    Classifier Chain - Classifier Chain method for multi-label classification
'''


class ClassifierChain(BaseModelProblemTransformation):
        
    def __init__(self, params, args):
        super().__init__(params, args)

        base_model = get_base_model(self.params, args, "binary")
        self.model = ClassifierChain(base_model, order="random", random_state=args.seed)


'''
    Label Powerset - Label Powerset method for multi-label classification
'''


class LabelPowerset(BaseModelProblemTransformation):
        
    def __init__(self, params, args):
        super().__init__(params, args)

        base_model = get_base_model(self.params, args, "classification")
        self.model = LabelPowerset(classifier=base_model, require_dense=[True, True])
