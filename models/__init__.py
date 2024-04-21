all_models = ["LinearModel", "KNN", "DecisionTree", "RandomForest", "XGBoost", "CatBoost", "LightGBM", "ModelTree",
               "MLP", "TabNet", "VIME", "TabTransformer", "NODE", "DeepGBM", "RLN", "DNFNet", "STG", "NAM", "DeepFM",
               "SAINT", "DANet"] # Need to update for new models


def str2model(args):

    if args.problem_transformation == "None":

        if args.model_name == "LinearModel":
            from models.baseline_models import LinearModel
            return LinearModel

        elif args.model_name == "KNN":
            from models.baseline_models import KNN
            return KNN

        elif args.model_name == "RandomForest":
            from models.baseline_models import RandomForest
            return RandomForest

        elif args.model_name == "XGBoost":
            from models.tree_models import XGBoost
            return XGBoost

        elif args.model_name == "CatBoost":
            from models.tree_models import CatBoost
            return CatBoost

        elif args.model_name == "TabNet":
            from models.tabnet import TabNet
            return TabNet

        elif args.model_name == "VIME":
            from models.vime import VIME
            return VIME

        elif args.model_name == "TabTransformer":
            from models.tabtransformer import TabTransformer
            return TabTransformer

        elif args.model_name == "SAINT":
            from models.saint import SAINT
            return SAINT

        else:
            raise NotImplementedError("Model \"" + args.model + "\" not yet implemented")
    
    elif args.problem_transformation == "BinaryRelevance":
        from models.problem_transformation import BinaryRelevance
        return BinaryRelevance

    elif args.problem_transformation == "ClassifierChain":
        from models.problem_transformation import ClassifierChain
        return ClassifierChain

    elif args.problem_transformation == "LabelPowerset":
        from models.problem_transformation import LabelPowerset
        return LabelPowerset

    else:
        raise NotImplementedError("Probelm Transformation method \"" + args.problem_transformation + "\" not yet implemented")

