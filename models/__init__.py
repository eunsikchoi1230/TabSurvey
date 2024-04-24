all_models = ["LinearModel", "KNN", "DecisionTree", "RandomForest", "XGBoost", "CatBoost", "LightGBM", "ModelTree",
               "MLP", "TabNet", "VIME", "TabTransformer", "NODE", "DeepGBM", "RLN", "DNFNet", "STG", "NAM", "DeepFM",
               "SAINT", "DANet"] # Need to update for new models


def str2model(args):

    if args.problem_transformation == "None":

        if args.model_name == "MostFrequent":
            from models.baseline_models import DummyMostFrequent
            return DummyMostFrequent

        elif args.model_name == "Stratified":
            from models.baseline_models import DummyStratified
            return DummyStratified
        
        elif args.model_name == "MLKNN":
            from models.baseline_models import MLKNN
            return MLKNN

        elif args.model_name == "RandomForest":
            from models.baseline_models import RandomForest
            return RandomForest

        elif args.model_name == "CatBoost":
            from models.tree_models import CatBoost
            return CatBoost

        elif args.model_name == "FFN":
            from models.ffn import FFN
            return FFN

        elif args.model_name == "TabNet":
            from models.tabnet import TabNet
            return TabNet

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

