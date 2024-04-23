import numpy as np
import torch
import torch.nn as nn
from models.basemodel_torch import BaseModelTorch
from utils.io_utils import save_model_to_file, load_model_from_file

from pytorch_tabnet.metrics import Metric
import torch.nn.functional as F


class FFNClassifier(nn.Module):
    def __init__(self, input_size, output_size, layer_num, layer_size, dropout=0.0):
        super(FFNClassifier, self).__init__()

        layers = [nn.Linear(input_size, layer_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for i in range(layer_num - 1):
            layers.extend([
                nn.Linear(layer_size, layer_size),
                nn.ReLU()
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(layer_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FFN(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.params["input_size"] = args.num_features
        self.params["output_size"] = args.num_classes

        self.params["layer_num"] = self.params["layer_num"]
        self.params["layer_size"] = self.params["layer_size"]
        self.params["dropout"] = self.params["dropout"]

        self.params["learning_rate"] = self.params["learning_rate"]

        if args.objective == "multi-label_classification":
            self.model = FFNClassifier(input_size=self.params["input_size"], 
                                       output_size=self.params["output_size"], 
                                       layer_num=self.params["layer_num"], 
                                       layer_size=self.params["layer_size"], 
                                       dropout=self.params["dropout"])
    
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.to(self.device)
        return super().fit(X, y, X_val, y_val)


    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "layer_num": trial.suggest_int("layer_num", 1, 5),
            "layer_size": trial.suggest_int("layer_size", 8, 64),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
        }
        return params

