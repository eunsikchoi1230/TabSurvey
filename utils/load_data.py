import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd

# Need to check where is fuction is used
def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q


def load_data(args):
    print("Loading dataset " + args.dataset + "...")

    if args.dataset == "CMCscreening":  # Multi-label classification dataset with categorical data
        path = "data/cmc_screening/cmc_screening.csv"
        df = pd.read_csv(path)
        
        # Fill missing values 
        if args.fill_na:
            # Categorical columns with the most frequent value
            for idx in args.cat_idx:
                df.iloc[:, idx] = df.iloc[:, idx].fillna(df.iloc[:, idx].mode()[0])
            # Numerical columns with the mean
            df = df.fillna(df.mean())

        # X.shape: (163948, 41), y.shape: (163948, 4)
        label_columns = ['HTN_class', 'DM_class', 'CAD_class', 'Dyslipidemia_class']
        X = df.drop(label_columns, axis=1).to_numpy()
        y = df[label_columns].to_numpy()

    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")
    print(f"X.shape: {X.shape}")



    # Preprocess target - Need to be implemented

    num_idx = []
    args.cat_dims = []

    # Preprocess data
    ## Ordinal encoding for categorical features
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])
            args.cat_dims.append(len(le.classes_))
        else:
            num_idx.append(i)
            
    ## Standardization for numerical features
    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

    ## One-hot encoding for categorical features (optional)
    if args.one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.cat_idx])
        new_x2 = X[:, num_idx]
        X = np.concatenate([new_x1, new_x2], axis=1)
        print("New X.shape:", X.shape)

    return X, y
