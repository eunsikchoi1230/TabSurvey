import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


path = "data/cmc_screening/cmc_screening.csv"
df = pd.read_csv(path)
cat_idx = [0, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38] 

# Fill missing values of categorical columns with the most frequent value
for idx in cat_idx:
    df.iloc[:, idx] = df.iloc[:, idx].fillna(df.iloc[:, idx].mode()[0])
# Fill missing values of numerical columns with the mean
df = df.fillna(df.mean())

# X.shape: (163948, 41), y.shape: (163948, 4)
label_columns = ['HTN_class', 'DM_class', 'CAD_class', 'Dyslipidemia_class']
X = df.drop(label_columns, axis=1).to_numpy()
y = df[label_columns].to_numpy()



num_idx = []
cat_dims = []

 
for i in range(41):
    if cat_idx and i in cat_idx:
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])

        cat_dims.append(len(le.classes_))

    else:
        num_idx.append(i)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, stratify=y)

# Print ratio of each labelsets in y_train and y_test
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                labelset = [i, j, k, l]
                train_ratio = sum((y_train == labelset).all(axis=1)) / len(y_train)
                test_ratio = sum((y_test == labelset).all(axis=1)) / len(y_test)
                print(f"Labelset {labelset}: Train Ratio = {train_ratio}, Test Ratio = {test_ratio}")




# python train.py --config config/cmc_screening.yml --model_name NaiveBayes --problem_transformation BinaryRelevance



# iterative-stratification  0.1.7   pypi_0    pypi