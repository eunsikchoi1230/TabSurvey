import os
import sys
import numpy as np

dir_path = 'output/CMCscreening'

problem_transformations = ['BinaryRelevance', 'ClassifierChain', 'LabelPowerset', 'None', 'None']
problem_transformation_names = ['Binary\\\\Relevance', 'Classifier\\\\Chain', 'Label\\\\Powerset', 'Algorithm\\\\Adaptation', 'Neural\\\\Network']
models = [
            ['NaiveBayes', 'LogisticRegression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'],
            ['NaiveBayes', 'LogisticRegression', 'KNN', 'RandomForest', 'XGBoost'],
            ['NaiveBayes', 'LogisticRegression', 'KNN', 'RandomForest', 'XGBoost'],
            ['MLKNN', 'RandomForest', 'CatBoost'],
            ['FFN', 'TabNet'],
        ]


model_nums = []
metrics = [] 
for idx, problem_transformation in enumerate(problem_transformations):

    model_nums.append(len(models[idx]))
    for model in models[idx]:
        folder_dir = dir_path + '/' + problem_transformation + '/' + model
        with open(os.path.join(folder_dir, 'results.txt'), 'r') as f:
            lines = f.readlines()

        model_metrics = []
        for i in range(3, 12, 2):
            value = float(lines[i].split(": ")[1])
            model_metrics.append("{:.3f}".format(value))

        metrics.append(model_metrics)


# Format metrics
max_indices = np.argmax(np.array(metrics), axis=0)
min_indices = np.argmin(np.array(metrics), axis=0)
secondmax_indices = np.argsort(np.array(metrics), axis=0)[-2]
secondmin_indices = np.argsort(np.array(metrics), axis=0)[1]

def format_values(lst):
    formatted_list = []
    for item in lst:
        if isinstance(item, list):  
            formatted_list.append(format_values(item))  
        else:
            formatted_list.append("{:.3f}".format(item) if isinstance(item, float) or isinstance(item, int) else item)
    return formatted_list

metrics = format_values(metrics)

best_indices = [max_indices[0], min_indices[1], min_indices[2], min_indices[3], max_indices[4]]
second_best_indices = [secondmax_indices[0], secondmin_indices[1], secondmin_indices[2], secondmin_indices[3], secondmax_indices[4]]
for i, best_index in enumerate(best_indices):
    metrics[best_index][i] = "\\textbf{%s}" % metrics[best_index][i]

for i, second_best_index in enumerate(second_best_indices):
    metrics[second_best_index][i] = "\\underline{%s}" % metrics[second_best_index][i]


# Generate latex format
latex_format = ""
for problem_transformation_index, problem_transformation in enumerate(problem_transformations):
    latex_format += "\\multirow{%d}{*}{\\makecell{%s}}" % (model_nums[problem_transformation_index] , problem_transformation_names[problem_transformation_index])

    total = sum(model_nums[:problem_transformation_index])
    for model_index in range(model_nums[problem_transformation_index]):
        latex_format += " & %s & %s \\\\" % (models[problem_transformation_index][model_index], " & ".join(metrics[total + model_index]))
    latex_format += " \\hline \n\n"

print()
print(latex_format)
