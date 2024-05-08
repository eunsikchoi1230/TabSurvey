import os
import sys
import numpy as np

dir_path = 'output/CMCscreening'

problem_transformations = ['BinaryRelevance', 'ClassifierChain', 'LabelPowerset', 'None', 'None']
problem_transformation_names = ['Binary\\\\Relevance', 'Classifier\\\\Chain', 'Label\\\\Powerset', 'Algorithm\\\\Adaptation', 'Neural\\\\Network']
models = [
            ['NaiveBayes', 'LogisticRegression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'],
            ['NaiveBayes', 'LogisticRegression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'],
            ['NaiveBayes', 'LogisticRegression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'],
            ['MLKNN', 'RandomForest', 'CatBoost'],
            ['FFN', 'TabNet']
        ]

gpu_model_indices = [5, 11, 17, 20, 21, 22]

model_nums = []
metrics = [] 
for idx, problem_transformation in enumerate(problem_transformations):

    model_nums.append(len(models[idx]))
    for model in models[idx]:
        folder_dir = dir_path + '/' + problem_transformation + '/' + model
        with open(os.path.join(folder_dir, 'results.txt'), 'r') as f:
            lines = f.readlines()

        model_metrics = []
        for i in [3, 5, 7, 9, 11, 14, 15]:
            value = float(lines[i].split(": ")[1])
            model_metrics.append("{:.3f}".format(value))

        

        metrics.append(model_metrics)


# Format metrics
max_indices = np.argmax(np.array(metrics), axis=0)
min_indices = np.argmin(np.array(metrics), axis=0)
secondmax_indices = np.argsort(np.array(metrics), axis=0)[-2]
secondmin_indices = np.argsort(np.array(metrics), axis=0)[1]

def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if seconds < 1:
        return "\\textless 1s"
    if minutes == 0:
        return f"{remaining_seconds}s"
    else:
        return f"{minutes}m {remaining_seconds:02}s"

for i, row in enumerate(metrics):
    for j, value in enumerate(row):
        if j in [5, 6]:
            if i in gpu_model_indices:
                metrics[i][j] = format_time(float(value)) + " \\tiny (GPU)"
            else:
                metrics[i][j] = format_time(float(value))
        else:
            metrics[i][j] = "{:.3f}".format(float(value))

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
