import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import prepare_data

#Custom imports
from src import evaluate_models, run_models


def get_prepared_data(gene_name, seed):
    """Parameters:
    <gene_name> a string, either 'ryr2', 'kcnh2', 'kcnq1', or 'scn5a'"""

    all_features =['AA position', 'Conservation Score', 'LQTS/GnomAD','Original','Change','Functional Domain']
    #Prepare data and split in train and test for given gene_name
    data = prepare_data.PrepareData(gene_name, all_features, seed)
    return data

def run_evaluate(data, what_to_run, modeling_approach, seed):
    """Parameters:
    <data>: data
    <what_to_run> a string, one of:
        'grid_search': this will do a grid search over different model setups
            
    <modeling_approach>: a string, 'LR' for logistic regression"""
    
    print('\n*** Running ',modeling_approach,'***')
    if what_to_run == 'grid_search':
        run = run_models.RunModels(modeling_approach, data, what_to_run, seed)
        evaluate = evaluate_models.EvaluateModels(modeling_approach, data, run.best_model, seed)
    
    return evaluate

def run_evaluate_with_loop(gene_name, what_to_run, modeling_approach, iterations):
    results_dict_list = []
    for i in range(0, iterations):
        seed = np.random.randint(low=1, high=270620)
        data = get_prepared_data(gene_name, seed)  
        evaluate = run_evaluate(data, what_to_run, modeling_approach, seed)
        print(evaluate.results_dict)
        results_dict_list.append(evaluate.results_dict)

    results_df = pd.DataFrame.from_dict(results_dict_list)
    print(results_df.sort_values(by='Test AUC', ascending=False))
    sns.histplot(data=results_df, x='Test AUC')
    plt.show()

# seed = np.random.randint(low=1, high=280293)
# data = get_prepared_data('KCNQ1', seed) 
# run_evaluate(data, 'grid_search', 'LR', seed)
# run_evaluate(data, 'grid_search', 'DecisionTree', seed)
# run_evaluate(data, 'grid_search', 'RandomForest', seed)
# run_evaluate(data, 'grid_search', 'GradientBoosting', seed)

run_evaluate_with_loop('KCNQ1', 'grid_search', 'LR', 3)
#run_evaluate_with_loop('KCNQ1', 'grid_search', 'DecisionTree', 500)
# run_evaluate_with_loop('KCNQ1', 'grid_search', 'RandomForest', 5)