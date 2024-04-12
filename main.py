import datetime
import os

from data import prepare_data

#Custom imports
from src import evaluate_models, run_models


def get_prepared_data(gene_name):
    """Parameters:
    <gene_name> a string, either 'ryr2', 'kcnh2', 'kcnq1', or 'scn5a'"""

    all_features =['AA position', 'Conservation Score', 'LQTS/GnomAD','Original','Change','Functional Domain']
    #Prepare data and split in train and test for given gene_name
    data = prepare_data.PrepareData(gene_name, all_features)
    return data

def run(data, what_to_run, modeling_approach):
    """Parameters:
    <data>: data
    <what_to_run> a string, one of:
        'grid_search': this will do a grid search over different model setups
            
    <modeling_approach>: a string, 'LR' for logistic regression"""
    
    print('\n*** Running ',modeling_approach,'***')
    if what_to_run == 'grid_search':
        run = run_models.RunModels(modeling_approach, data, what_to_run)
        evaluate = evaluate_models.EvaluateModels(modeling_approach, data, run.best_model)


data = get_prepared_data('KCNQ1')   
run(data, 'grid_search', 'LR')
run(data, 'grid_search', 'DecisionTree')
# # run(data, 'grid_search', 'RandomForest')
# run(data, 'grid_search', 'GradientBoosting')