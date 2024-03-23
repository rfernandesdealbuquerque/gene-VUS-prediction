import datetime
import os

from data import prepare_data

#Custom imports
from src import run_models


def run(gene_name, what_to_run, modeling_approach):
    """Parameters:
    <gene_name> a string, either 'ryr2', 'kcnh2', 'kcnq1', or 'scn5a'
    <what_to_run> a string, one of:
        'grid_search': this will do a grid search over different model setups
        'test_pred': this will save the test set predictions for the best
            model setup identified in the grid_search (meaning that grid_search
            must be run before test_pred is run)
            
    <modeling_approach>: a string, 'LR' for logistic regression"""
    

    all_features =['AA position', 'Conservation Score', 'LQTS/GnomAD','Original','Change','Functional Domain']
    data = prepare_data.PrepareData(gene_name, all_features)

    if what_to_run == 'grid_search':
        run_models.RunPredictiveModels(gene_name, modeling_approach, data, what_to_run, testing=False)
    
run('KCNQ1', 'grid_search', 'LR')