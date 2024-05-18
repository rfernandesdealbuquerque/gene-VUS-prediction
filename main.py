import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import prepare_data

#Custom imports
from src import evaluate_models, make_figures, run_models


class RunStudy:

    def __init__(self, gene_name, what_to_run, modeling_approach, iterations):
        self.gene_name = gene_name
        self.modeling_approach = modeling_approach
        self.what_to_run = what_to_run
        self.iterations = iterations
        self.best_cross_validation_models = {} #this keeps a dictionary of the best model obtained for the cross-validation of the given seed. Key: seed, Value: best_model


        self.run_evaluate_with_various_seeds(self.gene_name, self.what_to_run, self.modeling_approach, self.iterations)
        make_figures.MakeFigures(self.gene_name, self.median_model, self.data_median_model, self.results_df, self.modeling_approach)

             
    def get_prepared_data(self, gene_name, seed):
        """Parameters:
        <gene_name> a string, either 'ryr2', 'kcnh2', 'kcnq1', or 'scn5a'"""

        all_features =['AA position', 'Conservation Score', 'LQTS/GnomAD','Original','Change','Functional Domain']
        #Prepare data and split in train and test for given gene_name
        data = prepare_data.PrepareData(gene_name, all_features, seed)
        return data

    def run_evaluate(self, data, what_to_run, modeling_approach, seed):
        """Parameters:
        <data>: data
        <what_to_run> a string, one of:
            'grid_search': this will do a grid search over different model setups
                
        <modeling_approach>: a string, 'LR' for logistic regression"""
        
        print('\n*** Running ',modeling_approach,'***')
        if what_to_run == 'grid_search':
            run = run_models.RunModels(modeling_approach, data, what_to_run, seed)
            evaluate = evaluate_models.EvaluateModels(modeling_approach, data, run.best_model, seed)
            self.best_cross_validation_models[seed] = run.best_model
        
        return evaluate

    def run_evaluate_with_various_seeds(self, gene_name, what_to_run, modeling_approach, iterations) -> pd.DataFrame:
        """This runs the run_evaluate method multiple times with different seeds in order to get the results for various train/test splits. It returns a DataFrame with the performance metrics and the hyperparameters picked for the given model."""
        results_dict_list = []
        for i in range(0, iterations):
            seed = np.random.choice(np.arange(1, 270620), replace=False)
            data = self.get_prepared_data(gene_name, seed)  
            evaluate = self.run_evaluate(data, what_to_run, modeling_approach, seed)
            print(evaluate.results_dict)
            results_dict_list.append(evaluate.results_dict)

        results_df = pd.DataFrame.from_dict(results_dict_list).sort_values(by='Test AUC', ascending=False)
        self.results_df = results_df
        seed_median_model = results_df.iloc[len(results_df)//2, 0]
        seed_best_model = results_df.iloc[0, 0]
        self.median_model = self.best_cross_validation_models[seed_median_model]
        self.best_model = self.best_cross_validation_models[seed_best_model]
        self.data_best_model = self.get_prepared_data(self.gene_name, seed_best_model)
        self.data_median_model = self.get_prepared_data(self.gene_name, seed_median_model)



    # seed = np.random.randint(low=1, high=280293)
    # data = get_prepared_data('KCNQ1', seed) 
    # run_evaluate(data, 'grid_search', 'LR', seed)
    # run_evaluate(data, 'grid_search', 'DecisionTree', seed)
    # run_evaluate(data, 'grid_search', 'RandomForest', seed)
    # run_evaluate(data, 'grid_search', 'GradientBoosting', seed)


RunStudy('KCNQ1', 'grid_search', 'LR', iterations=10)



#run_evaluate_with_loop('KCNQ1', 'grid_search', 'DecisionTree', 500)
# run_evaluate_with_loop('KCNQ1', 'grid_search', 'RandomForest', 5)