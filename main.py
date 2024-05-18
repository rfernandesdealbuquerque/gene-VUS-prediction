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
        self.best_models_from_cross_validation = {} #this keeps a dictionary of the best model obtained for the cross-validation of the given seed. Key: seed, Value: best_model

        self.run_evaluate_with_various_seeds(self.gene_name, self.what_to_run, self.modeling_approach, self.iterations)
        make_figures.MakeFigures(self.gene_name, self.median_model, self.data_median_model, self.results_df, self.modeling_approach)

    def run_evaluate_with_various_seeds(self, gene_name, what_to_run, modeling_approach, iterations) -> pd.DataFrame:
        """This runs the run_evaluate method multiple times with different seeds in order to get the results for various train/test splits. It returns a DataFrame with the performance metrics and the hyperparameters picked for the given model."""
        all_features =['AA position', 'Score', 'Signal To Noise','Original','Change','Functional Domain']
        results_dict_list = []
        unique_seeds = np.random.choice(np.arange(1, 270600), size=iterations, replace=False)
        for i, seed in enumerate(unique_seeds):
            data = prepare_data.PrepareData(gene_name, all_features, seed) #Get data object with train/test splits for given seed
            print('\n*** Running ', modeling_approach,'***')
            run = run_models.RunModels(modeling_approach, data, what_to_run, seed) 
            evaluate = evaluate_models.EvaluateModels(modeling_approach, data, run.best_model, seed)
            self.best_models_from_cross_validation[seed] = run.best_model
            print(evaluate.results_dict)
            results_dict_list.append(evaluate.results_dict)

        results_df = pd.DataFrame.from_dict(results_dict_list).sort_values(by='Test AUC', ascending=False)
        print(results_df)
        self.results_df = results_df
        #Get seed for the best and median test AUC models to get the models and the splits used for their generation
        seed_median_model = results_df.iloc[len(results_df)//2, 0]
        seed_best_model = results_df.iloc[0, 0]
        self.median_model = self.best_models_from_cross_validation[seed_median_model]
        self.best_model = self.best_models_from_cross_validation[seed_best_model]
        self.data_best_model = self.get_prepared_data(self.gene_name, seed_best_model)
        self.data_median_model = self.get_prepared_data(self.gene_name, seed_median_model)


RunStudy('KCNQ1', 'grid_search', 'LR', iterations=10)
# RunStudy('MYH7', 'grid_search', 'LR', iterations=100)
# RunStudy('RYR2', 'grid_search', 'LR', iterations=100)

