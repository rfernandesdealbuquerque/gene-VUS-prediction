import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from data import prepare_data

#Custom imports
from src import evaluate_models, make_figures, run_models
from utils import get_best_hyperparameters, get_median_model_seed


class RunStudy:

    def __init__(self, gene_name, what_to_run, modeling_approach, iterations):
        self.gene_name = gene_name
        self.modeling_approach = modeling_approach
        self.what_to_run = what_to_run
        self.iterations = iterations
        self.results_df = pd.DataFrame()
        self.best_hyperparameters = pd.DataFrame()
        self.best_model = None
        self.data_median_model = None #PrepareData object


        self.run_evaluate_with_various_seeds(self.gene_name, self.what_to_run, self.modeling_approach, self.iterations)
        make_figures.MakeFigures(self.gene_name, self.best_model, self.data_median_model, self.results_df, self.modeling_approach, self.best_hyperparameters)

    def run_evaluate_with_various_seeds(self, gene_name, what_to_run, modeling_approach, iterations):
        """This runs the algorithm, picks the best model through GridSearch and then evaluate it. It does this multiple times with different seeds in order to get the results for various train/test splits. It returns a DataFrame with the performance metrics and the hyperparameters picked for the given model."""
        all_features =['AA position', 'Score', 'Signal To Noise','Original','Change','Functional Domain']
        results_dict_list = []
        unique_seeds = np.random.choice(np.arange(1, 270600), size=iterations, replace=False)
        for i, seed in enumerate(unique_seeds):
            data = prepare_data.PrepareData(gene_name, all_features, seed) #Get data object with train/test splits for given seed
            print('\n*** Running ', modeling_approach,'***')
            run = run_models.RunModels(modeling_approach, data, what_to_run, seed, n_iter=20) 
            evaluate = evaluate_models.EvaluateModels(modeling_approach, data, run.best_model, seed) #Evaluate best_model obtained from cross validation
            #print(evaluate.results_dict)
            results_dict_list.append(evaluate.results_dict)

        results_df = pd.DataFrame.from_dict(results_dict_list)
        results_df = results_df.sort_values(by=list(results_df.columns[3:]), ascending=False)
        #We need to group by hyperparameters, see which group performed the best across all different seeds and then use that group as the best model.
        print(results_df)
        self.results_df = results_df
        #Returns the group of hyperparameters that showed up the most and performed the best among the various runs of different seeds.
        best_hyperparameters = get_best_hyperparameters.get_best_hyperparameters(results_df) 
        self.best_hyperparameters = best_hyperparameters

        #Returns seed of median model among the models with best hyperparameters
        seed_median_model = get_median_model_seed.get_median_model_seed(best_hyperparameters, results_df)

        if modeling_approach == 'LR':
            self.best_model = LogisticRegression(
                penalty=best_hyperparameters.at[0,'Penalty'],
                C=best_hyperparameters.at[0,'C'],
                solver=best_hyperparameters.at[0, 'Solver'],
            )
            self.data_median_model = prepare_data.PrepareData(self.gene_name, all_features, seed_median_model) #Here we use the seed that produced the median model to split the data. Then, we fit the final model to this data using the best hyperparameter group.

        elif modeling_approach == 'DecisionTree':
            self.best_model = DecisionTreeClassifier(
                max_depth=int(best_hyperparameters.at[0,'max_depth']),
                criterion=best_hyperparameters.at[0, 'criterion'],
            )
            self.data_median_model = prepare_data.PrepareData(self.gene_name, all_features, seed_median_model) #Here we use the seed that produced the median model to split the data. Then, we fit the final model to this data using the best hyperparameter group.

        elif modeling_approach == 'RandomForest':
            self.best_model = RandomForestClassifier(
                n_estimators=int(best_hyperparameters.at[0,'n_estimators']),
                bootstrap=best_hyperparameters.at[0,'bootstrap'],
                max_depth=int(best_hyperparameters.at[0,'max_depth']),
                max_features=(best_hyperparameters.at[0,'max_features']),
                min_samples_leaf=int(best_hyperparameters.at[0,'min_samples_leaf']),
                min_samples_split=int(best_hyperparameters.at[0,'min_samples_split']),
            )
            self.data_median_model = prepare_data.PrepareData(self.gene_name, all_features, seed_median_model) #Here we use the seed that produced the median model to split the data. Then, we fit the final model to this data using the best hyperparameter group.           

        elif modeling_approach == 'GradientBoosting':
            self.best_model = GradientBoostingClassifier(
                n_estimators=int(best_hyperparameters.at[0,'n_estimators']),
                learning_rate=best_hyperparameters.at[0,'learning_rate'],
                max_depth=int(best_hyperparameters.at[0,'max_depth']),
                min_samples_leaf=int(best_hyperparameters.at[0,'min_samples_leaf']),
                min_samples_split=int(best_hyperparameters.at[0,'min_samples_split']),
                subsample=(best_hyperparameters.at[0,'subsample'])
            )
            self.data_median_model = prepare_data.PrepareData(self.gene_name, all_features, seed_median_model) #Here we use the seed that produced the median model to split the data. Then, we fit the final model to this data using the best hyperparameter group.  
            
RunStudy('KCNQ1', 'grid_search', 'LR', iterations=1000)
RunStudy('MYH7', 'grid_search', 'LR', iterations=1000)
RunStudy('RYR2', 'grid_search', 'LR', iterations=1000)

RunStudy('KCNQ1', 'grid_search', 'DecisionTree', iterations=1000)
RunStudy('MYH7', 'grid_search', 'DecisionTree', iterations=1000)
RunStudy('RYR2', 'grid_search', 'DecisionTree', iterations=1000)

RunStudy('KCNQ1', 'rand_search', 'RandomForest', iterations=1000)
RunStudy('MYH7', 'rand_search', 'RandomForest', iterations=1000)
RunStudy('RYR2', 'rand_search', 'RandomForest', iterations=1000)

RunStudy('KCNQ1', 'rand_search', 'GradientBoosting', iterations=1000)
RunStudy('MYH7', 'rand_search', 'GradientBoosting', iterations=1000)
RunStudy('RYR2', 'rand_search', 'GradientBoosting', iterations=1000)
