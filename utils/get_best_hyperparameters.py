import numpy as np
import pandas as pd


def get_best_hyperparameters(results_df):
    '''This gets results_df, groups the models with same hyperparameters, averages the performance of each group and returns the hyperparameters with best average.'''
    results_df_grouped = results_df.groupby(results_df.columns.tolist()[4:])['Test AUC'].agg(np.mean)
    print(results_df_grouped)
    #We have to find a way to account for the quantity of models. If a group of hyperparameters appears much more among the best ones, their average must have more weight. Maybe multiple avg. by log? 
