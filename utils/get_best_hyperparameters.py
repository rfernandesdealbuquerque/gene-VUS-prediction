import numpy as np
import pandas as pd


def get_best_hyperparameters(results_df):
    '''This gets results_df, groups the models with same hyperparameters, averages the performance of each group and returns the hyperparameters with best average.'''
    grouped = results_df.groupby(results_df.columns.tolist()[4:])['Test AUC'].agg(['mean', 'count']).reset_index()
    print(grouped)
    #We have to account for the quantity of models. If a group of hyperparameters appears much more among the best ones, their average must have more weight.
    #Calculate the total number of models
    total_models = grouped['count'].sum()
    #Calculate the weighted mean by the proportion of each group
    grouped['weight'] = grouped['mean'] * (grouped['count'] / total_models)
    print(grouped)
    best_group = grouped.loc[grouped[['mean']].idxmax()]
    print(best_group)
    best_group.drop(['count'], inplace=True, axis=1)
    best_group.rename({'mean': 'mean AUC'}, inplace=True)
    print(best_group)
    return best_group.reset_index()

