import numpy as np
import pandas as pd


def get_best_hyperparameters(results_df) -> pd.DataFrame:
    '''This gets results_df, groups the models with same hyperparameters, averages the performance of each group and returns the hyperparameters with best average.'''
    mean_AUC_overall = results_df['Test AUC'].mean()
    grouped = results_df.groupby(results_df.columns.tolist()[4:])['Test AUC'].agg(['mean', 'count']).reset_index()
    grouped['mean'] = grouped['mean'].round(2)
    #We have to account for the quantity of models. If a group of hyperparameters appears much more among the best ones, their average must have more weight.
    #Calculate the total number of models
    total_models = grouped['count'].sum()
    #Calculate the weighted mean by the proportion of each group
    grouped['weight'] = grouped['count'] / total_models
    grouped['score'] = grouped['mean'] * grouped['weight']
    grouped['score'] = grouped['score'].round(3)
    grouped['weight'] = grouped['weight'].round(3)
    print(grouped)
    best_hyperparameters = grouped.loc[grouped[['score']].idxmax()]
    best_hyperparameters.drop(['count'], inplace=True, axis=1)
    best_hyperparameters.drop(['score'], inplace=True, axis=1)
    best_hyperparameters.rename(columns={'mean': 'best avg AUC'}, inplace=True)
    best_hyperparameters['overall avg AUC'] = mean_AUC_overall.round(2)
    best_hyperparameters.reset_index(inplace=True, drop=True)
    print(best_hyperparameters)
    return best_hyperparameters