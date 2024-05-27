import numpy as np
import pandas as pd


def get_median_model_seed(best_hyperparameters, results_df) -> int:
    '''Returns seed of the median model among the models with the best hyperparameters group.'''
    filtered_df = results_df

    print(list(best_hyperparameters.columns)[:-4])
    for col in list(best_hyperparameters.columns)[:-4]:
        filtered_df = filtered_df[filtered_df[f'{col}'] == best_hyperparameters.at[0, f'{col}']] #Get from results_df only the rows with the best hyperparameters group
    print(filtered_df)
    filtered_df.sort_values(by='Test AUC', ascending=False, inplace=True)
    seed_median_model = filtered_df.iloc[len(filtered_df)//2, 0]
    print(seed_median_model)
    return seed_median_model