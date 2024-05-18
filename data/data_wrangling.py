import copy
import sys

import config
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

gene_name = 'KCNQ1'
# gene_name = 'MYH7'
# gene_name = 'RYR2'

def check_symbols(df, key):
        """Check symbols so that everything in the Original and Change columns
        corresponds to a letter in AMINO_ACIDS"""
    
        print('running check_symbols()...')
        #Just print off when you make a change what that change is
        for col in ['Original','Change']:
            for i in df.index.values:
                current_aa = df.at[i, col]
                if current_aa not in AMINO_ACIDS:
                     print(f'{current_aa} in {col} not in AMINO_ACIDS')
                    
        return df

def remove_original_equals_change(df, key):
        """Remove anything that is not a real mutation (i.e. the change is the
        same as the original)"""
        #Document history of upcoming change
        print('running remove_original_equals_change()...')
        temp = df.loc[df['Original'] == df['Change']]
        if(temp.shape[0] > 0):
            print(temp)

        #Keep only rows that original != change
        df = df.loc[df['Original'] != df['Change']]

        return df

def remove_dups(df, keep):
        """Remove duplicates.
        if keep == 'first' then keep first occurrence
        if keep == False then drop all occurrences """
        #Document history of upcoming change
        cols = ['Original','AA position','Change']
        # get a dataframe of the duplicataed rows
        temp = df[df.duplicated(subset=cols,keep=keep)]
        
        print('running remove_dups()...')
        print('\tRows before:', df.shape[0])
        df = df.drop_duplicates(subset=cols, keep=keep)
        print('\tRows after:',df.shape[0])

        return df

def merge_healthy_and_diseased(healthy, diseased):
        """Return merged dataframe for healthy and diseased"""
        #add Label column
        healthy['Label'] = 0
        diseased['Label'] = 1
        
        #concat
        # helen = healthy.shape[0]
        # dislen = diseased.shape[0]
        # diseased.index = range(helen,helen+dislen)
        return pd.concat([healthy,diseased])
        
healthy_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_LB_B_Variants.csv').sort_values(by='AA position').reset_index(drop=True)
diseased_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_LP_P_Variants.csv').sort_values(by='AA position').reset_index(drop=True)
uncertain_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_Uncertain_Variants.csv').sort_values(by='AA position').reset_index(drop=True)

signal_to_noise_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_Signal_to_Noise.csv').sort_values(by='AA position').reset_index(drop=True)
signal_to_noise_df[['AA position', 'Functional Domain']] = signal_to_noise_df[['AA position', 'Functional Domain']].astype(int)
# print(signal_to_noise_df.head())
conservation_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_Conservation_Score.csv').sort_values(by='AA position').reset_index(drop=True)
# print(conservation_df.head())

        
#Clean the data and add in signal to noise
dfs = {'healthy': healthy_df, 'diseased': diseased_df,
                'uncertain': uncertain_df}
for key in dfs.keys():
    df = dfs[key]
    print('\n***Working on',key,'***')
    df = check_symbols(df, key)
    df = remove_original_equals_change(df, key)
    df = remove_dups(df, keep='first')
    dfs[key] = df

# update the variables with clean data
healthy_df = dfs['healthy']
diseased_df = dfs['diseased']
uncertain_df = dfs['uncertain']
        
#Concat healthy and diseased
print('\n***Merging healthy and diseased***')
concat_df = merge_healthy_and_diseased(healthy_df, diseased_df)
print('merged shape:', concat_df.shape)
concat_df = remove_dups(concat_df, keep=False)
concat_df = concat_df.sort_values(by='AA position').reset_index(drop=True)
concat_df = concat_df[['AA position', 'Original', 'Change', 'Label']]

#Merge Signal to Noise Ratio
merged_df = concat_df.merge(signal_to_noise_df, on='AA position', how='left').sort_values(by='AA position')

#Merge Conservation Score
merged_df = merged_df.merge(conservation_df, on='AA position', how='left').sort_values(by='AA position')

#Reorder columns
# merged_df = merged_df.reindex(columns=[['AA position', 'Original', 'Change', 'LQTS/GnomAD', 'Functional Domain', 'Score', 'Label']])
print(merged_df.head())
merged_df.to_csv(config.data + f'/{gene_name}/3. Clean/{gene_name}_dataset.csv')

#Here we make sure there are no overlaps between uncertain and merged:
#note that duplicates between healthy and diseased have been removed
#so those can stay in uncertain (makes sense since if they're listed
#as both healthy and sick we don't know the right answer.)
print('uncertain shape:', uncertain_df.shape)
temp = copy.deepcopy(merged_df).drop(columns='Label')
uncertain_merged = merge_healthy_and_diseased(temp, uncertain_df)
uncertain_merged = remove_dups(uncertain_merged, keep=False)
uncertain_df = (uncertain_merged[uncertain_merged['Label']==1]).drop(columns='Label')
print('uncertain shape after:', uncertain_df.shape)
        
