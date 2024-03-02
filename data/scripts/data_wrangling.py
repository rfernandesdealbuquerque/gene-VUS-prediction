import sys

import pandas as pd

sys.path.append('../../')
import config

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

gene_name = 'KCNQ1'

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

def remove_dups(df, key, keep):
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
        healthy['Label'] = -1
        diseased['Label'] = 1
        
        #concat
        # helen = healthy.shape[0]
        # dislen = diseased.shape[0]
        # diseased.index = range(helen,helen+dislen)
        return pd.concat([healthy,diseased])
        
healthy_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_LB_B_Variants.csv').sort_values(by='AA position').reset_index(drop=True)
diseased_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_LP_P_Variants.csv').sort_values(by='AA position').reset_index(drop=True)
uncertain_df = pd.read_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_Uncertain_Variants.csv').sort_values(by='AA position').reset_index(drop=True)

        
#Clean the data and add in signal to noise
dfs = {'healthy': healthy_df, 'pathologic': diseased_df,
                'uncertain': uncertain_df}
for key in dfs.keys():
    df = dfs[key]
    print('\n***Working on',key,'***')
    df = check_symbols(df, key)
    df = remove_original_equals_change(df, key)
    df = remove_dups(df, key, keep='first')
    dfs[key] = df

    # update the variables with clean data
healthy_df = dfs['healthy']
diseased_df = dfs['pathologic']
uncertain_df = dfs['uncertain']
        
#Merge healthy and diseased
merged_df = merge_healthy_and_diseased(healthy_df, diseased_df)
print('merged shape:', merged_df.shape)
merged_df = remove_dups(merged_df, 'merged', keep=False)
merged_df = merged_df.sort_values(by='AA position').reset_index(drop=True)
merged_df = merged_df[['AA position', 'Original', 'Change', 'Label']]

print(merged_df)
        
        # #make sure there are no overlaps between mystery and merged:
        # #note that duplicates between healthy and sick have been removed
        # #so those can stay in mysteryAAs (makes sense since if they're listed
        # #as both healthy and sick we don't know the right answer.)
        # print('mysteryAAs shape:',self.mysteryAAs.shape)
        # temp = copy.deepcopy(self.merged).drop(columns='Label')
        # mystmerged = self.merge_healthy_and_diseased(temp, self.mysteryAAs)
        # mystmerged = self.remove_dups(mystmerged, 'mystmerged', False, 'duplicate_across_mystery_and_healthysick_removed_both')
        # self.mysteryAAs = (mystmerged[mystmerged['Label']==1]).drop(columns='Label')
        # print('mysteryAAs shape after:',self.mysteryAAs.shape)
        
