import copy
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append('../../')
import config

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class PrepareData(object):
    def __init__(self, gene_name, all_features):
        """This class produces self.real_data_split and self.mysteryAAs_split
        which are needed for all the modeling."""
        print('*** Preparing data for',gene_name,'***')
        self.gene_name = gene_name
        
        #Load real data consisting of benign and pathologic mutations
        df = pd.read_csv(config.data + f'/{gene_name}/3. Clean/{gene_name}_dataset.csv')
        X = df[all_features]
        X = pd.get_dummies(X, drop_first=True, dtype=int)
        y = df['Label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test