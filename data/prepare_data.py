import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append('../../')
import data.config as config

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class PrepareData(object):
    def __init__(self, gene_name, all_features, seed):
        """This class produces self.real_data_split and self.mysteryAAs_split
        which are needed for all the modeling."""
        print('*** Preparing data for',gene_name,'***')
        self.gene_name = gene_name
        self.seed = seed
        
        #Load real data consisting of benign and pathologic mutations
        df = pd.read_csv(config.data + f'/{gene_name}/3. Clean/{gene_name}_dataset.csv').iloc[:,1:]
        df.iloc[:,[0,1,2,4,5,6,3]] #Putting Label as the last column
        X = df[all_features]
        X = pd.get_dummies(X, drop_first=True, dtype=int)
        #self.one_hot_encoding_top_n(X, ['Original', 'Change'], 5)
        # print(X.head(10))
        y = df['Label']

        
        # print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=self.seed, stratify=y)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def one_hot_encoding_top_n(self, data, columns, n):
        """This will get the top n most frequent categories of the input columns and create one hot encoding for those only and drop the original categorical columns and, consequently, the other less frequent values."""
        for col in columns:
            top_n = [k for k in data[col].value_counts().sort_values(ascending=False).head(n).index]

            for cat in top_n:
                data[col + '_'+ cat] = np.where(data[col]==cat, 1, 0)

        data.drop(columns, axis=1, inplace=True)