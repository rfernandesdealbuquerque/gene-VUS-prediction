import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from scipy import stats

matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt

plt.ioff() #turn interactive plotting off
import matplotlib.lines as mlines


class MakeFigures:
    def __init__(self, gene_name, model, data, results_df, modeling_approach):
        self.gene_name = gene_name
        self.save_dir = "C:/dev/Python/gene-VUS-prediction/figures/"
        self.model = model #model to plot ROC for. Can be the best model or median model across all the seeds.
        self.data = data 
        self.results_df = results_df #this is the DataFrame containing the evaluated metrics and hyperparameters for each seed used.
        self.modeling_approach = modeling_approach
        #Plot characteristics
        #COlors: https://matplotlib.org/tutorials/colors/colors.html
        self.color = 'crimson'
        self.linestyle = 'solid'
        self.neutral_color = 'k'
        self.neutral_linestyle = 'dashed'
        self.lw = 2

        #Plot
        self.plot_roc_curve()
        self.plot_performance_hist()

    def plot_roc_curve(self):
        #ROC curve for model evaluation
        model = self.model
        pred_probs = model.decision_function(self.data.X_test)
        true_labels = self.data.y_test
        fpr, tpr, _ = sklearn.metrics.roc_curve(true_labels, pred_probs,pos_label = 1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        line, = plt.plot(fpr, tpr, color=self.color, lw=self.lw, linestyle = self.linestyle)

        plt.plot([0, 1], [0, 1], color=self.neutral_color, lw=self.lw, linestyle=self.neutral_linestyle) #diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.gene_name} - ROC of Best Model - {self.modeling_approach}')
        plt.legend([line], [f'{self.modeling_approach}, AUROC=%0.2f' % roc_auc,], loc='lower right')
        plt.savefig(os.path.join(self.save_dir, self.gene_name+f'_Best_Model_{self.modeling_approach}_ROC_Curve.jpg'))
        plt.show()
        plt.close()

        #ROC curve for algorithm evaluation
        
    def plot_performance_hist(self):
        results_df = self.results_df
        sns.histplot(data=results_df, x='Test AUC')
        plt.xlabel('Test AUC')
        plt.title(f'{self.gene_name} - AUC for {len(results_df)} Different Random Seeds - {self.modeling_approach}')
        plt.savefig(os.path.join(self.save_dir, self.gene_name+'_Performance_Hist.jpg'))