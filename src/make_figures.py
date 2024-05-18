import os

import matplotlib
import numpy as np
import pandas as pd
import sklearn.metrics
from scipy import stats

matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt

plt.ioff() #turn interactive plotting off
import matplotlib.lines as mlines


class MakeFigures:
    def __init__(self, gene_name, save_dir, model, data):
        self.gene_name = gene_name
        self.save_dir = save_dir
        self.model = model
        self.data = data

        #Plot characteristics
        #COlors: https://matplotlib.org/tutorials/colors/colors.html
        self.color = 'crimson'
        self.linestyle = 'solid'
        self.neutral_color = 'k'
        self.neutral_linestyle = 'dashed'
        self.lw = 2

        #Plot
        self.plot_roc_curve()

    def plot_roc_curve(self):
        #ROC curve for model evaluation
        model = self.model
        pred_probs = model.decision_function(self.data.X_test)
        true_labels = self.data.Y_test
        fpr, tpr, _ = sklearn.metrics.roc_curve(true_labels, pred_probs,pos_label = 1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        line, = plt.plot(fpr, tpr, color=self.color, lw=self.lw, linestyle = self.linestyle)

        plt.plot([0, 1], [0, 1], color=self.neutral_color, lw=self.lw, linestyle=self.neutral_linestyle) #diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristics')
        plt.legend([line], ['LR, AUROC=%0.2f' % roc_auc,], loc='lower right')
        plt.savefig(os.path.join(self.save_dir, self.gene_name+'_Best_Model_ROC_Curve.pdf'))
        plt.close()

        #ROC curve for algorithm evaluation
        
