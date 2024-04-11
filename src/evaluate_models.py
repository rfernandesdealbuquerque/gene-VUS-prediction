from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


class EvaluateModels(object):
    """Evaluate best models obtained in RunModels on the test set.
    Parameters:
    <gene_name>: a string e.g. 'ryr2'
    <modeling_approach>: a string, either 'MLP' or 'LR' (for logistic regression)
    <results_dir>: path to directory to save results
    <real_data_split>: data split defined in clean_data.py PrepareData class
    <what_to_run>: a string, either 'grid_search' (to perform grid search
        over many predefined model setups) or 'test_pred' (after a grid search
        is complete this will select the best model and then save the test set
        predictions of that model for use in e.g. visualization functions)
    <testing>: if True, and if <what_to_run>=='grid_search' then only run
        on a small number of possible settings (for code testing purposes)
    
    To run without ensembling, i.e. to run each architecture/hyperparameter
    grouping on only one instantiation of the model, set ensemble=[1]"""
    def __init__(self, modeling_approach,
                 data, best_model):
        
        self.best_model = best_model
        self.modeling_approach = modeling_approach
        assert self.modeling_approach in ['MLP','LR','DecisionTree', 'RandomForest', 'GradientBoosting']
        self.data = data
        
        if modeling_approach == 'LR':
            self._evaluate_lr()
        elif modeling_approach == 'DecisionTree':
            self._evaluate_decision_tree()

        elif modeling_approach == 'RandomForest':
            self._evaluate_random_forest()

        elif modeling_approach == 'GradientBoosting':
            self._evaluate_random_forest()        
    
    def _evaluate_lr(self):
        #this is the auc performance of the model with the best parameters trained on the whole training set.
        print('LR Train: ', self.best_model.score(self.data.X_train, self.data.y_train)) # this is the performance on the training set.
        print('LR Test: ',self.best_model.score(self.data.X_test, self.data.y_test)) # this is the performance on the test set.

    def _evaluate_decision_tree(self):
        #this is the auc performance of the model with the best parameters trained on the whole training set.
        print('DecisionTree Train: ', self.best_model.score(self.data.X_train, self.data.y_train)) # this is the performance on the training set.
        print('DecisionTree Test: ', self.best_model.score(self.data.X_test, self.data.y_test)) # this is the performance on the test set.

    def _evaluate_random_forest(self):
        #this is the auc performance of the model with the best parameters trained on the whole training set.
        print('RandomForest Train: ', self.best_model.score(self.data.X_train, self.data.y_train)) # this is the performance on the training set.
        print('RandomForest Test: ', self.best_model.score(self.data.X_test, self.data.y_test)) # this is the performance on the test set.
    
    def _evaluate_gradient_boosting(self):
        #this is the auc performance of the model with the best parameters trained on the whole training set.
        print('GradientBoosting Train: ', self.best_model.score(self.data.X_train, self.data.y_train)) # this is the performance on the training set.
        print('GradientBoosting Test: ', self.best_model.score(self.data.X_test, self.data.y_test)) # this is the performance on the test set.

        