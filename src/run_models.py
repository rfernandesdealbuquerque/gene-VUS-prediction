from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


#####################################################################
# Grid Search - Calculate Performance Across Different Model Setups #-----------
#####################################################################
class RunPredictiveModels(object):
    """Run models using cross validation
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
    def __init__(self, gene_name, modeling_approach,
                 data, what_to_run, testing):
        self.gene_name = gene_name
        self.modeling_approach = modeling_approach
        assert self.modeling_approach in ['MLP','LR']
        self.data = data
        if testing:
            self.number_of_cv_folds = 2
        else:
            self.number_of_cv_folds = 10 #ten fold cross validation
        self.max_epochs = 1000
        self.what_to_run = what_to_run
        self.testing = testing
        
        if what_to_run == 'grid_search':
            self._run_grid_search()
        elif what_to_run == 'test_pred':
            self._run_test_pred()
            
    # Grid Search Method #------------------------------------------------------
    def _run_grid_search(self):
        """Perform a grid search across predefined architectures and
        hyperparameters for a given gene <gene_name> to determine the best
        model setup for the given modeling approach."""
            
        if self.modeling_approach == 'LR':
            if self.testing:
                self._initialize_search_params_lr_testing()
            else:
                self._initialize_grid_search_params_lr()
            self._run_lr_grid_search()

    # LR Methods #--------------------------------------------------------------
    def _initialize_grid_search_params_lr_testing(self):
        """Initialize a small list of hyperparameters to try for testing purposes"""
        # Define hyperparameters grid for grid search
        param_grid = {
        'penalty': ['l1'],  # Regularization penalty
        'C': [0.1],  # Inverse of regularization strength
        'solver': ['liblinear', 'saga']  # Optimization algorithm
        }
    
    def _initialize_grid_search_params_lr(self):
        """Initialize lists of hyperparmeters to assess"""
        # Define hyperparameters grid for grid search
        self.param_grid = {
        'penalty': ['l1', 'l2'],  # Regularization penalty
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'solver': ['liblinear', 'saga']  # Optimization algorithm
        }

    
    def _run_lr_grid_search(self):
        """Run grid search with logistic regression model for each combination of hyperparameters"""
        # Define GridSearchCV object
        grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=self.param_grid, cv=self.number_of_cv_folds, scoring='roc_auc', n_jobs=-1)

        # Perform GridSearchCV
        grid_search.fit(self.data.X_train, self.data.y_train)
        y_pr=grid_search.decision_function(self.data.X_train)
        print(grid_search.score(self.data.X_train, self.data.y_train))
        print(roc_auc_score(self.data.y_train, y_pr))



        print('Best Auc:', grid_search.best_score_)
        print('Best Params:', grid_search.best_params_)
    