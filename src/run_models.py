import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


#####################################################################
# Grid Search - Calculate Performance Across Different Model Setups #-----------
#####################################################################
class RunModels(object):
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
    def __init__(self, modeling_approach,
                 data, what_to_run, seed, n_iter):
        
        self.modeling_approach = modeling_approach
        assert self.modeling_approach in ['MLP','LR', 'DecisionTree', 'RandomForest', 'GradientBoosting']
        self.data = data
       
        self.number_of_cv_folds = 10 #ten fold cross validation
        self.what_to_run = what_to_run
        self.best_model = None
        self.seed = seed
        self.n_iter = n_iter
        
        if what_to_run == 'grid_search':
            self._run_grid_search()
        elif what_to_run == 'rand_search':
            self._run_rand_search()
            
    # Grid Search Method #------------------------------------------------------
    def _run_grid_search(self):
        """Perform a grid search across predefined architectures and
        hyperparameters for a given gene <gene_name> to determine the best
        model setup for the given modeling approach."""
            
        if self.modeling_approach == 'LR':

            self._initialize_grid_search_params_lr()
            self._run_lr_grid_search()

        elif self.modeling_approach == 'DecisionTree':

            self._initialize_grid_search_params_decision_tree()
            self._run_decision_tree_grid_search()

        elif self.modeling_approach == 'RandomForest':

            self._initialize_grid_search_params_random_forest()
            self._run_random_forest_grid_search()   

        elif self.modeling_approach == 'GradientBoosting':

            self._initialize_grid_search_params_gradient_boosting()
            self._run_gradient_boosting_grid_search() 


    # LR Methods #--------------------------------------------------------------
    
    def _initialize_grid_search_params_lr(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for grid search
        self.param_grid = {
        'penalty': ['l1', 'l2'],  # Regularization penalty
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],  # Inverse of regularization strength
        'solver': ['liblinear', 'saga'],  # Optimization algorithm
        'max_iter': [100]
        }

    
    def _run_lr_grid_search(self):
        """Run grid search with logistic regression model for each combination of hyperparameters"""
        # Define GridSearchCV object
        gs = GridSearchCV(estimator=LogisticRegression(), param_grid=self.param_grid, cv=self.number_of_cv_folds, scoring='roc_auc', n_jobs=-1)

        # Perform GridSearchCV
        gs.fit(self.data.X_train, self.data.y_train)
        print('Best K-fold Auc:', gs.best_score_) #this is the avg. auc performance across the 10 folds of the best model. The model with this score is used to get the best_params_
        print('Best LR Params:', gs.best_params_)
        self.best_model = gs #save GridSearch object


    # Decision Tree Methods #--------------------------------------------------------------
    def _initialize_grid_search_params_decision_tree(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for grid search
        self.param_grid = {
            'max_depth': [1, 2, 3, 4, 5, 6, 7, None],
            'criterion': ['gini', 'entropy']
            }
    
    def _run_decision_tree_grid_search(self):
        """Run grid search with decision tree model for each combination of hyperparameters"""
        # Define GridSearchCV object
        gs = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=self.param_grid, cv=self.number_of_cv_folds, scoring='roc_auc', n_jobs=-1)

        # Perform GridSearchCV
        gs.fit(self.data.X_train, self.data.y_train)
        print('Best K-Fold Auc:', gs.best_score_) #this is the best model avg. auc performance across the 10 folds. The model with this score is used to get the best_params_
        print('Best DecisionTree Params:', gs.best_params_)
        self.best_model = gs #save GridSearch object

    # RandomForest Methods #--------------------------------------------------------------
    def _initialize_grid_search_params_random_forest(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for grid search
        self.param_grid = {'n_estimators': [100,   200, 500],
                            'bootstrap': [True, False],
                            'max_depth': [10, 20, 30, None],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 5, 10],
                            'min_samples_split': [2, 5, 10]}
    
    def _run_random_forest_grid_search(self):
        """Run grid search with random forest model for each combination of hyperparameters"""
        # Define GridSearchCV object
        gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid=self.param_grid, cv=self.number_of_cv_folds, scoring='roc_auc', n_jobs=-1)

        # Perform GridSearchCV
        gs.fit(self.data.X_train, self.data.y_train)
        print('Best K-Fold Auc:', gs.best_score_) #this is the best model avg. auc performance across the 10 folds. The model with this score is used to get the best_params_
        print('Best RandomForest Params:', gs.best_params_)
        self.best_model = gs #save GridSearch object

     # GradientBoosting Methods #--------------------------------------------------------------
    def _initialize_grid_search_params_gradient_boosting(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for grid search
        self.param_grid = { 'max_depth': [10, 20, None],
                            'max_features': ['sqrt'],
                            'min_samples_leaf': [1, 2],
                            'min_samples_split': [2, 5],
                            'n_estimators': [200, 400]}
    
    def _run_gradient_boosting_grid_search(self):
        """Run grid search with gradient boosting model for each combination of hyperparameters"""
        # Define GridSearchCV object
        gs = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=self.param_grid, cv=self.number_of_cv_folds, scoring='roc_auc', n_jobs=-1)

        # Perform GridSearchCV
        gs.fit(self.data.X_train, self.data.y_train)
        print('Best K-Fold Auc:', gs.best_score_) #this is the best model avg. auc performance across the 10 folds. The model with this score is used to get the best_params_
        print('Best GradientBoosting Params:', gs.best_params_)
        self.best_model = gs #save GridSearch object

        


    # Randomized Search Method #------------------------------------------------------
    def _run_rand_search(self):
        """Perform a grid search across predefined architectures and
        hyperparameters for a given gene <gene_name> to determine the best
        model setup for the given modeling approach."""
            
        if self.modeling_approach == 'LR':

            self._initialize_rand_search_params_lr()
            self._run_lr_rand_search()

        elif self.modeling_approach == 'DecisionTree':

            self._initialize_rand_search_params_decision_tree()
            self._run_decision_tree_rand_search()

        elif self.modeling_approach == 'RandomForest':

            self._initialize_rand_search_params_random_forest()
            self._run_random_forest_rand_search()   

        elif self.modeling_approach == 'GradientBoosting':

            self._initialize_rand_search_params_gradient_boosting()
            self._run_gradient_boosting_rand_search() 

    
    # LR Methods #--------------------------------------------------------------
    
    def _initialize_rand_search_params_lr(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for RandomizedSearch
        self.param_grid = {
        'penalty': ['l1', 'l2'],  # Regularization penalty
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],  # Inverse of regularization strength
        'solver': ['liblinear', 'saga'],  # Optimization algorithm
        'max_iter': [100]
        }

    
    def _run_lr_rand_search(self):
        """Run grid search with logistic regression model for each combination of hyperparameters"""
        # Define RandomizedSearchCV object
        rs = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions=self.param_grid, cv=self.number_of_cv_folds, n_iter=self.n_iter, scoring='roc_auc', n_jobs=-1)

        # Perform RandomizedSearchCV
        rs.fit(self.data.X_train, self.data.y_train)
        print('Best K-fold Auc:', rs.best_score_) #this is the avg. auc performance across the 10 folds of the best model. The model with this score is used to get the best_params_
        print('Best LR Params:', rs.best_params_)
        self.best_model = rs #save RandomizedSearch object


    # Decision Tree Methods #--------------------------------------------------------------
    def _initialize_rand_search_params_decision_tree(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for RandomizedSearch
        self.param_grid = {
            'max_depth': [1, 2, 3, 4, 5, 6, 7, None],
            'criterion': ['gini', 'entropy']
            }
    
    def _run_decision_tree_rand_search(self):
        """Run grid search with decision tree model for each combination of hyperparameters"""
        # Define RandomizedSearch object
        rs = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=self.param_grid, cv=self.number_of_cv_folds, n_iter=self.n_iter, scoring='roc_auc', n_jobs=-1)

        # Perform RandomizedSearchCV
        rs.fit(self.data.X_train, self.data.y_train)
        print('Best K-Fold Auc:', rs.best_score_) #this is the best model avg. auc performance across the 10 folds. The model with this score is used to get the best_params_
        print('Best DecisionTree Params:', rs.best_params_)
        self.best_model = rs #save RandomizedSearch object

    # RandomForest Methods #--------------------------------------------------------------
    def _initialize_rand_search_params_random_forest(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for RandomizedSearch
        self.param_grid = {'n_estimators': [100,   200, 500],
                            'bootstrap': [True, False],
                            'max_depth': [10, 20, 30, None],
                            'max_features': ['auto', 'sqrt'],
                            'min_samples_leaf': [1, 2, 5, 10],
                            'min_samples_split': [2, 5, 10]}
    
    def _run_random_forest_rand_search(self):
        """Run grid search with random forest model for each combination of hyperparameters"""
        # Define RandomizedSearchCV object
        rs = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=self.param_grid, cv=self.number_of_cv_folds, n_iter=self.n_iter, scoring='roc_auc', n_jobs=-1)

        # Perform RandomizedSearchCV
        rs.fit(self.data.X_train, self.data.y_train)
        print('Best K-Fold Auc:', rs.best_score_) #this is the best model avg. auc performance across the 10 folds. The model with this score is used to get the best_params_
        print('Best RandomForest Params:', rs.best_params_)
        self.best_model = rs #save RandomizedSearch object

     # GradientBoosting Methods #--------------------------------------------------------------
    def _initialize_rand_search_params_gradient_boosting(self):
        """Initialize lists of hyperparameters to assess"""
        # Define hyperparameters grid for RandomizedSearch
        self.param_grid = { 'max_depth': [10, 20, None],
                            'max_features': ['auto', 'sqrt'],
                            'min_samples_leaf': [1, 2],
                            'min_samples_split': [2, 5],
                            'n_estimators': [200, 400]}
    
    def _run_gradient_boosting_rand_search(self):
        """Run grid search with gradient boosting model for each combination of hyperparameters"""
        # Define RandomizedSearch object
        rs = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=self.param_grid, cv=self.number_of_cv_folds, n_iter=self.n_iter, scoring='roc_auc', n_jobs=-1)

        # Perform RandomizedSearchCV
        rs.fit(self.data.X_train, self.data.y_train)
        print('Best K-Fold Auc:', rs.best_score_) #this is the best model avg. auc performance across the 10 folds. The model with this score is used to get the best_params_
        print('Best GradientBoosting Params:', rs.best_params_)
        self.best_model = rs #save RandomizedSearch object