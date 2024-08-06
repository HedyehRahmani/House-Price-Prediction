import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.rmsI import *

def train_linear_regression(x_train, y_train):
    """
     Train a linear regression model. This is a wrapper for the LR model used to train the linear regression model
     
     @param x_train - pandas DataFrame with columns of training data
     @param y_train - pandas Series with columns of training labels
     
     @return A trained LR model that can be used to predict the label of the data points in x_
    """
    lrmodel = LinearRegression().fit(x_train, y_train)
    return lrmodel

def train_decision_tree(x_train, y_train):
    """
     Train a decision tree. This is a wrapper for DecisionTreeRegressor. The input is a list of feature vectors and the output is a list of decision tree features
     
     @param x_train - list of feature vectors to train the decision tree on
     @param y_train - list of target vectors for training the decision tree
     
     @return a dictionary containing the decision tree features and their associated label as well as the training set's decision
    """
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
    dtmodel = dt.fit(x_train, y_train)
    return dtmodel

def train_random_forest(x_train, y_train):
    """
     Train a random forest regressor. This is a wrapper around RandomForestRegressor.
     
     @param x_train - pandas DataFrame with columns : id ( int ) train_time ( int )
     @param y_train - pandas Series with columns : id ( int ) train_time ( int )
     
     @return model with trained RF model ( sklearn. model. Regressor ) and training data ( sklearn. model. Estimator
    """
    rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    rfmodel = rf.fit(x_train, y_train)
    return rfmodel

def save_model(model, filename):
    """
     Saves a pickled model to a file. This is a convenience function for use in tests that need to be run on a model that has been loaded into memory before it can be used.
     
     @param model - The model to be saved. Should be a : py : class : ` ~gensim. models. Model ` instance.
     @param filename - The name of the file to save the model to
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """
     Load a pickled model from a file. This is useful for debugging purposes. If you want to run a model on multiple machines you should use : py : func : ` load_model_multiprocessing `
     
     @param filename - Name of the file to load.
     
     @return The model loaded from the file or ` ` None ` ` if the file doesn't exist or could not be
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
