import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.rmsI import *

def evaluate_model(model, x_train, y_train, x_test, y_test):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_mae = mean_absolute_error(train_pred, y_train)
    test_mae = mean_absolute_error(test_pred, y_test)
    return train_mae, test_mae

def plot_tree(model, filename):
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, feature_names=model.feature_names_in_)
    plt.savefig(filename, dpi=300)
    plt.close()
