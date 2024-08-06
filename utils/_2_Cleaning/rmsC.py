import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.rmsI import *
    
def load_data(file_path):
    """
     Loads data from a CSV file. This is a convenience function for use in testing. It will take a path to a CSV file and load it into a pandas dataframe
     
     @param file_path - Path to the CSV file
     
     @return DataFrame with the data from the CSV file as a column and the columns as a row. Example :
    """
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    """
     Split data into training and test sets.
     
     @param df - pandas dataframe with data to split
     
     @return x_train x_test y_train y_test : numpy arrays of data for training and
    """
    x = df.drop('price', axis=1) 
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
