# main.py
from utils._1_Imports.rmsI import *
from utils._2_Cleaning.rmsC import load_data, split_data
from utils._3_ModelTraining.rmsT import train_linear_regression, train_decision_tree, train_random_forest, save_model, load_model
from utils._4_ModelEvaluation.rmsE import evaluate_model, plot_tree


# Load and Split Data
file_path = 'Dataset/final.csv'
df = load_data(file_path)
x_train, x_test, y_train, y_test = split_data(df)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#Linear Regression MODEL
lrmodel = train_linear_regression(x_train, y_train)
lr_train_mae, lr_test_mae = evaluate_model(lrmodel, x_train, y_train, x_test, y_test)
print(f'Linear Regression - Train MAE: {lr_train_mae}, Test MAE: {lr_test_mae}')


#Decision Tree MODEL
dtmodel = train_decision_tree(x_train, y_train)
dt_train_mae, dt_test_mae = evaluate_model(dtmodel, x_train, y_train, x_test, y_test)
plot_tree(dtmodel, 'tree.png')
print(f'Decision Tree - Train MAE: {dt_train_mae}, Test MAE: {dt_test_mae}')

#Random Forest MODEL
rfmodel = train_random_forest(x_train, y_train)
rf_train_mae, rf_test_mae = evaluate_model(rfmodel, x_train, y_train, x_test, y_test)
print(f'Random Forest - Train MAE: {rf_train_mae}, Test MAE: {rf_test_mae}')

# # Save and load random forest model
# save_model(rfmodel, 'RE_Model')
# loaded_model = load_model('RE_Model')

# # Use loaded model for prediction
# prediction = loaded_model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]])
# print(f'Loaded model prediction: {prediction}')
