from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pickle as pkl
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import json


X_train_scaled = pd.read_csv('./data/processed_data/X_train_scaled.csv', index_col=0)
X_test_scaled = pd.read_csv('./data/processed_data/X_test_scaled.csv', index_col=0)

y_train = pd.read_csv('./data/processed_data/y_train.csv', index_col=0)
y_test = pd.read_csv('./data/processed_data/y_test.csv', index_col=0)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

with open('./models/best_model.pkl', 'rb') as file:
    best_model = pkl.load(file)

y_pred = best_model.predict(X_test_scaled)

y_pred_df = pd.DataFrame(y_pred, columns=['Predictions'])

rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
mae = mean_absolute_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)

y_pred_df.to_csv('./data/predictions.csv')

metrics = {
    'RMSE': rmse,
    'MAE': mae,
    'R²': r2
}

with open('./metrics/metrics.json', 'w') as json_file:
    json.dump(metrics, json_file, indent=4)
