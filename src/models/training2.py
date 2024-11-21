from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pickle as pkl
import pandas as pd


X_train_scaled = pd.read_csv('./data/processed_data/X_train_scaled.csv', index_col=0)
X_test_scaled = pd.read_csv('./data/processed_data/X_test_scaled.csv', index_col=0)

y_train = pd.read_csv('./data/processed_data/y_train.csv', index_col=0)
y_test = pd.read_csv('./data/processed_data/y_test.csv', index_col=0)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

with open('./models/best_params.pkl', 'rb') as file:
    best_params = pkl.load(file)

best_model = SVR(**best_params)

best_model.fit(X_train_scaled, y_train)

print(best_model.score(X_train_scaled, y_train))

with open('./models/best_model.pkl', 'wb') as file:
    pkl.dump(best_model, file)