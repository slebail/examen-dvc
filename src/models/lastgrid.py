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

params = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'epsilon': [0.1, 0.2, 0.5, 1.0]
}

scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}

model = SVR()


grid = GridSearchCV(model, param_grid=params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

grid.fit(X_train_scaled, y_train)

print("Meilleurs paramètres :", grid.best_params_)
print("Meilleure erreur quadratique moyenne négative :", grid.best_score_)


best_model = grid.best_estimator_

with open('./models/best_params.pkl', 'wb') as file:
    pkl.dump(grid.best_params_, file)

