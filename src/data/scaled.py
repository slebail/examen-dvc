import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('./data/processed_data/X_train.csv', index_col=0)
X_test = pd.read_csv('./data/processed_data/X_test.csv', index_col=0)



scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(X_train_scaled.head())
print(X_test_scaled.head())

X_train_scaled.to_csv('./data/processed_data/X_train_scaled.csv')
X_test_scaled.to_csv('./data/processed_data/X_test_scaled.csv')

print(X_train_scaled.shape)
print(X_test_scaled.shape)