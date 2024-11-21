import pandas as pd
from sklearn.model_selection import train_test_split
import os

output_dir = './data/processed_data'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('./data/raw_data/raw.csv')

X = df.drop(['silica_concentrate', 'date'], axis=1)
y = df.silica_concentrate

print(df.drop('date', axis=1).corr())
from scipy import stats

# Test de normalité de Shapiro-Wilk
stat, p_value = stats.shapiro(y)
print(f"Statistique de Shapiro-Wilk : {stat}")
print(f"P-value : {p_value}")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=34, test_size=0.2)

X_train.to_csv(f'{output_dir}/X_train.csv')
X_test.to_csv(f'{output_dir}/X_test.csv')
y_train.to_csv(f'{output_dir}/y_train.csv')
y_test.to_csv(f'{output_dir}/y_test.csv')


