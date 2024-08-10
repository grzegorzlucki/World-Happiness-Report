import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
from pathlib import Path

base_dir = Path(__file__).resolve().parent
file_path = base_dir.parent / 'data' / 'data_2022.csv'
HappinessScore = pd.read_csv(file_path, encoding='utf-8')

HappinessScore = HappinessScore.iloc[:, 1:]

X = HappinessScore.iloc[:, 1:].values
y = HappinessScore.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

regressor = XGBRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted values')
plt.scatter(range(len(y_test)), y_test, color='red', label='Real values')
plt.xlabel('Data points')
plt.ylabel('Happiness Score values')
plt.title('Scatter plot of Happiness Score')
plt.legend()
plt.show()

file_path = base_dir.parent / 'models'
joblib.dump(regressor, f'{file_path}/xgboost_model.pkl')
joblib.dump(sc, f'{file_path}/scaler.pkl')