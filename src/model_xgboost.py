import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


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

params = {'n_estimators': 130, 'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.9, 'colsample_bytree': 0.85, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

regressor = XGBRegressor(**params)
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

model_dir = base_dir / 'model'
model_dir.mkdir(parents=True, exist_ok=True)

with open(model_dir / 'xgboost_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

with open(model_dir / 'scaler.pkl', 'wb') as file:
    pickle.dump(sc, file)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "mean_absolute_error": mae,
    "mean_squared_error": mse,
    "r2_score": r2
            }
print(metrics)