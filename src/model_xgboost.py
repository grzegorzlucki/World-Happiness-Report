import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow



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

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("XGBR-evaluation")

param_list = [
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1},
    {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 2, 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1},
    {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 1},
    {'n_estimators': 120, 'learning_rate': 0.08, 'max_depth': 3, 'min_child_weight': 3, 'subsample': 0.85, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 0.5},
    {'n_estimators': 110, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2, 'subsample': 0.8, 'colsample_bytree': 0.75, 'gamma': 0.3, 'reg_alpha': 0.2, 'reg_lambda': 0.8},
    {'n_estimators': 130, 'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.9, 'colsample_bytree': 0.85, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1},
    {'n_estimators': 160, 'learning_rate': 0.06, 'max_depth': 6, 'min_child_weight': 2, 'subsample': 0.75, 'colsample_bytree': 0.9, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.2},
    {'n_estimators': 180, 'learning_rate': 0.04, 'max_depth': 5, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0.3, 'reg_lambda': 1},
    {'n_estimators': 140, 'learning_rate': 0.07, 'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.85, 'colsample_bytree': 0.9, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1.1},
    {'n_estimators': 170, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 2, 'subsample': 0.75, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 0.9},
    ]
for i, params in enumerate(param_list, 1):
    with mlflow.start_run(run_name=f"test-{i}") as run:      
        mlflow.log_params(params)
        
        regressor = XGBRegressor(**params)
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)
        # np.set_printoptions(precision=2)

        # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

        # plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted values')
        # plt.scatter(range(len(y_test)), y_test, color='red', label='Real values')
        # plt.xlabel('Data points')
        # plt.ylabel('Happiness Score values')
        # plt.title('Scatter plot of Happiness Score')
        # plt.legend()
        # plt.show()

        # file_path = base_dir.parent / 'models'
        # joblib.dump(regressor, f'{file_path}/xgboost_model.pkl')
        # joblib.dump(sc, f'{file_path}/scaler.pkl')

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "r2_score": r2
            }
        
        mlflow.log_metrics(metrics)