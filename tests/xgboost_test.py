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
print(f'base: {base_dir}')
file_path = base_dir.parent / 'data'
print(f'file_path: {file_path}')

HappinessScore = pd.read_csv(file_path / 'data_2023.csv', encoding='utf-8')

X = HappinessScore.iloc[:, 2:].values  
y = HappinessScore.iloc[:, 1].values 

path = base_dir.parent / 'models'
print(f'path: {path}')
try:
    with open(f'{path}/xgboost_model.pkl', 'rb') as file:
        # regressor = pickle.load(file)
        regressor = joblib.load(f'{path}/xgboost_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found.")
    regressor = None

try:
    with open(f'{path}/scaler.pkl', 'rb') as file:
        sc = joblib.load(f'{path}/scaler.pkl')
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Scaler file not found.")
    sc = None

if regressor is None or sc is None:
    print("Error: Model or scaler could not be loaded.")
else:
    X = sc.transform(X)

    y_pred = regressor.predict(X)
    
    plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted values')
    plt.scatter(range(len(y)), y, color='red', label='Real values')
    plt.xlabel('Data points')
    plt.ylabel('Happiness Score values')
    plt.title('Scatter plot of Happiness Score')
    plt.legend()
    plt.show()

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    metrics = {
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "r2_score": r2
    }
    print(metrics)