import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import joblib

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

model_name = "HScore-XGBRegressor"
model_version = "1"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

base_dir = Path(__file__).resolve().parent
file_path = base_dir.parent / 'data' / 'data_2023.csv'

dataset = pd.read_csv(file_path, encoding='utf-8')

dataset = dataset.iloc[:, 1:]


file_path = base_dir.parent / 'models' / 'scaler.pkl'
sc = joblib.load(f'{file_path}')

X_test = dataset.iloc[:, 1:].values
y_test = dataset.iloc[:, 1].values
X_test = sc.transform(X_test)

y_pred = model.predict(X_test)

plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted values')
plt.scatter(range(len(y_test)), y_test, color='red', label='Real values')
plt.xlabel('Data points')
plt.ylabel('Happiness Score values')
plt.title('Scatter plot of Happiness Score')
plt.legend()
plt.show()