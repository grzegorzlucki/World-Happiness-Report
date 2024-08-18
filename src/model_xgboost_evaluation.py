import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
from pathlib import Path
from sklearn.metrics import r2_score
import mlflow
import optuna
import pickle
from mlflow.models import infer_signature

base_dir = Path(__file__).resolve().parent
file_path = base_dir.parent / 'data' / 'data_2022.csv'
HappinessScore = pd.read_csv(file_path, encoding='utf-8')

HappinessScore = HappinessScore.iloc[:, 1:]
X_train = HappinessScore.iloc[:, 1:].values
y_train = HappinessScore.iloc[:, 1].values

file_path = base_dir.parent / 'data' / 'data_2023.csv'
HappinessScore = pd.read_csv(file_path, encoding='utf-8')

HappinessScore = HappinessScore.iloc[:, 1:]
X_test = HappinessScore.iloc[:, 1:].values
y_test = HappinessScore.iloc[:, 1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("XGBR-evaluation")

params = {
    'n_estimators': 130, 
    'learning_rate': 0.03,
    'max_depth': 4, 
    'min_child_weight': 1,
    'subsample': 0.9, 
    'colsample_bytree': 0.85,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1
}

def evaluation(trial):
    with mlflow.start_run(run_name=f"trial-{trial.number}") as run:
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1)
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
        params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 6)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        params['gamma'] = trial.suggest_float('gamma', 0, 0.5)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0, 1.0)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.1, 1.0)
        
        mlflow.log_params(params)
        
        regressor = XGBRegressor(**params)
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "r2_score": r2
        }
        mlflow.log_metrics(metrics)     
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(evaluation, n_trials = 150)

best_trial = study.best_trial
print(f"Best trial number: {best_trial.number}")
print(f"Best trial value (r2_score): {best_trial.value}")

with mlflow.start_run(run_name = "best_model"):
    best_params = best_trial.params
    
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    signature = infer_signature(X_train, best_model.predict(X_train))
    
    mlflow.xgboost.log_model(
            xgb_model = best_model,
            artifact_path = "model",
            signature = signature,
            input_example = X_train
            )
    mlflow.log_params(best_params)
    
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("r2_score", r2)
    
    model_dir = base_dir.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # with open(model_dir / 'xgboost_model.pkl', 'wb') as file:
    #     pickle.dump(best_model, file)

print("Best model has been logged to MLflow.")