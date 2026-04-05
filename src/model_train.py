from sklearn.model_selection import KFold
import pandas as pd
from src.pipeline import build_pipeline
from sklearn.metrics import mean_squared_error
import time
import numpy as np

def _time_prediction(X_test, pipeline):
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    end_time = time.time()

    latency = ( (end_time - start_time) / len(X_test) ) * 1000

    return y_pred, latency

def train_model(n_splits, num_scaler, model_instance, target_col, df):
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    X = df.copy().drop(columns=[target_col])
    Y = df.copy()[target_col]

    mses, latencies = [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        pipeline = build_pipeline(df, num_scaler, model_instance, target_col)
        pipeline.fit(X_train, Y_train)

        y_pred, latency = _time_prediction(X_test, pipeline)
        mse = mean_squared_error(Y_test, y_pred)
        
        mses.append(mse)
        latencies.append(latency)

    rmse = np.sqrt(np.mean(mses))
    avg_latency = np.mean(latencies)

    return rmse, avg_latency
