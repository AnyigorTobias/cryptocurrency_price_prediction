import pandas as pd
import numpy as np
import os
import pandas as pd

import argparse
import pickle
import requests
import json
import mlflow
from prefect import flow, task

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


@task(retries = 3, retry_delay_seconds = 3)    
def read_data(filename: str):
    
    print("getting your data from folder for model training")
   
    df = pd.read_parquet(filename)
    
   
    return df
      
    
def MAPE(d_test, d_pred):
   MAPE= np.average(np.abs((d_test - d_pred) / d_test)) * 100
   return MAPE
   
def rmse(d_test,d_pred):
    rmse  = np.sqrt(mean_squared_error(d_test, d_pred))
    return rmse

@task(log_prints = True)    
def train_best_model(train_data, test_data):
    print("currently training your model and logging into mlflow")
  
    with mlflow.start_run():
        y_train = train_data['Close'].values#.reshape(-1,1)
        X_train = train_data.drop('Close', axis = 1).values
        
        y_test = test_data['Close'].values.reshape(-1,1)
        X_test = test_data.drop('Close', axis = 1).values
        params = {"C": 1000, "kernel":'linear'}
        mlflow.log_params(params)
        
        pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**params))])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test).reshape(-1,1)
        
        #MAPE = np.average(np.abs((y_test - y_pred) / y_test)) * 100
        #rmse  = rmse(y_test,y_pred)
        #MAPE= np.average(np.abs((y_test - y_pred) / y_test)) * 100
        #rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        mape= MAPE(y_test, y_pred)
        RMSE = rmse(y_test, y_pred)
        
        mlflow.log_metric('MAPE', mape)
        mlflow.log_metric('RMSE', RMSE)
        
        mlflow.sklearn.log_model(pipe, artifact_path = 'models')
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
        
        return None
    
@flow
def main(train_path:str = 'output/train_data2022-12-31.parquet',
        test_path: str = 'output/test_data2023-05-31.parquet'):
     
    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///backend.db")
    mlflow.set_experiment("crypto_price_prediction")

    
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    
    train_best_model(train_data, test_data)
    
           
if __name__ == "__main__":
   main()