
import pandas as pd
import numpy as np
import os

import yfinance as yf
import argparse
import mlflow


from typing import List, Tuple
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from prefect import flow, task

@task(retries = 3, retry_delay_seconds=2)
def load_data(start_date, end_date) -> pd.DataFrame:

    btc_data = yf.download('BTC-USD', start = start_date, end= end_date)
    btc_df = pd.DataFrame(btc_data[['Open', 'High', 'Low', 'Close']])
    
    return btc_df

@task
def save_data(df,end_date):
    output_file = f'output/data{end_date}.parquet'
    os.makedirs('output', exist_ok = True)
    df.to_parquet(
            output_file, 
            engine='pyarrow', 
            index=True)

    print("you are doing a good job savvy engineer")
    
    
@task
def get_data(data_path):
    data = pd.read_parquet("./output/data2023-06-30.parquet")
    return data  

@task
def split_data(df,train_date_end,test_date_start):
    train_data = df.loc[:train_date_end]
    test_data = df.loc[test_date_start:]
    
    return train_data, test_data
    
@task
def transform_crypto_data(data: pd.DataFrame, input_seq_len: int) -> Tuple[pd.DataFrame, pd.Series]:
    # Extract the 'Close' column
    close_prices = data['Close']

    # Initialize lists to store features and targets
    features = []
    targets = []

    # Loop through the data to create sequences
    for i in range(len(close_prices) - input_seq_len):
        features.append(close_prices[i:i + input_seq_len].values)
        targets.append(close_prices[i + input_seq_len])

    # Convert lists to NumPy arrays
    x = np.array(features)
    y = np.array(targets)

    # Create DataFrame for features
    feature_columns = [f'price_{i + 1}_day_ago' for i in reversed(range(input_seq_len))]
    features_df = pd.DataFrame(x, columns=feature_columns)

    # Create Series for target
    target_series = pd.Series(y, name='target_price_next_day')

    return features_df, target_series

@task(log_prints = True)
def train_best_model(w,x,y,z):
    
    with mlflow.start_run():
        
        mlflow.set_tag("developer", "Tobs")
        
        mlflow.log_param("train_data_path", "output/data2023-06-30.parquet")
    
        params = {"C": 1000, "kernel":'linear'}
        mlflow.log_params(params)
        pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**params))])    
        pipe.fit(w,x)
        y_pred = pipe.predict(y)
        MAPE = np.average(np.abs((z - y_pred) /z)) * 100
        rmse  = np.sqrt(mean_squared_error(z, y_pred))
        
        mlflow.log_metric('MAPE', MAPE)
        mlflow.log_metric('RMSE', rmse)
        
        mlflow.sklearn.log_model(pipe,artifact_path = 'models')
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
        
        print(MAPE)
        print(rmse)
        
        return None


@flow
def main(data_path:str = f'output/data2023-06-30.parquet'):
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("crypto_price_prediction_3")
    
    
    start_date = '2017-01-01'
    end_date = '2023-06-30'
    
    train_date_end = '2022-12-31'
    test_date_start = '2023-01-01'
    
    btc_df = load_data(start_date, end_date)
    save_data(btc_df,end_date)
    data_path = f'output/data{end_date}.parquet'
    data = get_data(data_path)

    train_data, test_data = split_data(data,train_date_end,test_date_start)
    
    input_seq_len = 7

    X_train, y_train = transform_crypto_data(train_data, input_seq_len)

    X_test, y_test = transform_crypto_data(test_data, input_seq_len)

    train_best_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()





