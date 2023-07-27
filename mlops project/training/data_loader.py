import os
import pandas as pd
import yfinance as yf
import argparse




def fetch_data(start_date, end_date):
    """This will fetch data from yfinance and then merge the btc data with eth and s&p500 close data
    this will result in a six features dataset containing BTC OHLC, ETH close and s&p500 close"""
    
    btc_data = yf.download('BTC-USD', start = start_date, end= end_date)
    
    
    eth_data = yf.download('ETH-USD', start = start_date, end= end_date)
    
    SandP_data = yf.download('^GSPC', start = start_date, end= end_date)
    
    
    
    return btc_data, eth_data, SandP_data

def prepare_data(btc_data, eth_data, SandP_data):
    #forming a new btc data frame
    btc_df = pd.DataFrame(btc_data[['Open', 'High', 'Low', 'Close']])
    # Rename columns for ETH and S&P 500 in their respective DataFrames
    eth_data.rename(columns={'Close': 'eth'}, inplace=True)
    SandP_data.rename(columns={'Close': 's&p500'}, inplace=True)
    df = pd.merge(btc_df, eth_data['eth'], on='Date', how='left')
    df = pd.merge(df, SandP_data['s&p500'], on='Date', how='left')
    
    

    #handling missing data in the dataframe

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df
    
    
def save_data(df, usecase,end_date):
    output_file = f'output/{usecase}_data{end_date}.parquet'
    os.makedirs('output', exist_ok = True)
    df.to_parquet(
            output_file, 
            engine='pyarrow', 
            index=True)

    print("you are doing a good job savvy engineer")
    
    
def main(start_date, end_date,usecase):
   
    # Fetch data
    btc_data, eth_data, SandP_data = fetch_data(start_date, end_date)

    # Prepare data
    df = prepare_data(btc_data, eth_data, SandP_data)
    
    # Save data
    save_data(df,usecase,end_date)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", type=str, help="input the year e.g yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help = 'input the month of the data e.g yyyy-mm-dd')
    parser.add_argument("usecase", type = str,help = 'enter train or test')
    args = parser.parse_args()
    
    start_date = args.start_date
    end_date = args.end_date
    usecase = args.usecase
    main(start_date,end_date, usecase)



