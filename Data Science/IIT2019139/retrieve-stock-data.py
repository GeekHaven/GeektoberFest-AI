# Install the yfinance if not already installed
!pip install yfinance

import numpy as np
import pandas as pd

#The yfinance module has the download method
#which can be used to download the stock market data.
import yfinance as yf

import datetime 
from datetime import datetime

def retrieve_stock_data_csv(ticker, start_date, end_date):
    
    
    #Fetching the stock market data for MSFT
    #for a period of 7 days of 1-minute frequency
    
    data = yf.download(tickers=ticker,start = start_date,
                       end = end_date, interval='1m')
    #print(data)
    
    
    #Reseting the index and making Datetime as a column
    
    data['Datetime']=data.index.to_series()
    data.reset_index(drop=True, inplace=True)
    #print(data.head())
    
    
    #Converting local (American/New York) to UTC.
    
    data_UTC = data
    data_UTC['Datetime'] = data['Datetime'].dt.tz_convert('UTC')
    
    return data_UTC.to_csv("Intradata.csv")


ticker = input(("Enter ticker (Enter 'x' for MSFT ticker) : "))

if ticker == 'x':
    ticker = 'MSFT'

# Enter initial date
# As we want 1m interval date must be from last 30 days
start_date = input(("Enter starting date (yyyy-mm-dd) : "))

# Enter final date
end_date = input(("Enter ending date (yyyy-mm-dd) : "))
                   
                   
retrieve_stock_data_csv(ticker, start_date, end_date)
