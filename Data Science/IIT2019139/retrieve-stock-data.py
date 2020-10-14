# Install the yfinance if not already installed
!pip install yfinance

import numpy as np
import pandas as pd

#The yfinance module has the download method
#which can be used to download the stock market data.
import yfinance as yf

#Fetching the stock market data for MSFT
#for a period of 7 days of 1-minute frequency

data = yf.download(tickers='MSFT',start = '2020-10-01',
                   end = '2020-10-08',interval='1m')
print(data)

import datetime 
from datetime import datetime

#Reseting the index and making Datetime as a column
data['Datetime']=data.index.to_series()
data.reset_index(drop=True, inplace=True)
print(data.head())

#Converting local (American/New York) to UTC.
data_UTC = data
data_UTC['Datetime'] = data['Datetime'].dt.tz_convert('UTC')
print(data_UTC.head())

