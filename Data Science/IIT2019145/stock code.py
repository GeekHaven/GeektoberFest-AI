import yfinance as yf
import numpy as np
import pandas as pd

def stock_callculator(Ticker):
    data = yf.download(tickers=Ticker, start='2020-10-01', end='2020-10-08', interval="1m")
    data.to_csv('Firstly add the location where you want to store the data\\' + Ticker+ '.csv')
    
ls=["^NSEI","^BSESN","MCX.NS","ETH-INR","^DJI"]
for i in ls:
    stock_callculator(i)
