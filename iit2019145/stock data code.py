# import yfinance as yf
# start_date = input(("Enter start date (yyyy-mm-dd) : "))
# # end_date=input(("Enter end sate (yyyy-mm-dd) : "))
# # print(start_date,end_date)
# ls=["^NSEI","^BSESN","MCX.NS","ETH-INR","^DJI"]
# # for i in ls:
# data = yf.download(tickers="^NSEI", start ='2000-04-11',end = '2000-04-16', interval="1m")
# data.to_csv('D:\\qui\\'+"^NSEI"+'.csv')
# print(data)
#
# # Print the data
# # print(data)


import yfinance as yf
import numpy as np
import pandas as pd
## succesfully extracted the data for dates im range  2020-09-25 to 2020-09-29

start_date = input("Enter start date (yyyy-mm-dd) : ")
end_date=input(("Enter end date (yyyy-mm-dd) : "))
print(start_date[-1])
ls=["^NSEI","^BSESN","MCX.NS","ETH-INR","^DJI"]
for i in ls:
    data = yf.download(tickers=i, start = str(start_date),end =str(end_date), interval="1m")
    data.to_csv('/*firstly add the location where you want to store the data*/\\' + i + '.csv')