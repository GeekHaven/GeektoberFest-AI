# # Data Pre-processing for the Bitcoin Historical Data (using RL)

# Importing all the required libraries

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# In[2]:

# The dataset can be downloaded from :-
# https://www.kaggle.com/mczielinski/bitcoin-historical-data

# Path for the dataset is specified which can be
# changed depending on where it is saved/extracted from...
data_path = '../../../bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv'

# Reading the csv file
df = pd.read_csv(data_path)

df.head()


# In[3]:
df.info()


# In[4]:
df.describe()


# In[5]:

# Display positive and negative correlation between columns
df.corr()


# In[6]:

# Checking whether the dataset contains any NULL values or not
df.isnull().values.any()


# In[7]:

# Display column-wise NULL value count
df.isnull().sum()


# In[8]:

# Fixing data where there is no trade
# Filling NANs with zeroes for relevant fields
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
df['Weighted_Price'].fillna(value=0, inplace=True)


# In[9]:

# Filling forward OHLC data as it is continuous timeseries...
df['Open'].fillna(method='ffill', inplace=True)
df['High'].fillna(method='ffill', inplace=True)
df['Low'].fillna(method='ffill', inplace=True)
df['Close'].fillna(method='ffill', inplace=True)


# In[10]:
df.head()
