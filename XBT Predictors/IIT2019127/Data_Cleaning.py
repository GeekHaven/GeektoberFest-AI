
# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Reading CSV File
data = pd.read_csv(
    r'C:\Users\Mridul\Desktop\bitstampUSD.csv')

# Checkin first and last few records of the dataset.
data.head()
data.tail()

# Checking if null values are present.
data.isnull().sum()

# Checking number of records.
data.shape

# Dropping rows with nan value
data = data.dropna()

# No null values present.
data.isnull().sum()

# Remaining dataset size
data.shape

# Checkin first few records of the dataset.
data.head()

# As the index of rows are mismatched we use reset_index
data = data.reset_index()
data.head()


# The previous index are stored in the column which are no longer needed
del data['index']

# Current timestamp is in UTC format. So we convert it in UNIX format
data.Timestamp = pd.to_datetime(data.Timestamp, unit='s')


# Separating the differentvalues from the Timestamp columns
data["year"] = pd.DatetimeIndex(data["Timestamp"]).year
data["Month"] = pd.DatetimeIndex(data["Timestamp"]).month
data["Day"] = pd.DatetimeIndex(data["Timestamp"]).day
data["Hours"] = pd.DatetimeIndex(data["Timestamp"]).hour
data["Minutes"] = pd.DatetimeIndex(data["Timestamp"]).minute
data["Seconds"] = pd.DatetimeIndex(data["Timestamp"]).second

# deleting timestamp column.
del data['Timestamp']

# Cleaned data set.
data.head()
