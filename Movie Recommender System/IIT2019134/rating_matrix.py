import pandas as pd
import numpy as np
import warnings

#Dataframe for array
df = pd.read_csv('../datasets/ratings.csv')

#Getting out the required column for the array
df = df[['userId','movieId','rating']]

#Using Pandas Pivot for Generating The Required DataFrame
df = df.pivot('userId','movieId','rating')

#Converting The DataFrame to Array
arr = np.array(df)

#Displaying the Array
#print(arr)