# Importing the Required libraries

import numpy as np
import pandas as pd
import math
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Reading the csv file for ratings and making use of
# only the userId, movieId & ratings columns

# Comment the below lines as they were executed in Kaggle Notebook
# ratings = pd.read_csv('../input/movie-lens-dataset/ratings.csv',
#                       sep=',', encoding='latin-1', usecols=
#                       ['userId','movieId','rating'])

# While running on system, use these paths
# (uncomment these & comment the above ones)
ratings = pd.read_csv('../datasets/ratings.csv',
                      sep=',',
                      encoding='latin-1',
                      usecols=['userId', 'movieId', 'rating'])

# Displaying the first 5 rows of the data
ratings.head()

# Displaying the no. of users and the no. of movies inorder to get an idea
# of the rows and columns of the matrix that will be calculated in the future

print('Number of users = ' + str(ratings.userId.unique().shape[0]))
print('Number of movies = ' + str(ratings.movieId.unique().shape[0]))

# Using Pandas Pivot for Generating The Required
# DataFrame & filling the NaN values with 0

# ratings_matrix_nan is the resultant table having NaN values
# whereas ratings_matrix has NaN values replaced with 0...
ratings_matrix_nan = ratings.pivot_table(index='userId',
                                         columns='movieId',
                                         values='rating')

# print(ratings_matrix)
ratings_matrix = ratings_matrix_nan.fillna(0)
print(ratings_matrix)

# De-normalize the data (normalize by each users mean)
# and convert it from a dataframe to a numpy array

ratings_np = ratings_matrix.to_numpy()
ratings_np_nan = ratings_matrix_nan.to_numpy()

user_ratings_mean = np.mean(ratings_np, axis=1)

# Making the user_ratings_mean vertical by reshaping
ratings_demeaned = ratings_np - user_ratings_mean.reshape(-1, 1)

# The ratings matrix has been Normalized and is Formatted

# Importint svd library from scipy function inorder to choose
# number of latent factors which is NOT possible in svd library of numpy

from scipy.sparse.linalg import svds
# 3 matrices - U, sigma and V-transpose with latent factors as 50

# U is an m × r orthogonal matrix
# S (sigma) is an r × r diagonal matrix
# V is an r × n orthogonal matrix
U, sigma, V_t = svds(ratings_demeaned, k=50)
print('Size of sigma: ', sigma.size)

# Converting sigma to Diagonal matrix form
sigma = np.diag(sigma)
print(sigma)

print('Shape of sigma: ', sigma.shape)
print('Shape of U: ', U.shape)
print('Shape of V_t: ', V_t.shape)

# Making predictions from matrices

user_predicted_ratings = np.dot(np.dot(U, sigma),
                                V_t) + user_ratings_mean.reshape(-1, 1)

print('Rating Dataframe column names', ratings_matrix.columns)

# Using column names from the ratings df
preds = pd.DataFrame(user_predicted_ratings, columns=ratings_matrix.columns)
print(preds)

# Converting predictions to numpy array for calculating RMSE
preds_np = preds.to_numpy()
print(preds_np)

# Now RMSE can be calculated ONLY for those elements which are not NaN
# So, we find the difference of the actual & pred numpy array, where actual is
# 'ratings_np_nan' whereas predicted one is 'preds_np'...

diff_act_pred = np.subtract(ratings_np_nan, preds_np)
sq_diff_act_pred = np.square(diff_act_pred)
# sq_diff

# We use numpy.isnan() -> (Boolean) & ~ OPERATOR to REMOVE all NaN from nparray

# Basically ~ operator to invert array so that indices with NaN
# are now marked as False.
# we call indexing syntax arr[n_arr] with n_arr as the result
# of the last step to get a new array with all NaNs filtered out.

mse = sq_diff_act_pred[~np.isnan(sq_diff_act_pred)].mean()
rmse = sqrt(mse)
print('MSE = ', mse)
print('RMSE = ', rmse)

# OUTPUT (for k=50)
# MSE =  3.9862007683156677
# RMSE =  1.9965472116420557
