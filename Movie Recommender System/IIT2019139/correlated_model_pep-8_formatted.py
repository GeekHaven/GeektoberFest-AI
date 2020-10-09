import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)


# function takes movie name and number of recommendations as input and
# returns a dataframe of recommended movies
def corr_model(name, num_recommends):
    ratings = pd.read_csv('../datasets/ratings.csv',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings = ratings.iloc[1:]
    movies = pd.read_csv('../datasets/movies.csv',
                         names=['movie_id', 'movie_name', 'genres'])
    movies = movies.iloc[1:]
    # merge movie and rating info
    data = pd.merge(ratings, movies, on='movie_id')
    data['rating'] = data['rating'].astype(float)
    ratings = pd.DataFrame(data.groupby('movie_name')['rating'].mean())
    # ratings dataframe used to store mean ratings for each
    # movie and number of ratings it recieved
    ratings['number_of_ratings'] = data.groupby('movie_name')['rating'].count()
    rating_matrix = data.pivot_table(
        index='user_id',
        columns='movie_name',
        values='rating')  # matrix of movie ratings
    user_rating = rating_matrix[name]
    # compute correlation column-wise with given movie column
    similarity = rating_matrix.corrwith(user_rating)
    corr = pd.DataFrame(similarity, columns=['correlation'])
    corr.dropna(inplace=True)
    corr = corr.join(ratings['number_of_ratings'])
    # drop the movies with less than 50 ratings received
    recommendations = corr[corr['number_of_ratings'] > 50]
    recommendations = recommendations.sort_values(
        by='correlation',
        ascending=False)  # sort correlations in descending order
    # exclude first row since correlation with itself is highest, i.e, 1
    recommendations = recommendations.iloc[1:num_recommends + 1]
    recommendations = recommendations.drop(columns=['number_of_ratings'])
    return recommendations
