#Importing the Required libraries

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# DATAPATHS

# While running on Kaggle, the following paths are used
# movies_df = pd.read_csv('../input/movie-lens-dataset/movies.csv', sep=',',
#                         encoding='latin-1', usecols=
#                         ['movieId','title','genres'])

# ratings_df = pd.read_csv('../input/movie-lens-dataset/ratings.csv'
#                         , sep=',', encoding='latin-1', usecols=
#                          ['userId','movieId','rating'])

# tags_df = pd.read_csv('../input/movie-lens-dataset/tags.csv')


# While running on system, use these paths (uncomment these & comment the above ones)
movies_df = pd.read_csv('../datasets/movies.csv', sep=',',
                        encoding='latin-1', usecols=
                        ['movieId','title','genres'])

ratings_df = pd.read_csv('../datasets/ratings.csv'
                         , sep=',', encoding='latin-1', usecols=
                         ['userId','movieId','rating'])

tags_df = pd.read_csv('../datasets/tags.csv')

df_movies=movies_df

# Joining the genre by replacing the '|' symbol with ' '
movies_df['genres'] = movies_df['genres'].str.replace('|',' ')

# Merging all the columns of the Movies and Tags dataframe to be processed further
merged = pd.merge(movies_df, ratings_df, on='movieId', how='inner')
print(merged.head())

# TF is simply the frequency of a word in a document.
# IDF is the inverse of the document frequency among the documents. 
# TF-IDF weighting negates the effect of high frequency words in
# determining the importance of an item (document).

# We consider genres as an important parameter to recommend user the
# movie he/she watches based on generes of movie ,user has already watched.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Defining TF-IDF Vectorizer Object
tfidf_M_genres = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')

#Replacing NaN values in 'genre' with an empty string...
movies_df['genres'] = movies_df['genres'].replace(
    to_replace="(no genres listed)", value="")

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_M_genres_matrix = tfidf_M_genres.fit_transform(movies_df['genres'])

#print(tfidf_movies_genres.get_feature_names())


#Cosine similarity is a measure of similarity between two non-zero vectors
#of an inner product space that measures the cosine of the angle between them.
# Cosine distance is used here as we are interested in similarity i.e.
# higher the value, higher will be the similarity between them...
# But as the function gives the distance, we will subtract it from 1,

# Compute the cosine similarity matrix
cosine_simi = linear_kernel(tfidf_M_genres_matrix,
                                  tfidf_M_genres_matrix)
print(cosine_simi)

# Function that returns 10 movie titles recommended to a user
# based on the genre given a movie title and cosine similarity...

def get_recom_genres(movie_title, cosine_simi=cosine_simi):
    """
    Calculates top 10 movies to recommend based on given movie titles genres. 
    """
    # Get the index of the movie that matches the title
    idx_movie = movies_df.loc[movies_df['title'].isin([movie_title])]
    idx_movie = idx_movie.index
    
    # Pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_simi[idx_movie][0]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 2 most similar movies
    return movies_df['title'].iloc[movie_indices]


# Function that recommendes movies based on the
# movies that the user has watched in the past...

def get_recom_model(userId):
    """
    Calculates top movies to be recommended to user based on movie user has watched.  
    """
    # Declaring 2 empty lists for storing recommendations and movies...
    recom_list = []
    movie_list = []
    
    rating_filter_df = ratings_df[ratings_df["userId"]== userId]
    
    for key, row in rating_filter_df.iterrows():
        movie_list.append((movies_df["title"][row["movieId"]==movies_df["movieId"]]).values) 
    
    for idx, movie in enumerate(movie_list):
        for key, movie_recommended in get_recom_genres(movie[0]).iteritems():
            recom_list.append(movie_recommended)

    # removing already watched movie from recommended list    
    for movie_title in recom_list:
        if movie_title in movie_list:
            recom_list.remove(movie_title)
    
    return set(recom_list)

# print(get_recom_model(1))

# Import libraries from Surprise package 
# for ready-to-use SVD() Algorithm.

import surprise
from surprise import Reader, Dataset, SVD, SVDpp
from surprise.model_selection import cross_validate

# Load Reader library
reader = Reader()

# Load ratings dataset with Dataset library
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

#SVD algorithm.
svd = SVD()

#Computing RMSE of SVD.
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
# svd.train(trainset)
svd.fit(trainset)


# HYBRID MODEL

# Now that we have developed the functions/models
# for CONTENT BASED FILTERING & SVD, we will combine them
# to get better and accurate results...

# We run the Content Based Filtering and determine the
# movies that are to be recommended to the user & then 
# filter and sort these recommendations based on SVD predicted ratings.

def hybrid_content_svd_model(userId):
    
    recom_M_model = get_recom_model(userId)
    
    recom_M_model = movies_df[movies_df.apply(lambda movie: movie["title"]
                                              in recom_M_model, axis=1)]
    
    for i, col in recom_M_model.iterrows():
        pred = svd.predict(userId, col["movieId"])
        recom_M_model.loc[i, "SVD_rating"] = pred.est
    
    return recom_M_model.sort_values("SVD_rating", ascending=False).iloc[0:11]
        
    
# MOVIE RECOMMENDATIONS FOR USER NO. 50    
print(hybrid_content_svd_model(50))
