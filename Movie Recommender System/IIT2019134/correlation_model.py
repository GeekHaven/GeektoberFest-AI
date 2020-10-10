import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)

def corr_model(name, num_recommends):                                                                            #function takes movie name and number of recommendations as input and returns a dataframe of recommended movies
    ratings = pd.read_csv('../datasets/ratings.csv', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings = ratings.iloc[1:]
    movies = pd.read_csv('../datasets/movies.csv', names=['movie_id', 'movie_name', 'genres'])
    movies = movies.iloc[1:]
    data = pd.merge(ratings, movies, on='movie_id')                                                              #merge movie and rating info
    data['rating'] = data['rating'].astype(float)
    ratings = pd.DataFrame(data.groupby('movie_name')['rating'].mean())                                          
    ratings['number_of_ratings'] = data.groupby('movie_name')['rating'].count()                                  #ratings dataframe used to store mean ratings for each movie and number of ratings it recieved
    rating_matrix = data.pivot_table(index='user_id', columns='movie_name', values='rating')                     #matrix of movie ratings
    user_rating = rating_matrix[name]                                   
    similarity = rating_matrix.corrwith(user_rating)                                                             #compute correlation column-wise with given movie column 
    corr = pd.DataFrame(similarity, columns=['correlation'])
    corr.dropna(inplace=True)
    corr = corr.join(ratings['number_of_ratings'])
    recommendations = corr[corr['number_of_ratings'] > 50]                                                       #drop the movies with less than 50 ratings received
    recommendations = recommendations.sort_values(by='correlation', ascending=False)                             #sort correlations in descending order
    recommendations = recommendations.iloc[1:num_recommends+1]                                                   #exclude first row since correlation with itself is highest, i.e, 1
    recommendations = recommendations.drop(columns=['number_of_ratings'])
    return recommendations

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