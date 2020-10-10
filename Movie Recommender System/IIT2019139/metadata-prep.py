#!/usr/bin/python
# -*- coding: utf-8 -*-
# Importing the required libraries for preparing metadata dataframe

import os
import sys

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# While running on Kaggle, the following paths are used
# movies_df = pd.read_csv('../input/movie-lens-dataset/movies.csv')
# ratings_df = pd.read_csv('../input/movie-lens-dataset/ratings.csv')
# tags_df = pd.read_csv('../input/movie-lens-dataset/tags.csv')

# While running on system, use these paths
# (uncomment these & comment the above ones)

movies_df = pd.read_csv('../datasets/movies.csv')
ratings_df = pd.read_csv('../datasets/ratings.csv')
tags_df = pd.read_csv('../datasets/tags.csv')

movies_df.head()

ratings_df.head()

tags_df.head()

movies_df.info

tags_df.info

# Joining the genre by replacing the '|' symbol with ' '

movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')

movies_df.head()

# Merging the **Movies** & the **Tags** dataframe
# and creating a **metadata** tag for each movie:

# Merging all the columns of the Movies and Tags
# dataframe to be processed further...

merged = pd.merge(movies_df, tags_df, on='movieId', how='left')
merged.head()

# Creating Metadata

merged.fillna('', inplace=True)
merged = pd.DataFrame(
    merged.groupby('movieId')['tag'].apply(lambda x: '%s' % ' '.join(x)))

post_merge = pd.merge(movies_df, merged, on='movieId', how='left')

# Joining genre & tag column

post_merge['metadata'] = post_merge[['tag',
                                     'genres']].apply(lambda x: ' '.join(x),
                                                      axis=1)
post_merge[['movieId', 'title', 'metadata']].head()

# Deleting the columns which are no longer required

del post_merge['movieId']
del post_merge['genres']
del post_merge['tag']

metadata_df = post_merge
metadata_df.head(10)

metadata_df.info

# Removing duplicate tags and genres if present
# For example. in Toy Story, tags contained multiple occurence of 'pixar'

from collections import OrderedDict

metadata_df['Metadata'] = metadata_df['metadata'].str.split().apply(
    lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ')
del metadata_df['metadata']

metadata_df.rename(columns={'Metadata': 'metadata'})

metadata_df
