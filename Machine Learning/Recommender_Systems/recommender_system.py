# Recommender system

import numpy as np
import pandas as pd

column_names = ['user_id','item_id','rating','timestamp']

movie_ratings = pd.read_csv('Datasets/movie_ratings.csv', sep='\t', names=column_names)

movie_titles = pd.read_csv('Datasets/Movie_Id_Titles.csv')

# merging movie_titles and movie_ratings based on item_id into one dataset
dataset = pd.merge(movie_titles, movie_ratings, on='item_id')

# movies with the best rating
dataset.groupby('title')['rating'].mean().sort_values(ascending=False).head()

# movies with the most ratings
dataset.groupby('title')['rating'].count().sort_values(ascending=False).head()

# creating ratings dataset
ratings = pd.DataFrame(dataset.groupby('title')['rating'].mean())

# number of ratings
ratings['num of ratings'] = dataset.groupby('title')['rating'].count()
