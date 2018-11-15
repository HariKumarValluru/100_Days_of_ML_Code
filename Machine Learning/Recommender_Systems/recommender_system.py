# Recommender system

import numpy as np
import pandas as pd

column_names = ['user_id','item_id','rating','timestamp']

movie_ratings = pd.read_csv('Datasets/movie_ratings.csv', sep='\t', names=column_names)

movie_titles = pd.read_csv('Datasets/Movie_Id_Titles.csv')
