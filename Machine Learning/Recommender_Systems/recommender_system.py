# Recommender system

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

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

# visualising the num of ratings
sns.distplot(ratings['num of ratings'], kde=False)

# visualising the rating
sns.distplot(ratings['rating'], kde=False, bins=70, hist_kws={"color": "g"})

# relationship between avg. rating vs num of ratings
sns.jointplot('rating','num of ratings', data=ratings, alpha=0.6)

# creating a matrix with user_id as index & title as columns
movie_mat = dataset.pivot_table(index='user_id', columns='title', values='rating')

ratings.sort_values('num of ratings', ascending=False).head(10)

# grabbing ratings for two movies
starwars_user_ratings = movie_mat['Star Wars (1977)']
liarliar_user_ratings = movie_mat['Liar Liar (1997)']

# finding Correlation between movies
similar_to_starwars = movie_mat.corrwith(starwars_user_ratings)
similar_to_liarliar = movie_mat.corrwith(liarliar_user_ratings)

#removing null values
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)

corr_liarliar= pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)

# sorting
corr_starwars.sort_values('Correlation', ascending=False).head(10)
corr_liarliar.sort_values('Correlation', ascending=False).head(10)

# adding num of ratings column to movies
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])

# finding related movies
corr_starwars[corr_starwars['num of ratings']].sort_values('Correlation',
             ascending=False).head()
corr_liarliar[corr_liarliar['num of ratings']].sort_values('Correlation',
             ascending=False).head()

#setting the threshold limit to 100
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',
             ascending=False).head()
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',
             ascending=False).head()