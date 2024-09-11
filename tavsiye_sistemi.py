import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('users.data', sep='\t', names=column_names)

movie_titles = pd.read_csv("movie_id_titles.csv")

df = pd.merge(df, movie_titles, on='item_id')  # "join" işlemi yapıldı.

# Pivot tablosu yapımı.
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# Star wars filminin user ratingleri
starwars_user_ratings = moviemat["Star Wars (1977)"]

similiar_to_starwars = moviemat.corrwith(starwars_user_ratings)


corr_starwars = pd.DataFrame(similiar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
# print(corr_starwars.sort_values('Correlation', ascending=False).head(10))

df.drop(['timestamp'], axis = 1)


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
# print(ratings.sort_values('rating', ascending=False).head())

ratings['rating_oy_sayisi'] = pd.DataFrame(df.groupby('title')['rating'].count())
# print(ratings.sort_values(by='rating_oy_sayisi', ascending=False))

corr_starwars = corr_starwars.join(ratings['rating_oy_sayisi'])
print(corr_starwars[corr_starwars['rating_oy_sayisi']>100].sort_values(by='Correlation', ascending=False).iloc[1:, :])