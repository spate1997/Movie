import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
data = pd.merge(ratings, movies, on='movieId')

# User-Item Matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Calculate Similarity
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
