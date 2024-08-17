import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
data = pd.merge(ratings, movies, on='movieId')
