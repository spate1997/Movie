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

# Predict Ratings
def predict_ratings(user_id, movie_title):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users[similar_users > 0.5]  # Threshold for similarity

    ratings = []
    similarities = []
    for similar_user in similar_users.index:
        if pd.notna(user_movie_matrix.loc[similar_user, movie_title]):
            ratings.append(user_movie_matrix.loc[similar_user, movie_title])
            similarities.append(similar_users[similar_user])
    
    if len(ratings) == 0:
        return 0  # If no similar users have rated the movie

    predicted_rating = sum([r * s for r, s in zip(ratings, similarities)]) / sum(similarities)
    return predicted_rating
    
# Recommend Movies
def recommend_movies(user_id, num_recommendations):
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index

    predictions = []
    for movie in unrated_movies:
        predicted_rating = predict_ratings(user_id, movie)
        predictions.append((movie, predicted_rating))
    
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]
    return recommendations

# Example Usage
user_id = 1
num_recommendations = 5
print(recommend_movies(user_id, num_recommendations))