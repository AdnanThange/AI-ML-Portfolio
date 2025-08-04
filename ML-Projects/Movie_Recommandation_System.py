import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movie_data = pd.read_csv(
    "u.item", 
    sep="|", 
    encoding="latin-1", 
    header=None, 
    names=[
        'movie_id', 'title', 'release_date', 'video_release', 'imdb_url',
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
)

rating_data = pd.read_csv("u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

merged_data = pd.merge(movie_data, rating_data, on="movie_id", how="inner")

user_movie_matrix = merged_data.pivot_table(index="user_id", columns="title", values="rating")
user_movie_matrix = user_movie_matrix.fillna(0)

rating_matrix = user_movie_matrix.values
user_mean = np.mean(rating_matrix, axis=1).reshape(-1, 1)
mean_normalized_matrix = rating_matrix - user_mean

similarity_matrix = cosine_similarity(mean_normalized_matrix)

def get_top_recommendations(user_id, matrix, similarity_matrix, top_n=3):
    similarity_scores = list(enumerate(similarity_matrix[user_id - 1]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_users = similarity_scores[1:top_n + 1]

    target_user_ratings = matrix.iloc[user_id - 1]
    recommended_movies = []

    for user_index, _ in top_users:
        similar_user_ratings = matrix.iloc[user_index]
        unrated = similar_user_ratings[target_user_ratings == 0]
        high_rated = unrated[unrated >= 4]

        for movie in high_rated.index:
            if movie not in recommended_movies:
                recommended_movies.append(movie)
                if len(recommended_movies) >= top_n:
                    return recommended_movies

    return recommended_movies

target_user_id = 2
print(get_top_recommendations(target_user_id, user_movie_matrix, similarity_matrix))
