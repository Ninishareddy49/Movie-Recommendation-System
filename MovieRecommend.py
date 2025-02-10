import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load the data files
try:
    ratings = pd.read_csv('C:/Users/DELL/PycharmProjects/PythonProject3/ratings.csv')
    movies = pd.read_csv('C:/Users/DELL/PycharmProjects/PythonProject3/movies.csv')
except FileNotFoundError:
    print("Error: Ensure 'ratings.csv' and 'movies.csv' are in the same directory as the script.")
    exit()

# Merge ratings with movies to get movie titles
ratings = ratings.merge(movies, on='movieId')

# Filter data: include movies with at least 5 ratings and users with at least 5 ratings
def filter_data():
    global filtered_ratings, user_item_matrix, user_item_sparse, similarity_df
    movie_counts = ratings['movieId'].value_counts()
    user_counts = ratings['userId'].value_counts()
    filtered_ratings = ratings[
        (ratings['movieId'].isin(movie_counts[movie_counts >= 2].index)) &
        (ratings['userId'].isin(user_counts[user_counts >= 2].index))
    ]

    # Create a user-item matrix
    user_item_matrix = filtered_ratings.pivot_table(index="userId", columns="title", values="rating").fillna(0)
    user_item_sparse = csr_matrix(user_item_matrix)

    # Compute cosine similarity
    if user_item_sparse.shape[0] > 0 and user_item_sparse.shape[1] > 0:
        similarity_matrix = cosine_similarity(user_item_sparse)
        similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    else:
        similarity_df = None

# Initial data filtering
filter_data()

# Recommendation function
def recommend_items(user, num_recommendations=5):
    if similarity_df is None or user not in similarity_df.index:
        return "User not found or insufficient data for recommendations."

    # Get the most similar users
    similar_users = similarity_df[user].sort_values(ascending=False).drop(user).index

    # Find items rated by similar users that the target user hasn't rated
    user_rated_items = set(user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index)
    recommendations = {}

    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        for item in similar_user_ratings.index:
            if item not in user_rated_items and similar_user_ratings[item] > 0:
                recommendations[item] = recommendations.get(item, 0) + similar_user_ratings[item]

    # Sort recommendations by rating and return the top N
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_recommendations[:num_recommendations]]

# Add or update movie ratings
def add_or_update_rating(user, movie, rating):
    global ratings
    if movie not in movies['title'].values:
        print(f"Movie '{movie}' not found in the database.")
        return

    movie_id = movies[movies['title'] == movie]['movieId'].values[0]
    new_row = {"userId": user, "movieId": movie_id, "rating": rating, "timestamp": pd.Timestamp.now().timestamp()}
    ratings = pd.concat([ratings, pd.DataFrame([new_row])], ignore_index=True)

    # Update filtered data and similarity matrix
    filter_data()

# Example usage with user input
while True:
    print("\nMenu:")
    print("1. Add/Update a movie rating")
    print("2. Get recommendations")
    print("3. Exit")
    choice = input("Enter your choice: ")

    if choice == "1":
        try:
            user = int(input("Enter your user ID: "))
            movie = input("Enter the movie title: ")
            rating = float(input("Enter your rating (0.5-5): "))
            if 0.5 <= rating <= 5:
                add_or_update_rating(user, movie, rating)
                print(f"Rating added/updated for User {user}: {movie} - {rating}")
            else:
                print("Error: Rating must be between 0.5 and 5.")
        except ValueError:
            print("Invalid input. Please enter valid user ID and rating.")
    elif choice == "2":
        try:
            user_to_recommend = int(input("Enter your user ID to get recommendations: "))
            recommendations = recommend_items(user_to_recommend)
            print(f"Recommendations for User {user_to_recommend}: {recommendations}")
        except ValueError:
            print("Invalid input. Please enter a valid user ID.")
    elif choice == "3":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")
