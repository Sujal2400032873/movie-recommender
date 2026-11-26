from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD


PROJECT_DIR = Path(__file__).resolve().parent
RATINGS_PATH = PROJECT_DIR / "data" / "ratings.csv"
MOVIES_PATH = PROJECT_DIR / "data" / "movies.csv"

ratings_df = pd.read_csv(RATINGS_PATH)
movies_df = pd.read_csv(MOVIES_PATH)

rating_min = float(ratings_df["rating"].min())
rating_max = float(ratings_df["rating"].max())
reader = Reader(rating_scale=(rating_min, rating_max))
surprise_dataset = Dataset.load_from_df(
    ratings_df[["userId", "movieId", "rating"]],
    reader,
)

training_data = surprise_dataset.build_full_trainset()
svd_model = SVD(random_state=42)
svd_model.fit(training_data)


def get_collaborative_recommendations(
    user_id: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top N unseen movies for a user by predicted SVD rating."""
    if not isinstance(user_id, (int, str)):
        raise TypeError("user_id must be an integer or numeric string")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    try:
        surprise_user_id = int(user_id)
    except (TypeError, ValueError) as error:
        raise ValueError("user_id must be numeric") from error

    rated_movie_ids = set(
        ratings_df.loc[
            ratings_df["userId"] == surprise_user_id,
            "movieId",
        ].tolist()
    )
    if not rated_movie_ids:
        raise ValueError(f"User ID not found: {user_id}")

    unrated_movies = movies_df.loc[
        ~movies_df["movieId"].isin(rated_movie_ids),
        ["movieId", "title", "genres"],
    ].copy()

    unrated_movies["predicted_rating"] = unrated_movies["movieId"].apply(
        lambda movie_id: svd_model.predict(
            surprise_user_id,
            int(movie_id),
        ).est
    )

    return (
        unrated_movies
        .sort_values("predicted_rating", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
