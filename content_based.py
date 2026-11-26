from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MOVIES_PATH = Path(__file__).resolve().parent / "data" / "movies.csv"

movies_df = pd.read_csv(MOVIES_PATH)
movies_df["genres"] = (
    movies_df["genres"]
    .fillna("")
    .astype(str)
    .str.replace("|", " ", regex=False)
)

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_content_recommendations(title: str, top_n: int = 5) -> pd.DataFrame:
    """Return the top N movies most similar to the given movie title."""
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    title_matches = movies_df[
        movies_df["title"].str.casefold() == title.casefold()
    ]
    if title_matches.empty:
        raise ValueError(f"Movie title not found: {title}")

    movie_index = title_matches.index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = [
        (index, score)
        for index, score in similarity_scores
        if index != movie_index
    ]
    similarity_scores.sort(key=lambda item: item[1], reverse=True)

    top_matches = similarity_scores[:top_n]
    recommendation_indices = [index for index, _ in top_matches]
    scores = [float(score) for _, score in top_matches]

    recommendations = movies_df.loc[
        recommendation_indices, ["movieId", "title", "genres"]
    ].copy()
    recommendations["similarity_score"] = scores
    recommendations.reset_index(drop=True, inplace=True)
    return recommendations
