import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st

from collaborative import (
    get_collaborative_recommendations,
    movies_df as collaborative_movies_df,
)
from content_based import get_content_recommendations, movies_df as content_movies_df


PROJECT_DIR = Path(__file__).resolve().parent
LINKS_PATH = PROJECT_DIR / "data" / "links.csv"
POSTER_FALLBACK_URL = "https://via.placeholder.com/500x750?text=No+Image"
movie_to_tmdb = (
    pd.read_csv(LINKS_PATH)
    .set_index("movieId")["tmdbId"]
    .to_dict()
)

st.set_page_config(
    page_title="Movie Recommender",
    page_icon=":movie_camera:",
    layout="wide",
)

st.title("Movie Recommendation System")

top_n = st.radio(
    "Number of recommendations",
    options=[5, 10],
    index=0,
    horizontal=True,
)


def fetch_poster(tmdb_id) -> str:
    """Return a TMDB poster URL, or a placeholder when none is available."""
    api_key = os.getenv("TMDB_API_KEY")
    if pd.isna(tmdb_id) or not api_key:
        return POSTER_FALLBACK_URL

    try:
        movie_id = int(tmdb_id)
        url = (
            f"https://api.themoviedb.org/3/movie/{movie_id}"
            f"?api_key={api_key}"
        )
        with urlopen(url, timeout=5) as response:
            movie_data = json.loads(response.read().decode("utf-8"))
    except (TypeError, ValueError, URLError, TimeoutError, json.JSONDecodeError):
        return POSTER_FALLBACK_URL

    poster_path = (
        movie_data.get("poster_path")
        if isinstance(movie_data, dict)
        else None
    )
    if not poster_path:
        return POSTER_FALLBACK_URL
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


def render_recommendations(recommendations: pd.DataFrame) -> None:
    score_labels = {
        "predicted_rating": ("Predicted rating", 2),
        "similarity_score": ("Similarity score", 4),
        "hybrid_score": ("Hybrid score", 4),
    }
    columns = st.columns(5)

    for index, (_, recommendation) in enumerate(recommendations.iterrows()):
        with columns[index % 5]:
            tmdb_id = movie_to_tmdb.get(recommendation["movieId"])
            st.image(fetch_poster(tmdb_id), width="stretch")
            st.markdown(f"**{recommendation['title']}**")
            for score_column, (label, precision) in score_labels.items():
                if score_column in recommendations.columns:
                    score = float(recommendation[score_column])
                    st.caption(f"{label}: {score:.{precision}f}")


tab_content, tab_collaborative, tab_hybrid = st.tabs(
    [
        "Movie-based Recommendations",
        "User-based Recommendations",
        "Hybrid Recommendations",
    ]
)

with tab_content:
    st.subheader("Content-Based Recommendations")
    titles = sorted(content_movies_df["title"].dropna().unique())
    selected_title = st.selectbox(
        "Select a movie",
        options=titles,
    )

    if st.button("Get Content-Based Recommendations", type="primary"):
        try:
            recommendations = get_content_recommendations(
                selected_title,
                top_n=top_n,
            )
            render_recommendations(recommendations)
        except (TypeError, ValueError) as error:
            st.error(str(error))

with tab_collaborative:
    st.subheader("Collaborative Filtering Recommendations")
    user_id = st.number_input(
        "User ID",
        min_value=1,
        step=1,
        value=1,
    )

    if st.button("Get Collaborative Recommendations", type="primary"):
        try:
            recommendations = get_collaborative_recommendations(
                int(user_id),
                top_n=top_n,
            )
            render_recommendations(recommendations)
        except (TypeError, ValueError) as error:
            st.error(str(error))

with tab_hybrid:
    st.subheader("Hybrid Recommendations")
    hybrid_titles = sorted(content_movies_df["title"].dropna().unique())
    hybrid_title = st.selectbox(
        "Select a reference movie",
        options=hybrid_titles,
        key="hybrid_title",
    )
    hybrid_user_id = st.number_input(
        "User ID",
        min_value=1,
        step=1,
        value=1,
        key="hybrid_user_id",
    )
    content_weight = st.slider(
        "Content-based weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="1.0 favors content similarity; 0.0 favors predicted collaborative ratings.",
    )

    if st.button("Get Hybrid Recommendations", type="primary"):
        try:
            content_results = get_content_recommendations(
                hybrid_title,
                top_n=len(content_movies_df),
            )

            collaborative_results = get_collaborative_recommendations(
                int(hybrid_user_id),
                top_n=len(collaborative_movies_df),
            )

            content_scores = content_results[
                ["movieId", "title", "genres", "similarity_score"]
            ]
            rating_scores = collaborative_results[
                ["movieId", "predicted_rating"]
            ]

            hybrid = content_scores.merge(
                rating_scores,
                on="movieId",
                how="inner",
            )
            if hybrid.empty:
                st.warning("No movies are available to combine.")
            else:
                rating_min = hybrid["predicted_rating"].min()
                rating_max = hybrid["predicted_rating"].max()
                if rating_max == rating_min:
                    hybrid["normalized_rating"] = 1.0
                else:
                    hybrid["normalized_rating"] = (
                        hybrid["predicted_rating"] - rating_min
                    ) / (rating_max - rating_min)

                hybrid["hybrid_score"] = (
                    content_weight * hybrid["similarity_score"]
                    + (1.0 - content_weight) * hybrid["normalized_rating"]
                )
                hybrid = (
                    hybrid.sort_values("hybrid_score", ascending=False)
                    .head(top_n)
                    .loc[
                        :,
                        [
                            "movieId",
                            "title",
                            "genres",
                            "similarity_score",
                            "predicted_rating",
                            "hybrid_score",
                        ],
                    ]
                    .reset_index(drop=True)
                )
                render_recommendations(hybrid)
        except (TypeError, ValueError) as error:
            st.error(str(error))
