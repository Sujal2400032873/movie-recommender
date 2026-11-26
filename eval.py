from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


RATINGS_PATH = Path(__file__).resolve().parent / "data" / "ratings.csv"
RELEVANCE_THRESHOLD = 4.0


def precision_recall_at_k(
    predictions,
    k: int,
    threshold: float = RELEVANCE_THRESHOLD,
) -> tuple[float, float]:
    """Return macro-averaged Precision@K and Recall@K across users."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")

    user_ratings = {}
    for prediction in predictions:
        user_ratings.setdefault(prediction.uid, []).append(
            (prediction.est, prediction.r_ui)
        )

    user_precisions = []
    user_recalls = []
    for ratings in user_ratings.values():
        ratings.sort(key=lambda rating: rating[0], reverse=True)
        top_k = ratings[:k]
        relevant_count = sum(
            actual_rating >= threshold for _, actual_rating in ratings
        )
        relevant_recommended = sum(
            estimated_rating >= threshold and actual_rating >= threshold
            for estimated_rating, actual_rating in top_k
        )

        user_precisions.append(
            relevant_recommended / len(top_k) if top_k else 0.0
        )
        user_recalls.append(
            relevant_recommended / relevant_count if relevant_count else 0.0
        )

    if not user_precisions:
        raise ValueError("predictions must contain at least one user")

    return (
        sum(user_precisions) / len(user_precisions),
        sum(user_recalls) / len(user_recalls),
    )


def evaluate() -> None:
    ratings = pd.read_csv(RATINGS_PATH)
    rating_min = float(ratings["rating"].min())
    rating_max = float(ratings["rating"].max())
    reader = Reader(rating_scale=(rating_min, rating_max))
    dataset = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader,
    )
    trainset, testset = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
    )

    model = SVD(random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    print("SVD Evaluation Metrics")
    print("----------------------")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"Relevance threshold: rating >= {RELEVANCE_THRESHOLD:.1f}")
    for k in (5, 10):
        precision, recall = precision_recall_at_k(
            predictions,
            k,
            threshold=RELEVANCE_THRESHOLD,
        )
        print(f"Precision@{k}: {precision:.4f}")
        print(f"Recall@{k}:    {recall:.4f}")


if __name__ == "__main__":
    evaluate()
