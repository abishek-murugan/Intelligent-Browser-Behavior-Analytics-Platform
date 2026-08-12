import pandas as pd

from src.recommendation.pipeline import RecommendationPipeline


def test_recommendation_pipeline_ranks_forecast_first(tmp_path):
    sessions = pd.DataFrame(
        {"dominant_category": ["Work", "Work", "Work", "Learning", "Learning", "Learning"]}
    )
    predictions = pd.DataFrame({"target_session_id": [10], "predicted_category": ["Learning"]})
    sessions_path, predictions_path = (
        tmp_path / "sessions.parquet",
        tmp_path / "predictions.parquet",
    )
    sessions.to_parquet(sessions_path, index=False)
    predictions.to_parquet(predictions_path, index=False)

    result = RecommendationPipeline(
        sessions_path=sessions_path,
        predictions_path=predictions_path,
        output_path=tmp_path / "recommendations.parquet",
        top_k=2,
        track_mlflow=False,
    ).run()

    assert result.iloc[0]["recommended_category"] == "Learning"
    assert (tmp_path / "recommendations.parquet").is_file()
