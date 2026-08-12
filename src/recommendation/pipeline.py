"""Generate transparent category recommendations from LSTM predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exceptions import DataValidationError, FileReadError, RecommendationEngineError
from src.utils.config_loader import get_models, get_paths
from src.utils.logger import get_logger
from src.utils.mlflow_utils import get_experiment_id, get_experiment_name, setup_mlflow

logger = get_logger(__name__)


class RecommendationPipeline:
    """Rank browsing categories using model forecasts and historical affinity.

    This is deliberately a category recommender: the collected data stores
    behavioral categories, not a licensed item/content catalog.
    """

    def __init__(
        self,
        sessions_path: str | Path | None = None,
        predictions_path: str | Path | None = None,
        output_path: str | Path | None = None,
        top_k: int | None = None,
        track_mlflow: bool = True,
    ) -> None:
        paths = get_paths()["paths"]
        config = get_models()["recommendation"]
        self.sessions_path = Path(sessions_path or paths["session_features_gold"]).expanduser()
        self.predictions_path = Path(predictions_path or paths["lstm_predictions"]).expanduser()
        self.output_path = Path(output_path or paths["recommendations"]).expanduser()
        self.top_k = int(top_k or config["top_k"])
        self.min_sessions = int(config["min_sessions_for_catalog"])
        self.track_mlflow = track_mlflow

    def run(self) -> pd.DataFrame:
        """Build and save recommendations for each LSTM forecast."""
        sessions, predictions = self._load()
        catalog = self._catalog(sessions)
        recommendations = self.recommend(predictions, catalog)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_parquet(self.output_path, index=False)
        self._log_mlflow(recommendations, len(catalog))
        logger.info("Recommendations generated | rows=%d", len(recommendations))
        return recommendations

    def recommend(self, predictions: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        """Return top-k recommendations, guaranteeing the forecast is first."""
        if catalog.empty:
            raise RecommendationEngineError("No categories satisfy the catalog threshold.")
        rows: list[dict[str, object]] = []
        for _, prediction in predictions.iterrows():
            forecast = str(prediction["predicted_category"])
            ranked = catalog.copy()
            ranked["score"] = ranked["historical_affinity"]
            ranked.loc[ranked["category"] == forecast, "score"] += 1.0
            if forecast not in set(ranked["category"]):
                ranked = pd.concat(
                    [
                        ranked,
                        pd.DataFrame(
                            [
                                {
                                    "category": forecast,
                                    "session_count": 0,
                                    "historical_affinity": 0.0,
                                    "score": 1.0,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            for rank, (_, item) in enumerate(
                ranked.sort_values(["score", "session_count"], ascending=False)
                .head(self.top_k)
                .iterrows(),
                start=1,
            ):
                reason = (
                    "Matches the LSTM next-session forecast"
                    if item["category"] == forecast
                    else "Frequently observed in your browsing history"
                )
                rows.append(
                    {
                        "target_session_id": prediction["target_session_id"],
                        "predicted_category": forecast,
                        "recommendation_rank": rank,
                        "recommended_category": item["category"],
                        "score": float(item["score"]),
                        "reason": reason,
                    }
                )
        return pd.DataFrame(rows)

    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        for path, label in (
            (self.sessions_path, "Session features"),
            (self.predictions_path, "LSTM predictions"),
        ):
            if not path.is_file():
                raise FileReadError(f"{label} dataset not found: {path}")
        sessions, predictions = (
            pd.read_parquet(self.sessions_path),
            pd.read_parquet(self.predictions_path),
        )
        if "dominant_category" not in sessions or "predicted_category" not in predictions:
            raise DataValidationError(
                "Sessions require dominant_category and predictions require predicted_category."
            )
        return sessions, predictions

    def _catalog(self, sessions: pd.DataFrame) -> pd.DataFrame:
        counts = (
            sessions["dominant_category"]
            .astype(str)
            .value_counts()
            .rename_axis("category")
            .reset_index(name="session_count")
        )
        counts = counts[counts["session_count"] >= self.min_sessions].copy()
        counts["historical_affinity"] = counts["session_count"] / counts["session_count"].sum()
        return counts

    def _log_mlflow(self, recommendations: pd.DataFrame, catalog_size: int) -> None:
        if not self.track_mlflow:
            return
        import mlflow

        setup_mlflow()
        experiment_id = get_experiment_id(get_experiment_name("recommendation"))
        with mlflow.start_run(experiment_id=experiment_id, run_name="category-recommendations"):
            mlflow.log_params({"top_k": self.top_k, "catalog_size": catalog_size})
            mlflow.log_metric("recommendation_rows", len(recommendations))
            mlflow.log_artifact(str(self.output_path))
