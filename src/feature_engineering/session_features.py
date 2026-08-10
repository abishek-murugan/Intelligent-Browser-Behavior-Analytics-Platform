"""
Session feature engineering.

Aggregates sessionized event data into one row per session,
including duration, event count, domain/category counts and
RAM usage statistics.
"""

from __future__ import annotations

import pandas as pd

from src.exceptions import (
    DataValidationError,
    FeatureEngineeringError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionFeatureBuilder:
    """Build aggregate session features from event-level data."""

    REQUIRED_COLUMNS = {
        "session_id",
        "timestamp",
        "domain",
        "category",
        "visit_count",
        "used_mb",
        "available_mb",
        "usage_percent",
    }

    def build(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build session-level aggregate features.

        Parameters
        ----------
        dataframe:
            Sessionized browser event dataset.

        Returns
        -------
        pd.DataFrame
            One row per session with aggregate features.
        """

        missing_columns = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Sessionized dataset is missing required columns: {sorted(missing_columns)}"
            )

        try:
            result = (
                dataframe.groupby("session_id")
                .agg(
                    session_start=("timestamp", "min"),
                    session_end=("timestamp", "max"),
                    event_count=("timestamp", "size"),
                    unique_domains=("domain", "nunique"),
                    unique_categories=("category", "nunique"),
                    total_visit_count=("visit_count", "sum"),
                    min_used_mb=("used_mb", "min"),
                    avg_used_mb=("used_mb", "mean"),
                    max_used_mb=("used_mb", "max"),
                    avg_available_mb=("available_mb", "mean"),
                    avg_usage_percent=("usage_percent", "mean"),
                    max_usage_percent=("usage_percent", "max"),
                )
                .reset_index()
            )

        except Exception as exc:
            raise FeatureEngineeringError("Failed to compute session features.") from exc

        result["session_duration_seconds"] = (
            (result["session_end"] - result["session_start"]).dt.total_seconds()
        ).astype("int64")

        feature_columns = [
            "session_id",
            "session_duration_seconds",
            "event_count",
            "unique_domains",
            "unique_categories",
            "total_visit_count",
            "min_used_mb",
            "avg_used_mb",
            "max_used_mb",
            "avg_available_mb",
            "avg_usage_percent",
            "max_usage_percent",
        ]

        result = result[feature_columns].reset_index(
            drop=True,
        )

        logger.info(
            "Session features built | sessions=%d | columns=%d",
            len(result),
            len(feature_columns),
        )

        return result
