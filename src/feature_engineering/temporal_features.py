"""
Temporal feature engineering.

Derives time-based features for each browsing session from the
session event timestamps (UTC), including hour, day of week,
weekend flag, time-of-day bucket and session time span.
"""

from __future__ import annotations

import pandas as pd

from src.exceptions import (
    DataValidationError,
    FeatureEngineeringError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalFeatureBuilder:
    """Build temporal features for each browsing session."""

    REQUIRED_COLUMNS = {
        "session_id",
        "timestamp",
    }

    def build(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build session-level temporal features.

        Parameters
        ----------
        dataframe:
            Sessionized browser event dataset.

        Returns
        -------
        pd.DataFrame
            One row per session with temporal features.
        """

        missing_columns = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Sessionized dataset is missing required columns: {sorted(missing_columns)}"
            )

        try:
            session_bounds = (
                dataframe.groupby("session_id")["timestamp"]
                .agg(
                    session_start="min",
                    session_end="max",
                )
                .reset_index()
            )

        except Exception as exc:
            raise FeatureEngineeringError("Failed to compute temporal features.") from exc

        result = session_bounds.copy()

        result["hour"] = result["session_start"].dt.hour.astype("int64")

        result["day_of_week_num"] = result["session_start"].dt.dayofweek.astype("int64")

        result["is_weekend"] = result["day_of_week_num"].isin([5, 6])

        result["time_of_day"] = result["hour"].apply(self._time_of_day).astype("string")

        result["session_hour_span"] = (
            result["session_end"] - result["session_start"]
        ).dt.total_seconds() / 3600

        result["crosses_midnight"] = (
            result["session_start"].dt.date != result["session_end"].dt.date
        )

        feature_columns = [
            "session_id",
            "session_start",
            "session_end",
            "hour",
            "day_of_week_num",
            "is_weekend",
            "time_of_day",
            "session_hour_span",
            "crosses_midnight",
        ]

        result = result[feature_columns].reset_index(
            drop=True,
        )

        logger.info(
            "Temporal features built | sessions=%d | columns=%d",
            len(result),
            len(feature_columns),
        )

        return result

    @staticmethod
    def _time_of_day(
        hour: int,
    ) -> str:
        """
        Bucket an hour into a time-of-day category.

        Morning:   05:00 - 11:59
        Afternoon: 12:00 - 16:59
        Evening:   17:00 - 21:59
        Night:     22:00 - 04:59
        """

        if 5 <= hour <= 11:
            return "Morning"

        if 12 <= hour <= 16:
            return "Afternoon"

        if 17 <= hour <= 21:
            return "Evening"

        return "Night"
