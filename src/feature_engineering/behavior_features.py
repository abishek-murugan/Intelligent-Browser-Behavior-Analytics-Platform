"""
Behavioral feature engineering.

Builds behavioral metrics for each browsing session such as
repetition, browsing intensity and content diversity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.exceptions import (
    DataValidationError,
    FeatureEngineeringError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BehaviorFeatureBuilder:
    """Build behavioral features for each browsing session."""

    REQUIRED_COLUMNS = {
        "session_id",
        "session_event_index",
        "timestamp",
        "domain",
        "category",
    }

    def build(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build session-level behavioral features.

        Parameters
        ----------
        dataframe:
            Sessionized browser event dataset.

        Returns
        -------
        pd.DataFrame
            One row per session with behavioral features.
        """

        missing_columns = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Sessionized dataset is missing required columns: {sorted(missing_columns)}"
            )

        try:
            result = self._compute_behavioral_features(dataframe)

        except Exception as exc:
            raise FeatureEngineeringError("Failed to compute behavioral features.") from exc

        logger.info(
            "Behavioral features built | sessions=%d | columns=%d",
            len(result),
            len(result.columns),
        )

        return result

    def _compute_behavioral_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all behavioral feature columns.
        """

        data = dataframe.sort_values(["session_id", "session_event_index"]).copy()

        # --------------------------------------------------
        # Per-session basic aggregates
        # --------------------------------------------------

        session_bounds = (
            data.groupby("session_id")["timestamp"]
            .agg(
                session_start="min",
                session_end="max",
                event_count="size",
            )
            .reset_index()
        )

        session_bounds["duration_seconds"] = (
            session_bounds["session_end"] - session_bounds["session_start"]
        ).dt.total_seconds()

        domain_counts = data.groupby(["session_id", "domain"]).size().reset_index(name="count")

        category_counts = data.groupby(["session_id", "category"]).size().reset_index(name="count")

        unique_stats = data.groupby("session_id").agg(
            unique_domains=("domain", "nunique"),
            unique_categories=("category", "nunique"),
        )

        # --------------------------------------------------
        # Repetition
        # --------------------------------------------------

        data["is_repeat"] = data.duplicated(subset=["session_id", "domain"])

        repeat_stats = data.groupby("session_id").agg(
            repeat_visits=("is_repeat", "sum"),
        )

        top_domain = domain_counts.groupby("session_id")["count"].max().rename("max_domain_repeats")

        top_category = (
            category_counts.groupby("session_id")["count"].max().rename("max_category_repeats")
        )

        # --------------------------------------------------
        # Intensity
        # --------------------------------------------------

        data["gap_seconds"] = data.groupby("session_id")["timestamp"].diff().dt.total_seconds()

        gap_stats = data.groupby("session_id")["gap_seconds"].agg(
            avg_gap_seconds="mean",
            median_gap_seconds="median",
            max_gap_seconds="max",
        )

        # --------------------------------------------------
        # Diversity
        # --------------------------------------------------

        previous_category = data.groupby("session_id")["category"].shift()

        data["category_switch"] = previous_category.notna() & (
            data["category"] != previous_category
        )

        category_switch_stats = (
            data.groupby("session_id")["category_switch"].sum().rename("category_switch_count")
        )

        category_entropy = (
            data.groupby("session_id")["category"].apply(self._entropy).rename("category_entropy")
        )

        domain_entropy = (
            data.groupby("session_id")["domain"].apply(self._entropy).rename("domain_entropy")
        )

        # --------------------------------------------------
        # Merge into a single session-level frame
        # --------------------------------------------------

        result = (
            session_bounds.set_index("session_id")
            .join(unique_stats)
            .join(repeat_stats)
            .join(top_domain)
            .join(top_category)
            .join(gap_stats)
            .join(category_switch_stats)
            .join(category_entropy)
            .join(domain_entropy)
            .reset_index()
        )

        # --------------------------------------------------
        # Derived feature columns
        # --------------------------------------------------

        result["repeat_visit_ratio"] = result["repeat_visits"] / result["event_count"]

        result["top_domain_share"] = result["max_domain_repeats"] / result["event_count"]

        result["top_category_share"] = result["max_category_repeats"] / result["event_count"]

        duration_minutes = result["duration_seconds"] / 60

        result["events_per_minute"] = np.where(
            duration_minutes > 0,
            result["event_count"] / duration_minutes,
            0.0,
        )

        result["domain_diversity_index"] = result["unique_domains"] / result["event_count"]

        feature_columns = [
            "session_id",
            "repeat_visit_ratio",
            "max_domain_repeats",
            "top_domain_share",
            "top_category_share",
            "events_per_minute",
            "avg_gap_seconds",
            "median_gap_seconds",
            "max_gap_seconds",
            "category_entropy",
            "domain_entropy",
            "domain_diversity_index",
            "category_switch_count",
        ]

        result = result[feature_columns]

        # Fill gap statistics for sessions with a single event.
        result = result.fillna(
            {
                "avg_gap_seconds": 0.0,
                "median_gap_seconds": 0.0,
                "max_gap_seconds": 0.0,
            }
        )

        return result.reset_index(drop=True)

    @staticmethod
    def _entropy(series: pd.Series) -> float:
        """
        Compute Shannon entropy in bits for a categorical series.
        """

        counts = series.value_counts()

        probabilities = counts / counts.sum()

        return float(-(probabilities * np.log2(probabilities)).sum())
