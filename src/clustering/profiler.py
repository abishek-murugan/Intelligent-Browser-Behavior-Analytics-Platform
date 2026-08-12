"""
Segment profiling.

Summarizes each KMeans segment with size, share, dominant category
and mean feature values, and exports a human-readable CSV report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exceptions import (
    DataValidationError,
    FileWriteError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXCLUDED_PROFILE_COLUMNS = {
    "session_id",
    "session_start",
    "session_end",
}


class SegmentProfiler:
    """Build descriptive profiles for each session segment."""

    def __init__(
        self,
        segment_column: str = "segment_id",
    ) -> None:
        """
        Initialize the profiler.

        Parameters
        ----------
        segment_column:
            Name of the column holding the segment assignment.
        """

        self.segment_column = segment_column

    def profile(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-segment aggregate statistics.

        Parameters
        ----------
        dataframe:
            Gold session features dataset with a segment_id column.

        Returns
        -------
        pd.DataFrame
            One row per segment with size, share, dominant category
            and mean feature values.
        """

        if self.segment_column not in dataframe.columns:
            raise DataValidationError(
                f"Segment column '{self.segment_column}' not found in dataset."
            )

        if "dominant_category" not in dataframe.columns:
            raise DataValidationError("Segment dataset is missing 'dominant_category' column.")

        segments = sorted(dataframe[self.segment_column].dropna().unique())

        if not segments:
            raise DataValidationError("No segments found to profile.")

        rows: list[dict[str, object]] = []

        total = len(dataframe)

        for segment in segments:
            subset = dataframe[dataframe[self.segment_column] == segment]

            mean_values = subset.select_dtypes(include=["number"]).mean()

            row: dict[str, object] = {
                "segment_id": int(segment),
                "session_count": int(len(subset)),
                "share_pct": round(len(subset) / total * 100, 2),
                "dominant_category": subset["dominant_category"].mode().iloc[0],
            }

            row.update(
                {
                    column: round(value, 4)
                    for column, value in mean_values.items()
                    if column not in EXCLUDED_PROFILE_COLUMNS
                }
            )

            rows.append(row)

        result = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)

        logger.info(
            "Segment profiles built | segments=%d",
            len(result),
        )

        return result

    def save(
        self,
        profile: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """
        Save the segment profile report as CSV.

        Parameters
        ----------
        profile:
            Profile table produced by ``profile``.

        output_path:
            Destination path for the CSV report.

        Returns
        -------
        Path
            Path to the saved report.
        """

        path = Path(output_path).expanduser()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            profile.to_csv(
                path,
                index=False,
            )

        except OSError as exc:
            raise FileWriteError(f"Failed to save segment profile: {path}") from exc

        logger.info("Segment profile saved: %s", path)

        return path

    @staticmethod
    def dominant_feature_importances(
        profile: pd.DataFrame,
        top_n: int = 5,
    ) -> dict[int, list[tuple[str, float]]]:
        """
        Rank the numeric features that distinguish each segment from
        the dataset mean.

        Parameters
        ----------
        profile:
            Profile table produced by ``profile``.

        top_n:
            Number of top distinguishing features per segment.

        Returns
        -------
        dict[int, list[tuple[str, float]]]
            Per-segment list of (feature, deviation) pairs ranked by
            absolute deviation from the mean.
        """

        numeric_columns = profile.select_dtypes(include=["number"]).columns

        feature_columns = [
            column
            for column in numeric_columns
            if column
            not in {
                "segment_id",
                "session_count",
                "share_pct",
            }
        ]

        result: dict[int, list[tuple[str, float]]] = {}

        means = profile[feature_columns].mean()

        for _, row in profile.iterrows():
            deviations = {column: float(row[column] - means[column]) for column in feature_columns}

            ranked = sorted(
                deviations.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:top_n]

            result[int(row["segment_id"])] = ranked

        return result
