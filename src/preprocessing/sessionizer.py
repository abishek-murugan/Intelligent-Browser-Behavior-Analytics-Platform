"""
Browser event sessionization.

Groups consecutive browser events into sessions based on an
inactivity threshold. A new session starts whenever the gap between
two consecutive events exceeds the configured threshold.

For example, with a 15 minute threshold:

    09:00 Google
    09:02 GitHub
    09:05 Stack Overflow
    09:11 GitHub
           |
    Session 1

    10:00 YouTube
    10:03 Instagram
           |
    Session 2

The output adds:

- session_id
- session_event_index
- session_start
- session_end
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exceptions import (
    DataValidationError,
    FileReadError,
    FileWriteError,
)
from src.utils.config_loader import get_config, get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Sessionizer:
    """Group consecutive browser events into inactivity-based sessions."""

    REQUIRED_DATA_COLUMNS = {
        "timestamp",
        "url",
        "title",
        "domain",
        "visit_count",
        "total_mb",
        "used_mb",
        "available_mb",
        "usage_percent",
        "category",
    }

    SESSION_COLUMNS = [
        "session_id",
        "session_event_index",
        "session_start",
        "session_end",
    ]

    def __init__(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        inactivity_threshold_minutes: int | float | None = None,
    ) -> None:
        """
        Initialize the sessionizer.

        Parameters
        ----------
        input_path:
            Path to the categorized browser/RAM Parquet dataset.

        output_path:
            Path where the sessionized Parquet dataset will be saved.

        inactivity_threshold_minutes:
            Maximum allowed gap between consecutive events before a
            new session starts. Defaults to the value in config.yaml.
        """

        paths = get_paths()["paths"]

        self.input_path = Path(
            input_path
            if input_path is not None
            else paths.get(
                "browser_ram_categorized",
                "data/silver/browser_ram_categorized.parquet",
            )
        ).expanduser()

        self.output_path = Path(
            output_path
            if output_path is not None
            else paths.get(
                "browser_sessions",
                "data/silver/browser_sessions.parquet",
            )
        ).expanduser()

        if inactivity_threshold_minutes is None:
            inactivity_threshold_minutes = get_config()["session"]["inactivity_threshold_minutes"]

        if inactivity_threshold_minutes <= 0:
            raise ValueError("inactivity_threshold_minutes must be greater than zero.")

        self.inactivity_threshold_minutes = float(inactivity_threshold_minutes)

        self.inactivity_threshold = pd.Timedelta(minutes=self.inactivity_threshold_minutes)

    def run(self) -> pd.DataFrame:
        """
        Execute the complete sessionization pipeline.

        Returns
        -------
        pd.DataFrame
            Sessionized dataset.
        """

        logger.info("Starting sessionization.")

        dataframe = self._load_parquet()

        self._validate_schema(dataframe)

        dataframe = self._prepare_timestamps(dataframe)

        result = self.sessionize(dataframe)

        self.save(result)

        logger.info("Sessionization completed.")

        return result

    def sessionize(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assign session identifiers to consecutive browser events.

        A new session begins when the gap between two consecutive
        events exceeds the configured inactivity threshold.

        Parameters
        ----------
        dataframe:
            Categorized browser/RAM dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with session columns appended.
        """

        missing_columns = self.REQUIRED_DATA_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Browser dataset is missing required columns: {sorted(missing_columns)}"
            )

        if "timestamp" not in dataframe.columns:
            raise DataValidationError("Browser dataset is missing required column: timestamp")

        result = dataframe.sort_values("timestamp").reset_index(drop=True)

        # --------------------------------------------------
        # Session boundaries
        # --------------------------------------------------

        gap = result["timestamp"].diff()

        new_session = gap.isna() | (gap > self.inactivity_threshold)

        result["session_id"] = new_session.cumsum()

        # --------------------------------------------------
        # Per-event session position
        # --------------------------------------------------

        result["session_event_index"] = result.groupby("session_id").cumcount()

        # --------------------------------------------------
        # Session boundaries per event
        # --------------------------------------------------

        result["session_start"] = result.groupby("session_id")["timestamp"].transform("first")

        result["session_end"] = result.groupby("session_id")["timestamp"].transform("last")

        self._log_session_stats(result)

        return result

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the sessionized dataset as Parquet.

        Parameters
        ----------
        dataframe:
            Sessionized browser dataset.

        output_path:
            Optional output path.

        Returns
        -------
        Path
            Path to the saved dataset.
        """

        path = Path(output_path if output_path is not None else self.output_path).expanduser()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            dataframe.to_parquet(
                path,
                index=False,
            )

        except (
            OSError,
            ImportError,
        ) as exc:
            raise FileWriteError(f"Failed to save sessionized dataset: {path}") from exc

        logger.info(
            "Sessionized dataset saved: %s | records=%d",
            path,
            len(dataframe),
        )

        return path

    def _load_parquet(self) -> pd.DataFrame:
        """
        Load and validate the categorized Parquet dataset.
        """

        if not self.input_path.is_file():
            raise FileReadError(f"Categorized dataset not found: {self.input_path}")

        try:
            dataframe = pd.read_parquet(
                self.input_path,
            )

        except (
            OSError,
            ImportError,
        ) as exc:
            raise FileReadError(f"Unable to read categorized dataset: {self.input_path}") from exc

        if dataframe.empty:
            raise DataValidationError(f"Categorized dataset is empty: {self.input_path}")

        return dataframe

    def _validate_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate that all required columns are present.
        """

        missing_columns = self.REQUIRED_DATA_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Browser dataset is missing required columns: {sorted(missing_columns)}"
            )

    def _prepare_timestamps(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize and validate timestamps.
        """

        dataframe = dataframe.copy()

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
            utc=True,
        )

        invalid_count = int(dataframe["timestamp"].isna().sum())

        if invalid_count:
            raise DataValidationError(
                f"Browser dataset contains {invalid_count} invalid timestamps."
            )

        dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

        return dataframe

    def _log_session_stats(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Log summary statistics for the generated sessions.
        """

        session_stats = dataframe.groupby("session_id").agg(
            events=("timestamp", "count"),
            duration=("timestamp", lambda s: s.max() - s.min()),
        )

        logger.info(
            "Sessionization completed | events=%d | "
            "sessions=%d | events/session "
            "(avg=%.2f, max=%d) | duration median=%s",
            len(dataframe),
            len(session_stats),
            session_stats["events"].mean(),
            session_stats["events"].max(),
            session_stats["duration"].median(),
        )
