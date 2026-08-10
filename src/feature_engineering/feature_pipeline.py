"""
Feature engineering pipeline.

Orchestrates the temporal, session and behavioral feature builders
into a single session-level feature dataset.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

from src.exceptions import (
    DataValidationError,
    FeatureEngineeringError,
    FileReadError,
    FileWriteError,
)
from src.feature_engineering.behavior_features import (
    BehaviorFeatureBuilder,
)
from src.feature_engineering.session_features import (
    SessionFeatureBuilder,
)
from src.feature_engineering.temporal_features import (
    TemporalFeatureBuilder,
)
from src.utils.config_loader import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """Build the complete session-level feature dataset."""

    REQUIRED_INPUT_COLUMNS = {
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
        "session_id",
        "session_event_index",
        "session_start",
        "session_end",
    }

    def __init__(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the feature pipeline.

        Parameters
        ----------
        input_path:
            Path to the sessionized Parquet dataset.

        output_path:
            Path where the session-level feature dataset will be
            saved.
        """

        paths = get_paths()["paths"]

        self.input_path = Path(
            input_path
            if input_path is not None
            else paths.get(
                "browser_sessions",
                "data/silver/browser_sessions.parquet",
            )
        ).expanduser()

        self.output_path = Path(
            output_path
            if output_path is not None
            else paths.get(
                "session_features",
                "data/silver/session_features.parquet",
            )
        ).expanduser()

        self.temporal_builder = TemporalFeatureBuilder()
        self.session_builder = SessionFeatureBuilder()
        self.behavior_builder = BehaviorFeatureBuilder()

    def run(self) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline.

        Returns
        -------
        pd.DataFrame
            Session-level feature dataset.
        """

        logger.info("Starting feature engineering pipeline.")

        dataframe = self._load_input()

        self._validate_schema(dataframe)

        temporal_features = self.temporal_builder.build(dataframe)
        session_features = self.session_builder.build(dataframe)
        behavior_features = self.behavior_builder.build(dataframe)

        result = self._merge_features(
            temporal_features,
            session_features,
            behavior_features,
        )

        self._validate_result(result)

        self.save(result)

        logger.info("Feature engineering pipeline completed.")

        return result

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the feature dataset as Parquet.

        Parameters
        ----------
        dataframe:
            Session-level feature dataset.

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
            raise FileWriteError(f"Failed to save feature dataset: {path}") from exc

        logger.info(
            "Feature dataset saved: %s | sessions=%d | features=%d",
            path,
            len(dataframe),
            len(dataframe.columns) - 1,
        )

        return path

    def _load_input(self) -> pd.DataFrame:
        """
        Load and validate the sessionized Parquet dataset.
        """

        if not self.input_path.is_file():
            raise FileReadError(f"Sessionized dataset not found: {self.input_path}")

        try:
            dataframe = pd.read_parquet(self.input_path)

        except (OSError, ImportError) as exc:
            raise FileReadError(f"Unable to read sessionized dataset: {self.input_path}") from exc

        if dataframe.empty:
            raise DataValidationError(f"Sessionized dataset is empty: {self.input_path}")

        return dataframe

    def _validate_schema(self, dataframe: pd.DataFrame) -> None:
        """
        Validate that all required input columns are present.
        """

        missing_columns = self.REQUIRED_INPUT_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Sessionized dataset is missing required columns: {sorted(missing_columns)}"
            )

    @staticmethod
    def _merge_features(
        *feature_frames: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge session-level feature frames on session_id.
        """

        frames = list(feature_frames)

        for frame in frames:
            if "session_id" not in frame.columns:
                raise FeatureEngineeringError("Feature frame is missing session_id column.")

        result = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on="session_id",
                how="outer",
            ),
            frames,
        )

        return result

    @staticmethod
    def _validate_result(dataframe: pd.DataFrame) -> None:
        """
        Validate the merged feature dataset.
        """

        if dataframe["session_id"].isna().any():
            raise DataValidationError(
                "Merged feature dataset contains rows with missing session_id."
            )

        if dataframe["session_id"].duplicated().any():
            raise DataValidationError(
                "Merged feature dataset contains duplicate session_id values."
            )

        logger.info(
            "Feature dataset merged | sessions=%d | columns=%d",
            len(dataframe),
            len(dataframe.columns),
        )
