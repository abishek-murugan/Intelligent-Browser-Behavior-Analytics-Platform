"""
Gold / ML-ready dataset builder.

Converts the session-level feature dataset produced by the feature
engineering pipeline into the two reproducible datasets consumed by
the ML and deep learning models:

    data/gold/session_features.parquet     -> one row per session (clustering)
    data/gold/behavior_sequences.parquet   -> supervised sliding windows (LSTM)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import (
    DataValidationError,
    FileReadError,
    FileWriteError,
    SequenceGenerationError,
)
from src.utils.config_loader import get_models, get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """Build the Gold layer datasets from session-level features."""

    REQUIRED_INPUT_COLUMNS = {
        "session_id",
        "session_start",
        "session_end",
        "dominant_category",
    }

    EXCLUDED_COLUMNS = {
        "session_id",
        "session_start",
        "session_end",
        "time_of_day",
        "dominant_category",
    }

    TIME_OF_DAY_BUCKETS = ["Morning", "Afternoon", "Evening", "Night"]

    def __init__(
        self,
        input_path: str | Path | None = None,
        session_features_output_path: str | Path | None = None,
        behavior_sequences_output_path: str | Path | None = None,
        sequence_length: int | None = None,
    ) -> None:
        """
        Initialize the Gold dataset builder.

        Parameters
        ----------
        input_path:
            Path to the session-level feature dataset produced by the
            feature engineering pipeline.

        session_features_output_path:
            Output path for the Gold session features dataset.

        behavior_sequences_output_path:
            Output path for the Gold behavior sequences dataset.

        sequence_length:
            Number of previous sessions used as the input window for
            each sequence sample. Defaults to the configured LSTM
            sequence length.
        """

        paths = get_paths()["paths"]
        models = get_models()

        self.input_path = Path(
            input_path
            if input_path is not None
            else paths.get(
                "session_features",
                "data/silver/session_features.parquet",
            )
        ).expanduser()

        self.session_features_output_path = Path(
            session_features_output_path
            if session_features_output_path is not None
            else paths.get(
                "session_features_gold",
                "data/gold/session_features.parquet",
            )
        ).expanduser()

        self.behavior_sequences_output_path = Path(
            behavior_sequences_output_path
            if behavior_sequences_output_path is not None
            else paths.get(
                "behavior_sequences",
                "data/gold/behavior_sequences.parquet",
            )
        ).expanduser()

        self.sequence_length = (
            sequence_length
            if sequence_length is not None
            else models["deep_learning"]["lstm"]["sequence_length"]
        )

        if self.sequence_length < 1:
            raise DataValidationError(f"sequence_length must be >= 1, got {self.sequence_length}.")

        self.feature_columns: list[str] = []

    def run(self) -> dict[str, pd.DataFrame]:
        """
        Execute the complete Gold dataset build.

        Returns
        -------
        dict[str, pd.DataFrame]
            The session features and behavior sequences datasets keyed
            by their artifact names.
        """

        logger.info("Starting Gold dataset build.")

        dataframe = self._load_input()

        self._validate_input(dataframe)

        session_features = self.build_session_features(dataframe)
        behavior_sequences = self.build_behavior_sequences(session_features)

        self.save_session_features(session_features)
        self.save_behavior_sequences(behavior_sequences)

        logger.info("Gold dataset build completed.")

        return {
            "session_features": session_features,
            "behavior_sequences": behavior_sequences,
        }

    def build_session_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the Gold session features dataset.

        One row per session, ordered chronologically.

        Parameters
        ----------
        dataframe:
            Session-level feature dataset.

        Returns
        -------
        pd.DataFrame
            Gold session features dataset.
        """

        self._validate_input(dataframe)

        result = dataframe.sort_values(
            ["session_start", "session_id"],
        ).reset_index(drop=True)

        logger.info(
            "Session features built | sessions=%d | columns=%d",
            len(result),
            len(result.columns),
        )

        return result

    def build_behavior_sequences(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the Gold behavior sequences dataset.

        Each sample is a supervised sliding window of the previous
        `sequence_length` sessions' numeric feature vectors with the
        next session as target.

        Parameters
        ----------
        dataframe:
            Chronologically ordered session-level feature dataset.

        Returns
        -------
        pd.DataFrame
            Gold behavior sequences dataset.
        """

        self._validate_input(dataframe)

        ordered = dataframe.sort_values(
            ["session_start", "session_id"],
        ).reset_index(drop=True)

        if len(ordered) < self.sequence_length:
            raise SequenceGenerationError(
                f"Cannot build sequences: sessions={len(ordered)} < "
                f"sequence_length={self.sequence_length}"
            )

        feature_vectors, feature_columns = self._build_feature_matrix(ordered)

        self._validate_feature_matrix(feature_vectors, feature_columns, ordered)

        self.feature_columns = feature_columns

        rows: list[dict[str, Any]] = []

        for index in range(self.sequence_length, len(ordered)):
            window = ordered.iloc[index - self.sequence_length : index]

            rows.append(
                {
                    "sequence_id": index - self.sequence_length,
                    "session_ids": window["session_id"].tolist(),
                    "start_time": window["session_start"].iloc[0],
                    "end_time": ordered["session_start"].iloc[index],
                    "feature_vectors": feature_vectors[
                        index - self.sequence_length : index
                    ].tolist(),
                    "target_session_id": ordered["session_id"].iloc[index],
                    "target_category": ordered["dominant_category"].iloc[index],
                    "target_features": feature_vectors[index].tolist(),
                }
            )

        result = pd.DataFrame(rows)

        logger.info(
            "Behavior sequences built | samples=%d | sequence_length=%d | features=%d",
            len(result),
            self.sequence_length,
            len(feature_columns),
        )

        return result

    def save_session_features(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the Gold session features dataset as Parquet.

        Parameters
        ----------
        dataframe:
            Gold session features dataset.

        output_path:
            Optional output path.

        Returns
        -------
        Path
            Path to the saved dataset.
        """

        path = Path(
            output_path if output_path is not None else self.session_features_output_path
        ).expanduser()

        return self._save(dataframe, path, "session features")

    def save_behavior_sequences(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the Gold behavior sequences dataset as Parquet.

        Parameters
        ----------
        dataframe:
            Gold behavior sequences dataset.

        output_path:
            Optional output path.

        Returns
        -------
        Path
            Path to the saved dataset.
        """

        path = Path(
            output_path if output_path is not None else self.behavior_sequences_output_path
        ).expanduser()

        return self._save(dataframe, path, "behavior sequences")

    def _load_input(self) -> pd.DataFrame:
        """
        Load and validate the session-level feature dataset.
        """

        if not self.input_path.is_file():
            raise FileReadError(f"Session features dataset not found: {self.input_path}")

        try:
            dataframe = pd.read_parquet(self.input_path)

        except (OSError, ImportError) as exc:
            raise FileReadError(
                f"Unable to read session features dataset: {self.input_path}"
            ) from exc

        if dataframe.empty:
            raise DataValidationError(f"Session features dataset is empty: {self.input_path}")

        return dataframe

    def _validate_input(self, dataframe: pd.DataFrame) -> None:
        """
        Validate the session-level feature dataset.
        """

        missing_columns = self.REQUIRED_INPUT_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"Session features dataset is missing required columns: {sorted(missing_columns)}"
            )

        if dataframe["session_id"].isna().any():
            raise DataValidationError(
                "Session features dataset contains rows with missing session_id."
            )

        if dataframe["session_id"].duplicated().any():
            raise DataValidationError(
                "Session features dataset contains duplicate session_id values."
            )

        for column in ("session_start", "session_end"):
            if not pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype):
                raise DataValidationError(f"Column '{column}' must be a datetime column.")

            if dataframe[column].isna().any():
                raise DataValidationError(
                    f"Session features dataset contains rows with missing {column}."
                )

    def _build_feature_matrix(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build the numeric feature matrix for sequence generation.

        Numeric and boolean columns are kept as-is (bools coerced to
        float); the categorical time-of-day column is one-hot encoded
        in a fixed bucket order.
        """

        numeric_columns = [
            column
            for column in dataframe.columns
            if column not in self.EXCLUDED_COLUMNS
            and pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        ]

        numeric_values = dataframe[numeric_columns].astype("float64").to_numpy()

        time_one_hot = pd.get_dummies(
            dataframe["time_of_day"],
            prefix="time_of_day",
            dtype=float,
        ).reindex(
            columns=[f"time_of_day_{bucket}" for bucket in self.TIME_OF_DAY_BUCKETS],
            fill_value=0.0,
        )

        time_values = time_one_hot.to_numpy()

        matrix = np.hstack([numeric_values, time_values])

        feature_columns = numeric_columns + list(time_one_hot.columns)

        return matrix, feature_columns

    @staticmethod
    def _validate_feature_matrix(
        matrix: np.ndarray,
        feature_columns: list[str],
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate that the numeric feature matrix is ML-ready.
        """

        if matrix.shape[0] != len(dataframe):
            raise DataValidationError("Feature matrix row count does not match session count.")

        if matrix.shape[1] != len(feature_columns):
            raise DataValidationError("Feature matrix column count does not match feature columns.")

        if np.isnan(matrix).any():
            missing_columns = [
                column
                for index, column in enumerate(feature_columns)
                if np.isnan(matrix[:, index]).any()
            ]

            raise DataValidationError(
                f"Feature matrix contains missing values in columns: {missing_columns}"
            )

    @staticmethod
    def _save(
        dataframe: pd.DataFrame,
        path: Path,
        label: str,
    ) -> Path:
        """
        Save a Gold dataset as Parquet.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            dataframe.to_parquet(
                path,
                index=False,
            )

        except (OSError, ImportError) as exc:
            raise FileWriteError(f"Failed to save {label} dataset: {path}") from exc

        logger.info(
            "Gold %s dataset saved: %s | rows=%d | columns=%d",
            label,
            path,
            len(dataframe),
            len(dataframe.columns),
        )

        return path
