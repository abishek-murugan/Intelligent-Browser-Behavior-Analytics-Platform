"""
RAM usage data loader.

Loads and validates the historical RAM usage CSV dataset used by
the Browser Behavior Analytics project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exceptions import DataCollectionError, DataValidationError
from src.utils.config_loader import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAMDataLoader:
    """Load and validate historical RAM usage data."""

    REQUIRED_COLUMNS = {
        "timestamp",
        "total_mb",
        "used_mb",
        "available_mb",
        "usage_percent",
    }

    def __init__(
        self,
        input_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the RAM data loader.

        Parameters
        ----------
        input_path:
            Path to the source RAM CSV file.

            If omitted, the path configured in paths.yaml
            is used.
        """

        if input_path is None:
            paths = get_paths()
            input_path = paths["paths"]["ram_log"]

        self.input_path = Path(input_path).expanduser()

    def load(self) -> pd.DataFrame:
        """
        Load and validate the RAM usage dataset.

        Returns
        -------
        pandas.DataFrame
            Validated RAM usage data.

        Raises
        ------
        DataCollectionError
            If the source file cannot be read.

        DataValidationError
            If the dataset does not match the expected schema.
        """

        logger.info(
            "Loading RAM dataset from: %s",
            self.input_path,
        )

        if not self.input_path.is_file():
            raise DataCollectionError(f"RAM dataset not found: {self.input_path}")

        try:
            dataframe = pd.read_csv(self.input_path)

        except (OSError, pd.errors.ParserError) as exc:
            raise DataCollectionError(f"Failed to read RAM dataset: {self.input_path}") from exc

        self._validate_schema(dataframe)

        dataframe = self._prepare(dataframe)

        logger.info(
            "RAM dataset loaded successfully | records=%d | columns=%d",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    def _validate_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the RAM dataset schema."""

        actual_columns = set(dataframe.columns)

        missing_columns = self.REQUIRED_COLUMNS - actual_columns

        if missing_columns:
            raise DataValidationError(
                f"RAM dataset is missing required columns: {sorted(missing_columns)}"
            )

        if dataframe.empty:
            raise DataValidationError("RAM dataset is empty.")

    def _prepare(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare the RAM dataset for downstream processing."""

        dataframe = dataframe.copy()

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
            utc=True,
            format="mixed",
        )

        if dataframe["timestamp"].isna().any():
            invalid_count = int(dataframe["timestamp"].isna().sum())

            raise DataValidationError(f"RAM dataset contains {invalid_count} invalid timestamps.")

        numeric_columns = [
            "total_mb",
            "used_mb",
            "available_mb",
            "usage_percent",
        ]

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

        if dataframe[numeric_columns].isna().any().any():
            raise DataValidationError("RAM dataset contains invalid numeric values.")

        dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

        return dataframe

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the validated RAM dataset as Parquet.

        Parameters
        ----------
        dataframe:
            Validated RAM usage DataFrame.

        output_path:
            Destination path. If omitted, the configured raw RAM
            Parquet path is used.

        Returns
        -------
        Path
            Path to the saved Parquet dataset.
        """

        if output_path is None:
            paths = get_paths()

            output_path = paths["paths"].get(
                "ram_data_raw",
                "data/raw/ram_usage.parquet",
            )

        output_path = Path(output_path).expanduser()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            dataframe.to_parquet(
                output_path,
                index=False,
            )

        except (OSError, ImportError) as exc:
            raise DataCollectionError(f"Failed to save RAM dataset: {output_path}") from exc

        logger.info(
            "RAM dataset saved to: %s | records=%d",
            output_path,
            len(dataframe),
        )

        return output_path
