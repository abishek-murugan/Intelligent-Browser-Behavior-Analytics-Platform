"""
Browser and RAM data integration.

Aligns Chrome browsing events with the nearest RAM usage observation
using timestamp-based matching.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exceptions import DataValidationError, FileReadError, FileWriteError
from src.utils.config_loader import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrowserRAMIntegrator:
    """Integrate Chrome history and RAM usage data by timestamp."""

    CHROME_REQUIRED_COLUMNS = {
        "timestamp",
        "url",
        "title",
        "domain",
        "visit_count",
    }

    RAM_REQUIRED_COLUMNS = {
        "timestamp",
        "total_mb",
        "used_mb",
        "available_mb",
        "usage_percent",
    }

    def __init__(
        self,
        chrome_path: str | Path | None = None,
        ram_path: str | Path | None = None,
        output_path: str | Path | None = None,
        tolerance_seconds: int = 5,
    ) -> None:
        """
        Initialize the data integrator.

        Parameters
        ----------
        chrome_path:
            Path to the Chrome history Parquet dataset.

        ram_path:
            Path to the RAM usage Parquet dataset.

        output_path:
            Path for the integrated Silver dataset.

        tolerance_seconds:
            Maximum allowed time difference between a Chrome event
            and a RAM observation.
        """

        paths = get_paths()["paths"]

        self.chrome_path = Path(
            chrome_path if chrome_path is not None else paths["chrome_history_raw"]
        ).expanduser()

        self.ram_path = Path(
            ram_path
            if ram_path is not None
            else paths.get(
                "ram_data_raw",
                "data/raw/ram_usage.parquet",
            )
        ).expanduser()

        self.output_path = Path(
            output_path
            if output_path is not None
            else paths.get(
                "browser_ram_aligned",
                "data/silver/browser_ram_aligned.parquet",
            )
        ).expanduser()

        if tolerance_seconds <= 0:
            raise ValueError("tolerance_seconds must be greater than zero.")

        self.tolerance = pd.Timedelta(seconds=tolerance_seconds)

    def integrate(self) -> pd.DataFrame:
        """
        Load, validate, align, and return the integrated dataset.

        Returns
        -------
        pandas.DataFrame
            Chrome browsing events enriched with RAM metrics.
        """

        logger.info("Starting browser and RAM data integration.")

        chrome_data = self._load_parquet(
            self.chrome_path,
            "Chrome history",
        )

        ram_data = self._load_parquet(
            self.ram_path,
            "RAM usage",
        )

        self._validate_schema(
            chrome_data,
            self.CHROME_REQUIRED_COLUMNS,
            "Chrome history",
        )

        self._validate_schema(
            ram_data,
            self.RAM_REQUIRED_COLUMNS,
            "RAM usage",
        )

        chrome_data = self._prepare_timestamps(
            chrome_data,
            "Chrome history",
        )

        ram_data = self._prepare_timestamps(
            ram_data,
            "RAM usage",
        )

        self._validate_time_overlap(
            chrome_data,
            ram_data,
        )

        integrated_data = self._align(
            chrome_data,
            ram_data,
        )

        logger.info(
            "Data integration completed | "
            "Chrome records=%d | RAM records=%d | "
            "Integrated records=%d",
            len(chrome_data),
            len(ram_data),
            len(integrated_data),
        )

        return integrated_data

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the integrated dataset as Parquet.
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

        except (OSError, ImportError) as exc:
            raise FileWriteError(f"Failed to save integrated dataset: {path}") from exc

        logger.info(
            "Integrated dataset saved to: %s | records=%d",
            path,
            len(dataframe),
        )

        return path

    @staticmethod
    def _load_parquet(
        path: Path,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Load a Parquet dataset."""

        if not path.is_file():
            raise FileReadError(f"{dataset_name} dataset not found: {path}")

        try:
            dataframe = pd.read_parquet(path)

        except (OSError, ImportError) as exc:
            raise FileReadError(f"Unable to read {dataset_name} dataset: {path}") from exc

        if dataframe.empty:
            raise DataValidationError(f"{dataset_name} dataset is empty: {path}")

        return dataframe

    @staticmethod
    def _validate_schema(
        dataframe: pd.DataFrame,
        required_columns: set[str],
        dataset_name: str,
    ) -> None:
        """Validate that all required columns are present."""

        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise DataValidationError(
                f"{dataset_name} dataset is missing required columns: {sorted(missing_columns)}"
            )

    @staticmethod
    def _prepare_timestamps(
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Normalize and validate timestamps."""

        dataframe = dataframe.copy()

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
            utc=True,
        )

        invalid_count = int(dataframe["timestamp"].isna().sum())

        if invalid_count:
            raise DataValidationError(
                f"{dataset_name} contains {invalid_count} invalid timestamps."
            )

        dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

        return dataframe

    @staticmethod
    def _validate_time_overlap(
        chrome_data: pd.DataFrame,
        ram_data: pd.DataFrame,
    ) -> None:
        """Ensure the two datasets have an overlapping time period."""

        chrome_start = chrome_data["timestamp"].min()
        chrome_end = chrome_data["timestamp"].max()

        ram_start = ram_data["timestamp"].min()
        ram_end = ram_data["timestamp"].max()

        overlap_start = max(
            chrome_start,
            ram_start,
        )

        overlap_end = min(
            chrome_end,
            ram_end,
        )

        if overlap_start > overlap_end:
            raise DataValidationError(
                "Chrome history and RAM datasets have no overlapping time period."
            )

        logger.info(
            "Time overlap detected: %s → %s",
            overlap_start,
            overlap_end,
        )

    def _align(
        self,
        chrome_data: pd.DataFrame,
        ram_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Match every Chrome event with the nearest RAM observation.

        Only matches within the configured tolerance are accepted.
        """

        ram_columns = [
            "timestamp",
            "total_mb",
            "used_mb",
            "available_mb",
            "usage_percent",
        ]

        ram_data = ram_data[ram_columns].copy()

        integrated_data = pd.merge_asof(
            chrome_data,
            ram_data,
            on="timestamp",
            direction="nearest",
            tolerance=self.tolerance,
        )

        matched_count = int(integrated_data["usage_percent"].notna().sum())

        unmatched_count = len(integrated_data) - matched_count

        match_rate = matched_count / len(integrated_data) if len(integrated_data) else 0.0

        logger.info(
            "RAM alignment completed | matched=%d | unmatched=%d | match_rate=%.2f%%",
            matched_count,
            unmatched_count,
            match_rate * 100,
        )

        if match_rate == 0:
            raise DataValidationError(
                "No Chrome events could be matched with RAM "
                "observations within the configured tolerance."
            )

        return integrated_data
