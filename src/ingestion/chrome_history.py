"""
Chrome browser history data collector.

Extracts browsing history from the local Google Chrome SQLite database
and converts it into a pandas DataFrame suitable for downstream
preprocessing and feature engineering.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.exceptions import (
    ChromeHistoryError,
    HistoryDatabaseLockedError,
)
from src.utils.config_loader import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChromeHistoryCollector:
    """Collect browsing history from Google Chrome."""

    QUERY = """
        SELECT
            urls.url,
            urls.title,
            urls.visit_count,
            visits.visit_time
        FROM visits
        INNER JOIN urls
            ON visits.url = urls.id
        WHERE urls.url IS NOT NULL
        ORDER BY visits.visit_time ASC
    """

    CHROME_EPOCH_OFFSET = 11644473600

    def __init__(
        self,
        database_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the Chrome history collector.

        Parameters
        ----------
        database_path:
            Optional path to the Chrome History database.
            If omitted, the path from paths.yaml is used.
        """

        if database_path is None:
            paths = get_paths()
            database_path = paths["paths"]["chrome_history_database"]

        self.database_path = Path(database_path).expanduser()

    def collect(self) -> pd.DataFrame:
        """
        Extract browsing history from Chrome.
        """
        logger.info(
            "Starting Chrome history collection from: %s",
            self.database_path,
        )

        self._validate_database()

        temporary_database = self._create_database_copy()

        try:
            dataframe = self._read_database(temporary_database)
        finally:
            temporary_database.unlink(missing_ok=True)

        dataframe = self._transform(dataframe)

        logger.info(
            "Chrome history collection completed. Records collected: %d",
            len(dataframe),
        )

        return dataframe

    def _validate_database(self) -> None:
        """Validate that the Chrome History database exists."""

        if not self.database_path.is_file():
            raise ChromeHistoryError(f"Chrome History database not found: {self.database_path}")

    def _create_database_copy(self) -> Path:
        """
        Create a temporary copy of the Chrome History database.

        Chrome may keep the original SQLite database locked while
        the browser is running. Reading a copied database avoids
        interfering with Chrome.
        """

        try:
            temporary_file = tempfile.NamedTemporaryFile(
                suffix=".db",
                prefix="chrome_history_",
                delete=False,
            )

            temporary_path = Path(temporary_file.name)
            temporary_file.close()

            shutil.copy2(
                self.database_path,
                temporary_path,
            )

        except OSError as exc:
            raise HistoryDatabaseLockedError(
                "Unable to create a readable copy of the "
                "Chrome History database. "
                "Make sure Chrome has access to its History file."
            ) from exc

        logger.debug(
            "Created temporary Chrome History database: %s",
            temporary_path,
        )

        return temporary_path

    def _read_database(
        self,
        database_path: Path,
    ) -> pd.DataFrame:
        """Read browsing history from the SQLite database."""

        connection: sqlite3.Connection | None = None

        try:
            connection = sqlite3.connect(
                f"file:{database_path}?mode=ro",
                uri=True,
            )

            return pd.read_sql_query(
                self.QUERY,
                connection,
            )

        except sqlite3.Error as exc:
            raise ChromeHistoryError("Failed to read Chrome History SQLite database.") from exc

        finally:
            if connection is not None:
                connection.close()

    def _transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform raw Chrome history into a structured DataFrame.
        """

        if dataframe.empty:
            logger.warning("Chrome History database contains no visits.")

            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "url",
                    "title",
                    "domain",
                    "visit_count",
                ]
            )

        dataframe["timestamp"] = self._convert_chrome_timestamp(dataframe["visit_time"])

        dataframe["domain"] = dataframe["url"].map(self._extract_domain)

        dataframe = dataframe[
            [
                "timestamp",
                "url",
                "title",
                "domain",
                "visit_count",
            ]
        ]

        return dataframe

    def _convert_chrome_timestamp(
        self,
        timestamps: pd.Series,
    ) -> pd.Series:
        """
        Convert Chrome WebKit timestamps to UTC datetime.

        Chrome stores timestamps as microseconds since
        1601-01-01 00:00:00 UTC.
        """

        chrome_epoch = pd.Timestamp(
            "1601-01-01",
            tz="UTC",
        )

        return chrome_epoch + pd.to_timedelta(
            timestamps,
            unit="us",
        )

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save collected Chrome history to a Parquet file.
        """

        if output_path is None:
            paths = get_paths()
            output_path = paths["paths"]["chrome_history_raw"]

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

        except Exception as exc:
            raise ChromeHistoryError(f"Failed to save Chrome history to: {output_path}") from exc

        logger.info(
            "Chrome history saved to: %s | Records: %d",
            output_path,
            len(dataframe),
        )

        return output_path

    @staticmethod
    def _extract_domain(url: str) -> str | None:
        """Extract the network location/domain from a URL."""

        try:
            parsed = urlparse(url)

            return parsed.netloc or None

        except ValueError:
            return None
