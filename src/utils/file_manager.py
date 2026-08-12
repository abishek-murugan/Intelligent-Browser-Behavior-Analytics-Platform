"""
Centralized file and directory management utilities.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.constants import (
    BRONZE_DATA_DIR,
    GOLD_DATA_DIR,
    LOG_DIR,
    MODEL_DIR,
    RAW_DATA_DIR,
    SILVER_DATA_DIR,
)
from src.exceptions import (
    DirectoryCreationError,
    FileCopyError,
    FileDeletionError,
    FileMoveError,
    FileReadError,
    FileWriteError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """
    Utility class for common file and directory operations.
    """

    def create_directory(self, path: str | Path) -> Path:
        """Create a directory if it does not exist."""
        path = Path(path)

        try:
            path.mkdir(parents=True, exist_ok=True)

        except OSError as exc:
            raise DirectoryCreationError(f"Unable to create directory: {path}") from exc

        logger.info("Directory ensured: %s", path)

        return path

    def ensure_project_directories(self) -> None:
        """Create all required project directories."""

        directories = (
            LOG_DIR,
            MODEL_DIR,
            RAW_DATA_DIR,
            BRONZE_DATA_DIR,
            SILVER_DATA_DIR,
            GOLD_DATA_DIR,
        )

        for directory in directories:
            self.create_directory(directory)

        logger.info("Project directories initialized.")

    @staticmethod
    def file_exists(path: str | Path) -> bool:
        return Path(path).is_file()

    @staticmethod
    def directory_exists(path: str | Path) -> bool:
        return Path(path).is_dir()

    def copy_file(
        self,
        source: str | Path,
        destination: str | Path,
    ) -> Path:
        """Copy a file."""

        source = Path(source)
        destination = Path(destination)

        if not source.is_file():
            raise FileReadError(f"Source file not found: {source}")

        self.create_directory(destination.parent)

        try:
            shutil.copy2(source, destination)

        except OSError as exc:
            raise FileCopyError(f"Unable to copy '{source}' to '{destination}'.") from exc

        logger.info("Copied %s -> %s", source, destination)

        return destination

    def move_file(
        self,
        source: str | Path,
        destination: str | Path,
    ) -> Path:
        """Move a file."""

        source = Path(source)
        destination = Path(destination)

        if not source.is_file():
            raise FileReadError(f"Source file not found: {source}")

        self.create_directory(destination.parent)

        try:
            shutil.move(str(source), str(destination))

        except OSError as exc:
            raise FileMoveError(f"Unable to move '{source}' to '{destination}'.") from exc

        logger.info("Moved %s -> %s", source, destination)

        return destination

    @staticmethod
    def delete_file(path: str | Path) -> None:
        """Delete a file."""

        path = Path(path)

        if not path.exists():
            return

        try:
            path.unlink()

        except OSError as exc:
            raise FileDeletionError(f"Unable to delete file: {path}") from exc

        logger.info("Deleted file: %s", path)

    @staticmethod
    def delete_directory(path: str | Path) -> None:
        """Delete a directory recursively."""

        path = Path(path)

        if not path.exists():
            return

        try:
            shutil.rmtree(path)

        except OSError as exc:
            raise FileDeletionError(f"Unable to delete directory: {path}") from exc

        logger.info("Deleted directory: %s", path)

    @staticmethod
    def list_files(
        path: str | Path,
        pattern: str = "*",
    ) -> list[Path]:
        """List files matching a pattern."""

        return sorted(Path(path).glob(pattern))

    @staticmethod
    def get_file_size(path: str | Path) -> int:
        """Return file size in bytes."""

        path = Path(path)

        if not path.is_file():
            raise FileReadError(f"File not found: {path}")

        return path.stat().st_size

    def read_yaml(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """Read a YAML file."""

        path = Path(path)

        if not path.is_file():
            raise FileReadError(f"File not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}

        except yaml.YAMLError as exc:
            raise FileReadError(f"Invalid YAML file: {path}") from exc

    def write_yaml(
        self,
        path: str | Path,
        data: dict[str, Any],
    ) -> Path:
        """Write a YAML file."""

        path = Path(path)

        self.create_directory(path.parent)

        try:
            with path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    data,
                    file,
                    sort_keys=False,
                    allow_unicode=True,
                )

        except OSError as exc:
            raise FileWriteError(f"Unable to write YAML: {path}") from exc

        logger.info("YAML written: %s", path)

        return path

    def read_csv(
        self,
        path: str | Path,
    ) -> list[dict[str, str]]:
        """Read a CSV file."""

        path = Path(path)

        if not path.is_file():
            raise FileReadError(f"File not found: {path}")

        try:
            with path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                return list(csv.DictReader(file))

        except OSError as exc:
            raise FileReadError(f"Unable to read CSV: {path}") from exc

    def write_csv(
        self,
        path: str | Path,
        rows: list[dict[str, Any]],
        fieldnames: list[str],
    ) -> Path:
        """Write rows to a CSV file."""

        path = Path(path)

        self.create_directory(path.parent)

        try:
            with path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )

                writer.writeheader()
                writer.writerows(rows)

        except OSError as exc:
            raise FileWriteError(f"Unable to write CSV: {path}") from exc

        logger.info("CSV written: %s", path)

        return path


file_manager = FileManager()
