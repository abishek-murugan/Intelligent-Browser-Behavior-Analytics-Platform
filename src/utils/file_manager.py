"""
Centralized file and directory management utilities.
"""

from __future__ import annotations
import yaml
import shutil
from pathlib import Path
import csv
from src.constants import ( BRONZE_DATA_DIR, 
                           GOLD_DATA_DIR,
                            LOG_DIR, 
                            RAW_DATA_DIR, 
                            SILVER_DATA_DIR, 
                            MODEL_DIR )
from src.exceptions import ( 
    FileCopyError,
    FileManagerError,)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """Utility class for file and directory operations."""

    @staticmethod
    def create_directory(path: str | Path) -> Path:
        """
        Create a directory if it does not exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info("Directory ensured: %s", path)

        return path

    @staticmethod
    def directory_exists(path: str | Path) -> bool:
        """
        Check whether a directory exists.
        """
        return Path(path).is_dir()

    @staticmethod
    def file_exists(path: str | Path) -> bool:
        """
        Check whether a file exists.
        """
        return Path(path).is_file()

    @staticmethod
    def create_parent_directory(path: str | Path) -> Path:
        """
        Create the parent directory for a file.
        """
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)

        logger.info("Parent directory ensured: %s", parent)

        return parent

    @staticmethod
    def copy_file(source: str | Path, destination: str | Path) -> Path:
        """
        Copy a file.
        """
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(source)

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(source, destination)

        except Exception as exc:
            raise FileCopyError(
                f"Unable to copy '{source}' to '{destination}'."
            ) from exc

        logger.info("Copied %s -> %s", source, destination)

        return destination 
    
    @staticmethod
    def move_file(source: str | Path, destination: str | Path) -> Path:
        """
        Move a file.
        """
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(source)

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(source), str(destination))

        except Exception as exc:
            raise FileManagerError(
                f"Unable to move '{source}' to '{destination}'."
            ) from exc

        logger.info("Moved %s -> %s", source, destination)

        return destination

    @staticmethod
    def delete_file(path: str | Path) -> None:
        """
        Delete a file.
        """
        path = Path(path)

        if not path.exists():
            return

        try:
            path.unlink()

        except Exception as exc:
            raise FileManagerError(
                f"Unable to delete '{path}'."
            ) from exc

            logger.info("Deleted file %s", path)

    @staticmethod
    def delete_directory(path: str | Path) -> None:
        """
        Delete a directory recursively.
        """
        path = Path(path)

        try:
            shutil.rmtree(path)

        except Exception as exc:
            raise FileManagerError(
                f"Unable to delete '{path}'."
            ) from exc

            logger.info("Deleted directory %s", path)

    @staticmethod
    def list_files(path: str | Path, pattern: str = "*") -> list[Path]:
        """
        List files matching a pattern.
        """
        path = Path(path)

        return sorted(path.glob(pattern))

    @staticmethod
    def file_size(path: str | Path) -> int:
        """
        Return file size in bytes.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        return path.stat().st_size

    @staticmethod
    def touch(path: str | Path) -> Path:
        """
        Create an empty file if it does not exist.
        """
        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

        logger.info("Created file %s", path)

        return path

    @staticmethod
    def ensure_project_directories() -> None:
        """
        Create all required project directories.
        """

        directories = [
            LOG_DIR,
            MODEL_DIR,
            RAW_DATA_DIR,
            BRONZE_DATA_DIR,
            SILVER_DATA_DIR,
            GOLD_DATA_DIR,
         ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Project directories initialized.")

    @staticmethod
    def read_yaml(path: str | Path) -> dict:
        """
        Read a YAML file.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @staticmethod
    def write_yaml(path: str | Path, data: dict) -> Path:
        """
        Write data to a YAML file.
        """

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                data,
                file,
                sort_keys=False,
                allow_unicode=True,
                )

            logger.info("YAML written: %s", path)

        return path

    @staticmethod
    def write_csv(
        path: str | Path,
        rows: list[dict],
        fieldnames: list[str],
        ) -> Path:
        """
        Write rows to a CSV file.
        """

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        logger.info("CSV written: %s", path)

        return path

    @staticmethod
    def read_csv(path: str | Path) -> list[dict]:
        """
        Read a CSV file.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

        return list(reader)