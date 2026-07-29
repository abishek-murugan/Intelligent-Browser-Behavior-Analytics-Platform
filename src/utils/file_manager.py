"""
Centralized file and directory management utilities.
"""

from __future__ import annotations

import shutil
from pathlib import Path

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

        shutil.copy2(source, destination)

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

        shutil.move(str(source), str(destination))

        logger.info("Moved %s -> %s", source, destination)

        return destination

    @staticmethod
    def delete_file(path: str | Path) -> None:
        """
        Delete a file.
        """
        path = Path(path)

        if path.exists():
            path.unlink()

            logger.info("Deleted file %s", path)

    @staticmethod
    def delete_directory(path: str | Path) -> None:
        """
        Delete a directory recursively.
        """
        path = Path(path)

        if path.exists():
            shutil.rmtree(path)

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
