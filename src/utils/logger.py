"""
Centralized logging configuration.

Configures the application's logging system from config/logging.yaml
and provides a helper function for retrieving loggers.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from src.constants import LOG_DIR, PROJECT_ROOT
from src.exceptions import (
    ConfigurationError,
)
from src.utils.config_loader import get_logging


def setup_logging() -> None:
    """
    Configure the application's logging system.

    This function should be called once during application startup.

    Raises
    ------
    ConfigurationError
        If the logging configuration is invalid.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        config = get_logging()

        _resolve_handler_paths(config)

        logging.config.dictConfig(config)

    except Exception as exc:
        raise ConfigurationError("Failed to initialize the logging system.") from exc


def _resolve_handler_paths(config: dict[str, object]) -> None:
    """
    Make relative log file paths in the logging configuration absolute.

    ``dictConfig`` resolves relative ``filename`` values against the
    current working directory, which breaks when code runs from a
    different directory (e.g. a notebook). Handler file paths are
    instead resolved against the project root so logs always land in
    ``<project root>/logs``.
    """

    handlers = config.get("handlers")

    if not isinstance(handlers, dict):
        return

    for handler in handlers.values():
        if not isinstance(handler, dict):
            continue

        filename = handler.get("filename")

        if isinstance(filename, str) and not Path(filename).is_absolute():
            handler["filename"] = str(PROJECT_ROOT / filename)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    return logging.getLogger(name)


# Configure logging immediately when this module is imported.
setup_logging()
