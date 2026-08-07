"""
Centralized logging configuration.

Configures the application's logging system from config/logging.yaml
and provides a helper function for retrieving loggers.
"""

from __future__ import annotations

import logging
import logging.config

from src.constants import LOG_DIR
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
        logging.config.dictConfig(config)

    except Exception as exc:
        raise ConfigurationError(
            "Failed to initialize the logging system."
        ) from exc


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