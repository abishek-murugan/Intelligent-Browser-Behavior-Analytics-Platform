"""
Centralized logging configuration.

Configures the application's logging system from config/logging.yaml
and provides a helper function for retrieving loggers.
"""

from __future__ import annotations

import logging
import logging.config

from src.constants import CONFIG_DIR, LOG_DIR
from src.exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
)
from src.utils.config_loader import get_logging


def setup_logging() -> None:
    """
    Configure the application's logging system.

    Raises
    ------
    ConfigurationFileNotFoundError
        If logging.yaml cannot be found.

    ConfigurationError
        If the logging configuration is invalid.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config_path = CONFIG_DIR / "logging.yaml"

    if not config_path.exists():
        raise ConfigurationFileNotFoundError(
            f"Logging configuration not found: {config_path}"
        )

    try:
        config = get_logging()
        logging.config.dictConfig(config)

    except Exception as exc:
        raise ConfigurationError(
            "Failed to configure the logging system."
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
    """

    return logging.getLogger(name)