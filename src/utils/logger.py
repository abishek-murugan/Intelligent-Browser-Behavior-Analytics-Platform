"""
Centralized logging configuration.

This module configures the application's logging system from
config/logging.yaml and exposes a helper function to retrieve loggers.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from src.utils.config_loader import CONFIG_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging() -> None:
    """
    Configure the application's logging system.

    Raises
    ------
    FileNotFoundError
        If logging.yaml does not exist.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config_path = CONFIG_DIR / "logging.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Logging configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Usually __name__.

    Returns
    -------
    logging.Logger
    """

    return logging.getLogger(name)


setup_logging()