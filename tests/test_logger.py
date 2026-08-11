import logging
from pathlib import Path

from src.utils.logger import (
    get_logger,
    setup_logging,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_logs_directory_exists():
    """
    Logs directory should exist after setup.
    """

    setup_logging()

    assert (PROJECT_ROOT / "logs").exists()


def test_get_logger_returns_logger():
    """
    get_logger should return a Logger instance.
    """

    logger = get_logger(__name__)

    assert isinstance(logger, logging.Logger)


def test_logger_name():
    """
    Logger name should match.
    """

    logger = get_logger("test_logger")

    assert logger.name == "test_logger"


def test_logging_message(caplog):
    """
    Logger should write messages.
    """

    logger = get_logger("test")

    with caplog.at_level(logging.INFO):
        logger.info("Hello Logger")

    assert "Hello Logger" in caplog.text
