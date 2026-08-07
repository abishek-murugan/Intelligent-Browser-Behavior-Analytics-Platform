"""
Centralized configuration loader.

Loads YAML configuration files from the config directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.constants import CONFIG_DIR
from src.exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
)


class ConfigLoader:
    """
    Loads YAML configuration files from the project's config directory.
    """

    def __init__(self, config_dir: Path = CONFIG_DIR) -> None:
        self.config_dir = config_dir

    @lru_cache(maxsize=None)
    def load(self, filename: str) -> dict[str, Any]:
        """
        Load a YAML configuration file.

        Parameters
        ----------
        filename : str
            Name of the YAML configuration file.

        Returns
        -------
        dict[str, Any]
            Parsed configuration dictionary.

        Raises
        ------
        ConfigurationFileNotFoundError
            If the configuration file does not exist.

        ConfigurationError
            If the YAML file is invalid or empty.
        """

        config_path = self.config_dir / filename

        if not config_path.is_file():
            raise ConfigurationFileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML syntax in '{filename}'."
            ) from exc

        except OSError as exc:
            raise ConfigurationError(
                f"Unable to read configuration file '{filename}'."
            ) from exc

        if config is None:
            raise ConfigurationError(
                f"Configuration file '{filename}' is empty."
            )

        if not isinstance(config, dict):
            raise ConfigurationError(
                f"Configuration file '{filename}' must contain a YAML mapping."
            )

        return config


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    """
    Return a cached ConfigLoader instance.
    """
    return ConfigLoader()


_loader = get_config_loader()


def get_config() -> dict[str, Any]:
    """
    Return application configuration.
    """
    return _loader.load("config.yaml")


def get_paths() -> dict[str, Any]:
    """
    Return project paths configuration.
    """
    return _loader.load("paths.yaml")


def get_models() -> dict[str, Any]:
    """
    Return machine learning configuration.
    """
    return _loader.load("models.yaml")


def get_logging() -> dict[str, Any]:
    """
    Return logging configuration.
    """
    return _loader.load("logging.yaml")