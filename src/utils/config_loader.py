"""
Centralized configuration loader.

Loads YAML configuration files from the config directory.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from src.constants import CONFIG_DIR
from src.exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
)


class ConfigLoader:
    """Loads YAML configuration files."""

    def __init__(self, config_dir=CONFIG_DIR):
        self.config_dir = config_dir

    @lru_cache(maxsize=None)
    def load(self, filename: str) -> dict[str, Any]:
        """
        Load a YAML configuration file.

        Parameters
        ----------
        filename : str
            YAML file name.

        Returns
        -------
        dict[str, Any]
            Parsed configuration.

        Raises
        ------
        ConfigurationFileNotFoundError
            If the configuration file does not exist.

        ConfigurationError
            If the YAML is empty or invalid.
        """

        path = self.config_dir / filename

        if not path.exists():
            raise ConfigurationFileNotFoundError(
                f"Configuration file not found: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML in '{filename}'."
            ) from exc

        if config is None:
            raise ConfigurationError(
                f"Configuration file '{filename}' is empty."
            )

        return config


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    """Return a cached ConfigLoader instance."""
    return ConfigLoader()


loader = get_config_loader()


def get_config() -> dict[str, Any]:
    return loader.load("config.yaml")


def get_paths() -> dict[str, Any]:
    return loader.load("paths.yaml")


def get_model() -> dict[str, Any]:
    return loader.load("model.yaml")


def get_logging() -> dict[str, Any]:
    return loader.load("logging.yaml")


def get_dashboard() -> dict[str, Any]:
    return loader.load("dashboard.yaml")


def get_azure() -> dict[str, Any]:
    return loader.load("azure.yaml")