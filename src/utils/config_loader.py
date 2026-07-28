from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigLoader:
    """Loads YAML configuration files."""

    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.config_dir = config_dir

    def load(self, filename: str) -> dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            filename: Name of the YAML file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError
            ValueError
        """

        path = self.config_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with open(path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"{filename} is empty.")

        return config


@lru_cache
def get_config_loader() -> ConfigLoader:
    return ConfigLoader()

