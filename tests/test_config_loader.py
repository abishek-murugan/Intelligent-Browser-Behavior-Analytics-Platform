from pathlib import Path

import pytest

from src.exceptions import ConfigurationFileNotFoundError
from src.utils.config_loader import ConfigLoader


def test_load_config():
    loader = ConfigLoader()

    config = loader.load("config.yaml")

    assert isinstance(config, dict)
    assert "project" in config


def test_missing_file():
    loader = ConfigLoader()

    with pytest.raises(ConfigurationFileNotFoundError):
        loader.load("missing.yaml")


def test_config_directory_exists():
    loader = ConfigLoader()

    assert Path(loader.config_dir).exists()
