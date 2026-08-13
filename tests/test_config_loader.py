from pathlib import Path

import pytest

from src.constants import PROJECT_ROOT
from src.exceptions import ConfigurationFileNotFoundError
from src.utils.config_loader import ConfigLoader, get_paths


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


def test_paths_resolve_relative_to_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    paths = get_paths()["paths"]

    assert Path(paths["raw_data"]).is_absolute()
    assert Path(paths["raw_data"]) == PROJECT_ROOT / "data" / "raw"
    assert Path(paths["session_features_gold"]).is_absolute()
    assert Path(paths["chrome_history_database"]).is_absolute()
