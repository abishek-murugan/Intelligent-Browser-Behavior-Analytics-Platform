from pathlib import Path

import pandas as pd
import pytest

from src.exceptions import DataCollectionError, DataValidationError
from src.ingestion.ram_data_loader import RAMDataLoader


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _raw_string_frame(ram_frame) -> pd.DataFrame:
    frame = ram_frame()

    frame["timestamp"] = frame["timestamp"].astype(str)
    frame["total_mb"] = frame["total_mb"].astype(str)
    frame["used_mb"] = frame["used_mb"].astype(str)
    frame["available_mb"] = frame["available_mb"].astype(str)
    frame["usage_percent"] = frame["usage_percent"].astype(str)

    return frame


@pytest.fixture
def ram_csv(tmp_path, ram_frame) -> Path:
    path = tmp_path / "ram_usage.csv"

    _write_csv(path, ram_frame())

    return path


def test_load_returns_prepared_frame(ram_csv):
    dataframe = RAMDataLoader(ram_csv).load()

    assert len(dataframe) == 6
    assert set(dataframe.columns) == {
        "timestamp",
        "total_mb",
        "used_mb",
        "available_mb",
        "usage_percent",
    }
    assert dataframe["timestamp"].dt.tz is not None
    assert dataframe["timestamp"].dt.tz == pd.Timestamp.now(tz="UTC").tz
    assert dataframe["timestamp"].is_monotonic_increasing
    assert dataframe["used_mb"].dtype.kind == "f"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(DataCollectionError, match="not found"):
        RAMDataLoader(tmp_path / "missing.csv").load()


def test_load_missing_column_raises(tmp_path, ram_frame):
    path = tmp_path / "ram.csv"

    _write_csv(path, ram_frame().drop(columns=["usage_percent"]))

    with pytest.raises(DataValidationError, match="usage_percent"):
        RAMDataLoader(path).load()


def test_load_empty_dataset_raises(tmp_path):
    path = tmp_path / "ram.csv"

    pd.DataFrame(
        columns=[
            "timestamp",
            "total_mb",
            "used_mb",
            "available_mb",
            "usage_percent",
        ]
    ).to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="empty"):
        RAMDataLoader(path).load()


def test_load_invalid_timestamp_raises(tmp_path, ram_frame):
    path = tmp_path / "ram.csv"

    dataframe = _raw_string_frame(ram_frame)
    dataframe.loc[0, "timestamp"] = "not-a-date"

    _write_csv(path, dataframe)

    with pytest.raises(DataValidationError, match="timestamps"):
        RAMDataLoader(path).load()


def test_load_invalid_numeric_raises(tmp_path, ram_frame):
    path = tmp_path / "ram.csv"

    dataframe = _raw_string_frame(ram_frame)
    dataframe.loc[0, "used_mb"] = "abc"

    _write_csv(path, dataframe)

    with pytest.raises(DataValidationError, match="numeric"):
        RAMDataLoader(path).load()


def test_load_sorts_by_timestamp(tmp_path):
    path = tmp_path / "ram.csv"

    dataframe = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-10 09:10:00+00:00",
                "2026-05-10 09:00:00+00:00",
            ],
            "total_mb": [8192, 8192],
            "used_mb": [3000, 2800],
            "available_mb": [5192, 5392],
            "usage_percent": [36.6, 34.2],
        }
    )

    _write_csv(path, dataframe)

    result = RAMDataLoader(path).load()

    assert result.loc[0, "timestamp"] == pd.Timestamp(
        "2026-05-10 09:00:00",
        tz="UTC",
    )


def test_save_writes_parquet(ram_csv, tmp_path):
    loader = RAMDataLoader(ram_csv)

    dataframe = loader.load()

    output = tmp_path / "nested" / "ram.parquet"

    saved = loader.save(dataframe, output)

    assert saved == output

    loaded = pd.read_parquet(output)

    pd.testing.assert_frame_equal(loaded, dataframe)
