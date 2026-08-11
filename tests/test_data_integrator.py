from pathlib import Path

import pandas as pd
import pytest

from src.exceptions import DataValidationError, FileReadError
from src.preprocessing.data_integrator import BrowserRAMIntegrator


def _write_parquet(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)


@pytest.fixture
def integrator(tmp_path, chrome_frame, ram_frame) -> tuple[BrowserRAMIntegrator, pd.DataFrame]:
    chrome_path = tmp_path / "chrome.parquet"
    ram_path = tmp_path / "ram.parquet"

    _write_parquet(chrome_path, chrome_frame())
    _write_parquet(ram_path, ram_frame())

    return (
        BrowserRAMIntegrator(
            chrome_path=chrome_path,
            ram_path=ram_path,
            output_path=tmp_path / "integrated.parquet",
        ),
        chrome_frame(),
    )


def test_integrate_aligns_all_events(integrator):
    instance, chrome = integrator

    result = instance.integrate()

    assert len(result) == len(chrome)

    assert set(result.columns) == {
        "timestamp",
        "url",
        "title",
        "domain",
        "visit_count",
        "total_mb",
        "used_mb",
        "available_mb",
        "usage_percent",
    }
    assert result["usage_percent"].notna().all()


def test_integrate_matches_expected_ram_values(integrator, ram_frame):
    instance, _ = integrator

    result = instance.integrate()

    first_usage = ram_frame().loc[0, "usage_percent"]

    assert result.loc[0, "usage_percent"] == first_usage


def test_integrate_missing_chrome_raises(tmp_path, ram_frame):
    ram_path = tmp_path / "ram.parquet"

    _write_parquet(ram_path, ram_frame())

    with pytest.raises(FileReadError, match="not found"):
        BrowserRAMIntegrator(
            chrome_path=tmp_path / "missing.parquet",
            ram_path=ram_path,
        ).integrate()


def test_integrate_empty_dataset_raises(tmp_path, chrome_frame, ram_frame):
    chrome_path = tmp_path / "chrome.parquet"
    ram_path = tmp_path / "ram.parquet"

    _write_parquet(chrome_path, chrome_frame().iloc[0:0])
    _write_parquet(ram_path, ram_frame())

    with pytest.raises(DataValidationError, match="empty"):
        BrowserRAMIntegrator(
            chrome_path=chrome_path,
            ram_path=ram_path,
        ).integrate()


def test_integrate_missing_column_raises(tmp_path, chrome_frame, ram_frame):
    chrome_path = tmp_path / "chrome.parquet"
    ram_path = tmp_path / "ram.parquet"

    _write_parquet(chrome_path, chrome_frame().drop(columns=["title"]))
    _write_parquet(ram_path, ram_frame())

    with pytest.raises(DataValidationError, match="title"):
        BrowserRAMIntegrator(
            chrome_path=chrome_path,
            ram_path=ram_path,
        ).integrate()


def test_integrate_no_time_overlap_raises(tmp_path):
    chrome_path = tmp_path / "chrome.parquet"
    ram_path = tmp_path / "ram.parquet"

    chrome = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-10 09:00:00", tz="UTC")],
            "url": ["https://example.com/"],
            "title": ["Example"],
            "domain": ["example.com"],
            "visit_count": [1],
        }
    )
    ram = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-10 20:00:00", tz="UTC")],
            "total_mb": [8192],
            "used_mb": [3000],
            "available_mb": [5192],
            "usage_percent": [36.6],
        }
    )

    _write_parquet(chrome_path, chrome)
    _write_parquet(ram_path, ram)

    with pytest.raises(DataValidationError, match="overlapping"):
        BrowserRAMIntegrator(
            chrome_path=chrome_path,
            ram_path=ram_path,
        ).integrate()


def test_integrate_zero_match_rate_raises(tmp_path):
    chrome_path = tmp_path / "chrome.parquet"
    ram_path = tmp_path / "ram.parquet"

    chrome = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-05-10 09:00:00", tz="UTC"),
                pd.Timestamp("2026-05-10 09:10:00", tz="UTC"),
            ],
            "url": ["https://a.example.com/", "https://b.example.com/"],
            "title": ["A", "B"],
            "domain": ["a.example.com", "b.example.com"],
            "visit_count": [1, 1],
        }
    )
    ram = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-05-10 09:05:00", tz="UTC"),
                pd.Timestamp("2026-05-10 09:15:00", tz="UTC"),
            ],
            "total_mb": [8192, 8192],
            "used_mb": [3000, 3100],
            "available_mb": [5192, 5092],
            "usage_percent": [36.6, 37.8],
        }
    )

    _write_parquet(chrome_path, chrome)
    _write_parquet(ram_path, ram)

    with pytest.raises(DataValidationError, match="matched"):
        BrowserRAMIntegrator(
            chrome_path=chrome_path,
            ram_path=ram_path,
        ).integrate()


def test_invalid_tolerance_raises():
    with pytest.raises(ValueError, match="tolerance"):
        BrowserRAMIntegrator(
            chrome_path="c.parquet",
            ram_path="r.parquet",
            tolerance_seconds=0,
        )


def test_save_writes_parquet(integrator):
    instance, _ = integrator

    result = instance.integrate()

    path = instance.save(result)

    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(path), result)
