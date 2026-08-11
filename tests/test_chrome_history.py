import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.exceptions import ChromeHistoryError
from src.ingestion.chrome_history import ChromeHistoryCollector

CHROME_EPOCH = pd.Timestamp("1601-01-01", tz="UTC")


def _chrome_visit_time(timestamp: pd.Timestamp) -> int:
    return int(
        (timestamp.tz_convert("UTC") - CHROME_EPOCH).total_seconds() * 1_000_000
    )


def _write_history_database(
    path: Path,
    rows: list[tuple[str, str, int, pd.Timestamp]],
) -> None:
    connection = sqlite3.connect(path)

    connection.execute(
        "CREATE TABLE urls ("
        " id INTEGER PRIMARY KEY,"
        " url TEXT NOT NULL,"
        " title TEXT,"
        " visit_count INTEGER NOT NULL"
        ")"
    )
    connection.execute(
        "CREATE TABLE visits ("
        " id INTEGER PRIMARY KEY,"
        " url INTEGER NOT NULL,"
        " visit_time INTEGER NOT NULL"
        ")"
    )

    for index, (url, title, visit_count, visit_time) in enumerate(rows, start=1):
        connection.execute(
            "INSERT INTO urls (id, url, title, visit_count) "
            "VALUES (?, ?, ?, ?)",
            (index, url, title, visit_count),
        )
        connection.execute(
            "INSERT INTO visits (id, url, visit_time) VALUES (?, ?, ?)",
            (index, index, _chrome_visit_time(visit_time)),
        )

    connection.commit()
    connection.close()


@pytest.fixture
def history_database(tmp_path) -> Path:
    path = tmp_path / "History"

    _write_history_database(
        path,
        [
            (
                "https://www.google.com/search?q=browser",
                "Search",
                3,
                pd.Timestamp("2026-05-10 09:00:00", tz="UTC"),
            ),
            (
                "https://github.com/user/repo",
                "GitHub",
                2,
                pd.Timestamp("2026-05-10 09:05:00", tz="UTC"),
            ),
        ],
    )

    return path


def test_collect_returns_transformed_frame(history_database):
    dataframe = ChromeHistoryCollector(history_database).collect()

    assert list(dataframe.columns) == [
        "timestamp",
        "url",
        "title",
        "domain",
        "visit_count",
    ]
    assert len(dataframe) == 2

    assert dataframe.loc[0, "timestamp"] == pd.Timestamp(
        "2026-05-10 09:00:00",
        tz="UTC",
    )
    assert dataframe.loc[0, "domain"] == "www.google.com"
    assert dataframe.loc[1, "domain"] == "github.com"
    assert dataframe.loc[0, "visit_count"] == 3


def test_collect_missing_database_raises(tmp_path):
    with pytest.raises(ChromeHistoryError, match="not found"):
        ChromeHistoryCollector(tmp_path / "missing" / "History").collect()


def test_collect_empty_database_returns_empty_frame(tmp_path):
    path = tmp_path / "History"

    _write_history_database(path, [])

    dataframe = ChromeHistoryCollector(path).collect()

    assert dataframe.empty
    assert list(dataframe.columns) == [
        "timestamp",
        "url",
        "title",
        "domain",
        "visit_count",
    ]


def test_collect_orders_by_visit_time(tmp_path):
    path = tmp_path / "History"

    _write_history_database(
        path,
        [
            (
                "https://b.example.com/",
                "B",
                1,
                pd.Timestamp("2026-05-10 10:00:00", tz="UTC"),
            ),
            (
                "https://a.example.com/",
                "A",
                1,
                pd.Timestamp("2026-05-10 09:00:00", tz="UTC"),
            ),
        ],
    )

    dataframe = ChromeHistoryCollector(path).collect()

    assert dataframe["domain"].tolist() == [
        "a.example.com",
        "b.example.com",
    ]


def test_save_writes_parquet(history_database, tmp_path):
    collector = ChromeHistoryCollector(history_database)

    dataframe = collector.collect()

    output = tmp_path / "nested" / "chrome_history.parquet"

    saved = collector.save(dataframe, output)

    assert saved == output

    loaded = pd.read_parquet(output)

    pd.testing.assert_frame_equal(loaded, dataframe)


def test_extract_domain():
    extract = ChromeHistoryCollector._extract_domain

    assert extract("https://www.google.com/search?q=x") == "www.google.com"
    assert extract("https://example.com/path") == "example.com"
    assert extract("http://localhost:8000/x") == "localhost:8000"
    assert extract("file:///tmp/example.html") is None
    assert extract("") is None


def test_extract_domain_invalid_url_returns_none():
    assert ChromeHistoryCollector._extract_domain("http://[::1") is None
