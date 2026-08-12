"""
Shared test data builders.

Provides factory fixtures that construct synthetic browser / RAM
datasets matching the schemas produced by the ingestion,
preprocessing and feature engineering pipelines.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

CATEGORY_MAP = {
    "www.google.com": "Search/Reference",
    "github.com": "Development/Programming",
    "www.instagram.com": "Social Media",
    "stackoverflow.com": "Development/Programming",
}


def _base_events() -> pd.DataFrame:
    """Return six Chrome-style events spanning two browsing sessions."""

    timestamps = pd.to_datetime(
        [
            "2026-05-10 09:00:00",
            "2026-05-10 09:05:00",
            "2026-05-10 09:10:00",
            "2026-05-10 09:15:00",
            "2026-05-10 10:40:00",
            "2026-05-10 10:45:00",
        ],
        utc=True,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "url": [
                "https://www.google.com/search?q=browser",
                "https://www.google.com/search?q=analytics",
                "https://github.com/user/repo",
                "https://github.com/user/repo/pulls",
                "https://www.instagram.com/p/abc",
                "https://stackoverflow.com/questions/1",
            ],
            "title": [
                "Search",
                "Search",
                "GitHub",
                "GitHub Pulls",
                "Instagram",
                "Stack Overflow",
            ],
            "domain": [
                "www.google.com",
                "www.google.com",
                "github.com",
                "github.com",
                "www.instagram.com",
                "stackoverflow.com",
            ],
            "visit_count": [1, 2, 3, 4, 5, 6],
        }
    )


def _integrated(events: pd.DataFrame) -> pd.DataFrame:
    """Attach RAM metrics to Chrome-style events."""

    dataframe = events.copy()

    used = np.linspace(2800.0, 3400.0, len(dataframe))

    dataframe["total_mb"] = 8192
    dataframe["used_mb"] = used
    dataframe["available_mb"] = 8192 - used
    dataframe["usage_percent"] = used / 8192 * 100

    return dataframe


def _categorized(events: pd.DataFrame) -> pd.DataFrame:
    """Attach RAM metrics and a behavioral category to each event."""

    dataframe = _integrated(events)

    dataframe["category"] = (dataframe["domain"].map(CATEGORY_MAP).fillna("Uncategorized")).astype(
        "string"
    )

    return dataframe


def _sessionized(events: pd.DataFrame) -> pd.DataFrame:
    """Add session columns using a 15-minute inactivity threshold."""

    dataframe = _categorized(events).sort_values("timestamp").reset_index(drop=True)

    gap = dataframe["timestamp"].diff()

    new_session = gap.isna() | (gap > pd.Timedelta(minutes=15))

    dataframe["session_id"] = new_session.cumsum()
    dataframe["session_event_index"] = dataframe.groupby("session_id").cumcount()
    dataframe["session_start"] = dataframe.groupby("session_id")["timestamp"].transform("first")
    dataframe["session_end"] = dataframe.groupby("session_id")["timestamp"].transform("last")

    return dataframe


@pytest.fixture
def chrome_frame() -> Callable[[], pd.DataFrame]:
    """Factory returning a Chrome-style event frame."""

    return _base_events


@pytest.fixture
def ram_frame() -> Callable[[], pd.DataFrame]:
    """Factory returning a RAM usage frame."""

    def _build() -> pd.DataFrame:
        integrated = _integrated(_base_events())

        return integrated[
            [
                "timestamp",
                "total_mb",
                "used_mb",
                "available_mb",
                "usage_percent",
            ]
        ]

    return _build


@pytest.fixture
def integrated_frame() -> Callable[[pd.DataFrame | None], pd.DataFrame]:
    """Factory returning an integrated browser/RAM frame."""

    def _build(events: pd.DataFrame | None = None) -> pd.DataFrame:
        return _integrated(_base_events() if events is None else events)

    return _build


@pytest.fixture
def categorized_frame() -> Callable[[pd.DataFrame | None], pd.DataFrame]:
    """Factory returning an integrated frame with a category column."""

    def _build(events: pd.DataFrame | None = None) -> pd.DataFrame:
        return _categorized(_base_events() if events is None else events)

    return _build


@pytest.fixture
def sessionized_frame() -> Callable[[pd.DataFrame | None], pd.DataFrame]:
    """Factory returning a categorized frame with session columns."""

    def _build(events: pd.DataFrame | None = None) -> pd.DataFrame:
        return _sessionized(_base_events() if events is None else events)

    return _build
