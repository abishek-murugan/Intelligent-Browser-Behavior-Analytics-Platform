"""Tests for the synthetic raw data generator."""

from __future__ import annotations

import pandas as pd

from scripts.make_synthetic_data import DOMAIN_MAP, build_domain_map, build_history, build_ram


def test_history_schema_and_sorting():
    history = build_history()

    assert set(history.columns) == {"timestamp", "url", "title", "domain", "visit_count"}
    assert len(history) > 0
    assert history["timestamp"].is_monotonic_increasing
    assert set(history["domain"]).issubset(DOMAIN_MAP)


def test_history_is_deterministic():
    first = build_history(seed=7)
    second = build_history(seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_ram_schema_covers_history():
    history = build_history()
    ram = build_ram(history)

    assert set(ram.columns) == {"timestamp", "total_mb", "used_mb", "available_mb", "usage_percent"}
    assert ram["timestamp"].min() <= history["timestamp"].min()
    assert ram["timestamp"].max() >= history["timestamp"].max()
    assert (ram["total_mb"] == 8192.0).all()
    assert ram["used_mb"].between(0, ram["total_mb"]).all()


def test_domain_map_schema():
    mapping = build_domain_map()

    assert set(mapping.columns) == {"domain", "category"}
    assert mapping["domain"].is_unique
    assert "Uncategorized" in set(mapping["category"])
