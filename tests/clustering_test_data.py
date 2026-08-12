"""
Synthetic clustering test data.

Builds session-level feature frames matching the gold session
features schema so preprocessor, segmenter and profiler behavior can
be verified without real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "hour",
    "day_of_week_num",
    "is_weekend",
    "session_hour_span",
    "crosses_midnight",
    "session_duration_seconds",
    "event_count",
    "unique_domains",
    "unique_categories",
    "total_visit_count",
    "min_used_mb",
    "avg_used_mb",
    "max_used_mb",
    "avg_available_mb",
    "avg_usage_percent",
    "max_usage_percent",
    "repeat_visit_ratio",
    "max_domain_repeats",
    "top_domain_share",
    "top_category_share",
    "events_per_minute",
    "avg_gap_seconds",
    "median_gap_seconds",
    "max_gap_seconds",
    "category_entropy",
    "domain_entropy",
    "domain_diversity_index",
    "category_switch_count",
]

EXCLUDED = {
    "session_id",
    "session_start",
    "session_end",
    "hour",
    "day_of_week_num",
    "time_of_day",
    "dominant_category",
}


def build_features(
    n_sessions: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic session-level feature frame with 3 clusters."""

    rng = np.random.default_rng(seed)

    centers = np.array(
        [
            [3, 30, 1200],
            [9, 8, 300],
            [14, 60, 700],
        ]
    )

    cluster_ids = rng.integers(0, 3, size=n_sessions)

    base = centers[cluster_ids] + rng.normal(
        scale=2.0,
        size=(n_sessions, 3),
    )

    timestamps = pd.date_range(
        "2026-01-01",
        periods=n_sessions,
        freq="1h",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "session_id": np.arange(n_sessions) + 1,
            "session_start": timestamps,
            "session_end": timestamps + pd.Timedelta(minutes=45),
            "hour": base[:, 0].round().astype(int),
            "day_of_week_num": rng.integers(0, 7, size=n_sessions),
            "is_weekend": rng.integers(0, 2, size=n_sessions).astype(bool),
            "time_of_day": rng.choice(
                ["Morning", "Afternoon", "Evening", "Night"],
                size=n_sessions,
            ),
            "dominant_category": rng.choice(
                ["Search/Reference", "Social Media", "Development/Programming"],
                size=n_sessions,
            ),
            "session_hour_span": base[:, 0],
            "crosses_midnight": rng.integers(0, 2, size=n_sessions).astype(bool),
            "session_duration_seconds": base[:, 1] * 60,
            "event_count": base[:, 2].round().astype(int),
            "unique_domains": rng.integers(1, 10, size=n_sessions),
            "unique_categories": rng.integers(1, 4, size=n_sessions),
            "total_visit_count": rng.integers(1, 100, size=n_sessions),
            "min_used_mb": rng.normal(2000, 200, size=n_sessions),
            "avg_used_mb": rng.normal(3200, 300, size=n_sessions),
            "max_used_mb": rng.normal(4500, 400, size=n_sessions),
            "avg_available_mb": rng.normal(5000, 300, size=n_sessions),
            "avg_usage_percent": rng.normal(55, 10, size=n_sessions),
            "max_usage_percent": rng.normal(75, 8, size=n_sessions),
            "repeat_visit_ratio": rng.uniform(0, 1, size=n_sessions),
            "max_domain_repeats": rng.integers(1, 20, size=n_sessions),
            "top_domain_share": rng.uniform(0.1, 1.0, size=n_sessions),
            "top_category_share": rng.uniform(0.1, 1.0, size=n_sessions),
            "events_per_minute": rng.uniform(0.5, 20, size=n_sessions),
            "avg_gap_seconds": rng.uniform(10, 300, size=n_sessions),
            "median_gap_seconds": rng.uniform(10, 300, size=n_sessions),
            "max_gap_seconds": rng.uniform(30, 900, size=n_sessions),
            "category_entropy": rng.uniform(0, 1, size=n_sessions),
            "domain_entropy": rng.uniform(0, 2, size=n_sessions),
            "domain_diversity_index": rng.uniform(0, 1, size=n_sessions),
            "category_switch_count": rng.integers(0, 15, size=n_sessions),
        }
    )
