"""
Generate deterministic synthetic raw browsing data for the platform.

Produces the three raw inputs the rest of the pipeline consumes:

- ``data/raw/chrome_history.parquet``  (Chrome-style history events)
- ``data/raw/ram_usage.parquet``       (system RAM observations)
- ``data/raw/domain_category_map.csv`` (domain -> category mapping)

Useful for CI notebook execution, demos, and testing the pipeline end to
end without exposing a real browser profile.

Usage:
    uv run python scripts/make_synthetic_data.py
"""

from __future__ import annotations

import datetime as dt
import random

import numpy as np
import pandas as pd

from src.constants import PROJECT_ROOT
from src.utils.config_loader import get_paths

DOMAIN_MAP = {
    "www.google.com": "Search/Reference",
    "www.bing.com": "Search/Reference",
    "chat.openai.com": "Search/Reference",
    "github.com": "Development/Programming",
    "stackoverflow.com": "Development/Programming",
    "docs.python.org": "Development/Programming",
    "developer.mozilla.org": "Development/Programming",
    "www.youtube.com": "Entertainment",
    "www.netflix.com": "Entertainment",
    "open.spotify.com": "Entertainment",
    "www.reddit.com": "Social Media",
    "www.instagram.com": "Social Media",
    "www.linkedin.com": "Social Media",
    "x.com": "Social Media",
    "www.bbc.com": "News",
    "edition.cnn.com": "News",
    "www.amazon.com": "Shopping",
    "www.flipkart.com": "Shopping",
    "mail.google.com": "Email",
    "outlook.live.com": "Email",
    "docs.google.com": "Productivity",
    "calendar.google.com": "Productivity",
    "www.notion.so": "Productivity",
    "www.khanacademy.org": "Education",
    "en.wikipedia.org": "Education",
    "www.weather.com": "Uncategorized",
}

DOMAINS = list(DOMAIN_MAP)

CATEGORY_TITLES = {
    "Search/Reference": ["Search", "Query", "Reference"],
    "Development/Programming": ["Docs", "Code", "Issue", "Pull request", "Snippet"],
    "Entertainment": ["Watch", "Stream", "Listen", "Clip"],
    "Social Media": ["Home", "Explore", "Profile", "Timeline"],
    "News": ["Article", "Breaking", "Analysis", "Live"],
    "Shopping": ["Product", "Cart", "Deal", "Listing"],
    "Email": ["Inbox", "Compose", "Mail"],
    "Productivity": ["Document", "Sheet", "Meeting", "Note"],
    "Education": ["Lesson", "Course", "Article"],
    "Uncategorized": ["Homepage", "Site"],
}


def build_history(seed: int = 42, n_sessions: int = 220) -> pd.DataFrame:
    """Generate a synthetic Chrome-style history frame."""
    rng = random.Random(seed)

    now = dt.datetime(2026, 7, 25, tzinfo=dt.UTC)
    cursor = now - dt.timedelta(days=45)
    rows: list[dict[str, object]] = []

    for _ in range(n_sessions):
        domain = rng.choice(DOMAINS)
        category = DOMAIN_MAP[domain]
        n_events = rng.randint(3, 12)
        session_start = cursor + dt.timedelta(
            minutes=rng.randint(30, 720), seconds=rng.randint(0, 59)
        )

        for index in range(n_events):
            timestamp = session_start + dt.timedelta(minutes=index * rng.randint(1, 6))
            timestamp = timestamp.replace(second=0, microsecond=0)
            title = rng.choice(CATEGORY_TITLES[category])
            subpath = "search?q=" if domain in {"www.google.com", "www.bing.com"} else ""
            query = rng.choice(["pytorch", "mlflow", "optuna", "lstm", "python", "databricks"])
            url = f"https://{domain}/{subpath}{query}/item/{rng.randint(1, 9999)}"
            rows.append(
                {
                    "timestamp": timestamp,
                    "url": url,
                    "title": f"{title} {rng.randint(1, 999)}",
                    "domain": domain,
                    "visit_count": rng.randint(1, 20),
                }
            )

        cursor = session_start + dt.timedelta(minutes=15)

    frame = pd.DataFrame(rows)
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_ram(history: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate RAM usage observations covering the history time span."""
    rng = np.random.default_rng(seed)

    start = history["timestamp"].min().floor("min")
    end = history["timestamp"].max().ceil("min")
    timestamps = pd.date_range(start, end, freq="1min", tz=start.tz)

    base = 6200.0 + 1500.0 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps)))
    noise = rng.normal(0, 180, len(timestamps))
    used = np.clip(base + noise, 1200, 7800)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "total_mb": np.full(len(timestamps), 8192.0),
            "used_mb": used,
            "available_mb": 8192.0 - used,
            "usage_percent": used / 8192.0 * 100.0,
        }
    )


def build_domain_map() -> pd.DataFrame:
    """Return the domain -> category mapping frame."""
    return pd.DataFrame(
        [{"domain": domain, "category": category} for domain, category in DOMAIN_MAP.items()]
    )


def main() -> None:
    """Write all raw inputs to their configured locations."""
    paths = get_paths()["paths"]

    history = build_history()
    ram = build_ram(history)

    for relative, frame in (
        (paths["chrome_history_raw"], history),
        (paths["ram_data_raw"], ram),
    ):
        output = PROJECT_ROOT / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        print(f"wrote {output} ({len(frame):,} rows)")

    map_path = PROJECT_ROOT / paths["domain_category_map"]
    map_path.parent.mkdir(parents=True, exist_ok=True)
    build_domain_map().to_csv(map_path, index=False)
    print(f"wrote {map_path} ({len(DOMAIN_MAP)} mappings)")


if __name__ == "__main__":
    main()
