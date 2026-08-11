import math

import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.feature_engineering.behavior_features import BehaviorFeatureBuilder


@pytest.fixture
def builder() -> BehaviorFeatureBuilder:
    return BehaviorFeatureBuilder()


def test_build_returns_one_row_per_session(builder, sessionized_frame):
    result = builder.build(sessionized_frame())

    assert len(result) == 2
    assert result["session_id"].tolist() == [1, 2]


def test_build_repetition_features(builder, sessionized_frame):
    result = builder.build(sessionized_frame()).set_index("session_id")

    session_one = result.loc[1]

    assert session_one["repeat_visit_ratio"] == pytest.approx(0.5)
    assert session_one["max_domain_repeats"] == 2
    assert session_one["top_domain_share"] == pytest.approx(0.5)
    assert session_one["top_category_share"] == pytest.approx(0.5)


def test_build_intensity_features(builder, sessionized_frame):
    result = builder.build(sessionized_frame()).set_index("session_id")

    session_one = result.loc[1]

    assert session_one["avg_gap_seconds"] == pytest.approx(300.0)
    assert session_one["median_gap_seconds"] == pytest.approx(300.0)
    assert session_one["max_gap_seconds"] == pytest.approx(300.0)

    assert session_one["events_per_minute"] == pytest.approx(4 / 15)


def test_build_diversity_features(builder, sessionized_frame):
    result = builder.build(sessionized_frame()).set_index("session_id")

    session_one = result.loc[1]

    assert session_one["category_switch_count"] == 1
    assert session_one["category_entropy"] == pytest.approx(1.0)
    assert session_one["domain_entropy"] == pytest.approx(1.0)
    assert session_one["domain_diversity_index"] == pytest.approx(0.5)


def test_build_single_event_session_fills_gaps(builder, sessionized_frame):
    single = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-10 09:00:00", tz="UTC")],
            "url": ["https://example.com/"],
            "title": ["Example"],
            "domain": ["example.com"],
            "visit_count": [1],
        }
    )

    frame = sessionized_frame(single)

    result = builder.build(frame).iloc[0]

    assert result["avg_gap_seconds"] == 0.0
    assert result["median_gap_seconds"] == 0.0
    assert result["max_gap_seconds"] == 0.0
    assert result["events_per_minute"] == 0.0


def test_build_missing_column_raises(builder, sessionized_frame):
    frame = sessionized_frame().drop(columns=["session_event_index"])

    with pytest.raises(DataValidationError, match="session_event_index"):
        builder.build(frame)


def test_entropy():
    entropy = BehaviorFeatureBuilder._entropy

    assert entropy(pd.Series(["a", "a"])) == pytest.approx(0.0)
    assert entropy(pd.Series(["a", "b"])) == pytest.approx(1.0)
    assert entropy(pd.Series(["a", "b", "c"])) == pytest.approx(math.log2(3))
