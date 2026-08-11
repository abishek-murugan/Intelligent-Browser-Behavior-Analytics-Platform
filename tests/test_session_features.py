import pytest

from src.exceptions import DataValidationError
from src.feature_engineering.session_features import SessionFeatureBuilder


@pytest.fixture
def builder() -> SessionFeatureBuilder:
    return SessionFeatureBuilder()


def test_build_returns_one_row_per_session(builder, sessionized_frame):
    result = builder.build(sessionized_frame())

    assert len(result) == 2
    assert result["session_id"].tolist() == [1, 2]


def test_build_aggregate_values(builder, sessionized_frame):
    result = builder.build(sessionized_frame()).set_index("session_id")

    session_one = result.loc[1]

    assert session_one["event_count"] == 4
    assert session_one["unique_domains"] == 2
    assert session_one["unique_categories"] == 2
    assert session_one["total_visit_count"] == 10
    assert session_one["session_duration_seconds"] == 900
    assert session_one["min_used_mb"] == pytest.approx(2800.0)
    assert session_one["max_used_mb"] == pytest.approx(3160.0)
    assert session_one["max_usage_percent"] == pytest.approx(3160 / 8192 * 100)

    session_two = result.loc[2]

    assert session_two["event_count"] == 2
    assert session_two["total_visit_count"] == 11
    assert session_two["session_duration_seconds"] == 300


def test_build_dominant_category_is_valid(builder, sessionized_frame):
    result = builder.build(sessionized_frame())

    valid = {"Search/Reference", "Development/Programming", "Social Media"}

    assert set(result["dominant_category"]) <= valid


def test_build_missing_column_raises(builder, sessionized_frame):
    frame = sessionized_frame().drop(columns=["visit_count"])

    with pytest.raises(DataValidationError, match="visit_count"):
        builder.build(frame)


def test_build_output_columns(builder, sessionized_frame):
    result = builder.build(sessionized_frame())

    assert list(result.columns) == [
        "session_id",
        "session_duration_seconds",
        "event_count",
        "unique_domains",
        "unique_categories",
        "dominant_category",
        "total_visit_count",
        "min_used_mb",
        "avg_used_mb",
        "max_used_mb",
        "avg_available_mb",
        "avg_usage_percent",
        "max_usage_percent",
    ]
