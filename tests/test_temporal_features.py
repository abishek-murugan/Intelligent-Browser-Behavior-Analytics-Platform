import pytest

from src.exceptions import DataValidationError
from src.feature_engineering.temporal_features import TemporalFeatureBuilder


@pytest.fixture
def builder() -> TemporalFeatureBuilder:
    return TemporalFeatureBuilder()


EXPECTED_COLUMNS = [
    "session_id",
    "session_start",
    "session_end",
    "hour",
    "day_of_week_num",
    "is_weekend",
    "time_of_day",
    "session_hour_span",
    "crosses_midnight",
]


def test_build_returns_one_row_per_session(builder, sessionized_frame):
    result = builder.build(sessionized_frame())

    assert len(result) == 2
    assert result["session_id"].tolist() == [1, 2]


def test_build_temporal_values(builder, sessionized_frame):
    result = builder.build(sessionized_frame())

    assert result.loc[0, "hour"] == 9
    assert result.loc[0, "time_of_day"] == "Morning"

    assert result.loc[0, "session_hour_span"] == pytest.approx(0.25)
    assert result.loc[1, "session_hour_span"] == pytest.approx(5 / 60)

    assert not result["crosses_midnight"].any()

    assert result["is_weekend"].equals(result["day_of_week_num"].isin([5, 6]))


def test_build_output_columns(builder, sessionized_frame):
    result = builder.build(sessionized_frame())

    assert list(result.columns) == EXPECTED_COLUMNS


def test_build_missing_column_raises(builder, sessionized_frame):
    frame = sessionized_frame().drop(columns=["timestamp"])

    with pytest.raises(DataValidationError, match="timestamp"):
        builder.build(frame)


def test_time_of_day_buckets():
    bucket = TemporalFeatureBuilder._time_of_day

    assert bucket(3) == "Night"
    assert bucket(5) == "Morning"
    assert bucket(11) == "Morning"
    assert bucket(12) == "Afternoon"
    assert bucket(16) == "Afternoon"
    assert bucket(17) == "Evening"
    assert bucket(21) == "Evening"
    assert bucket(22) == "Night"
