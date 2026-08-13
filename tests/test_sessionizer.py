import pandas as pd
import pytest

from src.exceptions import DataValidationError, FileReadError
from src.preprocessing.sessionizer import Sessionizer


@pytest.fixture
def sessionizer(tmp_path) -> Sessionizer:
    return Sessionizer(
        input_path=tmp_path / "input.parquet",
        output_path=tmp_path / "output.parquet",
        inactivity_threshold_minutes=15,
    )


def test_sessionize_two_sessions(sessionizer, categorized_frame):
    result = sessionizer.sessionize(categorized_frame())

    assert result["session_id"].tolist() == [1, 1, 1, 1, 2, 2]
    assert result["session_event_index"].tolist() == [0, 1, 2, 3, 0, 1]

    first = result[result["session_id"] == 1]

    assert first["session_start"].iloc[0] == pd.Timestamp(
        "2026-05-10 09:00:00",
        tz="UTC",
    )
    assert first["session_end"].iloc[0] == pd.Timestamp(
        "2026-05-10 09:15:00",
        tz="UTC",
    )


def test_sessionize_custom_threshold(sessionizer, categorized_frame):
    sessionizer.inactivity_threshold = pd.Timedelta("1 minute")

    result = sessionizer.sessionize(categorized_frame())

    assert result["session_id"].nunique() == 6


def test_sessionize_sorts_by_timestamp(sessionizer, categorized_frame):
    frame = categorized_frame().sample(frac=1, random_state=42)

    result = sessionizer.sessionize(frame)

    assert result["timestamp"].is_monotonic_increasing


def test_sessionize_missing_column_raises(sessionizer, categorized_frame):
    frame = categorized_frame().drop(columns=["category"])

    with pytest.raises(DataValidationError, match="category"):
        sessionizer.sessionize(frame)


def test_invalid_threshold_raises(tmp_path):
    with pytest.raises(ValueError, match="inactivity_threshold_minutes"):
        Sessionizer(
            input_path=tmp_path / "in.parquet",
            output_path=tmp_path / "out.parquet",
            inactivity_threshold_minutes=0,
        )


def test_run_end_to_end(tmp_path, categorized_frame):
    input_path = tmp_path / "categorized.parquet"
    output_path = tmp_path / "sessions.parquet"

    categorized_frame().to_parquet(input_path, index=False)

    instance = Sessionizer(
        input_path=input_path,
        output_path=output_path,
        inactivity_threshold_minutes=15,
    )

    result = instance.run()

    assert result["session_id"].nunique() == 2
    assert output_path.exists()


def test_run_missing_input_raises(tmp_path):
    instance = Sessionizer(
        input_path=tmp_path / "missing.parquet",
        output_path=tmp_path / "out.parquet",
        inactivity_threshold_minutes=15,
    )

    with pytest.raises(FileReadError, match="not found"):
        instance.run()
