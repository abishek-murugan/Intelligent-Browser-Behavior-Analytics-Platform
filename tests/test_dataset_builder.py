import pandas as pd
import pytest

from src.deep_learning.dataset_builder import DatasetBuilder
from src.exceptions import (
    DataValidationError,
    FileReadError,
    SequenceGenerationError,
)


def _build_synthetic_features(n_sessions: int = 15) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=n_sessions,
        freq="2h",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "session_id": list(range(1, n_sessions + 1)),
            "session_start": timestamps,
            "session_end": timestamps + pd.Timedelta("30 minutes"),
            "hour": timestamps.hour,
            "is_weekend": (timestamps.dayofweek >= 5),
            "time_of_day": ["Morning"] * n_sessions,
            "dominant_category": ["Search/Reference"] * n_sessions,
            "session_duration_seconds": [1800] * n_sessions,
            "event_count": [5] * n_sessions,
            "unique_domains": [3] * n_sessions,
            "avg_used_mb": [1024.5] * n_sessions,
            "max_used_mb": [2048.25] * n_sessions,
            "category_entropy": [0.5] * n_sessions,
        }
    )


def _make_builder(tmp_path, input_path=None, **kwargs) -> DatasetBuilder:
    return DatasetBuilder(
        input_path=input_path,
        session_features_output_path=tmp_path / "session_features.parquet",
        behavior_sequences_output_path=tmp_path / "behavior_sequences.parquet",
        **kwargs,
    )


def test_build_gold_datasets(tmp_path):
    input_path = tmp_path / "input.parquet"

    _build_synthetic_features(15).to_parquet(input_path, index=False)

    builder = _make_builder(tmp_path, input_path=input_path, sequence_length=3)

    result = builder.run()

    session_features = result["session_features"]
    behavior_sequences = result["behavior_sequences"]

    assert session_features.shape[0] == 15
    assert session_features["session_id"].is_unique
    assert session_features["session_start"].is_monotonic_increasing

    assert behavior_sequences.shape[0] == 15 - 3
    assert set(behavior_sequences.columns) == {
        "sequence_id",
        "session_ids",
        "start_time",
        "end_time",
        "feature_vectors",
        "target_session_id",
        "target_category",
        "target_features",
    }

    first = behavior_sequences.iloc[0]

    assert first["session_ids"] == [1, 2, 3]
    assert first["target_session_id"] == 4
    assert first["target_category"] == "Search/Reference"

    vector = first["feature_vectors"]
    features = len(builder.feature_columns)

    assert len(vector) == 3
    assert all(len(row) == features for row in vector)
    assert len(first["target_features"]) == features


def test_feature_columns_exclude_categorical(tmp_path):
    input_path = tmp_path / "input.parquet"

    _build_synthetic_features(15).to_parquet(input_path, index=False)

    builder = _make_builder(tmp_path, input_path=input_path, sequence_length=3)

    builder.run()

    feature_columns = builder.feature_columns

    assert "time_of_day" not in feature_columns
    assert "session_id" not in feature_columns
    assert "session_start" not in feature_columns
    assert "session_end" not in feature_columns
    assert "is_weekend" in feature_columns
    assert "time_of_day_Morning" in feature_columns
    assert "time_of_day_Night" in feature_columns


def test_insufficient_sessions_raises(tmp_path):
    input_path = tmp_path / "input.parquet"

    _build_synthetic_features(2).to_parquet(input_path, index=False)

    builder = _make_builder(tmp_path, input_path=input_path, sequence_length=3)

    with pytest.raises(SequenceGenerationError):
        builder.build_behavior_sequences(
            builder.build_session_features(pd.read_parquet(input_path))
        )


def test_duplicate_session_id_raises(tmp_path):
    input_path = tmp_path / "input.parquet"

    dataframe = _build_synthetic_features(3)

    dataframe.loc[1, "session_id"] = 1

    dataframe.to_parquet(input_path, index=False)

    builder = _make_builder(tmp_path, input_path=input_path, sequence_length=2)

    with pytest.raises(DataValidationError):
        builder.run()


def test_missing_required_column_raises(tmp_path):
    input_path = tmp_path / "input.parquet"

    dataframe = _build_synthetic_features(3).drop(columns=["session_end"])

    dataframe.to_parquet(input_path, index=False)

    builder = _make_builder(tmp_path, input_path=input_path, sequence_length=2)

    with pytest.raises(DataValidationError):
        builder.run()


def test_missing_input_file_raises(tmp_path):
    builder = _make_builder(tmp_path, input_path=tmp_path / "missing.parquet")

    with pytest.raises(FileReadError):
        builder.run()


def test_sequence_length_must_be_positive(tmp_path):
    with pytest.raises(DataValidationError):
        DatasetBuilder(sequence_length=0)
