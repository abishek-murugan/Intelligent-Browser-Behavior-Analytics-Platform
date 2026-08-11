import pandas as pd
import pytest

from src.exceptions import (
    DataValidationError,
    FeatureEngineeringError,
    FileReadError,
)
from src.feature_engineering.feature_pipeline import FeaturePipeline


@pytest.fixture
def pipeline(tmp_path, sessionized_frame) -> FeaturePipeline:
    input_path = tmp_path / "sessions.parquet"
    output_path = tmp_path / "features.parquet"

    sessionized_frame().to_parquet(input_path, index=False)

    return FeaturePipeline(
        input_path=input_path,
        output_path=output_path,
    )


def test_run_end_to_end(pipeline, tmp_path):
    result = pipeline.run()

    assert len(result) == 2

    assert set(result.columns) == {
        "session_id",
        "session_start",
        "session_end",
        "hour",
        "day_of_week_num",
        "is_weekend",
        "time_of_day",
        "session_hour_span",
        "crosses_midnight",
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
    }

    assert (tmp_path / "features.parquet").exists()


def test_run_missing_input_raises(tmp_path):
    instance = FeaturePipeline(
        input_path=tmp_path / "missing.parquet",
        output_path=tmp_path / "out.parquet",
    )

    with pytest.raises(FileReadError, match="not found"):
        instance.run()


def test_validate_schema_missing_column_raises(pipeline, sessionized_frame):
    frame = sessionized_frame().drop(columns=["session_start"])

    with pytest.raises(DataValidationError, match="session_start"):
        pipeline._validate_schema(frame)


def test_merge_features_combines_frames():
    left = pd.DataFrame(
        {
            "session_id": [1, 2],
            "hour": [9, 10],
        }
    )
    right = pd.DataFrame(
        {
            "session_id": [1, 2],
            "event_count": [4, 2],
        }
    )

    result = FeaturePipeline._merge_features(left, right)

    assert list(result.columns) == ["session_id", "hour", "event_count"]
    assert len(result) == 2


def test_merge_features_missing_session_id_raises():
    frame = pd.DataFrame({"hour": [9]})

    with pytest.raises(FeatureEngineeringError, match="session_id"):
        FeaturePipeline._merge_features(frame)


def test_validate_result_rejects_duplicate_sessions(pipeline):
    frame = pd.DataFrame(
        {
            "session_id": [1, 1],
            "hour": [9, 10],
        }
    )

    with pytest.raises(DataValidationError, match="duplicate"):
        pipeline._validate_result(frame)
