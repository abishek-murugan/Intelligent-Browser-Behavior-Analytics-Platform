import pandas as pd
import pytest

from src.clustering.profiler import SegmentProfiler
from src.exceptions import DataValidationError


@pytest.fixture
def segmented_frame(session_features):
    dataframe = session_features(60)

    dataframe["segment_id"] = dataframe["session_id"] % 3

    return dataframe


def test_profile_one_row_per_segment(segmented_frame):
    profiler = SegmentProfiler()

    profile = profiler.profile(segmented_frame)

    assert len(profile) == 3

    assert set(profile["segment_id"]) == {0, 1, 2}

    assert "session_count" in profile.columns
    assert "share_pct" in profile.columns
    assert "dominant_category" in profile.columns

    assert profile["session_count"].sum() == 60
    assert pytest.approx(profile["share_pct"].sum(), rel=1e-2) == 100.0


def test_profile_missing_segment_column_raises(session_features):
    dataframe = session_features(30)

    profiler = SegmentProfiler()

    with pytest.raises(DataValidationError, match="not found"):
        profiler.profile(dataframe)


def test_profile_missing_category_raises(segmented_frame):
    dataframe = segmented_frame.drop(columns=["dominant_category"])

    with pytest.raises(DataValidationError, match="dominant_category"):
        SegmentProfiler().profile(dataframe)


def test_save_writes_csv(segmented_frame, tmp_path):
    profiler = SegmentProfiler()

    profile = profiler.profile(segmented_frame)

    output = tmp_path / "nested" / "profile.csv"

    saved = profiler.save(profile, output)

    assert saved == output

    loaded = pd.read_csv(output)

    assert list(loaded.columns) == list(profile.columns)


def test_dominant_feature_importances(segmented_frame):
    profiler = SegmentProfiler()

    profile = profiler.profile(segmented_frame)

    importances = profiler.dominant_feature_importances(profile, top_n=3)

    assert set(importances.keys()) == {0, 1, 2}

    for segment, ranked in importances.items():
        assert len(ranked) == 3

        for feature, deviation in ranked:
            assert feature in profile.columns
            assert isinstance(deviation, float)
