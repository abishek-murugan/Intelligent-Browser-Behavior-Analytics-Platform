import os

import mlflow
import pandas as pd
import pytest

from src.clustering.pipeline import ClusteringPipeline
from src.clustering.preprocessor import ClusteringPreprocessor
from src.clustering.segmenter import KMeansSegmenter
from tests.clustering_test_data import EXCLUDED


def _pipeline(tmp_path, track_mlflow=False, n_sessions=90):
    dataframe_path = tmp_path / "input.parquet"

    from tests.clustering_test_data import build_features

    build_features(n_sessions).to_parquet(dataframe_path, index=False)

    return ClusteringPipeline(
        input_path=dataframe_path,
        output_path=tmp_path / "segments.parquet",
        report_dir=tmp_path / "reports",
        preprocessor=ClusteringPreprocessor(
            exclude_columns=EXCLUDED,
            pca_variance_ratio=0.95,
            random_state=42,
        ),
        segmenter=KMeansSegmenter(
            min_clusters=2,
            max_clusters=5,
            random_state=42,
        ),
        track_mlflow=track_mlflow,
    )


def test_pipeline_run(tmp_path):
    pipeline = _pipeline(tmp_path)

    result = pipeline.run()

    segments = result["segments"]
    profile = result["profile"]
    plots = result["plots"]

    assert isinstance(segments, pd.DataFrame)
    assert {"session_id", "segment_id", "distance_to_centroid"} <= set(segments.columns)
    assert segments["segment_id"].nunique() == result["optimal_k"]

    assert len(profile) == result["optimal_k"]
    assert len(plots) >= 3

    saved = pd.read_parquet(tmp_path / "segments.parquet")

    pd.testing.assert_frame_equal(saved, segments)

    for name in ("elbow.png", "silhouette.png", "pca_segments.png", "profile_heatmap.png"):
        assert (tmp_path / "reports" / name).is_file(), name

    assert (tmp_path / "reports" / "segment_profile.csv").is_file()


def test_pipeline_logs_mlflow_run(tmp_path):
    os.environ["MLFLOW_TRACKING_URI"] = str(tmp_path / "mlruns")

    pipeline = _pipeline(tmp_path, track_mlflow=True)

    pipeline.run()

    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))

    experiment = mlflow.get_experiment_by_name("browser-behavior-clustering")

    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
    )

    assert len(runs) == 1

    assert runs[0].data.metrics.get("silhouette_score") is not None
    assert runs[0].data.params.get("k") is not None


def test_pipeline_missing_input_raises(tmp_path):
    pipeline = ClusteringPipeline(
        input_path=tmp_path / "missing.parquet",
        track_mlflow=False,
    )

    with pytest.raises(Exception):
        pipeline.run()
