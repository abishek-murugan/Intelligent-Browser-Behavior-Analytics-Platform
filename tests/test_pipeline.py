"""Tests for the end-to-end pipeline orchestrator."""

from __future__ import annotations

import pandas as pd

import src.pipeline as pipeline


def _sample_features(n_sessions: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": list(range(n_sessions)),
            "session_start": pd.date_range("2026-06-01", periods=n_sessions, freq="h"),
            "session_end": pd.date_range("2026-06-01", periods=n_sessions, freq="h")
            + pd.Timedelta("30 minutes"),
            "duration_minutes": [30.0] * n_sessions,
            "event_count": [5] * n_sessions,
            "dominant_category": ["Social Media"] * n_sessions,
        }
    )


def test_publish_gold_features_writes_gold_dataset(tmp_path, monkeypatch):
    features = _sample_features()
    gold_dir = tmp_path / "gold"
    silver_path = tmp_path / "silver"
    silver_path.mkdir(parents=True)

    features.to_parquet(silver_path / "session_features.parquet", index=False)

    monkeypatch.setattr(
        pipeline,
        "get_paths",
        lambda: {
            "paths": {
                "session_features": str(silver_path / "session_features.parquet"),
                "session_features_gold": str(gold_dir / "session_features.parquet"),
            }
        },
    )

    pipeline._publish_gold_features()

    result = pd.read_parquet(gold_dir / "session_features.parquet")
    assert len(result) == len(features)


def test_run_full_pipeline_visits_every_stage_in_order(monkeypatch):
    order: list[str] = []

    def _track(name: str):
        def _wrap(*args, **kwargs):
            order.append(name)

        return _wrap

    monkeypatch.setattr(pipeline, "_integrate", _track("_integrate"))
    monkeypatch.setattr(pipeline, "_categorize", _track("_categorize"))
    monkeypatch.setattr(pipeline, "_publish_gold_features", _track("_publish_gold_features"))
    monkeypatch.setattr(pipeline.Sessionizer, "run", _track("Sessionizer"))
    monkeypatch.setattr(pipeline.FeaturePipeline, "run", _track("FeaturePipeline"))
    monkeypatch.setattr(pipeline.ClusteringPipeline, "run", _track("ClusteringPipeline"))
    monkeypatch.setattr(pipeline.DatasetBuilder, "run", _track("DatasetBuilder"))
    monkeypatch.setattr(pipeline.LSTMPipeline, "run", _track("LSTMPipeline"))
    monkeypatch.setattr(pipeline.RecommendationPipeline, "run", _track("RecommendationPipeline"))

    pipeline.run_full_pipeline()

    assert order == [
        "_integrate",
        "_categorize",
        "Sessionizer",
        "FeaturePipeline",
        "_publish_gold_features",
        "ClusteringPipeline",
        "DatasetBuilder",
        "LSTMPipeline",
        "RecommendationPipeline",
    ]
