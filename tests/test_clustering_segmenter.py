import numpy as np
import pytest

from src.clustering.preprocessor import ClusteringPreprocessor
from src.clustering.segmenter import KMeansSegmenter
from tests.clustering_test_data import EXCLUDED


def _scaled_matrix(n_sessions=90, seed=42):
    from tests.clustering_test_data import build_features

    dataframe = build_features(n_sessions, seed=seed)

    return ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=None,
    ).fit_transform(dataframe)


def test_sweep_returns_analysis(session_features):
    dataframe = session_features(60)

    preprocessor = ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=None,
    )

    matrix = preprocessor.fit_transform(dataframe)

    segmenter = KMeansSegmenter(
        min_clusters=2,
        max_clusters=6,
    )

    analysis = segmenter.sweep(matrix)

    assert analysis.k_values == [2, 3, 4, 5, 6]
    assert len(analysis.inertia) == 5
    assert len(analysis.silhouette) == 5
    assert len(analysis.davies_bouldin) == 5
    assert len(analysis.calinski_harabasz) == 5
    assert analysis.optimal_k in analysis.k_values


def test_optimal_k_recovered_for_blobs():
    matrix = _scaled_matrix(n_sessions=150)

    segmenter = KMeansSegmenter(
        min_clusters=2,
        max_clusters=6,
        random_state=42,
    )

    analysis = segmenter.sweep(matrix)

    assert analysis.optimal_k == 3


def test_fit_returns_segment_result(session_features):
    matrix = _scaled_matrix(n_sessions=60)

    segmenter = KMeansSegmenter(
        min_clusters=2,
        max_clusters=5,
        random_state=7,
    )

    result = segmenter.fit(matrix, k=3)

    assert result.k == 3
    assert result.labels.shape == (60,)
    assert set(np.unique(result.labels)) == {0, 1, 2}
    assert result.distances.shape == (60, 3)
    assert result.inertia > 0
    assert result.model.n_clusters == 3
    assert segmenter.selected_k == 3


def test_fit_without_k_runs_sweep():
    matrix = _scaled_matrix(n_sessions=90)

    segmenter = KMeansSegmenter(
        min_clusters=2,
        max_clusters=5,
        random_state=42,
    )

    result = segmenter.fit(matrix)

    assert result.k in (2, 3, 4, 5)


def test_min_clusters_must_be_at_least_two():
    with pytest.raises(ValueError):
        KMeansSegmenter(min_clusters=1)


def test_max_clusters_below_min_raises():
    with pytest.raises(ValueError):
        KMeansSegmenter(min_clusters=4, max_clusters=3)
