import numpy as np
import pandas as pd
import pytest

from src.clustering.preprocessor import ClusteringPreprocessor
from src.exceptions import DataValidationError
from tests.clustering_test_data import EXCLUDED, NUMERIC_FEATURES


def test_fit_transform_selects_numeric_features(session_features):
    dataframe = session_features(60)

    preprocessor = ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=None,
    )

    matrix = preprocessor.fit_transform(dataframe)

    assert matrix.shape[0] == 60

    assert set(preprocessor.feature_columns).issubset(NUMERIC_FEATURES)

    for column in EXCLUDED:
        assert column not in preprocessor.feature_columns

    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)

    assert np.allclose(means, 0, atol=1e-9)
    assert np.allclose(stds, 1, atol=1e-6)


def test_pca_projection_shape(session_features):
    dataframe = session_features(60)

    preprocessor = ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=0.95,
    )

    matrix = preprocessor.fit_transform(dataframe)

    projected = preprocessor.project(matrix)

    assert preprocessor.n_components is not None
    assert projected.shape == (60, preprocessor.n_components)
    assert 0 < preprocessor.n_components <= len(preprocessor.feature_columns)


def test_project_without_pca_raises(session_features):
    dataframe = session_features(20)

    preprocessor = ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=None,
    )

    matrix = preprocessor.fit_transform(dataframe)

    with pytest.raises(DataValidationError):
        preprocessor.project(matrix)


def test_save_load_roundtrip(session_features, tmp_path):
    dataframe = session_features(40)

    preprocessor = ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=0.95,
        random_state=7,
    )

    preprocessor.fit(dataframe)

    artifact = tmp_path / "nested" / "preprocessor.pkl"

    preprocessor.save(artifact)

    loaded = ClusteringPreprocessor.load(artifact)

    assert loaded.feature_columns == preprocessor.feature_columns
    assert loaded.n_components == preprocessor.n_components

    original = preprocessor.transform(dataframe)
    restored = loaded.transform(dataframe)

    np.testing.assert_allclose(original, restored)


def test_empty_dataframe_raises():
    preprocessor = ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=None,
    )

    with pytest.raises(DataValidationError):
        preprocessor.fit_transform(pd.DataFrame())


def test_missing_values_raise(session_features):
    dataframe = session_features(30)

    dataframe.loc[3, "avg_used_mb"] = np.nan

    preprocessor = ClusteringPreprocessor(
        exclude_columns=EXCLUDED,
        pca_variance_ratio=None,
    )

    with pytest.raises(DataValidationError, match="missing values"):
        preprocessor.fit_transform(dataframe)
