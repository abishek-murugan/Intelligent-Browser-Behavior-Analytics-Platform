"""
Clustering feature preprocessing.

Selects the numeric feature set used for session clustering, scales
it with a StandardScaler and fits an optional PCA projector (used for
2D/3D visualization and as a compact latent representation).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.exceptions import (
    DataValidationError,
    FileReadError,
    FileWriteError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClusteringPreprocessor:
    """Select, scale and optionally reduce session clustering features."""

    def __init__(
        self,
        exclude_columns: list[str] | None = None,
        pca_variance_ratio: float | None = 0.95,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize the preprocessor.

        Parameters
        ----------
        exclude_columns:
            Columns to exclude from the feature matrix even when
            numeric (identifiers, timestamps, categoricals).

        pca_variance_ratio:
            Target cumulative explained variance for the PCA projector.
            Pass ``None`` to skip PCA fitting.

        random_state:
            Random seed used by the PCA solver.
        """

        self.exclude_columns = set(exclude_columns or [])

        self.pca_variance_ratio = pca_variance_ratio

        self.random_state = random_state

        self.scaler = StandardScaler()

        self.pca: PCA | None = None

        self.feature_columns: list[str] = []

        self.n_components: int | None = None

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> ClusteringPreprocessor:
        """
        Fit the scaler and optional PCA on the session features.

        Parameters
        ----------
        dataframe:
            Gold session features dataset.

        Returns
        -------
        ClusteringPreprocessor
            The fitted preprocessor.
        """

        matrix = self._build_matrix(dataframe)

        self.scaler.fit(matrix)

        if self.pca_variance_ratio is not None:
            self.pca = PCA(
                n_components=self.pca_variance_ratio,
                random_state=self.random_state,
            )

            self.pca.fit(self.scaler.transform(matrix))

            self.n_components = self.pca.n_components_

        logger.info(
            "Clustering preprocessor fitted | features=%d | pca_components=%s",
            len(self.feature_columns),
            self.n_components,
        )

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform session features into the scaled clustering matrix.

        Parameters
        ----------
        dataframe:
            Session features dataset.

        Returns
        -------
        np.ndarray
            Scaled feature matrix of shape (n_sessions, n_features).
        """

        matrix = self._build_matrix(dataframe)

        return self.scaler.transform(matrix)

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Fit the preprocessor and return the scaled matrix.

        Parameters
        ----------
        dataframe:
            Session features dataset.

        Returns
        -------
        np.ndarray
            Scaled feature matrix of shape (n_sessions, n_features).
        """

        return self.fit(dataframe).transform(dataframe)

    def project(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Project the scaled matrix into the PCA latent space.

        Parameters
        ----------
        matrix:
            Scaled feature matrix.

        Returns
        -------
        np.ndarray
            PCA coordinates of shape (n_sessions, n_components).
        """

        if self.pca is None:
            raise DataValidationError(
                "PCA projector not fitted. Configure pca_variance_ratio to enable projection."
            )

        return self.pca.transform(matrix)

    def save(
        self,
        output_path: str | Path,
    ) -> Path:
        """
        Persist the fitted preprocessor to disk.

        Parameters
        ----------
        output_path:
            Destination path for the pickle artifact.

        Returns
        -------
        Path
            Path to the saved artifact.
        """

        path = Path(output_path).expanduser()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            with path.open("wb") as file:
                pickle.dump(
                    {
                        "scaler": self.scaler,
                        "pca": self.pca,
                        "feature_columns": self.feature_columns,
                        "n_components": self.n_components,
                        "exclude_columns": sorted(self.exclude_columns),
                    },
                    file,
                )

        except OSError as exc:
            raise FileWriteError(f"Failed to save clustering preprocessor: {path}") from exc

        logger.info("Clustering preprocessor saved: %s", path)

        return path

    @classmethod
    def load(
        cls,
        input_path: str | Path,
    ) -> ClusteringPreprocessor:
        """
        Load a persisted preprocessor from disk.

        Parameters
        ----------
        input_path:
            Path to the pickle artifact.

        Returns
        -------
        ClusteringPreprocessor
            The loaded preprocessor.
        """

        path = Path(input_path).expanduser()

        if not path.is_file():
            raise FileReadError(f"Clustering preprocessor artifact not found: {path}")

        with path.open("rb") as file:
            payload = pickle.load(file)

        instance = cls(
            exclude_columns=payload["exclude_columns"],
            pca_variance_ratio=(None if payload["pca"] is None else 0.95),
            random_state=payload["scaler"].get_params().get("random_state"),
        )

        instance.scaler = payload["scaler"]
        instance.pca = payload["pca"]
        instance.feature_columns = payload["feature_columns"]
        instance.n_components = payload["n_components"]

        logger.info("Clustering preprocessor loaded: %s", path)

        return instance

    def _build_matrix(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Build the numeric feature matrix for clustering.
        """

        if dataframe.empty:
            raise DataValidationError("Session features dataset is empty.")

        numeric_columns = [
            column
            for column in dataframe.columns
            if column not in self.exclude_columns
            and pd.api.types.is_numeric_dtype(dataframe[column].dtype)
        ]

        if not numeric_columns:
            raise DataValidationError("No numeric clustering features found.")

        matrix = dataframe[numeric_columns].to_numpy(dtype="float64")

        if np.isnan(matrix).any():
            missing_columns = [
                column
                for index, column in enumerate(numeric_columns)
                if np.isnan(matrix[:, index]).any()
            ]

            raise DataValidationError(
                f"Clustering features contain missing values in: {missing_columns}"
            )

        self.feature_columns = numeric_columns

        return matrix
