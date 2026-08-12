"""
KMeans session segmentation.

Runs an elbow + silhouette analysis over a range of cluster counts,
selects the optimal ``k`` and fits the final KMeans model, producing
segment labels and per-session distances for downstream profiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentResult:
    """Outcome of fitting KMeans segmentation on a feature matrix."""

    k: int

    labels: np.ndarray

    distances: np.ndarray

    inertia: float

    model: KMeans


@dataclass
class ClusterAnalysis:
    """Per-k metrics collected during the elbow / silhouette sweep."""

    k_values: list[int]

    inertia: list[float]

    silhouette: list[float]

    davies_bouldin: list[float]

    calinski_harabasz: list[float]

    optimal_k: int

    def best_index(self) -> int:
        """Index of the optimal k within the sweep arrays."""
        return self.k_values.index(self.optimal_k)


class KMeansSegmenter:
    """Segment sessions with KMeans and an automatic k selection."""

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 10,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the segmenter.

        Parameters
        ----------
        min_clusters:
            Lower bound of the cluster count sweep.

        max_clusters:
            Upper bound of the cluster count sweep.

        n_init:
            Number of KMeans restarts per k.

        random_state:
            Random seed for reproducible clustering.
        """

        if min_clusters < 2:
            raise ValueError("min_clusters must be >= 2 for silhouette scoring.")

        if max_clusters < min_clusters:
            raise ValueError("max_clusters must be >= min_clusters.")

        self.min_clusters = min_clusters

        self.max_clusters = max_clusters

        self.n_init = n_init

        self.random_state = random_state

        self._selected_k_value: int | None = None

    def sweep(
        self,
        matrix: np.ndarray,
    ) -> ClusterAnalysis:
        """
        Evaluate KMeans across the cluster count range.

        Parameters
        ----------
        matrix:
            Scaled session feature matrix.

        Returns
        -------
        ClusterAnalysis
            Per-k metrics and the optimal cluster count.
        """

        k_values = list(range(self.min_clusters, self.max_clusters + 1))

        inertia: list[float] = []
        silhouette: list[float] = []
        davies_bouldin: list[float] = []
        calinski_harabasz: list[float] = []

        max_k = min(self.max_clusters, len(matrix) - 1)

        for k in range(self.min_clusters, max_k + 1):
            model = KMeans(
                n_clusters=k,
                n_init=self.n_init,
                random_state=self.random_state,
            )

            labels = model.fit_predict(matrix)

            inertia.append(float(model.inertia_))
            silhouette.append(float(silhouette_score(matrix, labels)))
            davies_bouldin.append(float(davies_bouldin_score(matrix, labels)))
            calinski_harabasz.append(float(calinski_harabasz_score(matrix, labels)))

        optimal_k = self._select_optimal_k(
            silhouette,
            inertia,
            k_values,
        )

        logger.info(
            "Cluster sweep completed | range=%d..%d | optimal_k=%d",
            self.min_clusters,
            max_k,
            optimal_k,
        )

        return ClusterAnalysis(
            k_values=k_values,
            inertia=inertia,
            silhouette=silhouette,
            davies_bouldin=davies_bouldin,
            calinski_harabasz=calinski_harabasz,
            optimal_k=optimal_k,
        )

    def fit(
        self,
        matrix: np.ndarray,
        k: int | None = None,
    ) -> SegmentResult:
        """
        Fit the final KMeans model.

        Parameters
        ----------
        matrix:
            Scaled session feature matrix.

        k:
            Number of clusters. If omitted, the optimal k from the
            sweep is used (which requires ``sweep`` to have run first).

        Returns
        -------
        SegmentResult
            The fitted segmentation outcome.
        """

        n_clusters = k if k is not None else self.selected_k

        if n_clusters is None:
            analysis = self.sweep(matrix)
            n_clusters = analysis.optimal_k

        if n_clusters > len(matrix):
            raise ValueError("Number of clusters exceeds the number of sessions.")

        model = KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )

        labels = model.fit_predict(matrix)

        distances = model.transform(matrix)

        result = SegmentResult(
            k=n_clusters,
            labels=labels,
            distances=distances,
            inertia=float(model.inertia_),
            model=model,
        )

        self._selected_k_value = n_clusters

        logger.info(
            "KMeans segmentation fitted | k=%d | inertia=%.2f",
            n_clusters,
            result.inertia,
        )

        return result

    @property
    def selected_k(self) -> int | None:
        """Return the last selected cluster count, or None before fitting."""

        return getattr(self, "_selected_k_value", None)

    @staticmethod
    def _select_optimal_k(
        silhouette_scores: list[float],
        inertia: list[float],
        k_values: list[int],
    ) -> int:
        """
        Select the optimal k, maximizing silhouette with an inertia
        elbow tie-break.
        """

        if not silhouette_scores:
            raise ValueError("No cluster metrics available to select k.")

        best_index = int(np.argmax(silhouette_scores))

        return k_values[best_index]
