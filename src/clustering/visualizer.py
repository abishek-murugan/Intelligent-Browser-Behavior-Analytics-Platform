"""
Clustering visualizations.

Renders the elbow and silhouette curves from the k sweep, the PCA
scatter of sessions colored by segment, and a segment profile
heatmap. All plots are saved as PNG files under the clustering
report directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clustering.segmenter import ClusterAnalysis
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_elbow(
    analysis: ClusterAnalysis,
    output_path: str | Path,
) -> Path:
    """
    Plot the KMeans inertia (elbow) curve.
    """

    path = Path(output_path).expanduser()

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        analysis.k_values,
        analysis.inertia,
        marker="o",
        linewidth=2,
    )

    axis.axvline(
        analysis.optimal_k,
        color="red",
        linestyle="--",
        label=f"optimal k={analysis.optimal_k}",
    )

    axis.set_title("KMeans Elbow Curve")
    axis.set_xlabel("Number of clusters (k)")
    axis.set_ylabel("Inertia (within-cluster SSE)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("Elbow plot saved: %s", path)

    return path


def plot_silhouette_curve(
    analysis: ClusterAnalysis,
    output_path: str | Path,
) -> Path:
    """
    Plot the silhouette score curve across k.
    """

    path = Path(output_path).expanduser()

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        analysis.k_values,
        analysis.silhouette,
        marker="o",
        linewidth=2,
    )

    axis.axvline(
        analysis.optimal_k,
        color="red",
        linestyle="--",
        label=f"optimal k={analysis.optimal_k}",
    )

    axis.set_title("KMeans Silhouette Score")
    axis.set_xlabel("Number of clusters (k)")
    axis.set_ylabel("Silhouette score")
    axis.grid(True, alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("Silhouette plot saved: %s", path)

    return path


def plot_pca_scatter(
    coordinates: np.ndarray,
    labels: np.ndarray,
    k: int,
    output_path: str | Path,
    title: str = "Sessions Projected onto First Two PCA Components",
) -> Path:
    """
    Plot the 2D PCA scatter of sessions colored by segment.
    """

    path = Path(output_path).expanduser()

    path.parent.mkdir(parents=True, exist_ok=True)

    if coordinates.shape[1] < 2:
        logger.warning("PCA projection has fewer than 2 components; skipping scatter plot.")

        return path

    fig, axis = plt.subplots(figsize=(9, 6))

    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        s=30,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.3,
    )

    axis.set_title(title)
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.grid(True, alpha=0.3)

    fig.colorbar(
        scatter,
        ax=axis,
        label="segment",
        ticks=range(k),
    )

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("PCA scatter saved: %s", path)

    return path


def plot_profile_heatmap(
    profile: pd.DataFrame,
    feature_columns: list[str],
    output_path: str | Path,
) -> Path:
    """
    Plot a heatmap of per-segment mean feature values (z-scored rows).
    """

    path = Path(output_path).expanduser()

    path.parent.mkdir(parents=True, exist_ok=True)

    available = [column for column in feature_columns if column in profile.columns]

    if not available:
        logger.warning("No profile features available; skipping heatmap.")

        return path

    values = profile.set_index("segment_id")[available]

    z_scores = values.apply(
        lambda row: (row - row.mean()) / (row.std() + 1e-12),
        axis=1,
    )

    fig, axis = plt.subplots(figsize=(max(10, len(available) * 0.5), max(3, len(profile) * 0.8)))

    heatmap = axis.imshow(
        z_scores.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
    )

    axis.set_xticks(range(len(available)))
    axis.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(z_scores)))
    axis.set_yticklabels(
        [f"Segment {index}" for index in z_scores.index],
        fontsize=9,
    )

    axis.set_title("Segment Profile Heatmap (row z-score)")
    fig.colorbar(heatmap, ax=axis, label="z-score")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("Profile heatmap saved: %s", path)

    return path
